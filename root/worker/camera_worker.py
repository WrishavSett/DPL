"""
Camera worker process entrypoint.

One of these runs per configured camera, spawned by app/camera_manager.py
via multiprocessing.Process(target=run_camera_worker, args=(...)). It owns
nothing the dashboard process needs to reach into directly — all
communication happens through the shared primitives passed in:

  frame_store   : dict-like, frame_store[camera_id] = (jpeg_bytes, ts_float)
  status_store  : dict-like, status_store[camera_id] = {status, last_heartbeat, fps, error}
  event_queue   : multiprocessing.Queue, receives worker.events.CountEvent
  stop_event    : multiprocessing.Event, set() requests graceful shutdown

These shared objects are constructed by app/shared_state.py; camera_worker
itself only needs them to behave like a dict / queue / event, so it has no
import-time dependency on that module — keeps this process's import graph
small and avoids pulling FastAPI into a process that doesn't need it.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone
from multiprocessing.synchronize import Event as MpEvent
from queue import Full
from typing import Any

import supervision as sv

from app.config import settings
from worker.annotate import annotate_frame, build_labels, encode_jpeg
from worker.events import extract_events
from worker.pipeline import (
    CameraConfig,
    init_line_counter,
    init_stream,
    init_tracker_and_annotators,
    load_model,
    preprocess,
    read_frame,
    reconnect_stream,
    resolve_classes,
    run_counting,
    run_inference,
    run_tracking,
)


def _configure_logging(camera_id: str) -> logging.Logger:
    logger = logging.getLogger(f"worker.{camera_id}")
    logger.setLevel(settings.log_level)
    if not logger.handlers:
        log_path = settings.logs_dir / f"{camera_id}.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s [{camera_id}] %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
    return logger


def _build_camera_config(camera_id: str, cfg: dict[str, Any]) -> CameraConfig:
    count_line = cfg.get("count_line")
    return CameraConfig(
        camera_id=camera_id,
        source=cfg["source"],
        model_path=cfg.get("model_path") or settings.model_path,
        device=cfg.get("device", settings.device),
        target_w=cfg.get("target_w", settings.target_w),
        target_h=cfg.get("target_h", settings.target_h),
        conf_threshold=cfg.get("conf_threshold", settings.conf_threshold),
        iou_threshold=cfg.get("iou_threshold", settings.iou_threshold),
        lost_track_buffer=cfg.get("lost_track_buffer", settings.lost_track_buffer),
        classes=cfg.get("classes"),
        count_line=tuple(count_line) if count_line else None,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_status(status_store, camera_id: str, **fields: Any) -> None:
    """Read-modify-write the whole status dict for this camera.

    Required because mutating a nested dict in place wouldn't propagate
    through a multiprocessing.Manager DictProxy — reassignment is what
    actually syncs the change to the other processes.
    """
    current = dict(status_store.get(camera_id, {}))
    current.update(fields)
    status_store[camera_id] = current


def run_camera_worker(
    camera_id: str,
    config: dict[str, Any],
    frame_store: dict,
    status_store: dict,
    event_queue,
    stop_event: MpEvent,
) -> None:
    """Entrypoint executed inside the child process for one camera."""
    logger = _configure_logging(camera_id)

    def _handle_sigterm(_signum, _frame) -> None:
        logger.info("SIGTERM received, shutting down")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    cam_cfg = _build_camera_config(camera_id, config)
    _set_status(status_store, camera_id, status="starting", error=None)

    cap = None
    try:
        cap = init_stream(cam_cfg.source)
        model = load_model(cam_cfg.model_path, cam_cfg.device)
        class_ids = resolve_classes(model, cam_cfg.classes)
        tracker, box_ann, label_ann = init_tracker_and_annotators(
            cam_cfg.conf_threshold, cam_cfg.lost_track_buffer
        )
        line_zone, line_ann = init_line_counter(cam_cfg.count_line)
    except Exception as exc:  # noqa: BLE001 - any setup failure marks the camera crashed
        logger.exception("Setup failed")
        _set_status(status_store, camera_id, status="crashed", error=str(exc))
        if cap is not None:
            cap.release()
        return

    fps_monitor = sv.FPSMonitor()
    _set_status(status_store, camera_id, status="running", error=None, last_heartbeat=_utc_now_iso())
    last_heartbeat = time.monotonic()
    logger.info("Worker running")

    try:
        while not stop_event.is_set():
            frame = read_frame(cap)
            if frame is None:
                try:
                    cap.release()
                    cap = reconnect_stream(cam_cfg.source)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Reconnect failed: %s", exc)
                    _set_status(status_store, camera_id, status="crashed", error=str(exc))
                    time.sleep(2)
                continue

            try:
                processed = preprocess(frame, cam_cfg.target_w, cam_cfg.target_h)
                detections = run_inference(
                    model, processed, cam_cfg.conf_threshold, cam_cfg.iou_threshold,
                    cam_cfg.device, classes=class_ids,
                )
                tracked = run_tracking(tracker, detections)
                counting_result = run_counting(tracked, line_zone)

                for event in extract_events(camera_id, tracked, counting_result, model.names):
                    try:
                        event_queue.put_nowait(event)
                    except Full:
                        logger.warning("Event queue full, dropping event")

                fps_monitor.tick()
                labels = build_labels(tracked, model.names)
                annotated = annotate_frame(
                    processed, camera_id, tracked, labels, box_ann, label_ann,
                    fps_monitor.fps, line_zone, line_ann, counting_result,
                    class_ids, model.names,
                )
                frame_store[camera_id] = (
                    encode_jpeg(annotated, settings.frame_jpeg_quality),
                    time.time(),
                )

            except Exception as exc:  # noqa: BLE001 - keep the worker alive on per-frame errors
                logger.exception("Frame processing error")
                _set_status(status_store, camera_id, error=str(exc))

            now = time.monotonic()
            if now - last_heartbeat >= settings.worker_heartbeat_interval:
                _set_status(
                    status_store, camera_id,
                    status="running", last_heartbeat=_utc_now_iso(), fps=round(fps_monitor.fps, 1),
                )
                last_heartbeat = now

    finally:
        if cap is not None:
            cap.release()
        _set_status(status_store, camera_id, status="stopped", last_heartbeat=_utc_now_iso())
        logger.info("Worker stopped")