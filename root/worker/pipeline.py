"""
Vision pipeline: stream IO, model loading, tracking, and line counting.

Refactored from the original count.py baseline. Differences from that
baseline:
  - No global constants or argparse — every parameter comes from a
    CameraConfig built per-camera from the `cameras` DB row (see
    worker/camera_worker.py).
  - No cv2.imshow / waitKey anywhere — workers are headless. Annotated
    frames are handed off to worker/annotate.py for JPEG encoding instead
    of being displayed.
  - run_counting() now also returns the raw per-frame crossed_in/crossed_out
    masks from LineZone.trigger(), so worker/events.py can turn them into
    persisted event rows rather than relying on the in-memory cumulative
    counters, which reset to zero on every worker restart.
  - print() replaced with the logging module so output from several worker
    processes doesn't interleave unreadably on shared stdout.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Per-camera settings, built from a `cameras` table row."""

    camera_id: str
    source: str | int
    model_path: str
    device: str = "cpu"
    target_w: int = 640
    target_h: int = 480
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    lost_track_buffer: int = 30
    classes: list[str] | None = None                      # None = all classes
    count_line: tuple[int, int, int, int] | None = None   # (x1, y1, x2, y2)


@dataclass
class CountingResult:
    """Output of one frame's counting step."""

    in_count_per_class: dict[int, int] = field(default_factory=dict)
    out_count_per_class: dict[int, int] = field(default_factory=dict)
    crossed_in: np.ndarray | None = None    # bool mask aligned to `tracked`
    crossed_out: np.ndarray | None = None   # bool mask aligned to `tracked`


# 1. Initialize Stream

def init_stream(source: str | int) -> cv2.VideoCapture:
    """
    Open an RTSP stream, local video file, or webcam.
    For RTSP sources, forces the FFmpeg backend and minimises buffer size so
    we always decode the latest frame.
    """
    logger.info("Connecting to stream: %s", source)
    is_rtsp = isinstance(source, str) and source.startswith("rtsp")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if is_rtsp else cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("Stream connected: %s", source)
    return cap


def reconnect_stream(source: str | int, retry_delay: float = 1.0) -> cv2.VideoCapture:
    """Sleep briefly, then attempt to reopen the stream. Raises if it fails."""
    logger.warning("Stream read failed, reconnecting in %.1fs: %s", retry_delay, source)
    time.sleep(retry_delay)
    return init_stream(source)


# 2. Load Model

def load_model(model_path: str, device: str) -> YOLO:
    """Load a YOLO model and move it to the target device."""
    logger.info("Loading model '%s' on %s", model_path, device.upper())
    model = YOLO(model_path)
    model.to(device)
    logger.info("Model ready, %d classes", len(model.names))
    return model


# 2b. Resolve class names

def resolve_classes(model: YOLO, classes: list[str] | None) -> list[int] | None:
    """Convert class-name strings to the integer IDs the model uses."""
    if classes is None:
        return None
    name_to_id = {v: k for k, v in model.names.items()}
    ids = []
    for name in classes:
        if name not in name_to_id:
            raise ValueError(
                f"Class '{name}' not found in model. Available: {list(name_to_id)}"
            )
        ids.append(name_to_id[name])
    return ids


# 3. Initialize ByteTrack Tracker + Supervision Annotators

def init_tracker_and_annotators(conf_threshold: float, lost_track_buffer: int):
    """
    ByteTrack via Supervision.
    Annotators (instantiated once at startup, reused every frame):
      - BoxAnnotator   : draws detection boxes
      - LabelAnnotator : prints tracker-ID + class name + confidence
    """
    tracker = sv.ByteTrack(
        track_activation_threshold=conf_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=0.8,
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    logger.info("ByteTrack tracker initialised")
    return tracker, box_annotator, label_annotator


# 3b. Initialize Line Counter

def init_line_counter(count_line: tuple[int, int, int, int] | None):
    """
    Build a LineZone from a (x1, y1, x2, y2) tuple.

    Returns
    -------
    line_zone : sv.LineZone | None
    line_ann  : sv.LineZoneAnnotator | None
    """
    if count_line is None:
        return None, None

    x1, y1, x2, y2 = count_line
    line_zone = sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2))
    line_ann = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=0.6,
        display_in_count=False,
        display_out_count=False,
    )
    logger.info("LineZone configured: (%d,%d) - (%d,%d)", x1, y1, x2, y2)
    return line_zone, line_ann


# 4 & 5. Read + Preprocess Frames

def read_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Grab the latest frame; return None on failure."""
    ret, frame = cap.read()
    return frame if ret else None


def preprocess(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to the inference resolution."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


# 6. Inference

def run_inference(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    classes: list[int] | None,
) -> sv.Detections:
    """Run YOLO inference and return a Supervision Detections object."""
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        classes=classes,
        verbose=False,
    )[0]
    return sv.Detections.from_ultralytics(results)


# 7. Object Tracking

def run_tracking(tracker: sv.ByteTrack, detections: sv.Detections) -> sv.Detections:
    """Update ByteTrack with current-frame detections."""
    return tracker.update_with_detections(detections)


# 7b. Line Counting

def run_counting(tracked: sv.Detections, line_zone: sv.LineZone | None) -> CountingResult:
    """
    Update the line counter and return both the cumulative dicts (useful for
    a live HUD overlay) and the per-frame crossing masks (the actual source
    of truth fed to worker/events.py for persistence).
    """
    if line_zone is None:
        return CountingResult()
    
    if len(tracked) == 0:
        # Nothing to trigger this frame, but the HUD should still reflect
        # the line zone's real cumulative counts rather than flickering to
        # 0/0 — those counts are untouched and still correct; we just have
        # no new crossings (hence crossed_in/crossed_out stay None) to
        # extract events from.
        return CountingResult(
            in_count_per_class=dict(line_zone.in_count_per_class),
            out_count_per_class=dict(line_zone.out_count_per_class),
        )

    crossed_in, crossed_out = line_zone.trigger(detections=tracked)
    return CountingResult(
        in_count_per_class=dict(line_zone.in_count_per_class),
        out_count_per_class=dict(line_zone.out_count_per_class),
        crossed_in=crossed_in,
        crossed_out=crossed_out,
    )