"""
CameraManager: the only place that spawns, stops, and tracks per-camera
worker processes, and the only place that mutates the `cameras` table's
config/runtime columns. Routers call into this; it never talks raw SQL or
multiprocessing primitives to the outside world.

Live status (running/crashed/fps/heartbeat) lives in the shared status
dict (app/shared_state.py), written by the worker itself every heartbeat
interval. CameraManager only persists status to the DB at well-defined
transition points — start, stop, detected crash, startup reconciliation —
not on every frame.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from app.database import get_db
from app.models import CameraCreate, CameraOut, CameraStatus, CameraUpdate, CountLine
from app.shared_state import get_shared_state
from worker.camera_worker import run_camera_worker

logger = logging.getLogger(__name__)

STOP_JOIN_TIMEOUT = 5.0  # seconds to wait for graceful exit before terminate()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: Optional[str]):
    return json.loads(value) if value else None


class CameraManager:
    def __init__(self) -> None:
        self._processes: dict[str, multiprocessing.Process] = {}

    # ---------- startup ----------

    def reconcile_on_startup(self) -> None:
        """No workers exist yet right after process boot — force DB status to match."""
        with get_db() as conn:
            conn.execute(
                "UPDATE cameras SET status = ?, pid = NULL, updated_at = ? WHERE status != ?",
                (CameraStatus.STOPPED.value, _utc_now_iso(), CameraStatus.DISABLED.value),
            )
            conn.commit()

    def start_all_enabled(self) -> None:
        with get_db() as conn:
            rows = conn.execute("SELECT camera_id FROM cameras WHERE enabled = 1").fetchall()
        for row in rows:
            self.start_camera(row["camera_id"])

    # ---------- CRUD ----------

    def add_camera(self, payload: CameraCreate) -> CameraOut:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO cameras (
                    camera_id, name, source, enabled, classes, count_line,
                    model_path, device, target_w, target_h,
                    conf_threshold, iou_threshold, lost_track_buffer, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.camera_id,
                    payload.name,
                    payload.source,
                    int(payload.enabled),
                    json.dumps(payload.classes) if payload.classes is not None else None,
                    json.dumps(list(payload.count_line.model_dump().values()))
                        if payload.count_line else None,
                    payload.model_path,
                    payload.device,
                    payload.target_w,
                    payload.target_h,
                    payload.conf_threshold,
                    payload.iou_threshold,
                    payload.lost_track_buffer,
                    CameraStatus.STOPPED.value,
                ),
            )
            conn.commit()

        if payload.enabled:
            self.start_camera(payload.camera_id)
        return self.get_camera(payload.camera_id)

    def update_camera(self, camera_id: str, payload: CameraUpdate) -> CameraOut:
        fields = payload.model_dump(exclude_unset=True)
        if not fields:
            return self.get_camera(camera_id)

        columns, values = [], []
        for key, value in fields.items():
            if key == "classes":
                value = json.dumps(value) if value is not None else None
            elif key == "count_line":
                value = json.dumps(list(value.values())) if value is not None else None
            elif key == "enabled":
                value = int(value)
            columns.append(f"{key} = ?")
            values.append(value)
        columns.append("updated_at = ?")
        values.append(_utc_now_iso())
        values.append(camera_id)

        with get_db() as conn:
            conn.execute(f"UPDATE cameras SET {', '.join(columns)} WHERE camera_id = ?", values)
            conn.commit()

        # Config changed — restart the worker so it picks up new settings, if it's running.
        if camera_id in self._processes:
            self.restart_camera(camera_id)
        return self.get_camera(camera_id)

    def remove_camera(self, camera_id: str) -> None:
        self.stop_camera(camera_id)
        with get_db() as conn:
            conn.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
            conn.commit()
        get_shared_state().clear_camera(camera_id)

    # ---------- process control ----------

    def start_camera(self, camera_id: str) -> CameraOut:
        if camera_id in self._processes and self._processes[camera_id].is_alive():
            return self.get_camera(camera_id)

        row = self._get_row(camera_id)
        if row is None:
            raise KeyError(f"Unknown camera_id: {camera_id}")

        config = {
            "source": row["source"],
            "model_path": row["model_path"],
            "device": row["device"],
            "target_w": row["target_w"],
            "target_h": row["target_h"],
            "conf_threshold": row["conf_threshold"],
            "iou_threshold": row["iou_threshold"],
            "lost_track_buffer": row["lost_track_buffer"],
            "classes": _loads(row["classes"]),
            "count_line": _loads(row["count_line"]),
        }

        state = get_shared_state()
        stop_event = state.new_stop_event(camera_id)
        process = multiprocessing.Process(
            target=run_camera_worker,
            args=(camera_id, config, state.frame_store, state.status_store, state.event_queue, stop_event),
            daemon=True,
            name=f"camera-worker-{camera_id}",
        )
        process.start()
        self._processes[camera_id] = process

        self._set_db_status(camera_id, CameraStatus.STARTING, pid=process.pid)
        logger.info("Started camera worker %s (pid=%s)", camera_id, process.pid)
        return self.get_camera(camera_id)

    def stop_camera(self, camera_id: str) -> CameraOut:
        process = self._processes.pop(camera_id, None)
        if process is not None and process.is_alive():
            stop_event = get_shared_state().get_stop_event(camera_id)
            if stop_event is not None:
                stop_event.set()
            process.join(STOP_JOIN_TIMEOUT)
            if process.is_alive():
                logger.warning("Camera %s did not exit gracefully, terminating", camera_id)
                process.terminate()
                process.join(STOP_JOIN_TIMEOUT)

        self._set_db_status(camera_id, CameraStatus.STOPPED, pid=None)
        return self.get_camera(camera_id)

    def restart_camera(self, camera_id: str) -> CameraOut:
        self.stop_camera(camera_id)
        return self.start_camera(camera_id)

    def enable_camera(self, camera_id: str) -> CameraOut:
        with get_db() as conn:
            conn.execute(
                "UPDATE cameras SET enabled = 1, updated_at = ? WHERE camera_id = ?",
                (_utc_now_iso(), camera_id),
            )
            conn.commit()
        return self.start_camera(camera_id)

    def disable_camera(self, camera_id: str) -> CameraOut:
        self.stop_camera(camera_id)
        with get_db() as conn:
            conn.execute(
                "UPDATE cameras SET enabled = 0, status = ?, updated_at = ? WHERE camera_id = ?",
                (CameraStatus.DISABLED.value, _utc_now_iso(), camera_id),
            )
            conn.commit()
        return self.get_camera(camera_id)

    # ---------- health ----------

    def check_health(self, auto_restart: bool = True) -> None:
        """Detect workers that died without going through stop_camera(), mark crashed, optionally restart."""
        for camera_id, process in list(self._processes.items()):
            if process.is_alive():
                continue
            logger.warning("Camera %s process exited unexpectedly", camera_id)
            self._processes.pop(camera_id, None)
            self._set_db_status(camera_id, CameraStatus.CRASHED, pid=None)
            if auto_restart:
                row = self._get_row(camera_id)
                if row is not None and row["enabled"]:
                    self.start_camera(camera_id)

    # ---------- reads ----------

    def get_camera(self, camera_id: str) -> CameraOut:
        row = self._get_row(camera_id)
        if row is None:
            raise KeyError(f"Unknown camera_id: {camera_id}")
        return self._build_camera_out(row)

    def list_cameras(self) -> list[CameraOut]:
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall()
        return [self._build_camera_out(row) for row in rows]

    # ---------- internals ----------

    def _get_row(self, camera_id: str) -> Optional[sqlite3.Row]:
        with get_db() as conn:
            return conn.execute(
                "SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)
            ).fetchone()

    def _set_db_status(self, camera_id: str, status: CameraStatus, pid: Optional[int]) -> None:
        with get_db() as conn:
            conn.execute(
                "UPDATE cameras SET status = ?, pid = ?, updated_at = ? WHERE camera_id = ?",
                (status.value, pid, _utc_now_iso(), camera_id),
            )
            conn.commit()

    def _build_camera_out(self, row: sqlite3.Row) -> CameraOut:
        live = get_shared_state().get_status(row["camera_id"])
        count_line = _loads(row["count_line"])
        camera_id = row["camera_id"]

        return CameraOut(
            camera_id=camera_id,
            name=row["name"],
            source=row["source"],
            enabled=bool(row["enabled"]),
            classes=_loads(row["classes"]),
            count_line=CountLine(x1=count_line[0], y1=count_line[1], x2=count_line[2], y2=count_line[3])
                if count_line else None,
            model_path=row["model_path"],
            device=row["device"],
            target_w=row["target_w"],
            target_h=row["target_h"],
            conf_threshold=row["conf_threshold"],
            iou_threshold=row["iou_threshold"],
            lost_track_buffer=row["lost_track_buffer"],
            status=CameraStatus(live.get("status", row["status"])),
            pid=self._processes[camera_id].pid if camera_id in self._processes else row["pid"],
            last_heartbeat=live.get("last_heartbeat", row["last_heartbeat"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# Module-level singleton — mirrors app/shared_state.py's pattern.
camera_manager = CameraManager()