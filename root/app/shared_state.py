"""
Shared multiprocessing primitives connecting the FastAPI process to camera
worker processes.

A single multiprocessing.Manager owns the proxied frame/status dicts and
the event queue, so any process holding a reference sees the same data.
Each camera additionally gets its own plain multiprocessing.Event for stop
signalling — that one doesn't need to go through the manager since it's
only ever shared between the main process and the one worker it spawned.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SharedState:
    """Created once at app startup; used by camera_manager, db_writer, and the streaming router."""

    manager: Any                  # multiprocessing.managers.SyncManager
    frame_store: Any              # camera_id -> (jpeg_bytes: bytes, ts: float)
    status_store: Any             # camera_id -> {status, last_heartbeat, fps, error}
    event_queue: Any              # holds worker.events.CountEvent objects
    stop_events: dict[str, Any]   # camera_id -> multiprocessing.Event

    @classmethod
    def create(cls) -> "SharedState":
        manager = multiprocessing.Manager()
        return cls(
            manager=manager,
            frame_store=manager.dict(),
            status_store=manager.dict(),
            event_queue=manager.Queue(maxsize=10_000),
            stop_events={},
        )

    def new_stop_event(self, camera_id: str) -> Any:
        """Create (or replace) the stop event for a camera, passed to its worker on spawn."""
        event = multiprocessing.Event()
        self.stop_events[camera_id] = event
        return event

    def get_stop_event(self, camera_id: str) -> Optional[Any]:
        return self.stop_events.get(camera_id)

    def get_status(self, camera_id: str) -> dict:
        return dict(self.status_store.get(camera_id, {"status": "stopped"}))

    def get_frame(self, camera_id: str) -> Optional[tuple[bytes, float]]:
        return self.frame_store.get(camera_id)

    def clear_camera(self, camera_id: str) -> None:
        """Drop all shared state for a camera. Called on remove, not on a plain stop."""
        self.frame_store.pop(camera_id, None)
        self.status_store.pop(camera_id, None)
        self.stop_events.pop(camera_id, None)


# Module-level singleton, initialised once during FastAPI startup (see app/main.py).
shared_state: Optional[SharedState] = None


def init_shared_state() -> SharedState:
    global shared_state
    if shared_state is None:
        shared_state = SharedState.create()
    return shared_state


def get_shared_state() -> SharedState:
    if shared_state is None:
        raise RuntimeError(
            "SharedState not initialised — call init_shared_state() during app startup first"
        )
    return shared_state