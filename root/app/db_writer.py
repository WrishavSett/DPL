"""
Drains the shared event queue (worker.events.CountEvent objects produced by
camera workers) into the `events` table.

Runs in a dedicated background thread, not an asyncio task — the queue is
a multiprocessing.Manager proxy, so reading from it blocks the calling
thread; doing that directly on the FastAPI event loop would stall every
request. This is also the single long-lived write connection: camera
workers never open their own SQLite connections, which sidesteps
multi-writer lock contention entirely rather than working around it with
retries.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from app.database import get_connection
from app.shared_state import get_shared_state

logger = logging.getLogger(__name__)

BATCH_MAX_SIZE = 200
BATCH_MAX_WAIT_S = 1.0
QUEUE_GET_TIMEOUT_S = 0.5

INSERT_SQL = """
    INSERT INTO events (camera_id, track_id, class_id, class_name, direction, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
"""


class DBWriter:
    """Owns the single SQLite write connection and the drain thread."""

    def __init__(self) -> None:
        self._conn = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._conn = get_connection()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._thread.start()
        logger.info("DB writer thread started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._conn is not None:
            self._conn.close()
        logger.info("DB writer thread stopped")

    def _run(self) -> None:
        event_queue = get_shared_state().event_queue
        batch: list[tuple] = []
        last_flush = time.monotonic()

        while not self._stop.is_set():
            try:
                event = event_queue.get(timeout=QUEUE_GET_TIMEOUT_S)
                batch.append(self._to_row(event))
            except queue.Empty:
                pass

            now = time.monotonic()
            should_flush = batch and (
                len(batch) >= BATCH_MAX_SIZE or (now - last_flush) >= BATCH_MAX_WAIT_S
            )
            if should_flush:
                self._flush(batch)
                batch = []
                last_flush = now

        if batch:  # final flush on shutdown
            self._flush(batch)

    @staticmethod
    def _to_row(event) -> tuple:
        return (
            event.camera_id,
            event.track_id,
            event.class_id,
            event.class_name,
            event.direction,
            event.timestamp,
        )

    def _flush(self, batch: list[tuple]) -> None:
        try:
            self._conn.executemany(INSERT_SQL, batch)
            self._conn.commit()
            logger.debug("Flushed %d event(s)", len(batch))
        except Exception:
            logger.exception("Failed to flush %d event(s)", len(batch))


# Module-level singleton — mirrors shared_state/camera_manager pattern.
db_writer = DBWriter()