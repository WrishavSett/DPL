"""
Tests for app/db_writer.py.

Uses a plain queue.Queue as a stand-in for the multiprocessing Manager
queue — DBWriter only ever calls .get(timeout=...) on it, which
queue.Queue satisfies, so there's no need to spin up a real
multiprocessing.Manager for these tests.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import app.shared_state as shared_state_module
from app.database import get_db, init_db
from app.db_writer import DBWriter


@dataclass
class FakeEvent:
    camera_id: str
    track_id: int
    class_id: int
    class_name: str
    direction: str
    timestamp: str


@pytest.fixture
def writer(tmp_path, monkeypatch):
    from app import config as config_module

    monkeypatch.setattr(config_module.settings, "db_path", tmp_path / "test.db")
    init_db()

    fake_state = SimpleNamespace(event_queue=queue.Queue())
    shared_state_module.shared_state = fake_state

    db_writer = DBWriter()
    yield db_writer
    db_writer.stop()


def _events_in_db():
    with get_db() as conn:
        return conn.execute("SELECT * FROM events ORDER BY id").fetchall()


def test_writer_flushes_single_event(writer):
    writer.start()
    shared_state_module.shared_state.event_queue.put(
        FakeEvent("cam_1", 1, 0, "person", "in", "2026-06-12T12:00:00.000000+00:00")
    )

    # Flush happens on a timer (BATCH_MAX_WAIT_S), so give it a moment.
    time.sleep(1.5)

    rows = _events_in_db()
    assert len(rows) == 1
    assert rows[0]["camera_id"] == "cam_1"
    assert rows[0]["direction"] == "in"


def test_writer_batches_multiple_events(writer):
    writer.start()
    for i in range(5):
        shared_state_module.shared_state.event_queue.put(
            FakeEvent("cam_1", i, 0, "person", "in", f"2026-06-12T12:00:0{i}.000000+00:00")
        )

    time.sleep(1.5)

    rows = _events_in_db()
    assert len(rows) == 5


def test_writer_flushes_remaining_batch_on_stop(writer):
    writer.start()
    shared_state_module.shared_state.event_queue.put(
        FakeEvent("cam_1", 1, 0, "car", "out", "2026-06-12T13:00:00.000000+00:00")
    )
    writer.stop()  # should flush whatever's left before the thread exits

    rows = _events_in_db()
    assert len(rows) == 1
    assert rows[0]["class_name"] == "car"