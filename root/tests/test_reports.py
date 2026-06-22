"""
Tests for app/scheduler.py's hourly report generation.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.database import get_db, init_db
from app.scheduler import ISO_FMT, _hour_boundaries, generate_hourly_report


@pytest.fixture
def report_env(tmp_path, monkeypatch):
    from app import config as config_module

    monkeypatch.setattr(config_module.settings, "db_path", tmp_path / "test.db")
    monkeypatch.setattr(config_module.settings, "reports_dir", tmp_path)
    monkeypatch.setattr(config_module.settings, "scheduler_timezone", "UTC")
    init_db()
    return config_module.settings


def test_hour_boundaries_truncate_to_the_hour():
    now = datetime(2026, 6, 12, 13, 47, 22)
    start, end = _hour_boundaries(now)
    assert end == datetime(2026, 6, 12, 13, 0, 0)
    assert start == datetime(2026, 6, 12, 12, 0, 0)


def _insert_event(camera_id: str, class_name: str, direction: str, ts: datetime) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO cameras (camera_id, name, source) VALUES (?, ?, ?) "
            "ON CONFLICT(camera_id) DO NOTHING",
            (camera_id, camera_id, "test-source"),
        )
        conn.execute(
            "INSERT INTO events (camera_id, track_id, class_id, class_name, direction, timestamp) "
            "VALUES (?, 1, 0, ?, ?, ?)",
            (camera_id, class_name, direction, ts.strftime(ISO_FMT)),
        )
        conn.commit()


def test_generate_hourly_report_aggregates_only_the_completed_hour(report_env):
    tz = ZoneInfo("UTC")
    now_local = datetime.now(tz)
    period_start, period_end = _hour_boundaries(now_local)

    # Inside the window — should be counted.
    _insert_event("cam_1", "person", "in", period_start + timedelta(minutes=5))
    _insert_event("cam_1", "person", "in", period_start + timedelta(minutes=10))
    _insert_event("cam_1", "person", "out", period_start + timedelta(minutes=15))

    # Outside the window — should NOT be counted.
    _insert_event("cam_1", "person", "in", period_end + timedelta(minutes=5))
    _insert_event("cam_1", "person", "in", period_start - timedelta(minutes=5))

    generate_hourly_report()

    with get_db() as conn:
        report_row = conn.execute("SELECT * FROM reports ORDER BY id DESC LIMIT 1").fetchone()
    assert report_row is not None

    csv_path = report_env.reports_dir / report_row["filename"]
    assert csv_path.is_file()

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    person_in = next(r for r in rows if r["class_name"] == "person" and r["direction"] == "in")
    person_out = next(r for r in rows if r["class_name"] == "person" and r["direction"] == "out")
    assert person_in["count"] == "2"
    assert person_out["count"] == "1"


def test_report_filename_matches_naming_convention(report_env):
    tz = ZoneInfo("UTC")
    now_local = datetime.now(tz)
    period_start, period_end = _hour_boundaries(now_local)
    _insert_event("cam_1", "person", "in", period_start + timedelta(minutes=1))

    generate_hourly_report()

    with get_db() as conn:
        report_row = conn.execute("SELECT * FROM reports ORDER BY id DESC LIMIT 1").fetchone()

    expected_start = period_start.strftime("%Y%m%d_%H%M%S")
    expected_end = (period_end - timedelta(seconds=1)).strftime("%H%M%S")
    assert report_row["filename"] == f"Report_{expected_start}_{expected_end}.csv"