"""
Hourly CSV report generation (Tab 4).

Runs via APScheduler's BackgroundScheduler — a separate thread, not an
asyncio job — since the report query and CSV write are both synchronous
(sqlite3, csv module). Fires at the top of every hour in
SCHEDULER_TIMEZONE and aggregates the just-completed hour's events into
Report_<period_start>_<period_end>.csv, e.g. Report_20260612_120000_125959.csv.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import settings
from app.database import get_db

logger = logging.getLogger(__name__)

# Must match worker/events.py's _utc_now_iso() format exactly so the stored
# TEXT timestamps and these query bounds compare correctly as plain strings.
ISO_FMT = "%Y-%m-%dT%H:%M:%S.%f+00:00"

EVENTS_QUERY = """
    SELECT camera_id, class_name, direction, COUNT(*) AS count
    FROM events
    WHERE timestamp >= ? AND timestamp < ?
    GROUP BY camera_id, class_name, direction
    ORDER BY camera_id, class_name, direction
"""


def _hour_boundaries(now_local: datetime) -> tuple[datetime, datetime]:
    """Return (period_start, period_end) for the hour that just completed.
    period_end is exclusive (the top of the current hour)."""
    period_end_local = now_local.replace(minute=0, second=0, microsecond=0)
    period_start_local = period_end_local - timedelta(hours=settings.report_interval_hours)
    return period_start_local, period_end_local


def generate_hourly_report() -> None:
    """Aggregate the just-completed hour's events into a CSV file."""
    tz = ZoneInfo(settings.scheduler_timezone)
    now_local = datetime.now(tz)
    period_start_local, period_end_local = _hour_boundaries(now_local)

    period_start_utc = period_start_local.astimezone(timezone.utc).strftime(ISO_FMT)
    period_end_utc = period_end_local.astimezone(timezone.utc).strftime(ISO_FMT)

    with get_db() as conn:
        rows = conn.execute(EVENTS_QUERY, (period_start_utc, period_end_utc)).fetchall()

    # Display the end boundary as the hour's last second (HH:59:59) to match
    # the requested naming convention, even though the query bound above is
    # the exclusive top of the next hour.
    display_end_local = period_end_local - timedelta(seconds=1)
    filename = (
        f"Report_{period_start_local.strftime('%Y%m%d_%H%M%S')}"
        f"_{display_end_local.strftime('%H%M%S')}.csv"
    )
    filepath = settings.reports_dir / filename

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_id", "class_name", "direction", "count"])
        for row in rows:
            writer.writerow([row["camera_id"], row["class_name"], row["direction"], row["count"]])

    with get_db() as conn:
        conn.execute(
            "INSERT INTO reports (filename, period_start, period_end) VALUES (?, ?, ?)",
            (filename, period_start_utc, period_end_utc),
        )
        conn.commit()

    logger.info("Generated report %s (%d row(s))", filename, len(rows))


def create_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=settings.scheduler_timezone)
    scheduler.add_job(
        generate_hourly_report,
        trigger=CronTrigger(minute=0, timezone=settings.scheduler_timezone),
        id="hourly_report",
        replace_existing=True,
    )
    return scheduler


# Module-level singleton, started/stopped from app/main.py's lifespan handler.
scheduler = create_scheduler()