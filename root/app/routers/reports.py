"""
Tab 4 endpoints: list and download hourly CSV reports.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.config import settings
from app.database import get_db
from app.models import ReportOut
from app.scheduler import generate_hourly_report

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("", response_model=list[ReportOut])
def list_reports() -> list[ReportOut]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM reports ORDER BY period_start DESC").fetchall()
    return [ReportOut.model_validate(dict(row)) for row in rows]


@router.post("/generate", response_model=ReportOut, status_code=status.HTTP_201_CREATED)
def generate_report_now() -> ReportOut:
    """Manually trigger generation for the most recently completed hour (mainly for testing/demo)."""
    generate_hourly_report()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM reports ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Report generation produced no row")
    return ReportOut.model_validate(dict(row))


@router.get("/{filename}")
def download_report(filename: str) -> FileResponse:
    """
    Download one CSV by filename. The filename must match an existing
    `reports` row — this also rules out path traversal, since we never
    touch the filesystem with a value we didn't insert into the table
    ourselves.
    """
    with get_db() as conn:
        row = conn.execute("SELECT filename FROM reports WHERE filename = ?", (filename,)).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown report: {filename}")

    filepath = settings.reports_dir / row["filename"]
    if not filepath.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report file missing on disk")

    return FileResponse(filepath, media_type="text/csv", filename=filepath.name)