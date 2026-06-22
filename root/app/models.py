"""
Pydantic schemas shared by the FastAPI routers.

These describe API request/response shapes. JSON-encoded columns in the
`cameras` table (`classes`, `count_line`) are typed here as real lists/
objects rather than left as raw strings — the routers handle the
serialize/deserialize step against SQLite.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CameraStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    CRASHED = "crashed"
    DISABLED = "disabled"


class CountLine(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class CameraCreate(BaseModel):
    """Payload for adding a new camera (Tab 2)."""

    camera_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(..., min_length=1, max_length=128)
    source: str = Field(..., min_length=1)
    enabled: bool = True
    classes: Optional[list[str]] = None
    count_line: Optional[CountLine] = None
    model_path: Optional[str] = None
    device: str = "cpu"
    target_w: int = 640
    target_h: int = 480
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    lost_track_buffer: int = 30


class CameraUpdate(BaseModel):
    """Partial update payload for an existing camera — all fields optional."""

    name: Optional[str] = None
    source: Optional[str] = None
    enabled: Optional[bool] = None
    classes: Optional[list[str]] = None
    count_line: Optional[CountLine] = None
    model_path: Optional[str] = None
    device: Optional[str] = None
    target_w: Optional[int] = None
    target_h: Optional[int] = None
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    lost_track_buffer: Optional[int] = None


class CameraOut(BaseModel):
    """A camera row as returned to the dashboard."""

    model_config = ConfigDict(from_attributes=True)

    camera_id: str
    name: str
    source: str
    enabled: bool
    classes: Optional[list[str]] = None
    count_line: Optional[CountLine] = None
    model_path: Optional[str] = None
    device: str
    target_w: int
    target_h: int
    conf_threshold: float
    iou_threshold: float
    lost_track_buffer: int
    status: CameraStatus
    pid: Optional[int] = None
    last_heartbeat: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class ClasswiseCount(BaseModel):
    """Aggregated IN/OUT counts for one class on one camera."""

    camera_id: str
    class_name: str
    in_count: int = 0
    out_count: int = 0


class CameraSummary(BaseModel):
    """Tab 1 row: camera info plus its current classwise counts."""

    camera: CameraOut
    counts: list[ClasswiseCount] = Field(default_factory=list)


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    track_id: int
    class_id: int
    class_name: str
    direction: str
    timestamp: datetime


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime


class CameraActionResult(BaseModel):
    """Generic response for start/stop/restart/enable/disable actions."""

    camera_id: str
    action: str
    status: CameraStatus
    detail: Optional[str] = None