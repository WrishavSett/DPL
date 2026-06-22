"""
Tab 1 (per-camera info) and Tab 2 (runtime camera management) endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.camera_manager import camera_manager
from app.database import get_db
from app.models import (
    CameraActionResult,
    CameraCreate,
    CameraSummary,
    CameraUpdate,
    ClasswiseCount,
)

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


def _counts_query(camera_id: str | None) -> tuple[str, tuple]:
    base = """
        SELECT camera_id, class_name,
               SUM(CASE WHEN direction = 'in' THEN 1 ELSE 0 END) AS in_count,
               SUM(CASE WHEN direction = 'out' THEN 1 ELSE 0 END) AS out_count
        FROM events
    """
    if camera_id is None:
        return base + " GROUP BY camera_id, class_name ORDER BY camera_id, class_name", ()
    return (
        base + " WHERE camera_id = ? GROUP BY class_name ORDER BY class_name",
        (camera_id,),
    )


def _counts_for_all_cameras() -> dict[str, list[ClasswiseCount]]:
    sql, params = _counts_query(None)
    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()

    result: dict[str, list[ClasswiseCount]] = {}
    for row in rows:
        result.setdefault(row["camera_id"], []).append(
            ClasswiseCount(
                camera_id=row["camera_id"],
                class_name=row["class_name"],
                in_count=row["in_count"],
                out_count=row["out_count"],
            )
        )
    return result


def _counts_for_camera(camera_id: str) -> list[ClasswiseCount]:
    sql, params = _counts_query(camera_id)
    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        ClasswiseCount(
            camera_id=camera_id,
            class_name=row["class_name"],
            in_count=row["in_count"],
            out_count=row["out_count"],
        )
        for row in rows
    ]


@router.get("", response_model=list[CameraSummary])
def list_cameras() -> list[CameraSummary]:
    """Tab 1: every camera with its cumulative classwise counts."""
    counts_by_camera = _counts_for_all_cameras()
    return [
        CameraSummary(camera=camera, counts=counts_by_camera.get(camera.camera_id, []))
        for camera in camera_manager.list_cameras()
    ]


@router.get("/{camera_id}", response_model=CameraSummary)
def get_camera(camera_id: str) -> CameraSummary:
    try:
        camera = camera_manager.get_camera(camera_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown camera_id: {camera_id}")
    return CameraSummary(camera=camera, counts=_counts_for_camera(camera_id))


@router.post("", response_model=CameraSummary, status_code=status.HTTP_201_CREATED)
def add_camera(payload: CameraCreate) -> CameraSummary:
    """Tab 2: add a new camera. Starts immediately if `enabled` is true."""
    try:
        camera = camera_manager.add_camera(payload)
    except Exception as exc:  # e.g. duplicate camera_id, sqlite constraint error
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    return CameraSummary(camera=camera, counts=[])


@router.patch("/{camera_id}", response_model=CameraSummary)
def update_camera(camera_id: str, payload: CameraUpdate) -> CameraSummary:
    """Tab 2: edit a camera's config. Restarts it automatically if currently running."""
    try:
        camera = camera_manager.update_camera(camera_id, payload)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown camera_id: {camera_id}")
    return CameraSummary(camera=camera, counts=_counts_for_camera(camera_id))


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_camera(camera_id: str) -> None:
    """Tab 2: stop (if running) and permanently remove a camera and its events."""
    try:
        camera_manager.get_camera(camera_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown camera_id: {camera_id}")
    camera_manager.remove_camera(camera_id)


def _action(camera_id: str, action: str, fn) -> CameraActionResult:
    try:
        camera = fn(camera_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown camera_id: {camera_id}")
    return CameraActionResult(camera_id=camera_id, action=action, status=camera.status)


@router.post("/{camera_id}/enable", response_model=CameraActionResult)
def enable_camera(camera_id: str) -> CameraActionResult:
    return _action(camera_id, "enable", camera_manager.enable_camera)


@router.post("/{camera_id}/disable", response_model=CameraActionResult)
def disable_camera(camera_id: str) -> CameraActionResult:
    return _action(camera_id, "disable", camera_manager.disable_camera)


@router.post("/{camera_id}/restart", response_model=CameraActionResult)
def restart_camera(camera_id: str) -> CameraActionResult:
    return _action(camera_id, "restart", camera_manager.restart_camera)