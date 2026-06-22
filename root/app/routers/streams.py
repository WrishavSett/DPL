"""
Tab 3 endpoints: per-camera MJPEG stream and single-frame snapshot, read
from the shared frame_store that camera workers publish into every frame.

The 2x2 grid itself is a frontend concern (CSS grid of <img> tags, one per
camera, each pointed at /mjpeg) — this router only ever serves one
camera's feed at a time, by design, so it scales to any grid layout
without backend changes.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from app.camera_manager import camera_manager
from app.shared_state import get_shared_state

router = APIRouter(prefix="/api/streams", tags=["streams"])

MJPEG_BOUNDARY = "frame"
STREAM_INTERVAL_S = 0.1  # ~10 fps to the browser, independent of the camera's inference fps


def _ensure_camera_exists(camera_id: str) -> None:
    try:
        camera_manager.get_camera(camera_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown camera_id: {camera_id}")


async def _mjpeg_generator(camera_id: str):
    state = get_shared_state()
    while True:
        frame = state.get_frame(camera_id)
        if frame is not None:
            jpeg_bytes, _ts = frame
            yield (
                b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n"
                + jpeg_bytes + b"\r\n"
            )
        await asyncio.sleep(STREAM_INTERVAL_S)


@router.get("/{camera_id}/mjpeg")
def mjpeg_stream(camera_id: str) -> StreamingResponse:
    """Tab 3: continuous MJPEG feed for one camera's grid tile."""
    _ensure_camera_exists(camera_id)
    return StreamingResponse(
        _mjpeg_generator(camera_id),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
    )


@router.get("/{camera_id}/snapshot")
def snapshot(camera_id: str) -> Response:
    """A single current JPEG frame — e.g. for polling-based or thumbnail use."""
    _ensure_camera_exists(camera_id)
    frame = get_shared_state().get_frame(camera_id)
    if frame is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No frame available yet")
    jpeg_bytes, _ts = frame
    return Response(content=jpeg_bytes, media_type="image/jpeg")