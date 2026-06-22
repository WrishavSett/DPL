"""
WebSocket push of live camera status — lets Tab 1 / Tab 2 update without
polling the REST endpoints on a tight timer. Each connected client runs
its own independent polling loop; for the handful of cameras and viewers
this dashboard is built for, that's simpler and plenty efficient compared
to building a pub/sub broadcaster.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.camera_manager import camera_manager
from app.models import CameraSummary
from app.routers.cameras import _counts_for_all_cameras  # internal reuse, same app package

router = APIRouter(tags=["websocket"])

logger = logging.getLogger(__name__)

PUSH_INTERVAL_S = 2.0


def _build_summaries() -> list[CameraSummary]:
    counts_by_camera = _counts_for_all_cameras()
    return [
        CameraSummary(camera=camera, counts=counts_by_camera.get(camera.camera_id, []))
        for camera in camera_manager.list_cameras()
    ]


@router.websocket("/ws/status")
async def camera_status_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            # Offload the sync sqlite calls so they don't block the event loop.
            summaries = await asyncio.to_thread(_build_summaries)
            payload = [summary.model_dump(mode="json") for summary in summaries]
            await websocket.send_json(payload)
            await asyncio.sleep(PUSH_INTERVAL_S)
    except WebSocketDisconnect:
        logger.debug("Status websocket client disconnected")