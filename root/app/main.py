"""
FastAPI application entrypoint: mounts the dashboard templates/static
files and the API routers, and owns process-wide startup/shutdown of the
camera workers, DB writer thread, hourly-report scheduler, and the
background health-check loop.

Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.camera_manager import camera_manager
from app.config import settings
from app.database import init_db
from app.db_writer import db_writer
from app.routers import cameras, reports, streams, websocket
from app.scheduler import scheduler
from app.shared_state import init_shared_state

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.logs_dir / "main.log"),
    ],
)
logger = logging.getLogger(__name__)

HEALTH_CHECK_INTERVAL_S = 5.0

templates = Jinja2Templates(directory=str(settings.base_dir / "app" / "templates"))


async def _health_check_loop() -> None:
    """Periodically detect worker processes that died unexpectedly and restart them."""
    while True:
        await asyncio.sleep(HEALTH_CHECK_INTERVAL_S)
        try:
            await asyncio.to_thread(camera_manager.check_health)
        except Exception:
            logger.exception("Health check iteration failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up")
    init_db()
    init_shared_state()
    camera_manager.reconcile_on_startup()
    camera_manager.start_all_enabled()
    db_writer.start()
    scheduler.start()
    health_task = asyncio.create_task(_health_check_loop())

    yield

    logger.info("Shutting down")
    health_task.cancel()
    scheduler.shutdown(wait=False)
    db_writer.stop()
    for camera in camera_manager.list_cameras():
        camera_manager.stop_camera(camera.camera_id)
    logger.info("Shutdown complete")


app = FastAPI(title="Multi-Camera Counting Dashboard", lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=str(settings.base_dir / "app" / "static")),
    name="static",
)

app.include_router(cameras.router)
app.include_router(streams.router)
app.include_router(reports.router)
app.include_router(websocket.router)


@app.get("/")
def tab1_info(request: Request):
    """Tab 1: per-camera information."""
    return templates.TemplateResponse(request, "tab1_info.html", {})


@app.get("/manage")
def tab2_management(request: Request):
    """Tab 2: runtime camera management."""
    return templates.TemplateResponse(request, "tab2_management.html", {})


@app.get("/visualization")
def tab3_visualization(request: Request):
    """Tab 3: live visualization grid."""
    return templates.TemplateResponse(request, "tab3_visualization.html", {})


@app.get("/reports")
def tab4_reports(request: Request):
    """Tab 4: hourly CSV reports."""
    return templates.TemplateResponse(request, "tab4_reports.html", {})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=(settings.app_env == "development"),
    )