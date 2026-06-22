"""
Central configuration for the dashboard app and camera workers.

Loads from a `.env` file at the project root (falling back to the defaults
in `.env.example`). Both `app/` and `worker/` import the `settings` singleton
from here so the two never drift apart on shared constants (paths, default
inference thresholds, etc).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_list(key: str, default: list[str]) -> list[str]:
    val = os.environ.get(key)
    if val is None:
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- App / server ---
    app_host: str = _env_str("APP_HOST", "0.0.0.0")
    app_port: int = _env_int("APP_PORT", 8000)
    app_env: str = _env_str("APP_ENV", "development")

    # --- Filesystem layout (resolved to absolute paths) ---
    base_dir: Path = BASE_DIR
    db_path: Path = field(default_factory=lambda: BASE_DIR / _env_str("DB_PATH", "db/tracking.db"))
    reports_dir: Path = field(default_factory=lambda: BASE_DIR / _env_str("REPORTS_DIR", "reports"))
    logs_dir: Path = field(default_factory=lambda: BASE_DIR / _env_str("LOGS_DIR", "logs"))
    models_dir: Path = field(default_factory=lambda: BASE_DIR / "models")

    # --- Model defaults (a camera row in the DB may override these) ---
    model_path: str = _env_str("MODEL_PATH", "models/yolo26n.pt")
    device: str = _env_str("DEVICE", "cpu")

    # --- Inference defaults ---
    target_w: int = _env_int("TARGET_W", 640)
    target_h: int = _env_int("TARGET_H", 480)
    conf_threshold: float = _env_float("CONF_THRESHOLD", 0.5)
    iou_threshold: float = _env_float("IOU_THRESHOLD", 0.5)
    lost_track_buffer: int = _env_int("LOST_TRACK_BUFFER", 30)
    default_classes: list[str] = field(
        default_factory=lambda: _env_list("DEFAULT_CLASSES", ["person"])
    )

    # --- Reports ---
    report_interval_hours: int = _env_int("REPORT_INTERVAL_HOURS", 1)

    # --- Logging ---
    log_level: str = _env_str("LOG_LEVEL", "INFO")

    # --- Worker / streaming ---
    frame_jpeg_quality: int = _env_int("FRAME_JPEG_QUALITY", 80)
    worker_heartbeat_interval: int = _env_int("WORKER_HEARTBEAT_INTERVAL", 5)
    worker_heartbeat_timeout: int = _env_int("WORKER_HEARTBEAT_TIMEOUT", 15)

    # --- Scheduler ---
    scheduler_timezone: str = _env_str("SCHEDULER_TIMEZONE", "UTC")

    def ensure_directories(self) -> None:
        """Create runtime directories if they don't already exist."""
        for path in (self.db_path.parent, self.reports_dir, self.logs_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()