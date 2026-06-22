"""
Tests for app/camera_manager.py.

Worker processes are never actually spawned — multiprocessing.Process is
replaced with a lightweight fake so these tests exercise CameraManager's
CRUD/DB and status-transition logic without loading a real YOLO model or
touching a real camera stream.
"""

from __future__ import annotations

import pytest

import app.shared_state as shared_state_module
from app.camera_manager import CameraManager
from app.database import init_db
from app.models import CameraCreate, CameraStatus, CameraUpdate


class FakeProcess:
    """Stands in for multiprocessing.Process: never actually runs `target`."""

    def __init__(self, target=None, args=(), daemon=None, name=None):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.name = name
        self.pid = 4242
        self._alive = True

    def start(self):
        self._alive = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        # Simulate the worker honoring the stop event promptly, as a real
        # process would after camera_worker.py's SIGTERM handler fires.
        self._alive = False

    def terminate(self):
        self._alive = False


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """A CameraManager wired to a throwaway SQLite DB and fake process spawning."""
    from app import config as config_module

    monkeypatch.setattr(config_module.settings, "db_path", tmp_path / "test.db")
    init_db()

    monkeypatch.setattr("app.camera_manager.multiprocessing.Process", FakeProcess)

    shared_state_module.shared_state = None
    shared_state_module.init_shared_state()

    return CameraManager()


def _sample_camera(camera_id="cam_1", enabled=True):
    return CameraCreate(
        camera_id=camera_id,
        name="Test Camera",
        source="rtsp://example/stream",
        enabled=enabled,
    )


def test_add_camera_persists_and_starts_when_enabled(manager):
    camera = manager.add_camera(_sample_camera())
    assert camera.camera_id == "cam_1"
    assert camera.status == CameraStatus.STARTING
    assert "cam_1" in manager._processes


def test_add_camera_disabled_does_not_start(manager):
    camera = manager.add_camera(_sample_camera(enabled=False))
    assert camera.status == CameraStatus.STOPPED
    assert "cam_1" not in manager._processes


def test_stop_camera_marks_stopped(manager):
    manager.add_camera(_sample_camera())
    camera = manager.stop_camera("cam_1")
    assert camera.status == CameraStatus.STOPPED
    assert "cam_1" not in manager._processes


def test_restart_camera_respawns_process(manager):
    manager.add_camera(_sample_camera())
    first_process = manager._processes["cam_1"]
    manager.restart_camera("cam_1")
    assert "cam_1" in manager._processes
    assert manager._processes["cam_1"] is not first_process


def test_disable_then_enable_camera(manager):
    manager.add_camera(_sample_camera())
    disabled = manager.disable_camera("cam_1")
    assert disabled.status == CameraStatus.DISABLED
    assert disabled.enabled is False
    assert "cam_1" not in manager._processes

    enabled = manager.enable_camera("cam_1")
    assert enabled.enabled is True
    assert "cam_1" in manager._processes


def test_update_camera_changes_fields(manager):
    manager.add_camera(_sample_camera())
    updated = manager.update_camera("cam_1", CameraUpdate(name="Renamed Camera"))
    assert updated.name == "Renamed Camera"


def test_remove_camera_deletes_row(manager):
    manager.add_camera(_sample_camera())
    manager.remove_camera("cam_1")
    with pytest.raises(KeyError):
        manager.get_camera("cam_1")


def test_get_camera_unknown_raises_keyerror(manager):
    with pytest.raises(KeyError):
        manager.get_camera("does_not_exist")


def test_check_health_marks_crashed_and_restarts(manager):
    manager.add_camera(_sample_camera())
    manager._processes["cam_1"]._alive = False  # simulate an unexpected death

    manager.check_health(auto_restart=True)

    # check_health should have noticed the death and respawned it.
    assert "cam_1" in manager._processes
    assert manager._processes["cam_1"].is_alive()