"""
Unit tests for worker/pipeline.py. Heavy dependencies (YOLO models, real
camera streams) are mocked out — these tests exercise the pure logic
around them: class-name resolution, preprocessing, and the line-counting
result plumbing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from worker.pipeline import (
    CameraConfig,
    CountingResult,
    init_line_counter,
    preprocess,
    resolve_classes,
    run_counting,
)


def test_preprocess_resizes_to_target_dimensions():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = preprocess(frame, width=640, height=480)
    assert result.shape == (480, 640, 3)


def test_resolve_classes_returns_none_for_no_filter():
    model = MagicMock()
    model.names = {0: "person", 1: "car"}
    assert resolve_classes(model, None) is None


def test_resolve_classes_maps_names_to_ids():
    model = MagicMock()
    model.names = {0: "person", 1: "car", 2: "truck"}
    assert resolve_classes(model, ["car", "person"]) == [1, 0]


def test_resolve_classes_raises_on_unknown_class():
    model = MagicMock()
    model.names = {0: "person"}
    with pytest.raises(ValueError):
        resolve_classes(model, ["bicycle"])


def test_init_line_counter_returns_none_when_no_line_configured():
    line_zone, line_ann = init_line_counter(None)
    assert line_zone is None
    assert line_ann is None


def test_init_line_counter_builds_zone_from_coordinates():
    line_zone, line_ann = init_line_counter((10, 20, 100, 20))
    assert isinstance(line_zone, sv.LineZone)
    assert isinstance(line_ann, sv.LineZoneAnnotator)


def test_run_counting_returns_empty_result_without_line_zone():
    tracked = sv.Detections.empty()
    result = run_counting(tracked, line_zone=None)
    assert isinstance(result, CountingResult)
    assert result.crossed_in is None
    assert result.crossed_out is None
    assert result.in_count_per_class == {}


def test_run_counting_returns_empty_result_for_no_detections():
    line_zone, _ = init_line_counter((0, 0, 100, 0))
    tracked = sv.Detections.empty()
    result = run_counting(tracked, line_zone)
    assert result.crossed_in is None


def test_camera_config_defaults():
    cfg = CameraConfig(camera_id="cam_1", source="rtsp://example", model_path="yolo11n.pt")
    assert cfg.device == "cpu"
    assert cfg.target_w == 640
    assert cfg.count_line is None