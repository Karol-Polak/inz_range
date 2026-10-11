import cv2
import numpy as np
import pytest

from controller.main_controller import AnalysisError, analyze_image, analyze_image_with_timing


@pytest.fixture
def blank_image_path(tmp_path):
    # Flat, featureless image - detect_target() will find nothing in it,
    # forcing analyze_image() down the manual-calibration path.
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.full((400, 400, 3), 200, dtype=np.uint8))
    return str(path)


def test_raises_analysis_error_when_no_target_and_no_manual_fallback(blank_image_path):
    with pytest.raises(AnalysisError):
        analyze_image(blank_image_path, metadata={})


def test_uses_manual_target_when_detection_fails(blank_image_path):
    metadata = {
        "manual_target": {"center_x": 200.0, "center_y": 200.0, "radius": 150.0},
    }

    session = analyze_image(blank_image_path, metadata=metadata)

    assert session.target.type == "manual"
    assert session.target.center_x == 200.0
    assert session.target.radius == 150.0


def test_raises_file_not_found_for_invalid_path():
    with pytest.raises(FileNotFoundError):
        analyze_image("does_not_exist.jpg", metadata={})


def test_analyze_image_with_timing_reports_each_stage(blank_image_path):
    metadata = {
        "manual_target": {"center_x": 200.0, "center_y": 200.0, "radius": 150.0},
    }

    session, timings = analyze_image_with_timing(blank_image_path, metadata=metadata)

    assert session.target.type == "manual"
    for key in ("load_image", "preprocess", "detect_target", "detect_hit", "total"):
        assert key in timings
        assert timings[key] >= 0.0
    # total must cover at least the measured stages (there's a tiny bit of
    # unmeasured glue code - stats + Session construction - in between).
    assert timings["total"] >= timings["load_image"] + timings["preprocess"] + timings["detect_target"] + timings["detect_hit"]


def test_analyze_image_with_timing_raises_the_same_error_as_analyze_image(blank_image_path):
    with pytest.raises(AnalysisError):
        analyze_image_with_timing(blank_image_path, metadata={})
