import cv2
import numpy as np
import pytest

from controller.main_controller import AnalysisError, analyze_image


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
