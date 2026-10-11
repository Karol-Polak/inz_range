import cv2
import numpy as np

from model.image import Image
from services.preprocessing import preprocess_image
from services.target_detection import detect_target, _detect_circular_target


def test_detect_circular_target_finds_hough_circle():
    # Testing the Hough sub-step directly: detect_target() tries the dark-
    # foreground heuristic first, and for an isolated ring/disk that
    # heuristic's cv2.findContours(RETR_EXTERNAL) traces only the outer
    # boundary, so cv2.contourArea sees it as if it were a filled disk
    # (it ignores the hole) and would also classify a ring as "foreground".
    # So we exercise the Hough path on its own, the way it behaves when
    # the foreground heuristic has already failed on a harder image.
    gray_background = 180
    image_array = np.full((600, 600, 3), gray_background, dtype=np.uint8)
    cv2.circle(image_array, (300, 300), 180, (60, 60, 60), thickness=6)

    image = Image(path="synthetic.png", original_data=image_array)
    image = preprocess_image(image)

    target = _detect_circular_target(image)

    assert target is not None
    assert target.type == "circular"
    assert abs(target.center_x - 300) <= 10
    assert abs(target.center_y - 300) <= 10
    assert abs(target.radius - 180) <= 15


def test_detect_target_finds_foreground_silhouette():
    image_array = np.full((500, 400, 3), 230, dtype=np.uint8)
    cv2.ellipse(image_array, (200, 250), (120, 200), 0, 0, 360, (40, 40, 40), -1)

    image = Image(path="synthetic.png", original_data=image_array)
    image = preprocess_image(image)

    target = detect_target(image)

    assert target is not None
    assert target.type == "foreground"
    assert abs(target.center_x - 200) <= 15
    assert abs(target.center_y - 250) <= 15


def test_detect_target_returns_none_when_nothing_present():
    image_array = np.full((400, 400, 3), 200, dtype=np.uint8)

    image = Image(path="synthetic.png", original_data=image_array)
    image = preprocess_image(image)

    assert detect_target(image) is None
