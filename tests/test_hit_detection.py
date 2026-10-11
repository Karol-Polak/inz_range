import cv2
import numpy as np

from evaluation.metrics import match_points
from model.image import Image
from model.target import Target
from services.hit_detection import detect_hit
from services.preprocessing import preprocess_image

_TARGET = Target(center_x=300.0, center_y=300.0, radius=250.0, type="circular")


def _build_target_image(size=600, face_value=50, bg_value=230, radius=250, center=(300, 300)):
    """A dark circular target face on a lighter background - no bullet holes yet."""
    image_array = np.full((size, size, 3), bg_value, dtype=np.uint8)
    cv2.circle(image_array, center, radius, (face_value,) * 3, -1)
    return image_array


def _detect(image_array, target=_TARGET):
    image = Image(path="synthetic.png", original_data=image_array)
    image = preprocess_image(image)
    hits = detect_hit(image, target)
    return [(target.center_x + hit.x, target.center_y + hit.y) for hit in hits]


def test_detects_known_bright_dots_at_their_true_positions():
    expected_dots = [(250, 250), (350, 260), (300, 370)]

    image_array = _build_target_image()
    for x, y in expected_dots:
        cv2.circle(image_array, (x, y), 7, (250, 250, 250), -1)

    detected = _detect(image_array)
    match = match_points(detected, expected_dots, max_distance=5.0)

    assert len(match["matches"]) == len(expected_dots)
    assert match["false_positives"] == []
    assert match["false_negatives"] == []


def test_two_touching_holes_are_separated_by_watershed():
    # Centers 16px apart with radius-8 dots - close enough to merge into one
    # connected blob, but with two distinguishable peaks in the distance
    # transform, which is exactly the case _split_touching_candidates()
    # handles via cv2.watershed.
    expected_dots = [(292, 300), (308, 300)]

    image_array = _build_target_image()
    for x, y in expected_dots:
        cv2.circle(image_array, (x, y), 8, (250, 250, 250), -1)

    detected = _detect(image_array)

    assert len(detected) == 2
    match = match_points(detected, expected_dots, max_distance=5.0)
    assert len(match["matches"]) == 2


def test_empty_target_has_no_hits():
    image_array = _build_target_image()

    assert _detect(image_array) == []


def test_hole_outside_target_boundary_is_excluded():
    # The target circle has radius 250; this dot sits at distance 280 from
    # the center, clearly outside the target - must not count as a hit.
    image_array = _build_target_image()
    cv2.circle(image_array, (300, 20), 7, (250, 250, 250), -1)

    assert _detect(image_array) == []


def test_printed_digit_fragment_with_two_nearby_strokes_is_filtered():
    # A single bright blob, round/filled enough to pass the normal hit
    # filters on its own (confirmed by the sibling test below), but with
    # two elongated bright strokes nearby - mimicking a printed digit on
    # the target face (e.g. a scoring-ring number). This is exactly the
    # pattern _looks_like_printed_digit_fragment() is meant to reject.
    image_array = _build_target_image()
    cv2.ellipse(image_array, (300, 250), (9, 6), 0, 0, 360, (250, 250, 250), -1)
    cv2.rectangle(image_array, (280, 232), (283, 262), (250, 250, 250), -1)  # vertical stroke
    cv2.rectangle(image_array, (288, 264), (318, 267), (250, 250, 250), -1)  # horizontal stroke

    assert _detect(image_array) == []


def test_same_fragment_without_a_second_stroke_is_a_normal_hit():
    # Sibling of the test above: only one nearby stroke isn't enough to
    # trigger the printed-digit heuristic (it requires >= 2), so the
    # fragment must still be detected as an ordinary hit. This confirms
    # the previous test is actually exercising the digit-fragment filter,
    # not some unrelated rejection.
    image_array = _build_target_image()
    cv2.ellipse(image_array, (300, 250), (9, 6), 0, 0, 360, (250, 250, 250), -1)
    cv2.rectangle(image_array, (280, 232), (283, 262), (250, 250, 250), -1)

    detected = _detect(image_array)

    assert len(detected) == 1
    assert abs(detected[0][0] - 300) <= 2
    assert abs(detected[0][1] - 250) <= 2
