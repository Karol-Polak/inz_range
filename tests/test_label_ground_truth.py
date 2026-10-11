import cv2
import numpy as np
import pytest

from evaluation.label_ground_truth import load_image_for_labeling
from services.image_loader import load_image


def test_load_image_for_labeling_matches_load_image_up_to_channel_order(tmp_path):
    path = tmp_path / "pattern.png"
    pattern = np.zeros((40, 60, 3), dtype=np.uint8)
    pattern[0:10, 0:10] = (10, 20, 30)  # BGR - asymmetric corner marker

    cv2.imwrite(str(path), pattern)

    reference = load_image(str(path))
    labeled_rgb = load_image_for_labeling(str(path))

    assert labeled_rgb.shape == reference.original_data.shape
    expected_rgb = cv2.cvtColor(reference.original_data, cv2.COLOR_BGR2RGB)
    assert np.array_equal(labeled_rgb, expected_rgb)


def test_load_image_for_labeling_matches_load_image_with_exif_orientation(tmp_path):
    """
    Regression test for the actual bug: matplotlib.image.imread() ignores
    EXIF orientation while cv2.imread() (used by load_image) honors it. A
    photo with EXIF orientation != 1 would previously put hand-labelled
    coordinates in a different frame than the detector's. Builds a real
    EXIF-orientation-6 JPEG (Pillow, only inside this test - not added to
    requirements.txt) and checks both the shape AND the marker position
    agree between load_image_for_labeling() and load_image().
    """

    PILImage = pytest.importorskip("PIL.Image")

    height, width = 100, 60  # asymmetric dimensions - a 90 deg rotation is unambiguous
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    pattern[0:15, 0:15] = 255  # bright asymmetric marker, top-left of the STORED pixel data

    path = tmp_path / "exif6.jpg"
    pil_image = PILImage.fromarray(pattern)  # stored as RGB, upright pixel data
    exif = PILImage.Exif()
    exif[0x0112] = 6  # Orientation = 6 ("Rotate 90 CW" needed to view correctly)
    pil_image.save(str(path), exif=exif)

    # Written tag actually present, otherwise this test would prove nothing.
    assert PILImage.open(str(path)).getexif().get(0x0112) == 6

    reference = load_image(str(path))
    labeled_rgb = load_image_for_labeling(str(path))

    assert labeled_rgb.shape == reference.original_data.shape

    gray_reference = reference.original_data.astype(float).mean(axis=2)
    gray_labeled = labeled_rgb.astype(float).mean(axis=2)

    reference_marker = np.unravel_index(np.argmax(gray_reference), gray_reference.shape)
    labeled_marker = np.unravel_index(np.argmax(gray_labeled), gray_labeled.shape)

    assert labeled_marker == reference_marker
