import numpy as np
import pytest
from model.image import Image
from services.preprocessing import preprocess_image

def test_preprocessing_sets_processed_data():
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    image = Image(
        path="dummy.jpg",
        original_data=dummy_image,
        width=100,
        height=100,
    )

    processed = preprocess_image(image)

    assert processed.processed_data is not None
    assert processed.processed_data.ndim == 2


def test_preprocessing_output_matches_input_dimensions():
    dummy_image = np.zeros((80, 120, 3), dtype=np.uint8)

    image = Image(path="dummy.jpg", original_data=dummy_image, width=120, height=80)
    processed = preprocess_image(image)

    assert processed.processed_data.shape == (80, 120)


def test_preprocessing_raises_when_no_image_data():
    image = Image(path="dummy.jpg", original_data=None)

    with pytest.raises(ValueError):
        preprocess_image(image)

