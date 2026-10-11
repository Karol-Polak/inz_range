import os

import pytest
from services.image_loader import load_image

def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_image("nie_istnieje.jpg")


def test_load_wrong_extension(tmp_path):
    fake_file = tmp_path / "test.txt"
    fake_file.write_text("not an image")

    with pytest.raises(ValueError):
        load_image(str(fake_file))


def test_load_webp():
    # cv2.imread() can decode WebP just fine (confirmed on the real asset) -
    # .webp was previously missing from SUPPORTED_EXTENSIONS even though
    # nothing stopped it from loading; this is the regression test for that.
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "1134047.webp")

    image = load_image(path)

    assert image.original_data is not None
    assert image.width > 0
    assert image.height > 0