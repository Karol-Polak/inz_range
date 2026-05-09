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