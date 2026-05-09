import os
import cv2
from model.image import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}

def load_image(path: str) -> Image:
    """
    Wczytuje obraz z pliku i zwraca obiekt Image
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Plik nie istnieje: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Nieobsługiwany format pliku: {ext}")

    image_data = cv2.imread(path)

    if image_data is None:
        raise ValueError("Nie udało się wczytać obrazu (cv2.imread zwróciło None")

    height, width = image_data.shape[:2]

    return Image(
        path=path,
        original_data=image_data,
        processed_data=None,
        width=width,
        height=height,
        scale=None,
        metadata={
            "format": ext,
            "loader": "opencv",
        }
    )
