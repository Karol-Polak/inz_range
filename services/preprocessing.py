import cv2
from model.image import Image

def preprocess_image(image: Image) -> Image:
    """
    Wykonuje preprocessing obrazu:
    - konwersja do skali szarości
    - redukcja szumów
    - normalizacja kontrastu (CLAHE)
    """

    if image.original_data is None:
        raise ValueError("Brak danych obrazu do przetwarzania")

    # skala szarości
    gray = cv2.cvtColor(image.original_data, cv2.COLOR_BGR2GRAY)

    # redukcja szumów (filtr medianowy)
    denoised = cv2.medianBlur(gray, 5)

    # normalizacja kontrastu (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    image.processed_data = enhanced
    return image