import cv2
import numpy as np
from typing import Optional
from model.image import Image
from model.target import Target

def detect_target(image: Image) -> Optional[Target]:
    """
    Wykrywa tarczę strzelecką na obrazie przy użyciu transformacji Hougha.
    Zwraca obiekt Target lub None, jeśli detekcja się nie powiedzie.
    """

    if image.processed_data is None:
        raise ValueError("Brak przetworzonego obrazu (processed_data)")

    gray = image.processed_data

    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray.shape[0] // 2,
        param1=100,
        param2=30,
        minRadius=int(min(gray.shape) * 0.3),
        maxRadius=int(min(gray.shape) * 0.48)
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    #wybór największego okręgu (odpowiada tarczy)
    best_circle = max(circles, key=lambda c: c[2])

    center_x, center_y, radius = best_circle

    return Target(
        center_x=float(center_x),
        center_y=float(center_y),
        radius=float(radius),
        rings=None,
        type="circular",
        confidence=1.0,
    )