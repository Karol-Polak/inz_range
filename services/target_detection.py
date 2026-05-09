import cv2
import numpy as np
from typing import Optional
from model.image import Image
from model.target import Target


def detect_target(image: Image) -> Optional[Target]:
    """
    Wykrywa obszar tarczy na obrazie.

    Najpierw probuje znalezc duzy ciemny ksztalt tarczy, co dziala dla
    sylwetek i tarcz nieregularnych. Gdy taki obszar nie istnieje,
    przechodzi do klasycznej detekcji okregu.

    Zwraca obiekt Target lub None, jeśli detekcja się nie powiedzie.
    """

    if image.processed_data is None:
        raise ValueError("Brak przetworzonego obrazu (processed_data)")

    foreground_target = _detect_dark_foreground_target(image)
    if foreground_target is not None:
        return foreground_target

    return _detect_circular_target(image)


def _detect_dark_foreground_target(image: Image) -> Optional[Target]:
    if image.original_data is None:
        return None

    gray = cv2.cvtColor(image.original_data, cv2.COLOR_BGR2GRAY)

    _, dark_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    image_area = gray.shape[0] * gray.shape[1]
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < image_area * 0.10:
        return None

    x, y, width, height = cv2.boundingRect(largest)
    if width == 0 or height == 0:
        return None

    fill_ratio = area / (width * height)
    if fill_ratio < 0.20:
        return None

    center_x = x + width / 2
    center_y = y + height / 2
    radius = max(width, height) / 2

    return Target(
        center_x=float(center_x),
        center_y=float(center_y),
        radius=float(radius),
        rings=None,
        type="foreground",
        confidence=0.75,
    )


def _detect_circular_target(image: Image) -> Optional[Target]:
    gray = image.processed_data
    height, width = gray.shape[:2]

    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=20,
        minRadius=int(min(gray.shape) * 0.20),
        maxRadius=int(min(gray.shape) * 0.49)
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    valid_circles = []
    image_center_x = width / 2
    image_center_y = height / 2

    for center_x, center_y, radius in circles:
        if not (width * 0.15 <= center_x <= width * 0.85):
            continue
        if not (height * 0.15 <= center_y <= height * 0.85):
            continue

        distance_from_image_center = np.hypot(
            center_x - image_center_x,
            center_y - image_center_y
        )
        score = radius - distance_from_image_center * 0.35
        valid_circles.append((score, center_x, center_y, radius))

    if not valid_circles:
        return None

    _, center_x, center_y, radius = max(valid_circles, key=lambda item: item[0])


    return Target(
        center_x=float(center_x),
        center_y=float(center_y),
        radius=float(radius),
        rings=None,
        type="circular",
        confidence=1.0,
    )
