import cv2
import numpy as np
from typing import Optional
from model.image import Image
from model.target import Target
from utils.config import (
    CIRCULAR_BLUR_KERNEL_SIZE,
    CIRCULAR_BLUR_SIGMA,
    CIRCULAR_CENTER_DISTANCE_PENALTY,
    CIRCULAR_CENTER_WINDOW_MAX_FRACTION,
    CIRCULAR_CENTER_WINDOW_MIN_FRACTION,
    CIRCULAR_CONFIDENCE,
    FOREGROUND_CLOSE_ITERATIONS,
    FOREGROUND_CLOSE_KERNEL_SIZE,
    FOREGROUND_CONFIDENCE,
    FOREGROUND_MIN_AREA_FRACTION,
    FOREGROUND_MIN_FILL_RATIO,
    HOUGH_DP,
    HOUGH_MAX_RADIUS_FRACTION,
    HOUGH_MIN_DIST_PX,
    HOUGH_MIN_RADIUS_FRACTION,
    HOUGH_PARAM1,
    HOUGH_PARAM2,
)


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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, FOREGROUND_CLOSE_KERNEL_SIZE)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=FOREGROUND_CLOSE_ITERATIONS)

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

    if area < image_area * FOREGROUND_MIN_AREA_FRACTION:
        return None

    x, y, width, height = cv2.boundingRect(largest)
    if width == 0 or height == 0:
        return None

    fill_ratio = area / (width * height)
    if fill_ratio < FOREGROUND_MIN_FILL_RATIO:
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
        confidence=FOREGROUND_CONFIDENCE,
    )


def _detect_circular_target(image: Image) -> Optional[Target]:
    gray = image.processed_data
    height, width = gray.shape[:2]

    blurred = cv2.GaussianBlur(gray, CIRCULAR_BLUR_KERNEL_SIZE, CIRCULAR_BLUR_SIGMA)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST_PX,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=int(min(gray.shape) * HOUGH_MIN_RADIUS_FRACTION),
        maxRadius=int(min(gray.shape) * HOUGH_MAX_RADIUS_FRACTION)
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    valid_circles = []
    image_center_x = width / 2
    image_center_y = height / 2

    for center_x, center_y, radius in circles:
        if not (width * CIRCULAR_CENTER_WINDOW_MIN_FRACTION <= center_x <= width * CIRCULAR_CENTER_WINDOW_MAX_FRACTION):
            continue
        if not (height * CIRCULAR_CENTER_WINDOW_MIN_FRACTION <= center_y <= height * CIRCULAR_CENTER_WINDOW_MAX_FRACTION):
            continue

        distance_from_image_center = np.hypot(
            center_x - image_center_x,
            center_y - image_center_y
        )
        score = radius - distance_from_image_center * CIRCULAR_CENTER_DISTANCE_PENALTY
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
        confidence=CIRCULAR_CONFIDENCE,
    )
