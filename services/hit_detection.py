import cv2
import math
from typing import List
from model.image import Image
from model.target import Target
from model.hit import Hit

def detect_hit(image: Image, target: Target) -> List[Hit]:
    """
    Wykrywa trafienia na tarczy strzeleckiej.
    Zwraca listę obiektów Hit (współrzędne względem środka tarczy).
    """

    if image.processed_data is None:
        raise ValueError("Brak przetworzonego obrazu")

    gray = image.processed_data

    #Binaryzacja adaptacyjna (otwory są ciemne)
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    #morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    #detekcja konturów
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    hits: List[Hit] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        #Filtr1 - powierzchnia
        if area < 20 or area > 2000:
            continue

        #Środek konturu
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        #Filtr2 - położenie względem tarczy
        dx = cx - target.center_x
        dy = cy - target.center_y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > target.radius:
            continue

        #Filtr3 - kształt(okrągłość)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter**2)
        if circularity < 0.4:
            continue

        #współrzędne względne względem środka tarczy
        rel_x = dx
        rel_y = dy

        hits.append(
            Hit(
                x=rel_x,
                y=rel_y,
                distance_from_center=distance,
                valid=True
            )
        )

    return hits
