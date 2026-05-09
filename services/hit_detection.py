import math
from typing import List

import cv2
import numpy as np

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

    if image.original_data is None:
        raise ValueError("Brak oryginalnych danych obrazu")

    gray = cv2.cvtColor(image.original_data, cv2.COLOR_BGR2GRAY)
    target_mask = _build_target_context_mask(gray, target)
    bright_candidates = _build_bright_hit_candidates(gray, target_mask)
    bright_candidates = _split_touching_candidates(bright_candidates)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright_candidates,
        connectivity=8
    )

    hits: List[Hit] = []
    min_area, max_area = _candidate_area_range(gray)

    for component_id in range(1, component_count):
        pixel_area = stats[component_id, cv2.CC_STAT_AREA]
        x = stats[component_id, cv2.CC_STAT_LEFT]
        y = stats[component_id, cv2.CC_STAT_TOP]
        width = stats[component_id, cv2.CC_STAT_WIDTH]
        height = stats[component_id, cv2.CC_STAT_HEIGHT]

        if pixel_area < min_area or pixel_area > max_area:
            continue

        if width == 0 or height == 0:
            continue

        aspect_ratio = width / height
        if aspect_ratio < 0.55 or aspect_ratio > 1.8:
            continue

        fill_ratio = pixel_area / (width * height)
        if fill_ratio < 0.38:
            continue

        component_mask = np.uint8(labels[y:y + height, x:x + width] == component_id) * 255
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter**2)
        if circularity < 0.65:
            continue

        cx, cy = centroids[component_id]

        if not _is_inside_target_area(target_mask, int(round(cx)), int(round(cy)), width, height):
            continue

        if _looks_like_printed_digit_fragment(
            gray,
            target_mask,
            x,
            y,
            width,
            height,
            circularity,
            fill_ratio,
            pixel_area,
        ):
            continue

        dx = cx - target.center_x
        dy = cy - target.center_y
        distance = math.sqrt(dx**2 + dy**2)

        if target.type == "circular" and distance > target.radius:
            continue

        hits.append(
            Hit(
                x=dx,
                y=dy,
                distance_from_center=distance,
                valid=True,
                confidence=float(min(1.0, circularity)),
            )
        )

    return _deduplicate_hits(hits)


def _build_target_context_mask(gray, target: Target):
    if target.type == "circular":
        return _build_central_circular_target_mask(gray, target)

    return _build_large_dark_foreground_mask(gray, target)


def _build_central_circular_target_mask(gray, target: Target):
    image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    dark_or_colored = np.uint8((gray < 245) | (hsv[:, :, 1] > 40)) * 255
    dark_or_colored = cv2.morphologyEx(
        dark_or_colored,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1
    )

    center_x = float(target.center_x)
    center_y = float(target.center_y)
    center_limit = target.radius * 0.62

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark_or_colored,
        connectivity=8
    )

    mask = np.zeros_like(gray, dtype=np.uint8)
    image_area = gray.shape[0] * gray.shape[1]

    for component_id in range(1, component_count):
        area = stats[component_id, cv2.CC_STAT_AREA]
        if area < image_area * 0.0004:
            continue

        cx, cy = centroids[component_id]
        distance = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        if distance > center_limit:
            continue

        mask[labels == component_id] = 255

    if cv2.countNonZero(mask) == 0:
        center = (int(round(target.center_x)), int(round(target.center_y)))
        cv2.circle(mask, center, int(round(target.radius * 0.62)), 255, -1)

    return mask


def _build_large_dark_foreground_mask(gray, target: Target):
    _, dark_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return dark_mask

    foreground = np.zeros_like(gray, dtype=np.uint8)
    image_area = gray.shape[0] * gray.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.001:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        fill_ratio = area / max(1, width * height)
        if fill_ratio < 0.08:
            continue

        aspect_ratio = width / max(1, height)
        if (aspect_ratio < 0.15 or aspect_ratio > 6.5) and fill_ratio < 0.22:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        distance = math.sqrt(
            (center_x - target.center_x) ** 2 + (center_y - target.center_y) ** 2
        )
        if distance > target.radius * 1.05:
            continue

        cv2.drawContours(foreground, [contour], -1, 255, -1)

    return foreground


def _build_bright_hit_candidates(gray, target_mask):
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    bright_detail = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, blackhat_kernel)

    _, absolute_bright = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
    _, local_bright = cv2.threshold(
        bright_detail,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    candidates = cv2.bitwise_or(absolute_bright, local_bright)

    target_neighborhood = cv2.dilate(
        target_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1
    )
    candidates = cv2.bitwise_and(candidates, target_neighborhood)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, open_kernel, iterations=1)

    return candidates


def _split_touching_candidates(candidates):
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates,
        connectivity=8
    )

    split = np.zeros_like(candidates)

    for component_id in range(1, component_count):
        x = stats[component_id, cv2.CC_STAT_LEFT]
        y = stats[component_id, cv2.CC_STAT_TOP]
        width = stats[component_id, cv2.CC_STAT_WIDTH]
        height = stats[component_id, cv2.CC_STAT_HEIGHT]
        area = stats[component_id, cv2.CC_STAT_AREA]

        component = np.uint8(labels[y:y + height, x:x + width] == component_id) * 255

        if area < 160 or max(width, height) < 18:
            split[y:y + height, x:x + width] = cv2.bitwise_or(
                split[y:y + height, x:x + width],
                component
            )
            continue

        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        _, peaks = cv2.threshold(
            distance,
            0.55 * distance.max(),
            255,
            cv2.THRESH_BINARY
        )
        peaks = np.uint8(peaks)

        peak_count, peak_labels = cv2.connectedComponents(peaks)
        if peak_count <= 2:
            split[y:y + height, x:x + width] = cv2.bitwise_or(
                split[y:y + height, x:x + width],
                component
            )
            continue

        markers = peak_labels + 1
        markers[component == 0] = 0

        roi_color = cv2.cvtColor(component, cv2.COLOR_GRAY2BGR)
        cv2.watershed(roi_color, markers)

        for marker_id in range(2, peak_count + 1):
            separated = np.uint8(markers == marker_id) * 255
            separated = cv2.dilate(
                separated,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1
            )
            separated = cv2.bitwise_and(separated, component)
            split[y:y + height, x:x + width] = cv2.bitwise_or(
                split[y:y + height, x:x + width],
                separated
            )

    return split


def _candidate_area_range(gray):
    image_area = gray.shape[0] * gray.shape[1]
    min_area = max(18, int(image_area * 0.000006))
    max_area = max(900, int(image_area * 0.0016))
    return min_area, max_area


def _is_inside_target_area(target_mask, cx, cy, width, height):
    radius = max(7, int(round(max(width, height) * 1.8)))
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(target_mask.shape[1], cx + radius + 1)
    y2 = min(target_mask.shape[0], cy + radius + 1)

    neighborhood = target_mask[y1:y2, x1:x2]
    if neighborhood.size == 0:
        return False

    target_ratio = cv2.countNonZero(neighborhood) / neighborhood.size
    return target_ratio > 0.35


def _looks_like_printed_digit_fragment(
    gray,
    target_mask,
    x,
    y,
    width,
    height,
    circularity,
    fill_ratio,
    pixel_area,
):
    if circularity > 0.82 and fill_ratio > 0.55 and pixel_area >= 45:
        return False

    padding = max(4, int(round(max(width, height) * 0.75)))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(gray.shape[1], x + width + padding)
    y2 = min(gray.shape[0], y + height + padding)

    roi = gray[y1:y2, x1:x2]
    roi_target = target_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    bright_pixels = cv2.inRange(roi, 150, 255)
    bright_pixels = cv2.bitwise_and(bright_pixels, roi_target)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        bright_pixels,
        connectivity=8
    )

    nearby_printed_strokes = 0
    for component_id in range(1, component_count):
        area = stats[component_id, cv2.CC_STAT_AREA]
        comp_width = stats[component_id, cv2.CC_STAT_WIDTH]
        comp_height = stats[component_id, cv2.CC_STAT_HEIGHT]

        if area < 8:
            continue

        elongated = comp_width > width * 1.5 or comp_height > height * 1.5
        thin = area / max(1, comp_width * comp_height) < 0.45
        if elongated or thin:
            nearby_printed_strokes += 1

    return nearby_printed_strokes >= 2


def _deduplicate_hits(hits: List[Hit]) -> List[Hit]:
    unique_hits: List[Hit] = []

    for hit in sorted(hits, key=lambda item: item.distance_from_center or 0):
        duplicate = False
        for existing in unique_hits:
            dx = hit.x - existing.x
            dy = hit.y - existing.y
            if math.sqrt(dx**2 + dy**2) < 12:
                duplicate = True
                break

        if not duplicate:
            unique_hits.append(hit)

    return unique_hits
