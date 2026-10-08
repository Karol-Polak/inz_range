import math
from typing import List, Optional, Tuple

from model.hit import Hit
from model.target import Target

_REMOVE_RADIUS_PX = 15.0


def map_display_point_to_image(
    display_point: Tuple[float, float],
    label_size: Tuple[float, float],
    pixmap_size: Tuple[float, float],
    image_size: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """
    Przelicza punkt kliknięcia na widgecie (QLabel) na współrzędne w
    oryginalnym obrazie, uwzględniając wyśrodkowanie wyskalowanego
    podglądu (letterboxing) wewnątrz etykiety.

    Zwraca None, jeśli kliknięcie padło poza wyświetlanym podglądem.
    """

    label_w, label_h = label_size
    pixmap_w, pixmap_h = pixmap_size
    image_w, image_h = image_size

    if pixmap_w <= 0 or pixmap_h <= 0:
        return None

    offset_x = max(0.0, (label_w - pixmap_w) / 2.0)
    offset_y = max(0.0, (label_h - pixmap_h) / 2.0)

    x_in_pixmap = display_point[0] - offset_x
    y_in_pixmap = display_point[1] - offset_y

    if not (0.0 <= x_in_pixmap <= pixmap_w and 0.0 <= y_in_pixmap <= pixmap_h):
        return None

    scale_x = image_w / pixmap_w
    scale_y = image_h / pixmap_h

    return x_in_pixmap * scale_x, y_in_pixmap * scale_y


def toggle_hit_at(
    hits: List[Hit],
    target: Target,
    image_point: Tuple[float, float],
    remove_radius_px: float = _REMOVE_RADIUS_PX,
) -> List[Hit]:
    """
    Jeśli w pobliżu image_point znajduje się już trafienie, usuwa je.
    W przeciwnym razie dodaje nowe, ręczne trafienie w tym miejscu.

    Zwraca nową listę trafień (oryginalna lista nie jest modyfikowana).
    """

    x_img, y_img = image_point

    for hit in hits:
        hit_x = target.center_x + hit.x
        hit_y = target.center_y + hit.y

        if math.hypot(hit_x - x_img, hit_y - y_img) <= remove_radius_px:
            return [existing for existing in hits if existing is not hit]

    dx = x_img - target.center_x
    dy = y_img - target.center_y

    new_hit = Hit(
        x=dx,
        y=dy,
        distance_from_center=math.hypot(dx, dy),
        valid=True,
        confidence=1.0,
        source="manual",
    )

    return hits + [new_hit]
