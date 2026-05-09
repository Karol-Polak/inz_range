import math
from typing import List, Dict
from model.hit import Hit


def calculate_statistics(hits: List[Hit]) -> Dict[str, float]:
    """
    Oblicza statystyki dla listy trafień.
    Współrzędne trafień zakładane są względem środka tarczy (0,0).
    """

    valid_hits = [hit for hit in hits if hit.valid]

    if not valid_hits:
        return {
            "count": 0,
            "mean_radius": 0.0,
            "std_radius": 0.0,
            "max_radius": 0.0,
            "cep_50": 0.0,
            "mean_x": 0.0,
            "mean_y": 0.0,
            "std_x": 0.0,
            "std_y": 0.0,
            "extreme_spread": 0.0,
            "outliers_count": 0,
            "p25": 0.0,
            "p75": 0.0
        }

    xs = [hit.x for hit in valid_hits]
    ys = [hit.y for hit in valid_hits]
    count = len(xs)

    # Odległości od środka
    radii = [math.sqrt(x**2 + y**2) for x, y in zip(xs, ys)]

    mean_radius = sum(radii) / count
    variance = sum((r - mean_radius) ** 2 for r in radii) / count
    std_radius = math.sqrt(variance)

    max_radius = max(radii)

    # Percentyle
    sorted_radii = sorted(radii)
    cep_50 = sorted_radii[int(0.5 * count)]
    p25 = sorted_radii[int(0.25 * count)]
    p75 = sorted_radii[int(0.75 * count)]

    mean_x = sum(xs) / count
    mean_y = sum(ys) / count

    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / count)
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / count)

    # Extreme Spread (maks. odległość między dwoma trafieniami)
    extreme_spread = 0.0
    for i in range(count):
        for j in range(i + 1, count):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx**2 + dy**2)
            extreme_spread = max(extreme_spread, dist)

    # Outliery
    k = 2.0
    outliers = [r for r in radii if r > mean_radius + k * std_radius]

    return {
        "count": count,
        "mean_radius": mean_radius,
        "std_radius": std_radius,
        "max_radius": max_radius,
        "cep_50": cep_50,
        "p25": p25,
        "p75": p75,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "std_x": std_x,
        "std_y": std_y,
        "extreme_spread": extreme_spread,
        "outliers_count": len(outliers)
    }
