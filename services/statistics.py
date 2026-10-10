import math
import statistics as stats_lib
from typing import List, Dict
from model.hit import Hit

_MAD_OUTLIER_K = 2.0


def calculate_statistics(hits: List[Hit]) -> Dict[str, float]:
    """
    Oblicza statystyki dla listy trafień.
    Współrzędne trafień zakładane są względem środka tarczy (0,0).

    Rozróżnia:
      - celność (accuracy_radius): odległość średniego punktu trafień (MPI)
        od środka tarczy - mówi, jak bardzo grupa jest "przesunięta".
      - precyzję (precision_radius / precision_std / cep_50): rozrzut trafień
        wokół MPI - mówi, jak "ciasna" jest sama grupa, niezależnie od tego,
        czy jest wycelowana w środek.
    """

    valid_hits = [hit for hit in hits if hit.valid]

    if not valid_hits:
        return {
            "count": 0,
            "mean_x": 0.0,
            "mean_y": 0.0,
            "accuracy_radius": 0.0,
            "precision_radius": 0.0,
            "precision_std": 0.0,
            "precision_max_radius": 0.0,
            "cep_50": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "std_x": 0.0,
            "std_y": 0.0,
            "extreme_spread": 0.0,
            "outliers_count": 0,
        }

    xs = [hit.x for hit in valid_hits]
    ys = [hit.y for hit in valid_hits]
    count = len(xs)

    # Średni punkt trafień (MPI - Mean Point of Impact)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count

    # Celność: jak daleko MPI leży od środka tarczy (0,0)
    accuracy_radius = math.hypot(mean_x, mean_y)

    # Precyzja: rozrzut trafień względem MPI (nie środka tarczy)
    radii_from_mpi = [math.hypot(x - mean_x, y - mean_y) for x, y in zip(xs, ys)]

    precision_radius = sum(radii_from_mpi) / count
    precision_variance = sum((r - precision_radius) ** 2 for r in radii_from_mpi) / count
    precision_std = math.sqrt(precision_variance)
    precision_max_radius = max(radii_from_mpi)

    cep_50 = stats_lib.median(radii_from_mpi)
    p25, p75 = _percentiles_25_75(radii_from_mpi)

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

    outliers_count = _count_mad_outliers(radii_from_mpi)

    return {
        "count": count,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "accuracy_radius": accuracy_radius,
        "precision_radius": precision_radius,
        "precision_std": precision_std,
        "precision_max_radius": precision_max_radius,
        "cep_50": cep_50,
        "p25": p25,
        "p75": p75,
        "std_x": std_x,
        "std_y": std_y,
        "extreme_spread": extreme_spread,
        "outliers_count": outliers_count,
    }


def _percentiles_25_75(values: List[float]):
    if len(values) < 2:
        return values[0], values[0]

    quartiles = stats_lib.quantiles(values, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def _count_mad_outliers(radii: List[float], k: float = _MAD_OUTLIER_K) -> int:
    """
    Wykrywa trafienia-odstające (outliery) metodą MAD (Median Absolute
    Deviation) względem reszty grupy, zamiast mean + k*std.

    mean + k*std jest podatne na ten sam outlier, który ma wykrywać (jeden
    odległy strzał podnosi odchylenie standardowe, przez co nie przekracza
    już progu). Mediana i MAD są odporne na takie przypadki.
    """

    if len(radii) < 2:
        return 0

    median_radius = stats_lib.median(radii)
    mad = stats_lib.median(abs(r - median_radius) for r in radii)
    threshold = median_radius + k * mad

    return sum(1 for r in radii if r > threshold)
