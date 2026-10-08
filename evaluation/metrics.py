import math
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]


def match_points(
    detected: List[Point],
    ground_truth: List[Point],
    max_distance: float,
) -> Dict[str, list]:
    """
    Dopasowuje wykryte punkty (np. trafienia) do punktów referencyjnych
    (ground truth) metodą zachłannego najbliższego sąsiada: w każdej
    iteracji łączy najbliższą dostępną parę detekcja-referencja, o ile
    odległość nie przekracza max_distance.

    Zwraca:
      - "matches": lista (detected_index, ground_truth_index, distance)
      - "false_positives": indeksy detected bez dopasowania (nadmiarowe wykrycia)
      - "false_negatives": indeksy ground_truth bez dopasowania (pominięte trafienia)
    """

    candidate_pairs = []
    for d_index, d_point in enumerate(detected):
        for g_index, g_point in enumerate(ground_truth):
            distance = math.hypot(d_point[0] - g_point[0], d_point[1] - g_point[1])
            if distance <= max_distance:
                candidate_pairs.append((distance, d_index, g_index))

    candidate_pairs.sort(key=lambda pair: pair[0])

    matched_detected = set()
    matched_ground_truth = set()
    matches = []

    for distance, d_index, g_index in candidate_pairs:
        if d_index in matched_detected or g_index in matched_ground_truth:
            continue
        matched_detected.add(d_index)
        matched_ground_truth.add(g_index)
        matches.append((d_index, g_index, distance))

    false_positives = [i for i in range(len(detected)) if i not in matched_detected]
    false_negatives = [i for i in range(len(ground_truth)) if i not in matched_ground_truth]

    return {
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compute_metrics(
    detected: List[Point],
    ground_truth: List[Point],
    max_distance: float = 15.0,
    scale_mm_per_px: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Oblicza precision, recall i średni błąd położenia (px, opcjonalnie mm)
    dla wykrytych punktów względem ręcznie oznaczonego ground truth.
    """

    result = match_points(detected, ground_truth, max_distance)

    true_positives = len(result["matches"])
    false_positives = len(result["false_positives"])
    false_negatives = len(result["false_negatives"])

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else None
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else None
    )

    distances = [distance for _, _, distance in result["matches"]]
    mean_error_px = sum(distances) / len(distances) if distances else 0.0
    mean_error_mm = mean_error_px * scale_mm_per_px if scale_mm_per_px is not None else None

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "mean_error_px": mean_error_px,
        "mean_error_mm": mean_error_mm,
    }
