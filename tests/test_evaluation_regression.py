"""
Opcjonalny test regresyjny jakości detekcji trafień względem ręcznie
oznaczonego ground truth (assets/ground_truth.json).

Pomijany, jeśli ten plik nie istnieje - stwórz go, uruchamiając
evaluation/label_ground_truth.py (wymaga środowiska z wyświetlaczem).

Progi precision/recall są konfigurowalne zmiennymi środowiskowymi
EVAL_MIN_PRECISION / EVAL_MIN_RECALL i domyślnie wynoszą 0.0 (test nie może
nigdy "fałszywie" nie przejść z powodu wymyślonego progu) - podnieś je,
kiedy będziesz mieć realny wynik bazowy z własnych zdjęć.
"""

import os

import pytest

from evaluation.evaluate_hit_detection import aggregate, evaluate_image, load_ground_truth

GROUND_TRUTH_PATH = "assets/ground_truth.json"
MIN_PRECISION = float(os.environ.get("EVAL_MIN_PRECISION", "0.0"))
MIN_RECALL = float(os.environ.get("EVAL_MIN_RECALL", "0.0"))


@pytest.mark.skipif(
    not os.path.exists(GROUND_TRUTH_PATH),
    reason=(
        f"{GROUND_TRUTH_PATH} nie istnieje - oznacz trafienia recznie "
        "(python -m evaluation.label_ground_truth <zdjecie>), aby wlaczyc ten test."
    ),
)
def test_hit_detection_meets_minimum_precision_and_recall():
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    results = []
    for image_path, entry in ground_truth.items():
        gt_hits = [tuple(point) for point in entry["hits"]]
        metrics = evaluate_image(image_path, gt_hits, scale_mm_per_px=entry.get("scale_mm_per_px"))
        if metrics is not None:
            results.append(metrics)

    assert results, "Wszystkie obrazy zostaly pominiete (nie wykryto tarczy) - sprawdz ground_truth.json"

    summary = aggregate(results)

    assert summary["precision"] is not None
    assert summary["recall"] is not None
    assert summary["precision"] >= MIN_PRECISION
    assert summary["recall"] >= MIN_RECALL
