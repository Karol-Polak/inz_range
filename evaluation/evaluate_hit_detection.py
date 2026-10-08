"""
Porownuje automatyczna detekcje trafien (services/hit_detection.py) z recznie
oznaczonym ground truth i raportuje precision, recall oraz sredni blad
polozenia (px, opcjonalnie mm).

Format pliku ground truth (JSON), np. assets/ground_truth.json:

    {
      "assets/idpa_przestrzeliny.jpg": {
        "hits": [[x1, y1], [x2, y2], ...],
        "scale_mm_per_px": 0.58
      }
    }

Wspolrzedne hits sa w pikselach oryginalnego obrazu (0,0 = lewy gorny rog).
"scale_mm_per_px" jest opcjonalne - jesli podane, blad polozenia jest
dodatkowo raportowany w milimetrach.

Uzyj evaluation/label_ground_truth.py, aby wygenerowac/rozszerzyc ten plik,
klikajac recznie na zdjeciu (wymaga srodowiska z wyswietlaczem).

Uzycie:
    python -m evaluation.evaluate_hit_detection assets/ground_truth.json
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple

from evaluation.metrics import compute_metrics
from services.hit_detection import detect_hit
from services.image_loader import load_image
from services.preprocessing import preprocess_image
from services.target_detection import detect_target


def evaluate_image(
    image_path: str,
    ground_truth_hits: List[Tuple[float, float]],
    max_distance: float = 15.0,
    scale_mm_per_px: Optional[float] = None,
) -> Optional[Dict]:
    image = load_image(image_path)
    image = preprocess_image(image)
    target = detect_target(image)

    if target is None:
        return None

    hits = detect_hit(image, target)
    detected_points = [(target.center_x + hit.x, target.center_y + hit.y) for hit in hits]

    return compute_metrics(detected_points, ground_truth_hits, max_distance, scale_mm_per_px)


def load_ground_truth(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as ground_truth_file:
        return json.load(ground_truth_file)


def aggregate(results: List[Dict]) -> Dict:
    true_positives = sum(r["true_positives"] for r in results)
    false_positives = sum(r["false_positives"] for r in results)
    false_negatives = sum(r["false_negatives"] for r in results)

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

    matched_errors = [r["mean_error_px"] for r in results if r["true_positives"] > 0]
    mean_error_px = sum(matched_errors) / len(matched_errors) if matched_errors else 0.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "mean_error_px": mean_error_px,
    }


def _format(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ground_truth", help="Sciezka do pliku ground_truth.json")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=15.0,
        help="Maks. odleglosc (px) uznawana za dopasowanie wykrycia do ground truth (domyslnie 15)",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    results = []

    for image_path, entry in ground_truth.items():
        gt_hits = [tuple(point) for point in entry["hits"]]
        scale_mm_per_px = entry.get("scale_mm_per_px")

        metrics = evaluate_image(image_path, gt_hits, args.max_distance, scale_mm_per_px)

        if metrics is None:
            print(f"[SKIP] {image_path}: nie wykryto tarczy, pominieto.")
            continue

        results.append(metrics)

        mm_part = (
            f" ({metrics['mean_error_mm']:.1f}mm)" if metrics["mean_error_mm"] is not None else ""
        )
        print(
            f"[{image_path}] "
            f"TP={metrics['true_positives']} FP={metrics['false_positives']} "
            f"FN={metrics['false_negatives']} "
            f"precision={_format(metrics['precision'])} recall={_format(metrics['recall'])} "
            f"mean_error={metrics['mean_error_px']:.1f}px{mm_part}"
        )

    if not results:
        print("Brak wynikow do zagregowania (wszystkie obrazy pominieto).")
        return

    summary = aggregate(results)
    print("\n--- Podsumowanie ---")
    print(
        f"TP={summary['true_positives']} FP={summary['false_positives']} "
        f"FN={summary['false_negatives']}"
    )
    print(f"precision={_format(summary['precision'])} recall={_format(summary['recall'])}")
    print(f"sredni blad polozenia: {summary['mean_error_px']:.1f}px")


if __name__ == "__main__":
    main()
