"""
Porownuje automatyczna detekcje trafien (services/hit_detection.py) z recznie
oznaczonym ground truth i raportuje precision, recall, sredni blad
polozenia (px, opcjonalnie mm) oraz czas analizy.

Format pliku ground truth (JSON), np. assets/ground_truth.json:

    {
      "assets/idpa_przestrzeliny.jpg": {
        "hits": [[x1, y1], [x2, y2], ...],
        "scale_mm_per_px": 0.58,
        "target_type": "foreground"
      }
    }

Wspolrzedne hits sa w pikselach obrazu W ORIENTACJI, KTORA PRODUKUJE
services.image_loader.load_image() (cv2.imread, honoruje orientacje EXIF) -
to ta sama funkcja, ktorej uzywa analyze_image() i evaluation/label_ground_truth.py
do wczytania zdjecia. Jesli etykietujesz recznie zdjecie z telefonu z
orientacja EXIF != 1, uzyj label_ground_truth.py (nie innego narzedzia) -
inaczej klikniete wspolrzedne beda w innym ukladzie niz detekcje.
"scale_mm_per_px" jest opcjonalne - jesli podane, blad polozenia jest
dodatkowo raportowany w milimetrach.
"target_type" jest opcjonalne (circular/foreground/manual) i sluzy tylko do
grupowania wynikow w raporcie - jesli nie podane, uzywany jest typ
faktycznie zwrocony przez detektor. Pliki ground truth bez tego pola dzialaja
bez zmian (pole jest w pelni opcjonalne).

Uzyj evaluation/label_ground_truth.py, aby wygenerowac/rozszerzyc ten plik,
klikajac recznie na zdjeciu (wymaga srodowiska z wyswietlaczem).

Uzycie:
    python -m evaluation.evaluate_hit_detection assets/ground_truth.json
    python -m evaluation.evaluate_hit_detection assets/ground_truth.json --benchmark 5
"""

import argparse
import json
import statistics as stats_lib
from typing import Dict, List, Optional, Tuple

from controller.main_controller import AnalysisError, analyze_image_with_timing
from evaluation.metrics import compute_metrics


def evaluate_image(
    image_path: str,
    ground_truth_hits: List[Tuple[float, float]],
    max_distance: float = 15.0,
    scale_mm_per_px: Optional[float] = None,
) -> Optional[Dict]:
    """
    Analizuje jeden obraz (przez analyze_image_with_timing - ten sam kod,
    ktory uzywa GUI) i porownuje wykryte trafienia z ground truth.

    Zwraca None, jesli nie wykryto tarczy (odpowiednik AnalysisError bez
    recznej kalibracji - w ewaluacji nie ma reki do kalibracji, wiec taki
    obraz jest pomijany).
    """

    try:
        session, timings = analyze_image_with_timing(image_path, metadata={})
    except AnalysisError:
        return None

    target = session.target
    detected_points = [(target.center_x + hit.x, target.center_y + hit.y) for hit in session.hits]

    metrics = compute_metrics(detected_points, ground_truth_hits, max_distance, scale_mm_per_px)
    metrics["target_type"] = target.type
    metrics["time_total_s"] = timings["total"]
    metrics["time_breakdown_s"] = timings
    return metrics


def benchmark_image(image_path: str, repeats: int) -> Dict:
    """
    Powtarza analize tego samego obrazu `repeats` razy (bez liczenia
    precision/recall - tylko czas) i zwraca liste czasow calkowitych oraz
    ich srednia/odchylenie standardowe. Do stabilnego pomiaru wydajnosci
    (patrz flaga --benchmark), bo pojedynczy pomiar jest podatny na szum
    systemowy (inne procesy, stan cache'u CPU itp.).
    """

    times = []
    for _ in range(repeats):
        try:
            _, timings = analyze_image_with_timing(image_path, metadata={})
        except AnalysisError:
            continue
        times.append(timings["total"])

    if not times:
        return {"times": [], "mean": None, "std": None}

    mean = stats_lib.mean(times)
    std = stats_lib.stdev(times) if len(times) >= 2 else 0.0
    return {"times": times, "mean": mean, "std": std}


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

    times = [r["time_total_s"] for r in results if r.get("time_total_s") is not None]

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "mean_error_px": mean_error_px,
        "mean_time_s": stats_lib.mean(times) if times else None,
        "median_time_s": stats_lib.median(times) if times else None,
        "max_time_s": max(times) if times else None,
    }


def aggregate_by_group(results: List[Dict]) -> Dict[str, Dict]:
    """
    Jak aggregate(), ale osobno dla kazdej grupy wynikajacej z pola
    "target_type" w wynikach evaluate_image() (circular/foreground/manual).
    """

    groups: Dict[str, List[Dict]] = {}
    for result in results:
        group = result.get("target_type") or "nieznany"
        groups.setdefault(group, []).append(result)

    return {group: aggregate(group_results) for group, group_results in groups.items()}


def _format(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _format_time(seconds: Optional[float]) -> str:
    return "n/a" if seconds is None else f"{seconds * 1000:.1f}ms"


def _print_summary(title: str, summary: Dict) -> None:
    print(f"\n--- {title} ---")
    print(
        f"TP={summary['true_positives']} FP={summary['false_positives']} "
        f"FN={summary['false_negatives']}"
    )
    print(f"precision={_format(summary['precision'])} recall={_format(summary['recall'])}")
    print(f"sredni blad polozenia: {summary['mean_error_px']:.1f}px")
    print(
        f"czas analizy - mean={_format_time(summary['mean_time_s'])} "
        f"median={_format_time(summary['median_time_s'])} "
        f"max={_format_time(summary['max_time_s'])}"
    )


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
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Jesli >0, dla kazdego obrazu dodatkowo powtarza analize N razy "
            "(tylko czas, bez precision/recall) i raportuje sredni czas oraz "
            "odchylenie standardowe."
        ),
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

        # ground truth's own "target_type" (jesli podane) jest traktowane
        # jako prawda odniesienia - nadpisuje typ zwrocony przez detektor.
        if "target_type" in entry:
            metrics["target_type"] = entry["target_type"]

        results.append(metrics)

        mm_part = (
            f" ({metrics['mean_error_mm']:.1f}mm)" if metrics["mean_error_mm"] is not None else ""
        )
        print(
            f"[{image_path}] "
            f"TP={metrics['true_positives']} FP={metrics['false_positives']} "
            f"FN={metrics['false_negatives']} "
            f"precision={_format(metrics['precision'])} recall={_format(metrics['recall'])} "
            f"mean_error={metrics['mean_error_px']:.1f}px{mm_part} "
            f"time={_format_time(metrics['time_total_s'])} "
            f"typ={metrics['target_type']}"
        )

    if not results:
        print("Brak wynikow do zagregowania (wszystkie obrazy pominieto).")
        return

    _print_summary("Podsumowanie", aggregate(results))

    by_group = aggregate_by_group(results)
    print("\n--- wg typu tarczy ---")
    for group, summary in by_group.items():
        _print_summary(f"typ: {group}", summary)

    if args.benchmark > 0:
        print(f"\n--- Benchmark (N={args.benchmark} powtorzen na obraz) ---")
        for image_path in ground_truth:
            bench = benchmark_image(image_path, args.benchmark)
            if bench["mean"] is None:
                print(f"[{image_path}] brak udanych przebiegow")
                continue
            print(
                f"[{image_path}] mean={_format_time(bench['mean'])} "
                f"std={_format_time(bench['std'])} (N={len(bench['times'])})"
            )


if __name__ == "__main__":
    main()
