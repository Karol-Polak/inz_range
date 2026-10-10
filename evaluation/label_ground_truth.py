"""
Interaktywne narzedzie do recznego oznaczania polozenia trafien na zdjeciu
tarczy - generuje/rozszerza plik ground_truth.json uzywany przez
evaluation/evaluate_hit_detection.py.

Wymaga srodowiska z wyswietlaczem (nie zadziala przez SSH bez X11 / headless).

Obsluga:
  - lewy przycisk myszy  -> dodaj punkt trafienia
  - prawy przycisk myszy -> usun najblizszy oznaczony punkt
  - zamknij okno, aby zapisac wynik do pliku JSON

Uzycie:
    python -m evaluation.label_ground_truth assets/idpa_przestrzeliny.jpg \\
        --scale-mm-per-px 0.58
"""

import argparse
import json
import math
import os
from typing import List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def label_image(image_path: str) -> List[Tuple[float, float]]:
    image = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(
        "Lewy klik: dodaj trafienie | Prawy klik: usun najblizsze | "
        "Zamknij okno, aby zapisac"
    )

    points: List[Tuple[float, float]] = []
    markers = []

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            points.append((event.xdata, event.ydata))
            marker, = ax.plot(event.xdata, event.ydata, "rx", markersize=10, markeredgewidth=2)
            markers.append(marker)
        elif event.button == 3 and points:
            distances = [math.hypot(px - event.xdata, py - event.ydata) for px, py in points]
            closest_index = distances.index(min(distances))
            points.pop(closest_index)
            markers.pop(closest_index).remove()

        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    return points


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Sciezka do zdjecia tarczy do oznaczenia")
    parser.add_argument(
        "--output",
        default="assets/ground_truth.json",
        help="Plik ground truth do zapisu/rozszerzenia (domyslnie assets/ground_truth.json)",
    )
    parser.add_argument(
        "--scale-mm-per-px",
        type=float,
        default=None,
        help="Opcjonalna skala mm/px dla tego zdjecia, zapisywana w ground truth",
    )
    args = parser.parse_args()

    points = label_image(args.image)

    ground_truth = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as ground_truth_file:
            ground_truth = json.load(ground_truth_file)

    entry = {"hits": [[round(x, 1), round(y, 1)] for x, y in points]}
    if args.scale_mm_per_px is not None:
        entry["scale_mm_per_px"] = args.scale_mm_per_px

    ground_truth[args.image] = entry

    with open(args.output, "w", encoding="utf-8") as ground_truth_file:
        json.dump(ground_truth, ground_truth_file, indent=2, ensure_ascii=False)

    print(f"Zapisano {len(points)} trafien do {args.output} dla {args.image}")


if __name__ == "__main__":
    main()
