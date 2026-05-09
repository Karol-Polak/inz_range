import math
import matplotlib.pyplot as plt
from typing import List
from model.hit import Hit


def plot_hits_scatter(hits: List[Hit], cep_radius: float = None) -> None:
    """
    Rysuje wykres rozrzutu trafień (scatter plot).
    """

    valid_hits = [hit for hit in hits if hit.valid]
    if not valid_hits:
        print("Brak trafień do wizualizacji.")
        return

    xs = [hit.x for hit in valid_hits]
    ys = [hit.y for hit in valid_hits]

    plt.figure()
    plt.scatter(xs, ys)
    plt.scatter(0, 0, marker='x')  # środek tarczy

    if cep_radius is not None:
        circle = plt.Circle((0, 0), cep_radius, fill=False)
        plt.gca().add_patch(circle)

    plt.axhline(0)
    plt.axvline(0)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Rozrzut trafień")

    plt.show()

def plot_radius_histogram(hits: List[Hit], mean_radius: float = None, cep_radius: float = None) -> None:
    """
    Rysuje histogram odległości trafień od środka tarczy.
    """

    valid_hits = [hit for hit in hits if hit.valid]
    if not valid_hits:
        print("Brak trafień do wizualizacji.")
        return

    radii = [math.sqrt(hit.x**2 + hit.y**2) for hit in valid_hits]

    plt.figure()
    plt.hist(radii, bins=10)

    if mean_radius is not None:
        plt.axvline(mean_radius, linestyle='--', label="Średnia")

    if cep_radius is not None:
        plt.axvline(cep_radius, linestyle=':', label="CEP 50%")

    plt.xlabel("Odległość od środka")
    plt.ylabel("Liczba trafień")
    plt.title("Histogram odległości trafień")
    plt.legend()

    plt.show()

def plot_xy_histograms(hits: List[Hit]) -> None:
    """
    Rysuje histogramy odchyleń X i Y.
    """

    valid_hits = [hit for hit in hits if hit.valid]
    if not valid_hits:
        print("Brak trafień do wizualizacji.")
        return

    xs = [hit.x for hit in valid_hits]
    ys = [hit.y for hit in valid_hits]

    plt.figure()
    plt.hist(xs, bins=10, alpha=0.7, label="X")
    plt.hist(ys, bins=10, alpha=0.7, label="Y")

    plt.xlabel("Odchylenie")
    plt.ylabel("Liczba trafień")
    plt.title("Histogramy odchyleń X i Y")
    plt.legend()

    plt.show()
