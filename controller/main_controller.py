import time
from typing import Dict, Tuple
from services.image_loader import load_image
from services.preprocessing import preprocess_image
from services.target_detection import detect_target
from services.hit_detection import detect_hit
from services.statistics import calculate_statistics
from services.manual_calibration import create_target_from_manual_input
from model.session import Session

class AnalysisError(Exception):
    """Błąd analizy obrazu"""
    pass



def analyze_image(path: str, metadata: Dict) -> Session:
    """
    Wykonuje pełną analizę obrazu tarczy strzeleckiej
    Zwraca obiekt Session
    """

    session, _timings = analyze_image_with_timing(path, metadata)
    return session


def analyze_image_with_timing(path: str, metadata: Dict) -> Tuple[Session, Dict[str, float]]:
    """
    To samo co analyze_image(), ale dodatkowo zwraca słownik z czasem
    (w sekundach) każdego etapu analizy: load_image, preprocess,
    detect_target, detect_hit, total.

    Używane przez evaluation/evaluate_hit_detection.py do pomiaru
    wydajności detekcji - samo analyze_image() nie zmienia zwracanego typu,
    więc GUI nie wymaga żadnych zmian.
    """

    timings: Dict[str, float] = {}
    overall_start = time.perf_counter()

    #wczytanie obrazu
    start = time.perf_counter()
    image = load_image(path)
    timings["load_image"] = time.perf_counter() - start

    #preprocessing
    start = time.perf_counter()
    image = preprocess_image(image)
    timings["preprocess"] = time.perf_counter() - start

    #detekcja tarczy
    start = time.perf_counter()
    target = detect_target(image)
    timings["detect_target"] = time.perf_counter() - start

    if target is None:
        if "manual_target" not in metadata:
            raise AnalysisError("Nie udało się wykryć tarczy i brak danych do kalibracji ręcznej.")

        manual = metadata["manual_target"]
        target = create_target_from_manual_input(
            center_x=manual["center_x"],
            center_y=manual["center_y"],
            radius=manual["radius"]
        )

    #detekcja trafień
    start = time.perf_counter()
    hits = detect_hit(image, target)
    timings["detect_hit"] = time.perf_counter() - start

    #Statystyki
    stats = calculate_statistics(hits)

    #utworzenie sesji
    session = Session(
        id=None,
        image=image,
        target=target,
        hits=hits,
        statistics=stats,
        metadata=metadata
    )

    timings["total"] = time.perf_counter() - overall_start
    return session, timings
