from typing import Dict
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

    #wczytanie obrazu
    image = load_image(path)

    #preprocessing
    image = preprocess_image(image)

    #detekcja tarczy
    target = detect_target(image)

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
    hits = detect_hit(image, target)

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
    return session
