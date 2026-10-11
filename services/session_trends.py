from typing import Dict, List, Optional

from database.models import TrainingSession

# Metrics that are stored in px and need converting to mm using a session's
# own scale_mm_per_px when showing them in the Sessions page.
_SCALED_METRICS = ("accuracy_radius", "precision_radius", "cep_50")


def scale_warning(sessions: List[TrainingSession]) -> Optional[str]:
    """
    Sesje bez ustawionej skali mm/px nie da się rzetelnie zmieszać w mm z
    sesjami, które skalę mają - te samo wartość w px może oznaczać inną
    odległość w mm. Zwraca komunikat ostrzegawczy, jeśli PRZYNAJMNIEJ JEDNA
    z sesji nie ma skali; jeśli każda ma swoją (nawet różną) skalę, zwraca
    None - każdą da się bezpiecznie przeliczyć do mm jej własną skalą.
    """

    if any(session.scale_mm_per_px is None for session in sessions):
        return (
            "Przynajmniej jedna z porównywanych sesji nie ma ustawionej "
            "skali kalibracji - wyniki pokazane w px."
        )
    return None


def build_trend_series(sessions: List[TrainingSession]) -> Dict:
    """
    Sortuje sesje chronologicznie (od najstarszej) i zwraca punkty gotowe
    do narysowania wykresu trendu (CEP 50%, celność, precyzja w czasie).

    Każdy punkt jest przeliczany do mm WŁASNĄ skalą tej sesji (różne sesje
    mogą mieć różną odległość strzelania / kalibrację). Jeśli choć jedna
    sesja nie ma ustawionej skali, cały wykres wraca w px z ostrzeżeniem -
    nie da się bezpiecznie zmieszać mm i px na jednej osi.
    """

    warning = scale_warning(sessions)
    unit = "px" if warning else "mm"

    points = []
    for session in sorted(sessions, key=lambda s: s.created_at):
        scale = 1.0 if warning else session.scale_mm_per_px
        points.append(
            {
                "created_at": session.created_at,
                "cep_50": session.cep_50 * scale,
                "accuracy_radius": session.accuracy_radius * scale,
                "precision_radius": session.precision_radius * scale,
            }
        )

    return {"unit": unit, "warning": warning, "points": points}


def compare_sessions(first: TrainingSession, second: TrainingSession) -> Dict:
    """
    Zwraca wartości i różnicę (second - first) kluczowych metryk.

    Jeśli OBIE sesje mają ustawioną własną skalę (nawet różną), każda jest
    przeliczana do mm JEJ WŁASNĄ skalą - to właśnie czyni je porównywalnymi
    mimo różnych odległości/kalibracji. W przeciwnym razie wynik jest w px
    z ostrzeżeniem (mieszanie mm i px dałoby nieprawdziwy wynik).
    """

    warning = scale_warning([first, second])
    scale_first = 1.0 if warning else first.scale_mm_per_px
    scale_second = 1.0 if warning else second.scale_mm_per_px

    result = {
        "unit": "px" if warning else "mm",
        "warning": warning,
        "hit_count_first": first.hit_count,
        "hit_count_second": second.hit_count,
        "hit_count_delta": second.hit_count - first.hit_count,
    }

    for metric in _SCALED_METRICS:
        first_value = getattr(first, metric) * scale_first
        second_value = getattr(second, metric) * scale_second
        result[f"{metric}_first"] = first_value
        result[f"{metric}_second"] = second_value
        result[f"{metric}_delta"] = second_value - first_value

    return result


def filter_sessions(sessions: List[TrainingSession], query: str) -> List[TrainingSession]:
    """
    Proste filtrowanie po treści - dopasowuje zapytanie (bez rozróżniania
    wielkości liter) do rodzaju broni i odległości strzelania. Pusty query
    zwraca wszystkie sesje bez zmian.
    """

    query = query.strip().lower()
    if not query:
        return list(sessions)

    def matches(session: TrainingSession) -> bool:
        haystack = f"{session.weapon_type or ''} {session.distance_m or ''}".lower()
        return query in haystack

    return [session for session in sessions if matches(session)]
