from typing import Dict, List, Optional

from database.models import TrainingSession
from services.units import LengthUnit, UnitContext, common_unit, convert_length

# Metrics that are stored in px and need converting (mm/MOA/mrad) using a
# session's own scale_mm_per_px and distance_m when showing them in the
# Sessions page.
_SCALED_METRICS = ("accuracy_radius", "precision_radius", "cep_50", "extreme_spread")
_TREND_METRICS = ("cep_50", "accuracy_radius", "precision_radius")


def unit_context(session: TrainingSession) -> UnitContext:
    """Skala i odległość zapisane z TĄ sesją (bez żadnych wartości zastępczych)."""
    return UnitContext(scale_mm_per_px=session.scale_mm_per_px, distance_m=session.distance_m)


def _resolve_unit(sessions: List[TrainingSession], unit: Optional[LengthUnit]):
    """
    unit=None - dotychczasowe zachowanie: mm, gdy każda sesja ma skalę,
    w przeciwnym razie px z ostrzeżeniem scale_warning().
    W przeciwnym razie żądana jednostka albo najlepsza wspólna (common_unit).
    """
    if unit is None:
        warning = scale_warning(sessions)
        return (LengthUnit.PX if warning else LengthUnit.MM), warning
    return common_unit(LengthUnit(unit), [unit_context(s) for s in sessions])


def _convert(session: TrainingSession, metric: str, unit: LengthUnit) -> float:
    return convert_length(getattr(session, metric), unit, unit_context(session))


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


def build_trend_series(sessions: List[TrainingSession], unit: Optional[LengthUnit] = None) -> Dict:
    """
    Sortuje sesje chronologicznie (od najstarszej) i zwraca punkty gotowe
    do narysowania wykresu trendu (CEP 50%, celność, precyzja w czasie).

    Każdy punkt jest przeliczany do mm WŁASNĄ skalą tej sesji (różne sesje
    mogą mieć różną odległość strzelania / kalibrację). Jeśli choć jedna
    sesja nie ma ustawionej skali, cały wykres wraca w px z ostrzeżeniem -
    nie da się bezpiecznie zmieszać mm i px na jednej osi.

    Z parametrem unit (mm/MOA/mrad) każdy punkt jest przeliczany własną
    skalą I własną odległością sesji; gdy którejś sesji czegoś brakuje,
    wykres wraca w najlepszej wspólnej jednostce z ostrzeżeniem.
    """

    shown_unit, warning = _resolve_unit(sessions, unit)

    points = []
    for session in sorted(sessions, key=lambda s: s.created_at):
        point = {"created_at": session.created_at}
        for metric in _TREND_METRICS:
            point[metric] = _convert(session, metric, shown_unit)
        points.append(point)

    return {"unit": shown_unit.value, "warning": warning, "points": points}


def compare_sessions(
    first: TrainingSession, second: TrainingSession, unit: Optional[LengthUnit] = None
) -> Dict:
    """
    Zwraca wartości i różnicę (second - first) kluczowych metryk.

    Jeśli OBIE sesje mają ustawioną własną skalę (nawet różną), każda jest
    przeliczana do mm JEJ WŁASNĄ skalą - to właśnie czyni je porównywalnymi
    mimo różnych odległości/kalibracji. W przeciwnym razie wynik jest w px
    z ostrzeżeniem (mieszanie mm i px dałoby nieprawdziwy wynik).

    Z parametrem unit (mm/MOA/mrad) każda sesja jest przeliczana własną
    skalą i odległością; brakujące dane => najlepsza wspólna jednostka
    z ostrzeżeniem (common_unit).
    """

    shown_unit, warning = _resolve_unit([first, second], unit)

    result = {
        "unit": shown_unit.value,
        "warning": warning,
        "hit_count_first": first.hit_count,
        "hit_count_second": second.hit_count,
        "hit_count_delta": second.hit_count - first.hit_count,
    }

    for metric in _SCALED_METRICS:
        first_value = _convert(first, metric, shown_unit)
        second_value = _convert(second, metric, shown_unit)
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
