from typing import Dict, List

from database.models import TrainingSession


def build_trend_series(sessions: List[TrainingSession]) -> List[Dict]:
    """
    Sortuje sesje chronologicznie (od najstarszej) i zwraca listę punktów
    gotowych do narysowania wykresu trendu (CEP 50%, celność, precyzja w czasie).
    """

    sorted_sessions = sorted(sessions, key=lambda s: s.created_at)

    return [
        {
            "created_at": session.created_at,
            "cep_50": session.cep_50,
            "accuracy_radius": session.accuracy_radius,
            "precision_radius": session.precision_radius,
        }
        for session in sorted_sessions
    ]


def compare_sessions(first: TrainingSession, second: TrainingSession) -> Dict[str, float]:
    """Zwraca różnicę kluczowych metryk (second - first)."""

    return {
        "hit_count_delta": second.hit_count - first.hit_count,
        "accuracy_radius_delta": second.accuracy_radius - first.accuracy_radius,
        "precision_radius_delta": second.precision_radius - first.precision_radius,
        "cep_50_delta": second.cep_50 - first.cep_50,
    }
