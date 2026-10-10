from datetime import datetime, timedelta

from database.models import TrainingSession
from services.session_trends import build_trend_series, compare_sessions


def _session(days_ago, hit_count=5, cep_50=10.0, accuracy_radius=2.0, precision_radius=8.0):
    return TrainingSession(
        id=days_ago,
        created_at=datetime(2026, 1, 10) - timedelta(days=days_ago),
        image_path="x.jpg",
        target_center_x=0.0,
        target_center_y=0.0,
        target_radius=100.0,
        target_type="circular",
        hit_count=hit_count,
        accuracy_radius=accuracy_radius,
        precision_radius=precision_radius,
        precision_std=1.0,
        cep_50=cep_50,
        extreme_spread=5.0,
        outliers_count=0,
        mean_x=0.0,
        mean_y=0.0,
    )


def test_build_trend_series_sorts_oldest_first():
    newer = _session(days_ago=1, cep_50=5.0)
    older = _session(days_ago=10, cep_50=20.0)

    series = build_trend_series([newer, older])

    assert [point["cep_50"] for point in series] == [20.0, 5.0]
    assert series[0]["created_at"] < series[1]["created_at"]


def test_compare_sessions_computes_deltas_from_first_to_second():
    first = _session(days_ago=10, cep_50=20.0, accuracy_radius=5.0, precision_radius=15.0, hit_count=5)
    second = _session(days_ago=1, cep_50=12.0, accuracy_radius=3.0, precision_radius=9.0, hit_count=8)

    comparison = compare_sessions(first, second)

    assert comparison["hit_count_delta"] == 3
    assert comparison["cep_50_delta"] == -8.0
    assert comparison["accuracy_radius_delta"] == -2.0
    assert comparison["precision_radius_delta"] == -6.0
