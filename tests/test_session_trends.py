from datetime import datetime, timedelta

from database.models import TrainingSession
from services.session_trends import build_trend_series, compare_sessions, filter_sessions, scale_warning


def _session(
    days_ago,
    hit_count=5,
    cep_50=10.0,
    accuracy_radius=2.0,
    precision_radius=8.0,
    scale_mm_per_px=None,
    weapon_type=None,
    distance_m=None,
):
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
        scale_mm_per_px=scale_mm_per_px,
        weapon_type=weapon_type,
        distance_m=distance_m,
    )


# ---------------------------------------------------------------------------
# scale_warning
# ---------------------------------------------------------------------------

def test_scale_warning_none_when_every_session_has_a_scale():
    a = _session(days_ago=1, scale_mm_per_px=0.5)
    b = _session(days_ago=2, scale_mm_per_px=0.8)  # different scale is fine

    assert scale_warning([a, b]) is None


def test_scale_warning_set_when_any_session_is_missing_a_scale():
    a = _session(days_ago=1, scale_mm_per_px=0.5)
    b = _session(days_ago=2, scale_mm_per_px=None)

    assert scale_warning([a, b]) is not None


# ---------------------------------------------------------------------------
# build_trend_series
# ---------------------------------------------------------------------------

def test_build_trend_series_sorts_oldest_first():
    newer = _session(days_ago=1, cep_50=5.0)
    older = _session(days_ago=10, cep_50=20.0)

    result = build_trend_series([newer, older])

    assert [point["cep_50"] for point in result["points"]] == [20.0, 5.0]
    assert result["points"][0]["created_at"] < result["points"][1]["created_at"]


def test_build_trend_series_converts_each_point_with_its_own_scale():
    older = _session(days_ago=10, cep_50=20.0, scale_mm_per_px=0.5)
    newer = _session(days_ago=1, cep_50=10.0, scale_mm_per_px=1.0)

    result = build_trend_series([newer, older])

    assert result["unit"] == "mm"
    assert result["warning"] is None
    assert [p["cep_50"] for p in result["points"]] == [10.0, 10.0]  # 20*0.5, 10*1.0


def test_build_trend_series_warns_and_uses_px_when_any_session_missing_scale():
    older = _session(days_ago=10, cep_50=20.0, scale_mm_per_px=0.5)
    newer = _session(days_ago=1, cep_50=10.0, scale_mm_per_px=None)

    result = build_trend_series([newer, older])

    assert result["unit"] == "px"
    assert result["warning"] is not None
    assert [p["cep_50"] for p in result["points"]] == [20.0, 10.0]  # unconverted


# ---------------------------------------------------------------------------
# compare_sessions
# ---------------------------------------------------------------------------

def test_compare_sessions_computes_deltas_from_first_to_second():
    first = _session(days_ago=10, cep_50=20.0, accuracy_radius=5.0, precision_radius=15.0, hit_count=5)
    second = _session(days_ago=1, cep_50=12.0, accuracy_radius=3.0, precision_radius=9.0, hit_count=8)

    comparison = compare_sessions(first, second)

    assert comparison["hit_count_delta"] == 3
    assert comparison["cep_50_delta"] == -8.0
    assert comparison["accuracy_radius_delta"] == -2.0
    assert comparison["precision_radius_delta"] == -6.0


def test_compare_sessions_converts_to_mm_when_both_have_the_same_scale():
    first = _session(days_ago=10, cep_50=20.0, scale_mm_per_px=0.5)
    second = _session(days_ago=1, cep_50=12.0, scale_mm_per_px=0.5)

    comparison = compare_sessions(first, second)

    assert comparison["unit"] == "mm"
    assert comparison["warning"] is None
    assert comparison["cep_50_first"] == 10.0  # 20 * 0.5
    assert comparison["cep_50_second"] == 6.0  # 12 * 0.5
    assert comparison["cep_50_delta"] == -4.0


def test_compare_sessions_converts_each_session_with_its_own_different_scale():
    # Different shooting distances / calibrations - converting each to mm
    # with its OWN scale is exactly what makes them comparable.
    first = _session(days_ago=10, cep_50=20.0, scale_mm_per_px=0.5)
    second = _session(days_ago=1, cep_50=10.0, scale_mm_per_px=1.0)

    comparison = compare_sessions(first, second)

    assert comparison["unit"] == "mm"
    assert comparison["warning"] is None
    assert comparison["cep_50_first"] == 10.0   # 20 * 0.5
    assert comparison["cep_50_second"] == 10.0  # 10 * 1.0
    assert comparison["cep_50_delta"] == 0.0


def test_compare_sessions_warns_and_uses_px_when_one_session_has_no_scale():
    first = _session(days_ago=10, cep_50=20.0, scale_mm_per_px=0.5)
    second = _session(days_ago=1, cep_50=12.0, scale_mm_per_px=None)

    comparison = compare_sessions(first, second)

    assert comparison["unit"] == "px"
    assert comparison["warning"] is not None
    assert comparison["cep_50_first"] == 20.0
    assert comparison["cep_50_second"] == 12.0
    assert comparison["cep_50_delta"] == -8.0


# ---------------------------------------------------------------------------
# filter_sessions
# ---------------------------------------------------------------------------

def test_filter_sessions_empty_query_returns_all():
    a = _session(days_ago=1, weapon_type="Glock 17")
    b = _session(days_ago=2, weapon_type="AR-15")

    assert filter_sessions([a, b], "") == [a, b]


def test_filter_sessions_matches_weapon_type_case_insensitively():
    a = _session(days_ago=1, weapon_type="Glock 17")
    b = _session(days_ago=2, weapon_type="AR-15")

    assert filter_sessions([a, b], "glock") == [a]


def test_filter_sessions_matches_distance():
    a = _session(days_ago=1, weapon_type="Glock 17", distance_m=10.0)
    b = _session(days_ago=2, weapon_type="AR-15", distance_m=25.0)

    assert filter_sessions([a, b], "25") == [b]


def test_filter_sessions_excludes_sessions_without_a_match():
    a = _session(days_ago=1, weapon_type="Glock 17")

    assert filter_sessions([a], "AR-15") == []
