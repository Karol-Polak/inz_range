import ast
import math
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from database.models import TrainingSession
from services.session_trends import build_trend_series, compare_sessions
from services.units import (
    LengthUnit,
    MissingInput,
    UnitContext,
    available_units,
    common_unit,
    convert_length,
    convert_statistics,
    default_unit,
    format_length,
    format_number,
    is_available,
    missing_inputs,
    missing_reason_text,
    mm_to_moa,
    mm_to_mrad,
    moa_to_mm,
    mrad_to_mm,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FULL = UnitContext(scale_mm_per_px=1.0, distance_m=100.0)


# ---------------------------------------------------------------------------
# Known values (definitions from the specification)
# ---------------------------------------------------------------------------

def test_one_moa_at_100_m_is_29_0888_mm():
    assert mm_to_moa(29.0888, 100.0) == pytest.approx(1.0, abs=1e-5)
    assert moa_to_mm(1.0, 100.0) == pytest.approx(100000.0 * math.tan(math.pi / 10800), rel=1e-12)
    assert moa_to_mm(1.0, 100.0) == pytest.approx(29.0888, abs=1e-4)


def test_one_moa_at_25_m_is_7_2722_mm():
    assert mm_to_moa(7.2722, 25.0) == pytest.approx(1.0, abs=1e-4)


def test_ten_mm_at_10_m_is_one_mrad():
    assert mm_to_mrad(10.0, 10.0) == pytest.approx(1.0, abs=1e-6)


def test_one_mrad_is_about_3_4377_moa():
    length_mm = mrad_to_mm(1.0, 50.0)
    assert mm_to_moa(length_mm, 50.0) == pytest.approx(3.4377, abs=1e-4)


def test_moa_is_angular_not_shooters_moa():
    # Shooter's MOA would be 1.047" at 100 yd = 26.594 mm at 91.44 m.
    # True MOA at 91.44 m is 26.599 mm - close, but not the same definition.
    assert moa_to_mm(1.0, 91.44) == pytest.approx(91440.0 * math.tan(math.pi / 10800), rel=1e-12)


def test_convert_length_px_scale_distance_chain():
    ctx = UnitContext(scale_mm_per_px=0.5, distance_m=25.0)
    # 14.5444 px * 0.5 mm/px = 7.2722 mm at 25 m = 1 MOA
    assert convert_length(14.5444, LengthUnit.MM, ctx) == pytest.approx(7.2722)
    assert convert_length(14.5444, LengthUnit.MOA, ctx) == pytest.approx(1.0, abs=1e-4)
    assert convert_length(20.0, LengthUnit.MRAD, UnitContext(0.5, 10.0)) == pytest.approx(1.0, abs=1e-6)
    assert convert_length(3.0, LengthUnit.PX, UnitContext()) == 3.0


def test_small_angle_difference_matches_documented_estimate():
    # Relative difference atan vs small-angle ~ (L/D)^2 / 3.
    length_mm, distance_m = 150.0, 10.0
    ratio = length_mm / (distance_m * 1000.0)
    exact = mm_to_mrad(length_mm, distance_m) / 1000.0
    relative = (ratio - exact) / exact
    assert relative == pytest.approx(ratio ** 2 / 3, rel=1e-3)


# ---------------------------------------------------------------------------
# Sign, round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit", [LengthUnit.MM, LengthUnit.MOA, LengthUnit.MRAD, LengthUnit.PX])
def test_sign_is_preserved(unit):
    ctx = UnitContext(scale_mm_per_px=0.3, distance_m=25.0)
    positive = convert_length(42.0, unit, ctx)
    negative = convert_length(-42.0, unit, ctx)
    assert positive > 0
    assert negative == pytest.approx(-positive)


@pytest.mark.parametrize("length_mm", [0.5, 7.2722, 50.0, 300.0])
@pytest.mark.parametrize("distance_m", [5.0, 25.0, 100.0])
def test_round_trip_mm_angular(length_mm, distance_m):
    assert moa_to_mm(mm_to_moa(length_mm, distance_m), distance_m) == pytest.approx(length_mm, rel=1e-12)
    assert mrad_to_mm(mm_to_mrad(length_mm, distance_m), distance_m) == pytest.approx(length_mm, rel=1e-12)


def test_moa_and_mrad_are_consistent_with_each_other():
    ctx = UnitContext(scale_mm_per_px=0.2, distance_m=25.0)
    moa = convert_length(100.0, LengthUnit.MOA, ctx)
    mrad = convert_length(100.0, LengthUnit.MRAD, ctx)
    assert moa == pytest.approx(mrad / 1000.0 * 180.0 / math.pi * 60.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Availability: missing scale / distance => None, never a guess
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [None, 0.0, -1.0])
def test_missing_scale_makes_mm_and_angular_unavailable(scale):
    ctx = UnitContext(scale_mm_per_px=scale, distance_m=25.0)
    assert convert_length(10.0, LengthUnit.MM, ctx) is None
    assert convert_length(10.0, LengthUnit.MOA, ctx) is None
    assert convert_length(10.0, LengthUnit.MRAD, ctx) is None
    assert missing_inputs(LengthUnit.MOA, ctx) == (MissingInput.SCALE,)
    assert convert_length(10.0, LengthUnit.PX, ctx) == 10.0


@pytest.mark.parametrize("distance", [None, 0.0, -5.0])
def test_missing_distance_makes_only_angular_unavailable(distance):
    ctx = UnitContext(scale_mm_per_px=0.5, distance_m=distance)
    assert convert_length(10.0, LengthUnit.MM, ctx) == 5.0
    assert convert_length(10.0, LengthUnit.MOA, ctx) is None
    assert convert_length(10.0, LengthUnit.MRAD, ctx) is None
    assert missing_inputs(LengthUnit.MRAD, ctx) == (MissingInput.DISTANCE,)


def test_missing_both():
    ctx = UnitContext()
    assert set(missing_inputs(LengthUnit.MOA, ctx)) == {MissingInput.SCALE, MissingInput.DISTANCE}
    assert available_units(ctx) == [LengthUnit.PX]
    assert "skali i odległości" in missing_reason_text(LengthUnit.MOA, ctx)


def test_reason_texts():
    assert missing_reason_text(LengthUnit.PX, UnitContext()) is None
    assert "Ustaw skalę" in missing_reason_text(LengthUnit.MM, UnitContext())
    assert "odległości strzelania" in missing_reason_text(LengthUnit.MOA, UnitContext(0.5, None))
    assert "skali" in missing_reason_text(LengthUnit.MRAD, UnitContext(None, 25.0))
    assert missing_reason_text(LengthUnit.MOA, FULL) is None


def test_default_unit():
    assert default_unit(UnitContext()) is LengthUnit.PX
    assert default_unit(UnitContext(scale_mm_per_px=0.5)) is LengthUnit.MM
    assert default_unit(FULL) is LengthUnit.MM  # angular only on request
    assert is_available(LengthUnit.MOA, FULL)


# ---------------------------------------------------------------------------
# Statistics dict conversion
# ---------------------------------------------------------------------------

def test_convert_statistics_converts_lengths_not_counts():
    stats = {
        "count": 9, "outliers_count": 1,
        "accuracy_radius": 10.0, "precision_radius": 20.0, "precision_std": 2.0,
        "precision_max_radius": 30.0, "cep_50": 18.0, "p25": 5.0, "p75": 25.0,
        "std_x": 4.0, "std_y": 6.0, "extreme_spread": 50.0, "mean_x": -3.0, "mean_y": 7.0,
    }
    ctx = UnitContext(scale_mm_per_px=0.5, distance_m=10.0)
    converted = convert_statistics(stats, LengthUnit.MRAD, ctx)

    assert converted["count"] == 9
    assert converted["outliers_count"] == 1
    for key in ("p25", "p75", "std_x", "std_y", "extreme_spread", "cep_50"):
        assert converted[key] == pytest.approx(mm_to_mrad(stats[key] * 0.5, 10.0))
    assert converted["mean_x"] < 0  # sign kept
    assert stats["cep_50"] == 18.0  # input untouched
    assert convert_statistics(stats, LengthUnit.MOA, UnitContext(0.5, None)) is None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def test_formatting_uses_decimal_comma_and_unit_decimals():
    assert format_number(7.7) == "7,7"
    assert format_number(-1.25, 2, signed=True) == "-1,25"
    assert format_number(1.25, 2, signed=True) == "+1,25"
    assert format_length(7.7, LengthUnit.MOA) == "7,70 MOA"
    assert format_length(0.5, LengthUnit.MRAD) == "0,50 mrad"
    assert format_length(-1.94, LengthUnit.MM) == "-1,9 mm"
    assert format_length(184.0, LengthUnit.PX) == "184,0 px"


# ---------------------------------------------------------------------------
# common_unit + sessions trend / comparison with per-session distance
# ---------------------------------------------------------------------------

def test_common_unit_fallbacks():
    full_a = UnitContext(0.5, 25.0)
    full_b = UnitContext(0.3, 50.0)
    no_distance = UnitContext(0.5, None)
    no_scale = UnitContext(None, 25.0)

    assert common_unit(LengthUnit.MOA, [full_a, full_b]) == (LengthUnit.MOA, None)

    unit, warning = common_unit(LengthUnit.MOA, [full_a, no_distance, full_b])
    assert unit is LengthUnit.MM
    assert "1 z 3 sesji nie ma zapisanej odległości strzelania" in warning
    assert "w mm zamiast MOA" in warning

    unit, warning = common_unit(LengthUnit.MRAD, [full_a, no_scale])
    assert unit is LengthUnit.PX
    assert "nie ma zapisanej skali" in warning

    assert common_unit(LengthUnit.PX, [no_scale]) == (LengthUnit.PX, None)


def _session(days_ago, cep_50, scale=None, distance=None, extreme_spread=40.0):
    return TrainingSession(
        id=days_ago,
        created_at=datetime(2026, 1, 10) - timedelta(days=days_ago),
        image_path="x.jpg",
        target_center_x=0.0, target_center_y=0.0, target_radius=100.0, target_type="circular",
        hit_count=5, accuracy_radius=4.0, precision_radius=8.0, precision_std=1.0,
        cep_50=cep_50, extreme_spread=extreme_spread, outliers_count=0, mean_x=0.0, mean_y=0.0,
        scale_mm_per_px=scale, distance_m=distance,
    )


def test_trend_converts_each_session_with_its_own_distance():
    # Same physical group (14.5444 px * 0.5 = 7.2722 mm) at 25 m and 50 m.
    near = _session(days_ago=5, cep_50=14.5444, scale=0.5, distance=25.0)
    far = _session(days_ago=1, cep_50=14.5444, scale=0.5, distance=50.0)

    trend = build_trend_series([far, near], unit=LengthUnit.MOA)

    assert trend["unit"] == "MOA"
    assert trend["warning"] is None
    assert [p["cep_50"] for p in trend["points"]] == pytest.approx([1.0, 0.5], abs=1e-4)


def test_trend_falls_back_to_mm_when_a_distance_is_missing():
    a = _session(days_ago=5, cep_50=10.0, scale=0.5, distance=25.0)
    b = _session(days_ago=1, cep_50=10.0, scale=0.5, distance=None)

    trend = build_trend_series([a, b], unit=LengthUnit.MOA)

    assert trend["unit"] == "mm"
    assert "odległości strzelania" in trend["warning"]
    assert [p["cep_50"] for p in trend["points"]] == [5.0, 5.0]


def test_trend_default_behaviour_unchanged():
    a = _session(days_ago=5, cep_50=10.0, scale=0.5, distance=25.0)
    b = _session(days_ago=1, cep_50=10.0, scale=None)
    trend = build_trend_series([a, b])
    assert trend["unit"] == "px"
    assert trend["points"][0]["cep_50"] == 10.0


def test_comparison_in_mrad_includes_extreme_spread():
    first = _session(days_ago=5, cep_50=20.0, scale=0.5, distance=10.0, extreme_spread=40.0)
    second = _session(days_ago=1, cep_50=10.0, scale=0.5, distance=20.0, extreme_spread=40.0)

    result = compare_sessions(first, second, unit=LengthUnit.MRAD)

    assert result["unit"] == "mrad"
    assert result["warning"] is None
    assert result["cep_50_first"] == pytest.approx(1.0, abs=1e-6)        # 10 mm @ 10 m
    assert result["cep_50_second"] == pytest.approx(0.25, abs=1e-6)      # 5 mm @ 20 m
    assert result["extreme_spread_first"] == pytest.approx(2.0, abs=1e-5)  # 20 mm @ 10 m
    assert result["extreme_spread_delta"] == pytest.approx(-1.0, abs=1e-5)
    assert result["hit_count_delta"] == 0  # counts never converted


def test_comparison_falls_back_to_px_without_scale():
    first = _session(days_ago=5, cep_50=20.0, scale=None, distance=10.0)
    second = _session(days_ago=1, cep_50=10.0, scale=0.5, distance=20.0)

    result = compare_sessions(first, second, unit=LengthUnit.MOA)

    assert result["unit"] == "px"
    assert "skali" in result["warning"]
    assert result["cep_50_first"] == 20.0


# ---------------------------------------------------------------------------
# Layering: pure service, view only displays
# ---------------------------------------------------------------------------

def test_units_service_imports_without_pyqt():
    code = (
        "import sys; import services.units; "
        "bad = [m for m in sys.modules if m.startswith('PyQt')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_units_service_has_no_qt_imports_in_source():
    modules = _imported_modules(PROJECT_ROOT / "services" / "units.py")
    assert not any(m.startswith("PyQt") for m in modules)


@pytest.mark.parametrize("view_file", ["main_window.py", "statistics_page.py", "sessions_page.py"])
def test_view_layer_uses_units_service(view_file):
    assert "services.units" in _imported_modules(PROJECT_ROOT / "view" / view_file)
