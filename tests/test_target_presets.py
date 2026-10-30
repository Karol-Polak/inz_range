import ast
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from services.target_presets import (
    ReferenceFeature,
    all_presets,
    feature_display_text,
    find_feature,
    get_preset,
    measurement_hint,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ISSF_IDS = ("issf_25m_precision", "issf_10m_air_pistol", "issf_10m_air_rifle", "issf_50m_rifle")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def test_preset_ids_are_unique():
    ids = [preset.id for preset in all_presets()]
    assert len(ids) == len(set(ids))


def test_all_sizes_positive_and_finite():
    for preset in all_presets():
        assert preset.features, preset.id
        for feature in preset.features:
            assert math.isfinite(feature.size_mm) and feature.size_mm > 0, (preset.id, feature)


def test_every_preset_has_source_and_confidence():
    for preset in all_presets():
        assert preset.name.strip()
        assert preset.source.strip()
        assert preset.confidence.strip()


def _ring(preset_id, prefix):
    preset = get_preset(preset_id)
    return next((f.size_mm for f in preset.features if f.label.startswith(prefix)), None)


@pytest.mark.parametrize("preset_id", ISSF_IDS)
def test_issf_ring_order_is_consistent(preset_id):
    ring_1 = _ring(preset_id, "pierścień 1")
    black = _ring(preset_id, "czerń")
    ten = _ring(preset_id, "dziesiątka")

    assert ring_1 is not None and black is not None
    assert ring_1 > black
    if ten is not None:
        assert black > ten


def test_needs_check_exactly_for_unverified_presets():
    flagged = {preset.id for preset in all_presets() if preset.needs_check}
    assert flagged == {"issf_10m_air_rifle", "issf_50m_rifle", "paper_17x17", "ipsc"}


def test_round_target_false_only_for_ipsc_and_paper():
    not_round = {preset.id for preset in all_presets() if not preset.round_target}
    assert not_round == {"ipsc", "paper_17x17"}


def test_no_unverified_targets_included():
    names = " ".join(p.name.lower() + " " + p.id for p in all_presets())
    assert "sylwetk" not in names and "police" not in names
    assert "szybkostrzel" not in names and "rapid" not in names


def test_specified_values_are_stored_exactly():
    assert find_feature("issf_25m_precision", "czerń (pierścienie 7–10)").size_mm == 200.0
    assert find_feature("issf_10m_air_pistol", "dziesiątka").size_mm == 11.5
    assert find_feature("issf_50m_rifle", "czerń (pierścienie 4–10)").size_mm == 112.4
    assert find_feature("paper_17x17", "bok arkusza").size_mm == 170.0
    assert find_feature("ipsc", "wysokość").size_mm == 580.0


def test_unknown_ids_and_labels():
    assert get_preset("nie_istnieje") is None
    assert find_feature("nie_istnieje", "dziesiątka") is None
    assert find_feature("issf_25m_precision", "nie ma takiego elementu") is None


def test_feature_display_text_uses_decimal_comma():
    assert feature_display_text(find_feature("issf_25m_precision", "czerń (pierścienie 7–10)")) == (
        "czerń (pierścienie 7–10) – 200 mm"
    )
    assert feature_display_text(ReferenceFeature("pierścień 1 (zewnętrzny)", 155.5)) == (
        "pierścień 1 (zewnętrzny) – 155,5 mm"
    )


def test_measurement_hints():
    assert measurement_hint(get_preset("issf_25m_precision")) is None
    assert "szerokość lub wysokość" in measurement_hint(get_preset("ipsc"))
    paper = measurement_hint(get_preset("paper_17x17"))
    assert paper.startswith("Zmierz bok arkusza")
    assert "szerokość" not in paper


def test_target_presets_module_has_no_qt():
    tree = ast.parse((PROJECT_ROOT / "services" / "target_presets.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("PyQt")
        elif isinstance(node, ast.Import):
            assert not any(alias.name.startswith("PyQt") for alias in node.names)

    code = (
        "import sys; import services.target_presets; "
        "sys.exit(1 if any(m.startswith('PyQt') for m in sys.modules) else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Settings page behaviour (offscreen Qt, no new dependencies)
# ---------------------------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qapp):
    from view.settings_page import SettingsPage

    return SettingsPage()


def _select_preset(page, preset_id):
    combo = page._preset_combo
    combo.setCurrentIndex(combo.findData(preset_id))


def _select_feature(page, label):
    combo = page._feature_combo
    combo.setCurrentIndex(combo.findData(label))


def _custom_selected(page):
    return page._preset_combo.currentData() is None and page._preset_combo.currentText() == "Własny wymiar…"


def test_custom_is_default_and_behaviour_unchanged(page):
    assert _custom_selected(page)
    assert page._diameter_mm.value() == 500.0
    assert page._diameter_px.value() == 800
    assert page.get_scale_mm_per_px() is None  # conversion off by default

    page._scale_enabled.setChecked(True)
    assert page.get_scale_mm_per_px() == pytest.approx(500.0 / 800)

    page.set_detected_radius(809.0)  # auto-fill still works with "Własny wymiar…"
    assert page._diameter_px.value() == 1618
    assert page.get_scale_mm_per_px() == pytest.approx(500.0 / 1618)


def test_preset_selection_fills_the_size_field(page):
    _select_preset(page, "issf_10m_air_pistol")
    _select_feature(page, "czerń (pierścienie 7–10)")

    assert page._diameter_mm.value() == 59.5
    assert page._feature_combo.currentText() == "czerń (pierścienie 7–10) – 59,5 mm"
    assert not _custom_selected(page)
    assert not page._scale_enabled.isChecked()  # checkbox never touched

    page._scale_enabled.setChecked(True)
    page.set_measured_pixel_distance(238.0)
    assert page.get_scale_mm_per_px() == pytest.approx(59.5 / 238)  # math unchanged


def test_manual_edit_switches_back_to_custom_keeping_value(page):
    _select_preset(page, "issf_25m_precision")
    _select_feature(page, "dziesiątka")
    assert page._diameter_mm.value() == 50.0

    page._diameter_mm.setValue(123.4)

    assert _custom_selected(page)
    assert page._diameter_mm.value() == 123.4


def test_preset_blocks_detector_autofill_but_not_two_point_measurement(page):
    page._diameter_px.setValue(700)
    _select_preset(page, "issf_25m_precision")

    page.set_detected_radius(809.0)
    assert page._diameter_px.value() == 700  # detector does not know which ring it found

    page.set_measured_pixel_distance(412.6)
    assert page._diameter_px.value() == 413


def test_measure_warning_until_measured_or_px_typed(page):
    _select_preset(page, "issf_25m_precision")
    assert page._awaiting_measurement

    page.set_measured_pixel_distance(400.0)
    assert not page._awaiting_measurement

    _select_feature(page, "dziesiątka")  # new feature -> measure again
    assert page._awaiting_measurement

    page._diameter_px.setValue(321)  # typed by hand
    assert not page._awaiting_measurement


def test_warning_source_line_follows_needs_check(page):
    _select_preset(page, "issf_25m_precision")
    assert page._source_line.text() == ""

    _select_preset(page, "ipsc")
    assert "podane przez autora" in page._source_line.text()
    assert "szerokość lub wysokość" in page._shape_hint.text()


def test_preset_change_applies_to_loaded_session_under_explicit_scale_rule(qapp, monkeypatch):
    """
    Documents the existing rule: a loaded session uses its OWN stored scale
    until the scale in Ustawienia is explicitly changed - a preset pick that
    changes the scale counts as such a change (same as the 2-point tool).
    With conversion off the scale stays None, so nothing changes.
    """
    import view.sessions_page as sessions_page

    monkeypatch.setattr(sessions_page, "list_sessions", lambda: [])  # no database access
    from view.main_window import MainWindow

    window = MainWindow()
    settings = window.settings_page

    # Simulate a displayed loaded session with its own stored scale.
    window._session_is_loaded = True
    window._loaded_scale = 0.25
    window._last_settings_scale = settings.get_scale_mm_per_px()

    # Conversion off -> preset pick does not change the scale -> session untouched.
    _select_preset(settings, "issf_10m_air_pistol")
    assert window._current_unit_context().scale_mm_per_px == 0.25

    # Conversion on + preset feature -> explicit scale change -> settings scale applies.
    settings._scale_enabled.setChecked(True)
    _select_preset(settings, "issf_25m_precision")
    assert window._session_is_loaded is False
    assert window._current_unit_context().scale_mm_per_px == pytest.approx(
        settings.get_scale_mm_per_px()
    )
    window.close()


def test_new_detection_with_preset_shows_measure_warning_again(page):
    _select_preset(page, "issf_25m_precision")
    page.set_measured_pixel_distance(400.0)
    assert not page._awaiting_measurement

    page.set_detected_radius(809.0)  # new photo analysed

    assert page._awaiting_measurement
    assert not page._measure_warning.parentWidget().isHidden()
    assert page._diameter_px.value() == 400  # previous value kept, not auto-filled


def test_new_detection_with_custom_autofills_without_warning(page):
    assert _custom_selected(page)

    page.set_detected_radius(809.0)

    assert page._diameter_px.value() == 1618
    assert not page._awaiting_measurement
    assert page._measure_warning.parentWidget().isHidden()
