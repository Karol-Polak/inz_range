from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from services.target_presets import all_presets, feature_display_text, get_preset, measurement_hint
from view import theme
from view.number_format import fmt_num


class SettingsPage(QWidget):
    """
    Settings page with two sections:
      1. Manual target calibration (center X/Y + radius)
      2. Scale calibration (real-world mm per pixel)

    The scale section works two ways:
      - type a known real-world distance (mm) + its length on the photo (px)
      - OR click two points on the analysed photo ("Zmierz na zdjęciu") -
        MainWindow measures the pixel distance and fills the px field via
        set_measured_pixel_distance(), you then just type the real mm value.

    Call .get_metadata()  → dict ready to pass to analyze_image()
    Call .get_scale_mm_per_px() → float or None for stats conversion
    """

    def __init__(
        self,
        on_settings_changed: Optional[Callable] = None,
        on_measure_requested: Optional[Callable] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._on_changed = on_settings_changed
        self._on_measure_requested = on_measure_requested
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict:
        """Return metadata dict for analyze_image(). Includes manual_target if enabled."""
        meta = {}
        if self._manual_enabled.isChecked():
            meta["manual_target"] = {
                "center_x": self._cx.value(),
                "center_y": self._cy.value(),
                "radius":   self._radius.value(),
            }
        return meta

    def get_scale_mm_per_px(self) -> Optional[float]:
        """Return mm-per-pixel scale factor, or None if scale is disabled."""
        if not self._scale_enabled.isChecked():
            return None
        diameter_mm = self._diameter_mm.value()
        diameter_px = self._diameter_px.value()
        if diameter_px <= 0 or diameter_mm <= 0:
            return None
        return diameter_mm / diameter_px

    def set_detected_radius(self, radius_px: float) -> None:
        """
        Called by MainWindow after a successful auto-detection so the
        diameter_px field is pre-filled with the detected target size.

        Skipped while a reference feature from the target catalogue is
        selected: the detector does not know WHICH feature it found (outer
        ring, black area, ...), so the px value must come from the 2-point
        measurement of that same feature. "Własny wymiar…" keeps auto-fill.
        """
        if self._selected_preset() is not None:
            return
        self._set_px_programmatically(int(round(radius_px * 2)))

    def set_measured_pixel_distance(self, distance_px: float) -> None:
        """
        Called by MainWindow after the user clicks two points on the photo
        ("Zmierz na zdjęciu"). Fills the px field and turns scale conversion
        on, so the user only has to type the real-world mm distance.
        """
        self._set_px_programmatically(max(1, int(round(distance_px))))
        self._set_awaiting_measurement(False)
        self._scale_enabled.setChecked(True)

    def set_measure_button_enabled(self, enabled: bool) -> None:
        """Called by MainWindow to lock the 'Zmierz na zdjęciu' button while
        a background analysis is running (it needs self.current_session)."""
        self._measure_button.setEnabled(enabled)

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(0, 0, theme.SPACE_M, 0)
        root.setSpacing(theme.SPACE_L)

        root.addWidget(self._build_calibration_section())
        root.addWidget(self._build_scale_section())
        root.addWidget(self._build_info_section())
        root.addStretch(1)

        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _build_calibration_section(self) -> QFrame:
        section = QFrame()
        section.setObjectName("panel")

        layout = QVBoxLayout(section)
        layout.setContentsMargins(theme.SPACE_XL, theme.SPACE_L, theme.SPACE_XL, theme.SPACE_L)
        layout.setSpacing(theme.SPACE_M)

        # Header
        title = QLabel("Ręczna kalibracja tarczy")
        title.setObjectName("sectionTitle")

        desc = QLabel(
            "Jeśli automatyczna detekcja tarczy nie powiedzie się, "
            "podaj ręcznie środek i promień tarczy (w pikselach). "
            "Możesz odczytać te wartości najeżdżając myszką na zdjęcie w edytorze graficznym."
        )
        desc.setObjectName("mutedText")
        desc.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(desc)

        # Toggle
        self._manual_enabled = QCheckBox("Włącz ręczną kalibrację")
        self._manual_enabled.stateChanged.connect(self._on_manual_toggled)
        layout.addWidget(self._manual_enabled)

        # Input fields container
        self._calibration_inputs = QWidget()
        inputs_layout = QVBoxLayout(self._calibration_inputs)
        inputs_layout.setContentsMargins(0, theme.SPACE_XS, 0, 0)
        inputs_layout.setSpacing(theme.SPACE_S)

        self._cx = self._make_spinbox(0, 9999, 960, "px")
        self._cy = self._make_spinbox(0, 9999, 540, "px")
        self._radius = self._make_spinbox(1, 9999, 400, "px")

        inputs_layout.addWidget(self._make_field_row("Środek X", self._cx))
        inputs_layout.addWidget(self._make_field_row("Środek Y", self._cy))
        inputs_layout.addWidget(self._make_field_row("Promień tarczy", self._radius))

        # Reset button
        reset_btn = QPushButton("Resetuj do domyślnych")
        reset_btn.setObjectName("secondaryButton")
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.clicked.connect(self._reset_calibration)
        inputs_layout.addLayout(_left_aligned(reset_btn))

        layout.addWidget(self._calibration_inputs)
        self._calibration_inputs.setEnabled(False)

        return section

    def _build_scale_section(self) -> QFrame:
        section = QFrame()
        section.setObjectName("panel")

        layout = QVBoxLayout(section)
        layout.setContentsMargins(theme.SPACE_XL, theme.SPACE_L, theme.SPACE_XL, theme.SPACE_L)
        layout.setSpacing(theme.SPACE_M)

        title = QLabel("Kalibracja skali")
        title.setObjectName("sectionTitle")

        desc = QLabel(
            "Podaj znaną odległość rzeczywistą w milimetrach (np. średnicę tarczy) "
            "oraz tę samą odległość w pikselach na zdjęciu - wykrytą automatycznie, "
            "zmierzoną przyciskiem 'Zmierz na zdjęciu', albo wpisaną ręcznie. "
            "Dzięki temu statystyki będą wyświetlane w milimetrach zamiast pikselach."
        )
        desc.setObjectName("mutedText")
        desc.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(desc)

        self._scale_enabled = QCheckBox("Włącz konwersję px → mm")
        self._scale_enabled.stateChanged.connect(self._on_scale_toggled)
        layout.addWidget(self._scale_enabled)

        self._scale_inputs = QWidget()
        scale_layout = QVBoxLayout(self._scale_inputs)
        scale_layout.setContentsMargins(0, theme.SPACE_XS, 0, 0)
        scale_layout.setSpacing(theme.SPACE_S)

        self._diameter_mm = QDoubleSpinBox()
        self._diameter_mm.setRange(1.0, 2000.0)
        self._diameter_mm.setValue(500.0)
        self._diameter_mm.setSuffix(" mm")
        self._diameter_mm.setDecimals(1)
        self._diameter_mm.valueChanged.connect(self._update_scale_preview)
        self._diameter_mm.valueChanged.connect(self._on_mm_edited)

        self._diameter_px = QSpinBox()
        self._diameter_px.setRange(1, 9999)
        self._diameter_px.setValue(800)
        self._diameter_px.setSuffix(" px")
        self._diameter_px.valueChanged.connect(self._update_scale_preview)
        self._diameter_px.valueChanged.connect(self._on_px_edited)

        self._scale_preview = QLabel()
        self._scale_preview.setObjectName("statValue")
        self._update_scale_preview()

        self._build_preset_controls(scale_layout)

        scale_layout.addWidget(self._make_field_row("Znana odległość rzeczywista (mm)", self._diameter_mm))
        scale_layout.addWidget(self._make_field_row("Ta sama odległość na zdjęciu (px)", self._diameter_px))
        scale_layout.addWidget(self._make_field_row("Skala", self._scale_preview))

        layout.addWidget(self._scale_inputs)
        self._scale_inputs.setEnabled(False)

        # Outside _scale_inputs on purpose: clicking it should work even
        # before "Włącz konwersję" is checked - it turns the checkbox on
        # itself (via set_measured_pixel_distance), so it must stay usable.
        self._measure_button = QPushButton("Zmierz na zdjęciu (2 punkty)")
        self._measure_button.setObjectName("secondaryButton")
        self._measure_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._measure_button.clicked.connect(self._request_measurement)
        layout.addLayout(_left_aligned(self._measure_button))

        return section

    # ------------------------------------------------------------------
    # Target catalogue (services/target_presets.py) - fills ONLY the mm
    # input; the scale is still mm / px exactly as before.
    # ------------------------------------------------------------------

    _CUSTOM_LABEL = "Własny wymiar…"

    def _build_preset_controls(self, scale_layout: QVBoxLayout) -> None:
        self._filling = False
        self._awaiting_measurement = False

        self._preset_combo = QComboBox()
        for preset in all_presets():
            self._preset_combo.addItem(preset.name, preset.id)
            self._preset_combo.setItemData(
                self._preset_combo.count() - 1, _source_tooltip(preset), Qt.ItemDataRole.ToolTipRole,
            )
        self._preset_combo.addItem(self._CUSTOM_LABEL, None)
        self._preset_combo.setItemData(
            self._preset_combo.count() - 1,
            "Wpisz wymiar ręcznie w polu poniżej.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._preset_combo.setCurrentIndex(self._preset_combo.count() - 1)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        self._feature_combo = QComboBox()
        self._feature_combo.currentIndexChanged.connect(self._on_feature_changed)

        scale_layout.addWidget(self._make_field_row("Typ tarczy", self._preset_combo, width=320))
        scale_layout.addWidget(self._make_field_row("Element odniesienia", self._feature_combo, width=320))

        self._preset_hint = QLabel(
            "Zmierz na zdjęciu ten sam element, którego wymiar wybrano. "
            "Przy wybranym elemencie wzorcowym rozmiar na zdjęciu mierzysz ręcznie "
            "(automatyczne wykrycie okręgu jest wyłączone)."
        )
        self._preset_hint.setObjectName("mutedText")

        self._shape_hint = QLabel()
        self._shape_hint.setObjectName("mutedText")

        self._source_line = QLabel()
        self._source_line.setObjectName("warningText")

        self._measure_warning = QLabel("Wymiar elementu wybrany – zmierz go na zdjęciu (przycisk poniżej).")
        self._measure_warning.setObjectName("warningText")

        for label in (self._preset_hint, self._shape_hint, self._source_line, self._measure_warning):
            label.setWordWrap(True)
            scale_layout.addWidget(_indented(label))

        self._on_preset_changed(self._preset_combo.currentIndex())  # fills "—" for "Własny wymiar…"

    def _selected_preset(self):
        preset_id = self._preset_combo.currentData()
        return get_preset(preset_id) if preset_id else None

    def _on_preset_changed(self, _index: int):
        preset = self._selected_preset()

        self._feature_combo.blockSignals(True)
        self._feature_combo.clear()
        if preset is None:
            self._feature_combo.addItem("—", None)
        else:
            for feature in preset.features:
                self._feature_combo.addItem(feature_display_text(feature), feature.label)
        self._feature_combo.blockSignals(False)

        if preset is not None:
            self._apply_feature()
        else:
            self._set_awaiting_measurement(False)
        self._refresh_preset_ui()

    def _on_feature_changed(self, _index: int):
        if self._selected_preset() is not None:
            self._apply_feature()

    def _apply_feature(self):
        preset = self._selected_preset()
        label = self._feature_combo.currentData()
        feature = next((f for f in preset.features if f.label == label), None)
        if feature is None:
            return
        self._filling = True
        try:
            self._diameter_mm.setValue(feature.size_mm)
        finally:
            self._filling = False
        self._set_awaiting_measurement(True)

    def _on_mm_edited(self, _value: float):
        if self._filling or self._selected_preset() is None:
            return
        # Hand edit: back to "Własny wymiar…", keeping the typed value.
        self._preset_combo.setCurrentIndex(self._preset_combo.count() - 1)

    def _on_px_edited(self, _value: int):
        if not self._filling:
            self._set_awaiting_measurement(False)  # px typed by hand

    def _set_px_programmatically(self, value: int):
        self._filling = True
        try:
            self._diameter_px.setValue(value)
        finally:
            self._filling = False

    def _set_awaiting_measurement(self, awaiting: bool):
        self._awaiting_measurement = awaiting
        _show(self._measure_warning, awaiting)

    def _refresh_preset_ui(self):
        preset = self._selected_preset()
        is_preset = preset is not None

        self._feature_combo.setEnabled(is_preset)
        _show(self._preset_hint, is_preset)

        tooltip = _source_tooltip(preset) if is_preset else "Wpisz wymiar ręcznie w polu poniżej."
        self._preset_combo.setToolTip(tooltip)
        self._feature_combo.setToolTip(tooltip if is_preset else "")

        hint = measurement_hint(preset) if is_preset else None
        self._shape_hint.setText(hint or "")
        _show(self._shape_hint, hint is not None)

        show_source = is_preset and preset.needs_check
        self._source_line.setText(
            f"Źródło: {preset.source} · {preset.confidence}" if show_source else ""
        )
        _show(self._source_line, show_source)

        _show(self._measure_warning, self._awaiting_measurement)

    def _request_measurement(self):
        if self._on_measure_requested:
            self._on_measure_requested()

    def _build_info_section(self) -> QFrame:
        section = QFrame()
        section.setObjectName("panel")

        layout = QVBoxLayout(section)
        layout.setContentsMargins(theme.SPACE_XL, theme.SPACE_L, theme.SPACE_XL, theme.SPACE_L)
        layout.setSpacing(theme.SPACE_S)

        title = QLabel("Wskazówki")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        tips = [
            "Środek i promień tarczy możesz odczytać otwierając zdjęcie w programie Paint lub GIMP i najeżdżając myszką.",
            "Po włączeniu kalibracji ręcznej i uruchomieniu analizy, podane wartości zostaną użyte zamiast automatycznej detekcji.",
            "Przy typie tarczy „Własny wymiar…” skala jest obliczana automatycznie po wykryciu tarczy – pole „Ta sama odległość na zdjęciu (px)” zostanie wypełnione automatycznie. "
            "Przy wybranym typie tarczy zmierz wybrany element przyciskiem „Zmierz na zdjęciu (2 punkty)”.",
        ]

        for tip in tips:
            text = QLabel(tip)
            text.setObjectName("mutedText")
            text.setWordWrap(True)
            layout.addWidget(text)

        return section

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_spinbox(self, min_val: int, max_val: int, default: int, suffix: str) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(default)
        sb.setSuffix(f" {suffix}")
        return sb

    def _make_field_row(self, label: str, widget: QWidget, width: int = 200) -> QWidget:
        row = QWidget()

        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_M)

        lbl = QLabel(label)
        lbl.setObjectName("mutedText")
        lbl.setFixedWidth(_FIELD_LABEL_WIDTH)

        widget.setFixedWidth(width)

        layout.addWidget(lbl)
        layout.addWidget(widget)
        layout.addStretch(1)

        return row

    def _on_manual_toggled(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self._calibration_inputs.setEnabled(enabled)
        if self._on_changed:
            self._on_changed()

    def _on_scale_toggled(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self._scale_inputs.setEnabled(enabled)
        if self._on_changed:
            self._on_changed()

    def _reset_calibration(self):
        self._cx.setValue(960)
        self._cy.setValue(540)
        self._radius.setValue(400)

    def _update_scale_preview(self):
        mm = self._diameter_mm.value()
        px = self._diameter_px.value()
        if px > 0:
            scale = mm / px
            self._scale_preview.setText(f"{fmt_num(scale, 4)} mm/px")
        else:
            self._scale_preview.setText("—")
        # Scale values changed - let MainWindow re-show results in mm/MOA/mrad.
        if self._on_changed:
            self._on_changed()


_FIELD_LABEL_WIDTH = 260


def _indented(widget: QWidget) -> QWidget:
    """Place a hint under the input column of the field rows."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(_FIELD_LABEL_WIDTH + theme.SPACE_M, 0, 0, 0)
    layout.addWidget(widget, 1)
    return row


def _show(label: QLabel, visible: bool) -> None:
    """Hints live in an indented wrapper row; hide the row, not just the label."""
    label.parentWidget().setVisible(visible)


def _source_tooltip(preset) -> str:
    return f"Źródło: {preset.source}\nPewność: {preset.confidence}"


def _left_aligned(widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(widget)
    row.addStretch(1)
    return row
