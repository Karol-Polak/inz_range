from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
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


class SettingsPage(QWidget):
    """
    Settings page with two sections:
      1. Manual target calibration (center X/Y + radius)
      2. Scale calibration (real-world mm per pixel)

    Call .get_metadata()  → dict ready to pass to analyze_image()
    Call .get_scale_mm_per_px() → float or None for stats conversion
    """

    def __init__(self, on_settings_changed: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_settings_changed
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
        """
        self._diameter_px.setValue(int(round(radius_px * 2)))

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(0, 0, 12, 0)
        root.setSpacing(22)

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
        section.setObjectName("imagePanel")

        layout = QVBoxLayout(section)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

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
        self._manual_enabled.setStyleSheet("color: #1d2935; font-weight: 700; font-size: 13px;")
        self._manual_enabled.stateChanged.connect(self._on_manual_toggled)
        layout.addWidget(self._manual_enabled)

        # Input fields container
        self._calibration_inputs = QWidget()
        inputs_layout = QVBoxLayout(self._calibration_inputs)
        inputs_layout.setContentsMargins(0, 4, 0, 0)
        inputs_layout.setSpacing(10)

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
        inputs_layout.addWidget(reset_btn)

        layout.addWidget(self._calibration_inputs)
        self._calibration_inputs.setEnabled(False)

        return section

    def _build_scale_section(self) -> QFrame:
        section = QFrame()
        section.setObjectName("imagePanel")

        layout = QVBoxLayout(section)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QLabel("Kalibracja skali")
        title.setObjectName("sectionTitle")

        desc = QLabel(
            "Podaj rzeczywistą średnicę tarczy w milimetrach oraz jej średnicę w pikselach "
            "(wykrytą automatycznie lub zmierzoną). "
            "Dzięki temu statystyki będą wyświetlane w milimetrach zamiast pikselach."
        )
        desc.setObjectName("mutedText")
        desc.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(desc)

        self._scale_enabled = QCheckBox("Włącz konwersję px → mm")
        self._scale_enabled.setStyleSheet("color: #1d2935; font-weight: 700; font-size: 13px;")
        self._scale_enabled.stateChanged.connect(self._on_scale_toggled)
        layout.addWidget(self._scale_enabled)

        self._scale_inputs = QWidget()
        scale_layout = QVBoxLayout(self._scale_inputs)
        scale_layout.setContentsMargins(0, 4, 0, 0)
        scale_layout.setSpacing(10)

        self._diameter_mm = QDoubleSpinBox()
        self._diameter_mm.setRange(1.0, 2000.0)
        self._diameter_mm.setValue(500.0)
        self._diameter_mm.setSuffix(" mm")
        self._diameter_mm.setDecimals(1)
        self._diameter_mm.valueChanged.connect(self._update_scale_preview)
        _style_spinbox(self._diameter_mm)

        self._diameter_px = QSpinBox()
        self._diameter_px.setRange(1, 9999)
        self._diameter_px.setValue(800)
        self._diameter_px.setSuffix(" px")
        self._diameter_px.valueChanged.connect(self._update_scale_preview)
        _style_spinbox(self._diameter_px)

        self._scale_preview = QLabel()
        self._scale_preview.setObjectName("statValue")
        self._update_scale_preview()

        scale_layout.addWidget(self._make_field_row("Średnica tarczy (mm)", self._diameter_mm))
        scale_layout.addWidget(self._make_field_row("Średnica tarczy (px)", self._diameter_px))
        scale_layout.addWidget(self._make_field_row("Skala:", self._scale_preview))

        layout.addWidget(self._scale_inputs)
        self._scale_inputs.setEnabled(False)

        return section

    def _build_info_section(self) -> QFrame:
        section = QFrame()
        section.setObjectName("imagePanel")

        layout = QVBoxLayout(section)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(8)

        title = QLabel("Wskazówki")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        tips = [
            "Środek i promień tarczy możesz odczytać otwierając zdjęcie w programie Paint lub GIMP i najeżdżając myszką.",
            "Typowe tarcze strzeleckie mają średnicę 26 cm (tarcza pistoletowa) lub 50 cm (karabinowa).",
            "Po włączeniu kalibracji ręcznej i uruchomieniu analizy, podane wartości zostaną użyte zamiast automatycznej detekcji.",
            "Skala jest obliczana automatycznie po wykryciu tarczy — pole 'Średnica px' zostanie wypełnione automatycznie.",
        ]

        for tip in tips:
            row = QHBoxLayout()
            row.setSpacing(8)

            dot = QLabel("•")
            dot.setStyleSheet("color: #2f6f73; font-size: 16px; font-weight: 700;")
            dot.setFixedWidth(14)
            dot.setAlignment(Qt.AlignmentFlag.AlignTop)

            text = QLabel(tip)
            text.setObjectName("mutedText")
            text.setWordWrap(True)

            row.addWidget(dot)
            row.addWidget(text, 1)
            layout.addLayout(row)

        return section

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_spinbox(self, min_val: int, max_val: int, default: int, suffix: str) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(default)
        sb.setSuffix(f" {suffix}")
        _style_spinbox(sb)
        return sb

    def _make_field_row(self, label: str, widget: QWidget) -> QFrame:
        row = QFrame()
        row.setObjectName("statRow")

        layout = QHBoxLayout(row)
        layout.setContentsMargins(14, 10, 14, 10)

        lbl = QLabel(label)
        lbl.setObjectName("statLabel")
        lbl.setMinimumWidth(180)

        layout.addWidget(lbl)
        layout.addWidget(widget, 1)

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
            self._scale_preview.setText(f"{scale:.4f} mm/px")
        else:
            self._scale_preview.setText("—")


def _style_spinbox(sb):
    sb.setStyleSheet("""
        QSpinBox, QDoubleSpinBox {
            background: #f8fafb;
            border: 1px solid #e1e6eb;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 13px;
            color: #1d2935;
        }
        QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #2f6f73;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            width: 20px;
        }
    """)