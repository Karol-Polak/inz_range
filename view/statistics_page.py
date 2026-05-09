from __future__ import annotations

import math
from typing import Optional

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from model.session import Session


# ---------------------------------------------------------------------------
# Colour tokens – must match main_window.py stylesheet
# ---------------------------------------------------------------------------
_BG       = "#ffffff"
_CARD_BG  = "#f8fafb"
_BORDER   = "#e1e6eb"
_TEAL     = "#2f6f73"
_TEAL2    = "#3f8c91"
_ACCENT   = "#e05c2a"
_TEXT     = "#1d2935"
_MUTED    = "#657282"
_GRID     = "#e8edf2"


def _base_fig(rows: int = 1, cols: int = 1, h: float = 3.6):
    """Return a Figure + axes array styled to match the app palette."""
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, h))
    fig.patch.set_facecolor(_BG)
    for ax in (np.array(axes).flat if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(_CARD_BG)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.xaxis.label.set_color(_MUTED)
        ax.yaxis.label.set_color(_MUTED)
        ax.title.set_color(_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)
        ax.grid(color=_GRID, linewidth=0.6, linestyle="--")
    fig.tight_layout(pad=1.6)
    return fig, axes


class _Canvas(FigureCanvas):
    def __init__(self, fig: Figure):
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(260)


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def _make_scatter(session: Session) -> _Canvas:
    hits = [h for h in session.hits if h.valid]
    fig, ax = _base_fig(h=4.0)

    if hits:
        xs = [h.x for h in hits]
        ys = [h.y for h in hits]
        radii = [math.sqrt(h.x**2 + h.y**2) for h in hits]
        max_r = max(radii) if radii else 1

        scatter = ax.scatter(
            xs, ys,
            c=radii,
            cmap="RdYlGn_r",
            vmin=0, vmax=max_r,
            s=80, zorder=3, edgecolors=_TEXT, linewidths=0.5,
        )
        fig.colorbar(scatter, ax=ax, label="Odległość od środka (px)", pad=0.02)

        cep = session.statistics.get("cep_50", 0)
        if cep:
            ax.add_patch(mpatches.Circle(
                (0, 0), cep, fill=False,
                edgecolor=_TEAL, linewidth=1.5, linestyle="--", label=f"CEP 50% ({cep:.0f} px)",
            ))

        for i, (x, y) in enumerate(zip(xs, ys), 1):
            ax.annotate(str(i), (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=_MUTED)

    ax.axhline(0, color=_BORDER, linewidth=1)
    ax.axvline(0, color=_BORDER, linewidth=1)
    ax.plot(0, 0, "x", color=_ACCENT, markersize=10, markeredgewidth=2, zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title("Rozrzut trafień", fontweight="bold")
    if hits:
        ax.legend(fontsize=8, facecolor=_BG, edgecolor=_BORDER)

    fig.tight_layout(pad=1.8)
    return _Canvas(fig)


def _make_radius_hist(session: Session) -> _Canvas:
    hits = [h for h in session.hits if h.valid]
    fig, ax = _base_fig(h=3.4)

    if hits:
        radii = [math.sqrt(h.x**2 + h.y**2) for h in hits]
        ax.hist(radii, bins=min(10, len(radii)), color=_TEAL, edgecolor=_BG, alpha=0.85)

        mean_r = session.statistics.get("mean_radius", 0)
        cep    = session.statistics.get("cep_50", 0)

        ax.axvline(mean_r, color=_ACCENT,  linewidth=1.8, linestyle="--", label=f"Średnia ({mean_r:.1f} px)")
        ax.axvline(cep,    color=_TEAL2,   linewidth=1.8, linestyle=":",  label=f"CEP 50% ({cep:.1f} px)")
        ax.legend(fontsize=8, facecolor=_BG, edgecolor=_BORDER)

    ax.set_xlabel("Odległość od środka (px)")
    ax.set_ylabel("Liczba trafień")
    ax.set_title("Histogram odległości", fontweight="bold")
    fig.tight_layout(pad=1.8)
    return _Canvas(fig)


def _make_xy_hist(session: Session) -> _Canvas:
    hits = [h for h in session.hits if h.valid]
    fig, ax = _base_fig(h=3.4)

    if hits:
        xs = [h.x for h in hits]
        ys = [h.y for h in hits]
        bins = min(10, len(hits))
        ax.hist(xs, bins=bins, color=_TEAL,   edgecolor=_BG, alpha=0.75, label="X")
        ax.hist(ys, bins=bins, color=_ACCENT,  edgecolor=_BG, alpha=0.75, label="Y")
        ax.legend(fontsize=8, facecolor=_BG, edgecolor=_BORDER)

    ax.set_xlabel("Odchylenie (px)")
    ax.set_ylabel("Liczba trafień")
    ax.set_title("Histogramy odchyleń X / Y", fontweight="bold")
    fig.tight_layout(pad=1.8)
    return _Canvas(fig)


# ---------------------------------------------------------------------------
# Stat-card helper
# ---------------------------------------------------------------------------

def _stat_card(label: str, value: str, unit: str = "") -> QFrame:
    card = QFrame()
    card.setObjectName("statRow")

    v = QVBoxLayout(card)
    v.setContentsMargins(16, 12, 16, 12)
    v.setSpacing(4)

    lbl = QLabel(label)
    lbl.setObjectName("statLabel")

    val_row = QHBoxLayout()
    val_row.setSpacing(4)

    val = QLabel(value)
    val.setObjectName("statValue")

    if unit:
        u = QLabel(unit)
        u.setObjectName("statLabel")
        val_row.addWidget(val)
        val_row.addWidget(u)
        val_row.addStretch()
    else:
        val_row.addWidget(val)
        val_row.addStretch()

    v.addWidget(lbl)
    v.addLayout(val_row)
    return card


# ---------------------------------------------------------------------------
# Main Statistics Page widget
# ---------------------------------------------------------------------------

class StatisticsPage(QWidget):
    """Full statistics page – replace the placeholder in MainWindow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session: Optional[Session] = None
        self._scale: Optional[float] = None
        self._build_empty_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(self, session: Optional[Session], scale_mm_per_px: Optional[float] = None) -> None:
        """Call this from MainWindow after a successful analysis."""
        self._session = session
        self._scale = scale_mm_per_px
        self._rebuild()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _clear(self):
        layout = self.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            # delete old layout
            QWidget().setLayout(layout)

    def _build_empty_state(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)

        title = QLabel("Statystyki")
        title.setObjectName("placeholderTitle")

        desc = QLabel("Uruchom analize tarczy na stronie Analiza tarczy, a tutaj pojawia sie wykresy i szczegolowe statystyki trafien.")

        desc.setObjectName("placeholderText")
        desc.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addStretch(1)

    def _rebuild(self):
        self._clear()

        if self._session is None or not self._session.hits:
            self._build_empty_state()
            return

        # Root scroll area so the page survives small windows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(0, 0, 12, 0)
        root.setSpacing(22)

        root.addWidget(self._build_summary_cards())
        root.addWidget(self._build_chart_grid())
        root.addStretch(1)

        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _build_summary_cards(self) -> QFrame:
        stats = self._session.statistics

        scale = self._scale
        unit = "mm" if scale else "px"

        def fmt(v, decimals=1):
            val = float(v)
            if scale:
                val = val * scale
            return f"{val:.{decimals}f}"

        cards_data = [
            ("Liczba trafień",   fmt(stats.get("count", 0), 0),      ""),
            ("Średni promień",   fmt(stats.get("mean_radius", 0)),    unit),
            ("Odch. std.",       fmt(stats.get("std_radius", 0)),     unit),
            ("CEP 50%",          fmt(stats.get("cep_50", 0)),         unit),
            ("Rozrzut maks.",    fmt(stats.get("extreme_spread", 0)), unit),
            ("Outliery",         fmt(stats.get("outliers_count", 0), 0), ""),
            ("Śr. odch. X",      fmt(stats.get("mean_x", 0)),         unit),
            ("Śr. odch. Y",      fmt(stats.get("mean_y", 0)),         unit),
        ]

        section = QFrame()
        section.setObjectName("imagePanel")
        sec_layout = QVBoxLayout(section)
        sec_layout.setContentsMargins(20, 18, 20, 18)
        sec_layout.setSpacing(12)

        title = QLabel("Podsumowanie sesji")
        title.setObjectName("sectionTitle")
        sec_layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(10)

        for i, (label, value, unit) in enumerate(cards_data):
            card = _stat_card(label, value, unit)
            grid.addWidget(card, i // 4, i % 4)

        sec_layout.addLayout(grid)
        return section

    def _build_chart_grid(self) -> QFrame:
        section = QFrame()
        section.setObjectName("imagePanel")
        sec_layout = QVBoxLayout(section)
        sec_layout.setContentsMargins(20, 18, 20, 18)
        sec_layout.setSpacing(16)

        title = QLabel("Wykresy")
        title.setObjectName("sectionTitle")
        sec_layout.addWidget(title)

        # Row 1: scatter (wide) | radius histogram
        row1 = QHBoxLayout()
        row1.setSpacing(16)
        row1.addWidget(_make_scatter(self._session), 3)
        row1.addWidget(_make_radius_hist(self._session), 2)
        sec_layout.addLayout(row1)

        # Row 2: X/Y histogram (half width, left-aligned)
        row2 = QHBoxLayout()
        row2.setSpacing(16)
        row2.addWidget(_make_xy_hist(self._session), 2)
        row2.addStretch(3)
        sec_layout.addLayout(row2)

        return section