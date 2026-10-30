from __future__ import annotations

import math
from typing import Callable, Optional

import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from model.session import Session
from services.units import (
    DECIMALS,
    LengthUnit,
    UnitContext,
    convert_length,
    convert_statistics,
    default_unit,
    is_available,
    missing_reason_text,
)
from view import theme
from view.number_format import fmt_num
from view.chart_style import (
    AIM as _AIM,
    AXIS as _AXIS,
    BG as _BG,
    MUTED as _MUTED,
    PRIMARY as _PRIMARY,
    SECONDARY as _SECONDARY,
    TEXT as _TEXT,
    Canvas as _Canvas,
    base_fig as _base_fig,
)
from view.unit_selector import UnitSelector


def _fmt(value: float, unit: LengthUnit, signed: bool = False) -> str:
    return fmt_num(value, DECIMALS[unit], signed)


# ---------------------------------------------------------------------------
# Individual chart builders - every length is converted to the shown unit
# ---------------------------------------------------------------------------

def _make_scatter(session: Session, unit: LengthUnit, ctx: UnitContext) -> _Canvas:
    def conv(value):
        return convert_length(value, unit, ctx)

    u = unit.value
    hits = [h for h in session.hits if h.valid]
    target = session.target
    fig, ax = _base_fig(h=4.0)

    if hits:
        xs = [conv(h.x) for h in hits]
        ys = [conv(h.y) for h in hits]

        ax.scatter(
            xs, ys,
            s=46, color=_PRIMARY, zorder=3, edgecolors=_BG, linewidths=0.8,
            label="Trafienia",
        )

        mpi_x = conv(session.statistics.get("mean_x", 0.0))
        mpi_y = conv(session.statistics.get("mean_y", 0.0))
        cep = conv(session.statistics.get("cep_50", 0))

        ax.plot(mpi_x, mpi_y, "+", color=_SECONDARY, markersize=14, markeredgewidth=2, zorder=4,
                label="MPI (śr. punkt trafień)")

        if cep:
            # CEP 50% to promień wokół MPI, nie wokół środka tarczy.
            ax.add_patch(mpatches.Circle(
                (mpi_x, mpi_y), cep, fill=False,
                edgecolor=_SECONDARY, linewidth=1.4, linestyle="--",
                label=f"CEP 50% ({_fmt(cep, unit)} {u})",
            ))

        aim_x = target.aim_offset_x or 0.0
        aim_y = target.aim_offset_y or 0.0
        if aim_x != 0.0 or aim_y != 0.0:
            # Punkt celowania różni się od środka tarczy (np. sylwetka) -
            # pokazujemy go osobno, tak jak na podglądzie analizy.
            ax.plot(conv(aim_x), conv(aim_y), "*", color=_AIM, markersize=14, markeredgewidth=0.8,
                    markeredgecolor=_TEXT, zorder=5, label="Punkt celowania")

        for i, (x, y) in enumerate(zip(xs, ys), 1):
            ax.annotate(str(i), (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=_MUTED)

    ax.axhline(0, color=_AXIS, linewidth=0.8)
    ax.axvline(0, color=_AXIS, linewidth=0.8)
    ax.plot(0, 0, "x", color=_TEXT, markersize=9, markeredgewidth=1.6, zorder=4,
            label="Środek tarczy")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"X ({u})")
    ax.set_ylabel(f"Y ({u}, dodatnie = w dół)")
    ax.set_title("Rozrzut trafień", fontweight="bold")
    if hits:
        # Legend outside the plot area so it never covers a hit.
        ax.legend(fontsize=8, frameon=False, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # Współrzędne trafień są w układzie obrazu (y rośnie w dół, tak jak na
    # zdjęciu) - bez tego "góra zdjęcia" wyświetlałaby się na dole wykresu.
    ax.invert_yaxis()

    fig.tight_layout(pad=1.8)
    return _Canvas(fig)


_LEGEND_HEADROOM = 1.35


def _legend_with_headroom(ax) -> None:
    """Reserve empty space above the tallest bar and put the legend there."""
    _, top = ax.get_ylim()
    ax.set_ylim(top=top * _LEGEND_HEADROOM)
    ax.legend(fontsize=8, ncol=2, loc="upper right", frameon=False)


def _make_radius_hist(session: Session, unit: LengthUnit, ctx: UnitContext) -> _Canvas:
    def conv(value):
        return convert_length(value, unit, ctx)

    u = unit.value
    hits = [h for h in session.hits if h.valid]
    fig, ax = _base_fig(h=3.4)

    if hits:
        mpi_x = session.statistics.get("mean_x", 0.0)
        mpi_y = session.statistics.get("mean_y", 0.0)
        radii = [conv(math.hypot(h.x - mpi_x, h.y - mpi_y)) for h in hits]
        ax.hist(radii, bins=min(10, len(radii)), color=_PRIMARY, edgecolor=_BG, alpha=0.9)

        mean_r = conv(session.statistics.get("precision_radius", 0))
        cep    = conv(session.statistics.get("cep_50", 0))

        ax.axvline(mean_r, color=_SECONDARY, linewidth=1.6, linestyle="--", ymax=1 / _LEGEND_HEADROOM,
                   label=f"Średnia ({_fmt(mean_r, unit)} {u})")
        ax.axvline(cep,    color=_TEXT,      linewidth=1.4, linestyle=":",  ymax=1 / _LEGEND_HEADROOM,
                   label=f"CEP 50% ({_fmt(cep, unit)} {u})")
        _legend_with_headroom(ax)

    ax.set_xlabel(f"Odległość od MPI ({u})")
    ax.set_ylabel("Liczba trafień")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Histogram precyzji (odległość od MPI)", fontweight="bold")
    fig.tight_layout(pad=1.8)
    return _Canvas(fig)


def _make_xy_hist(session: Session, unit: LengthUnit, ctx: UnitContext) -> _Canvas:
    def conv(value):
        return convert_length(value, unit, ctx)

    hits = [h for h in session.hits if h.valid]
    fig, ax = _base_fig(h=3.4)

    if hits:
        xs = [conv(h.x) for h in hits]
        ys = [conv(h.y) for h in hits]
        bins = min(10, len(hits))
        ax.hist(xs, bins=bins, color=_PRIMARY,   edgecolor=_BG, alpha=0.75, label="X")
        ax.hist(ys, bins=bins, color=_SECONDARY, edgecolor=_BG, alpha=0.65, label="Y")
        _legend_with_headroom(ax)

    ax.set_xlabel(f"Odchylenie ({unit.value})")
    ax.set_ylabel("Liczba trafień")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Histogramy odchyleń X / Y", fontweight="bold")
    fig.tight_layout(pad=1.8)
    return _Canvas(fig)


# ---------------------------------------------------------------------------
# Stat-card helper
# ---------------------------------------------------------------------------

def _stat_card(label: str, value: str, unit: str = "") -> QWidget:
    card = QWidget()

    v = QVBoxLayout(card)
    v.setContentsMargins(0, theme.SPACE_S, 0, theme.SPACE_S)
    v.setSpacing(2)

    lbl = QLabel(label)
    lbl.setObjectName("statLabel")

    val_row = QHBoxLayout()
    val_row.setSpacing(4)

    val = QLabel(value)
    val.setObjectName("kpiValue")

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

    def __init__(self, on_unit_changed: Optional[Callable[[LengthUnit], None]] = None, parent=None):
        super().__init__(parent)
        self._session: Optional[Session] = None
        self._ctx = UnitContext()
        self._unit = LengthUnit.PX
        self._on_unit_changed = on_unit_changed
        self._build_empty_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(
        self,
        session: Optional[Session],
        scale_mm_per_px: Optional[float] = None,
        distance_m: Optional[float] = None,
        unit: Optional[LengthUnit] = None,
    ) -> None:
        """
        Call this from MainWindow after a successful analysis. Lengths are
        shown in `unit` if it can be computed from the scale/distance,
        otherwise in the default unit (mm with a scale, else px).
        """
        self._session = session
        self._ctx = UnitContext(scale_mm_per_px=scale_mm_per_px, distance_m=distance_m)
        if unit is None or not is_available(unit, self._ctx):
            unit = default_unit(self._ctx)
        self._unit = LengthUnit(unit)
        self._rebuild()

    def unit(self) -> LengthUnit:
        return self._unit

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _clear(self):
        layout = self.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().hide()
                    item.widget().deleteLater()
            # delete old layout
            QWidget().setLayout(layout)

    def _build_empty_state(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(theme.SPACE_XL, theme.SPACE_L, theme.SPACE_XL, theme.SPACE_L)
        panel_layout.setSpacing(theme.SPACE_S)

        title = QLabel("Brak wyników analizy")
        title.setObjectName("sectionTitle")

        desc = QLabel(
            "Uruchom analizę tarczy w widoku „Analiza tarczy”, a tutaj pojawią się "
            "wykresy i szczegółowe statystyki trafień."
        )
        desc.setObjectName("mutedText")
        desc.setWordWrap(True)

        panel_layout.addWidget(title)
        panel_layout.addWidget(desc)

        layout.addWidget(panel)
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
        root.setContentsMargins(0, 0, theme.SPACE_M, 0)
        root.setSpacing(theme.SPACE_L)

        root.addWidget(self._build_summary_cards())
        root.addWidget(self._build_chart_grid())
        root.addStretch(1)

        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _selector_changed(self, unit: LengthUnit):
        if self._on_unit_changed:
            self._on_unit_changed(unit)  # MainWindow keeps Analiza + Statystyki in sync
        else:
            self.set_session(self._session, self._ctx.scale_mm_per_px, self._ctx.distance_m, unit)

    def _build_summary_cards(self) -> QFrame:
        unit = self._unit
        stats = convert_statistics(self._session.statistics, unit, self._ctx)
        u = unit.value

        def length(key, signed=False):
            return _fmt(stats.get(key, 0.0), unit, signed)

        def count(key):
            # Counts are never converted.
            return str(int(stats.get(key, 0)))

        cards_data = [
            ("Liczba trafień",                         count("count"),            ""),
            ("Celność (MPI od punktu celowania)",       length("accuracy_radius"), u),
            ("Precyzja – śr. odległość od MPI (MRD)",   length("precision_radius"), u),
            ("Odch. std. precyzji",                     length("precision_std"),   u),
            ("CEP 50%",                                 length("cep_50"),          u),
            ("Rozrzut maks. (ES, środek–środek)",       length("extreme_spread"),  u),
            ("Trafienia odstające",                     count("outliers_count"),   ""),
            ("Średnie odchylenie X",                    length("mean_x"),          u),
            ("Średnie odchylenie Y",                    length("mean_y"),          u),
        ]

        section = QFrame()
        section.setObjectName("panel")
        sec_layout = QVBoxLayout(section)
        sec_layout.setContentsMargins(theme.SPACE_XL, theme.SPACE_L, theme.SPACE_XL, theme.SPACE_L)
        sec_layout.setSpacing(theme.SPACE_S)

        header = QHBoxLayout()
        title = QLabel("Podsumowanie sesji")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch(1)

        unit_label = QLabel("Jednostka")
        unit_label.setObjectName("statLabel")
        header.addWidget(unit_label)
        header.addSpacing(theme.SPACE_S)

        selector = UnitSelector(unit)
        selector.set_context(self._ctx)
        selector.unit_changed.connect(self._selector_changed)
        header.addWidget(selector)
        sec_layout.addLayout(header)

        reason = missing_reason_text(LengthUnit.MM, self._ctx) or missing_reason_text(LengthUnit.MOA, self._ctx)
        if reason:
            note = QLabel(reason)
            note.setObjectName("mutedText")
            note.setWordWrap(True)
            sec_layout.addWidget(note)

        grid = QGridLayout()
        grid.setHorizontalSpacing(theme.SPACE_XL)
        grid.setVerticalSpacing(theme.SPACE_XS)

        for i, (label, value, card_unit) in enumerate(cards_data):
            card = _stat_card(label, value, card_unit)
            grid.addWidget(card, i // 3, i % 3)

        sec_layout.addLayout(grid)
        return section

    def _build_chart_grid(self) -> QFrame:
        section = QFrame()
        section.setObjectName("panel")
        sec_layout = QVBoxLayout(section)
        sec_layout.setContentsMargins(theme.SPACE_XL, theme.SPACE_L, theme.SPACE_XL, theme.SPACE_L)
        sec_layout.setSpacing(theme.SPACE_M)

        title = QLabel("Wykresy")
        title.setObjectName("sectionTitle")
        sec_layout.addWidget(title)

        session, unit, ctx = self._session, self._unit, self._ctx

        # Row 1: scatter (wide) | radius histogram
        row1 = QHBoxLayout()
        row1.setSpacing(theme.SPACE_L)
        row1.addWidget(_make_scatter(session, unit, ctx), 3)
        row1.addWidget(_make_radius_hist(session, unit, ctx), 2)
        sec_layout.addLayout(row1)

        # Row 2: X/Y histogram (half width, left-aligned)
        row2 = QHBoxLayout()
        row2.setSpacing(theme.SPACE_L)
        row2.addWidget(_make_xy_hist(session, unit, ctx), 2)
        row2.addStretch(3)
        sec_layout.addLayout(row2)

        return section
