from __future__ import annotations

from typing import Callable, List, Optional

import matplotlib.dates as mdates

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from database.models import TrainingSession
from database.repository import delete_session, list_sessions, load_session
from model.session import Session
from services.csv_export import export_sessions_list_to_csv
from services.session_trends import build_trend_series, compare_sessions, filter_sessions, unit_context
from services.units import DECIMALS, LengthUnit, convert_length, default_unit
from view import theme
from view.number_format import fmt_general, fmt_num
from view.chart_style import AXIS, BG, PRIMARY, SECONDARY, Canvas, base_fig
from view.unit_selector import UnitSelector


class SessionsPage(QWidget):
    """
    Historia zapisanych sesji treningowych: lista, wczytywanie,
    usuwanie, porównanie dwóch sesji oraz wykres trendu w czasie.
    """

    def __init__(self, on_session_loaded: Optional[Callable[[Session], None]] = None, parent=None):
        super().__init__(parent)
        self._on_session_loaded = on_session_loaded
        self._sessions: List[TrainingSession] = []
        self._visible_sessions: List[TrainingSession] = []
        self._filter_query = ""
        self._chart_canvas: Optional[Canvas] = None
        self._analysis_lock = False
        # Requested unit for trend + comparison; each block falls back on its
        # own (services.units.common_unit) when a session lacks scale/distance.
        self._unit = LengthUnit.MM
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_analysis_lock(self, locked: bool) -> None:
        """Called by MainWindow while a background analysis is running, so
        loading a different session can't race with it overwriting
        current_session."""
        self._analysis_lock = locked
        self._on_selection_changed()

    def refresh(self):
        self._sessions = list_sessions()
        self._apply_filter()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _on_filter_changed(self, text: str):
        self._filter_query = text
        self._apply_filter()

    def _apply_filter(self):
        self._visible_sessions = filter_sessions(self._sessions, self._filter_query)
        self._populate_list()
        self._rebuild_trend_chart()
        self._update_comparison()

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_L)

        layout.addWidget(self._build_list_panel(), 1)
        layout.addWidget(self._build_details_panel(), 2)

    def _build_list_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(theme.SPACE_L, theme.SPACE_L, theme.SPACE_L, theme.SPACE_L)
        layout.setSpacing(theme.SPACE_S)

        title = QLabel("Zapisane sesje")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        hint = QLabel("Ctrl+klik, aby wybrać dwie sesje do porównania.")
        hint.setObjectName("mutedText")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText("Filtruj po broni lub odległości...")
        self._filter_input.textChanged.connect(self._on_filter_changed)
        layout.addWidget(self._filter_input)

        self._list_widget = QListWidget()
        self._list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list_widget.setWordWrap(True)
        self._list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)  # re-wrap rows on resize
        self._list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list_widget, 1)

        buttons = QGridLayout()
        buttons.setSpacing(theme.SPACE_S)

        self._load_button = QPushButton("Wczytaj")
        self._load_button.setObjectName("secondaryButton")
        self._load_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._load_button.setEnabled(False)
        self._load_button.clicked.connect(self._load_selected)

        self._delete_button = QPushButton("Usuń")
        self._delete_button.setObjectName("secondaryButton")
        self._delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_button.setEnabled(False)
        self._delete_button.clicked.connect(self._delete_selected)

        refresh_button = QPushButton("Odśwież")
        refresh_button.setObjectName("secondaryButton")
        refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_button.clicked.connect(self.refresh)

        export_button = QPushButton("Eksportuj listę CSV")
        export_button.setObjectName("secondaryButton")
        export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        export_button.clicked.connect(self._export_list_csv)

        buttons.addWidget(self._load_button, 0, 0)
        buttons.addWidget(self._delete_button, 0, 1)
        buttons.addWidget(refresh_button, 1, 0)
        buttons.addWidget(export_button, 1, 1)
        layout.addLayout(buttons)

        return panel

    def _build_details_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(theme.SPACE_L, theme.SPACE_L, theme.SPACE_L, theme.SPACE_L)
        layout.setSpacing(theme.SPACE_M)

        header = QHBoxLayout()
        trend_title = QLabel("Trend w czasie (CEP 50% i celność)")
        trend_title.setObjectName("sectionTitle")
        header.addWidget(trend_title)
        header.addStretch(1)
        unit_label = QLabel("Jednostka")
        unit_label.setObjectName("statLabel")
        header.addWidget(unit_label)
        header.addSpacing(theme.SPACE_S)
        self._unit_selector = UnitSelector(self._unit)
        self._unit_selector.unit_changed.connect(self._on_unit_changed)
        header.addWidget(self._unit_selector)
        layout.addLayout(header)

        self._chart_container = QVBoxLayout()
        layout.addLayout(self._chart_container)

        comparison_title = QLabel("Porównanie dwóch sesji")
        comparison_title.setObjectName("sectionTitle")
        layout.addSpacing(theme.SPACE_S)
        layout.addWidget(comparison_title)

        self._comparison_warning = QLabel()
        self._comparison_warning.setObjectName("warningText")
        self._comparison_warning.setWordWrap(True)
        self._comparison_warning.hide()
        layout.addWidget(self._comparison_warning)

        self._comparison_label = QLabel()
        self._comparison_label.setObjectName("mutedText")
        self._comparison_label.setWordWrap(True)
        layout.addWidget(self._comparison_label)

        layout.addStretch(1)
        return panel

    # ------------------------------------------------------------------
    # List handling
    # ------------------------------------------------------------------

    def _populate_list(self):
        self._list_widget.clear()

        for session in self._visible_sessions:
            label = (
                f"{session.created_at:%Y-%m-%d %H:%M} — "
                f"trafienia: {session.hit_count}, CEP {self._list_cep(session)}"
            )

            extras = []
            if session.weapon_type:
                extras.append(session.weapon_type)
            if session.distance_m:
                extras.append(f"{fmt_general(session.distance_m)} m")
            if session.scale_mm_per_px:
                extras.append(f"{fmt_num(session.scale_mm_per_px, 3)} mm/px")
            if session.notes:
                note = session.notes if len(session.notes) <= 30 else session.notes[:30] + "…"
                extras.append(note)
            if extras:
                label += " | " + ", ".join(extras)

            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            self._list_widget.addItem(item)

        self._load_button.setEnabled(False)
        self._delete_button.setEnabled(False)

    @staticmethod
    def _list_cep(session: TrainingSession) -> str:
        """CEP in this session's own default unit (mm when it has a scale)."""
        ctx = unit_context(session)
        unit = default_unit(ctx)
        value = convert_length(session.cep_50, unit, ctx)
        return f"{fmt_num(value, 0 if unit is LengthUnit.PX else 1)} {unit.value}"

    def _on_unit_changed(self, unit: LengthUnit):
        self._unit = LengthUnit(unit)
        self._rebuild_trend_chart()
        self._update_comparison()

    def _selected_ids(self) -> List[int]:
        return [item.data(Qt.ItemDataRole.UserRole) for item in self._list_widget.selectedItems()]

    def _on_selection_changed(self):
        ids = self._selected_ids()
        self._load_button.setEnabled(len(ids) == 1 and not self._analysis_lock)
        self._delete_button.setEnabled(len(ids) == 1)
        self._update_comparison()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _load_selected(self):
        ids = self._selected_ids()
        if len(ids) != 1:
            return

        try:
            session = load_session(ids[0])
        except FileNotFoundError as error:
            QMessageBox.warning(self, "Nie można wczytać obrazu", str(error))
            return

        if session is None:
            QMessageBox.warning(self, "Sesja nie istnieje", "Wybrana sesja została już usunięta.")
            self.refresh()
            return

        if self._on_session_loaded:
            self._on_session_loaded(session)

    def _delete_selected(self):
        ids = self._selected_ids()
        if len(ids) != 1:
            return

        confirmed = QMessageBox.question(
            self,
            "Usuń sesję",
            "Czy na pewno chcesz usunąć wybraną sesję treningową?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        delete_session(ids[0])
        self.refresh()

    def _export_list_csv(self):
        if not self._sessions:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Eksportuj listę sesji do CSV", "", "CSV (*.csv)",
        )
        if not file_path:
            return

        try:
            export_sessions_list_to_csv(self._sessions, file_path)
        except Exception as error:
            QMessageBox.critical(self, "Błąd eksportu", str(error))

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _update_comparison(self):
        ids = self._selected_ids()

        if len(ids) != 2:
            self._comparison_warning.hide()
            self._comparison_label.setText(
                "Wybierz dokładnie dwie sesje na liście (Ctrl+klik), aby zobaczyć porównanie."
            )
            return

        by_id = {session.id: session for session in self._visible_sessions}
        first, second = sorted((by_id[ids[0]], by_id[ids[1]]), key=lambda s: s.created_at)
        delta = compare_sessions(first, second, unit=self._unit)

        self._comparison_warning.setText(delta["warning"] or "")
        self._comparison_warning.setVisible(bool(delta["warning"]))
        self._comparison_label.setText(self._format_comparison(first, second, delta))

    @staticmethod
    def _format_comparison(first: TrainingSession, second: TrainingSession, comparison: dict) -> str:
        unit = comparison["unit"]
        decimals = DECIMALS[LengthUnit(unit)]
        lines = []

        lines.append(f"{first.created_at:%Y-%m-%d %H:%M} → {second.created_at:%Y-%m-%d %H:%M}")
        lines.append("")
        lines.append(
            f"Trafienia: {comparison['hit_count_first']} → {comparison['hit_count_second']} "
            f"({comparison['hit_count_delta']:+d})"
        )

        for label, metric in (
            ("Celność (MPI od punktu celowania)", "accuracy_radius"),
            ("Precyzja – śr. odległość od MPI (MRD)", "precision_radius"),
            ("CEP 50%", "cep_50"),
            ("Rozrzut maks. (ES, środek–środek)", "extreme_spread"),
        ):
            first_value = comparison[f"{metric}_first"]
            second_value = comparison[f"{metric}_second"]
            delta_value = comparison[f"{metric}_delta"]
            lines.append(
                f"{label}: {fmt_num(first_value, decimals)} → {fmt_num(second_value, decimals)} {unit} "
                f"({fmt_num(delta_value, decimals, signed=True)} {unit})"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Trend chart
    # ------------------------------------------------------------------

    def _rebuild_trend_chart(self):
        while self._chart_container.count():
            item = self._chart_container.takeAt(0)
            if item.widget():
                item.widget().hide()  # deleteLater() alone leaves it painted until the next loop pass
                item.widget().deleteLater()
        self._chart_canvas = None

        if len(self._visible_sessions) < 2:
            placeholder = QLabel("Zapisz (lub odfiltruj) co najmniej dwie sesje, aby zobaczyć trend w czasie.")
            placeholder.setObjectName("mutedText")
            placeholder.setWordWrap(True)
            self._chart_container.addWidget(placeholder)
            return

        trend = build_trend_series(self._visible_sessions, unit=self._unit)
        unit = trend["unit"]
        points = trend["points"]

        if trend["warning"]:
            warning_label = QLabel(trend["warning"])
            warning_label.setObjectName("warningText")
            warning_label.setWordWrap(True)
            self._chart_container.addWidget(warning_label)

        fig, ax = base_fig(h=3.0)

        dates = [point["created_at"] for point in points]
        cep = [point["cep_50"] for point in points]
        accuracy = [point["accuracy_radius"] for point in points]

        ax.plot(dates, cep, marker="o", markersize=4, color=PRIMARY, linewidth=1.6,
                label=f"CEP 50% ({unit})")
        ax.plot(dates, accuracy, marker="o", markersize=4, color=SECONDARY, linewidth=1.6,
                label=f"Celność - MPI od punktu celowania ({unit})")

        ax.set_ylabel(unit)
        ax.set_title("CEP 50% i celność w czasie", fontweight="bold")
        ax.legend(fontsize=8, facecolor=BG, edgecolor=AXIS)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        # Numeric, locale-independent date labels (no English month names).
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
            locator,
            formats=["%Y", "%m.%Y", "%d.%m", "%H:%M", "%H:%M", "%S.%f"],
            zero_formats=["", "%Y", "%m.%Y", "%d.%m", "%d.%m", "%H:%M"],
            offset_formats=["", "%Y", "%Y-%m", "%Y-%m-%d", "%Y-%m-%d", "%Y-%m-%d %H:%M"],
        ))
        fig.tight_layout(pad=1.6)

        self._chart_canvas = Canvas(fig)
        self._chart_canvas.setMinimumHeight(220)
        self._chart_container.addWidget(self._chart_canvas)
