from __future__ import annotations

from typing import Callable, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
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
from services.session_trends import build_trend_series, compare_sessions
from view.chart_style import ACCENT, BG, BORDER, TEAL, TEAL2, Canvas, base_fig


class SessionsPage(QWidget):
    """
    Historia zapisanych sesji treningowych: lista, wczytywanie,
    usuwanie, porównanie dwóch sesji oraz wykres trendu w czasie.
    """

    def __init__(self, on_session_loaded: Optional[Callable[[Session], None]] = None, parent=None):
        super().__init__(parent)
        self._on_session_loaded = on_session_loaded
        self._sessions: List[TrainingSession] = []
        self._chart_canvas: Optional[Canvas] = None
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self):
        self._sessions = list_sessions()
        self._populate_list()
        self._rebuild_trend_chart()
        self._update_comparison()

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(22)

        layout.addWidget(self._build_list_panel(), 1)
        layout.addWidget(self._build_details_panel(), 2)

    def _build_list_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("imagePanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title = QLabel("Zapisane sesje")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        hint = QLabel("Ctrl+klik, aby wybrać dwie sesje do porównania.")
        hint.setObjectName("mutedText")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._list_widget = QListWidget()
        self._list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list_widget, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)

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

        buttons.addWidget(self._load_button)
        buttons.addWidget(self._delete_button)
        buttons.addWidget(refresh_button)
        layout.addLayout(buttons)

        return panel

    def _build_details_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("imagePanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(16)

        trend_title = QLabel("Trend w czasie (CEP 50% i celność)")
        trend_title.setObjectName("sectionTitle")
        layout.addWidget(trend_title)

        self._chart_container = QVBoxLayout()
        layout.addLayout(self._chart_container)

        comparison_title = QLabel("Porównanie dwóch sesji")
        comparison_title.setObjectName("sectionTitle")
        layout.addWidget(comparison_title)

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

        for session in self._sessions:
            label = (
                f"{session.created_at:%Y-%m-%d %H:%M} — "
                f"{session.hit_count} strzałów, CEP {session.cep_50:.0f}px"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            self._list_widget.addItem(item)

        self._load_button.setEnabled(False)
        self._delete_button.setEnabled(False)

    def _selected_ids(self) -> List[int]:
        return [item.data(Qt.ItemDataRole.UserRole) for item in self._list_widget.selectedItems()]

    def _on_selection_changed(self):
        ids = self._selected_ids()
        self._load_button.setEnabled(len(ids) == 1)
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

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _update_comparison(self):
        ids = self._selected_ids()

        if len(ids) != 2:
            self._comparison_label.setText(
                "Wybierz dokładnie dwie sesje na liście (Ctrl+klik), aby zobaczyć porównanie."
            )
            return

        by_id = {session.id: session for session in self._sessions}
        first, second = sorted((by_id[ids[0]], by_id[ids[1]]), key=lambda s: s.created_at)
        delta = compare_sessions(first, second)

        self._comparison_label.setText(self._format_comparison(first, second, delta))

    @staticmethod
    def _format_comparison(first: TrainingSession, second: TrainingSession, delta: dict) -> str:
        def signed(value: float) -> str:
            return f"{value:+.1f}"

        return (
            f"{first.created_at:%Y-%m-%d %H:%M} → {second.created_at:%Y-%m-%d %H:%M}\n\n"
            f"Trafienia: {first.hit_count} → {second.hit_count} "
            f"({delta['hit_count_delta']:+d})\n"
            f"Celność (MPI-środek): {first.accuracy_radius:.1f} → {second.accuracy_radius:.1f} px "
            f"({signed(delta['accuracy_radius_delta'])} px)\n"
            f"Precyzja (śr. od MPI): {first.precision_radius:.1f} → {second.precision_radius:.1f} px "
            f"({signed(delta['precision_radius_delta'])} px)\n"
            f"CEP 50%: {first.cep_50:.1f} → {second.cep_50:.1f} px "
            f"({signed(delta['cep_50_delta'])} px)"
        )

    # ------------------------------------------------------------------
    # Trend chart
    # ------------------------------------------------------------------

    def _rebuild_trend_chart(self):
        while self._chart_container.count():
            item = self._chart_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._chart_canvas = None

        series = build_trend_series(self._sessions)

        if len(series) < 2:
            placeholder = QLabel("Zapisz co najmniej dwie sesje, aby zobaczyć trend w czasie.")
            placeholder.setObjectName("mutedText")
            self._chart_container.addWidget(placeholder)
            return

        fig, ax = base_fig(h=3.0)

        dates = [point["created_at"] for point in series]
        cep = [point["cep_50"] for point in series]
        accuracy = [point["accuracy_radius"] for point in series]

        ax.plot(dates, cep, marker="o", color=TEAL, linewidth=1.8, label="CEP 50% (px)")
        ax.plot(dates, accuracy, marker="o", color=ACCENT, linewidth=1.8, label="Celność - offset MPI (px)")

        ax.set_ylabel("px")
        ax.set_title("CEP 50% i celność w czasie", fontweight="bold")
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER)
        fig.autofmt_xdate()

        self._chart_canvas = Canvas(fig)
        self._chart_canvas.setMinimumHeight(220)
        self._chart_container.addWidget(self._chart_canvas)
