import math

from PyQt6.QtCore import QObject, QSize, Qt, QThread, QTimer, pyqtSignal
import cv2

from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from controller.main_controller import AnalysisError, analyze_image
from services.analysis_overlay import draw_analysis_overlay
from services.manual_correction import map_display_point_to_image, set_aim_point_at, toggle_hit_at
from services.statistics import recompute_statistics
from services.units import (
    LengthUnit,
    UnitContext,
    convert_statistics,
    default_unit,
    format_length,
    is_available,
)
from view import theme
from view.number_format import fmt_num
from view.unit_selector import UnitSelector
from view.statistics_page import StatisticsPage
from view.settings_page import SettingsPage
from view.sessions_page import SessionsPage

from database.repository import save_session
from services.csv_export import export_session_to_csv


class _ClickableImageLabel(QLabel):
    """QLabel that reports click positions, used for manual hit correction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._click_callback = None

    def set_click_callback(self, callback):
        self._click_callback = callback

    def mousePressEvent(self, event):
        if self._click_callback:
            position = event.position()
            self._click_callback(position.x(), position.y())
        super().mousePressEvent(event)


class _NavItem(QFrame):
    """Custom nav item: QFrame + two QLabels. No QPushButton height constraints."""

    def __init__(self, label: str, description: str, parent=None):
        super().__init__(parent)
        self._checked = False
        self._callback = None
        self.setObjectName("navItem")
        self.setProperty("selected", False)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 8, 16, 8)  # + 3 px left accent bar = 16
        layout.setSpacing(2)

        self._title = QLabel(label)
        self._title.setObjectName("navTitle")

        self._caption = QLabel(description)
        self._caption.setObjectName("navHint")
        self._caption.setWordWrap(True)

        layout.addWidget(self._title)
        layout.addWidget(self._caption)

    # -- public API mimicking QPushButton.setChecked --
    def setChecked(self, checked: bool):
        self._checked = checked
        self.setProperty("selected", checked)
        # Re-evaluate the [selected="true"] selectors on the item and its labels.
        for widget in (self, self._title, self._caption):
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def clicked(self):
        pass  # overridden below via connect()

    def connect(self, slot):
        self._callback = slot

    # Allow item.clicked.connect(fn) pattern
    class _Signal:
        def __init__(self, item):
            self._item = item
        def connect(self, fn):
            self._item._callback = fn

    @property
    def clicked(self):
        return self._Signal(self)

    def mousePressEvent(self, event):
        if self._callback:
            self._callback()
        super().mousePressEvent(event)


class _AnalysisWorker(QObject):
    """
    Runs analyze_image() on a background QThread so the GUI thread stays
    responsive. Must not touch any widget directly - it only emits signals;
    all UI updates happen in slots running on the GUI thread (connected in
    MainWindow._start_analysis_worker).
    """

    finished = pyqtSignal(object)       # Session
    analysis_error = pyqtSignal(str)    # AnalysisError (needs manual calibration)
    unexpected_error = pyqtSignal(str)  # anything else

    def __init__(self, path: str, metadata: dict):
        super().__init__()
        self._path = path
        self._metadata = metadata

    def run(self):
        try:
            session = analyze_image(self._path, self._metadata)
        except AnalysisError as error:
            self.analysis_error.emit(str(error))
            return
        except Exception as error:
            self.unexpected_error.emit(str(error))
            return

        self.finished.emit(session)


class MainWindow(QMainWindow):
    """Main application shell with navigation prepared for future modules."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Inz Range – analiza treningu strzeleckiego")
        self.setMinimumSize(QSize(1180, 760))

        app = QApplication.instance()
        if app is not None:
            theme.apply_theme(app)

        self.nav_buttons = []
        self.nav_items = [
            ("Analiza tarczy", "Wczytaj zdjęcie i uruchom detekcję."),
            ("Statystyki", "Wyniki, wykresy i rozrzut trafień."),
            ("Sesje", "Historia treningów i porównania."),
            ("Ustawienia", "Kalibracja, skala i parametry analizy."),
        ]
        self.pages = QStackedWidget()
        self.selected_image_path = None
        self.original_preview_pixmap = None
        self.current_session = None
        self.stat_value_labels = {}
        self.correction_mode_enabled = False
        self.calibration_mode_enabled = False
        self.calibration_points = []
        self.aim_point_mode_enabled = False

        self._analysis_thread = None
        self._analysis_worker = None
        self._analysis_in_progress = False

        # Display units (in-memory per run). A loaded session uses ONLY its
        # own stored scale until the user explicitly changes the scale in
        # Ustawienia - no fallback to an unrelated global scale.
        self._unit_user_choice = None
        self._session_is_loaded = False
        self._loaded_scale = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(200)
        self._refresh_timer.timeout.connect(self._refresh_results)

        self._create_actions()
        self._build_layout()
        self._last_settings_scale = self.settings_page.get_scale_mm_per_px()
        self._apply_styles()
        self._select_page(0)

    def _create_actions(self):
        exit_action = QAction("Zamknij", self)
        exit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("Plik")
        file_menu.addAction(exit_action)

    def _build_layout(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = self._build_sidebar()
        content = self._build_content()

        root_layout.addWidget(sidebar)
        root_layout.addWidget(content, 1)

        self.setCentralWidget(root)

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(theme.SIDEBAR_WIDTH)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 20, 0, 16)
        layout.setSpacing(2)

        title = QLabel("Inz Range")
        title.setObjectName("appTitle")
        title.setContentsMargins(16, 0, 16, 0)

        subtitle = QLabel("Analiza trafień")
        subtitle.setObjectName("appSubtitle")
        subtitle.setContentsMargins(16, 0, 16, 0)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(20)

        for index, (label, description) in enumerate(self.nav_items):
            button = self._create_nav_button(label, description, index)
            self.nav_buttons.append(button)
            layout.addWidget(button)

        layout.addStretch(1)

        footer = QLabel("Projekt inżynierski\nPython, PyQt, OpenCV")
        footer.setObjectName("sidebarFooter")
        footer.setContentsMargins(16, 0, 16, 0)
        footer.setWordWrap(True)
        layout.addWidget(footer)

        return sidebar

    def _create_nav_button(self, label, description, index):
        item = _NavItem(label, description)
        item.clicked.connect(lambda: self._select_page(index))
        return item

    def _build_content(self):
        content = QFrame()
        content.setObjectName("content")

        layout = QVBoxLayout(content)
        layout.setContentsMargins(theme.SPACE_XL, 20, theme.SPACE_XL, theme.SPACE_XL)
        layout.setSpacing(theme.SPACE_L)

        header = self._build_header()
        layout.addWidget(header)

        self.pages.addWidget(self._build_analysis_page())

        # --- Real Statistics page ---
        self.statistics_page = StatisticsPage(on_unit_changed=self._on_unit_selected)
        self.pages.addWidget(self.statistics_page)

        # --- Real Sessions page ---
        self.sessions_page = SessionsPage(on_session_loaded=self._load_session_into_analysis)
        self.pages.addWidget(self.sessions_page)

        self.settings_page = SettingsPage(
            on_settings_changed=self._on_settings_changed,
            on_measure_requested=self._start_two_point_calibration,
        )
        self.pages.addWidget(self.settings_page)

        layout.addWidget(self.pages, 1)

        return content

    def _build_header(self):
        header = QFrame()
        header.setObjectName("header")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)

        text_area = QVBoxLayout()
        text_area.setSpacing(2)

        self.page_title = QLabel()
        self.page_title.setObjectName("pageTitle")

        self.page_hint = QLabel()
        self.page_hint.setObjectName("pageHint")

        text_area.addWidget(self.page_title)
        text_area.addWidget(self.page_hint)

        layout.addLayout(text_area, 1)

        # Plain status text in the window's status bar instead of a header badge.
        self.status_label = QLabel("Gotowe do pracy")
        self.status_label.setObjectName("statusText")
        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(False)
        status_bar.addWidget(self.status_label, 1)

        return header

    def _build_analysis_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_L)

        image_panel = self._build_image_panel()
        side_panel = self._build_analysis_side_panel()

        layout.addWidget(image_panel, 1)
        layout.addWidget(side_panel)

        return page

    def _build_image_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(theme.SPACE_L, theme.SPACE_L, theme.SPACE_L, theme.SPACE_L)
        layout.setSpacing(theme.SPACE_M)

        title = QLabel("Podgląd tarczy")
        title.setObjectName("sectionTitle")

        self.image_preview = _ClickableImageLabel("Wczytaj zdjęcie tarczy, aby zobaczyć podgląd analizy.")
        self.image_preview.setObjectName("imagePreview")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setWordWrap(True)
        self.image_preview.setMinimumHeight(420)
        self.image_preview.set_click_callback(self._handle_image_click)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(theme.SPACE_S)

        self.load_image_button = QPushButton("Wczytaj zdjęcie")
        self.load_image_button.setObjectName("secondaryButton")
        self.load_image_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_image_button.clicked.connect(self._choose_image)

        self.analyze_button = QPushButton("Analizuj")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._analyze_selected_image)

        self.correction_toggle = QPushButton("Koryguj trafienia")
        self.correction_toggle.setObjectName("secondaryButton")
        self.correction_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.correction_toggle.setCheckable(True)
        self.correction_toggle.setEnabled(False)
        self.correction_toggle.clicked.connect(self._toggle_correction_mode)

        self.aim_point_toggle = QPushButton("Punkt celowania")
        self.aim_point_toggle.setToolTip(
            "Ustaw punkt celowania: kliknij na tarczy miejsce, w które faktycznie celowano."
        )
        self.aim_point_toggle.setObjectName("secondaryButton")
        self.aim_point_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.aim_point_toggle.setCheckable(True)
        self.aim_point_toggle.setEnabled(False)
        self.aim_point_toggle.clicked.connect(self._toggle_aim_point_mode)

        toolbar.addWidget(self.load_image_button)
        toolbar.addWidget(self.analyze_button)
        toolbar.addSpacing(theme.SPACE_M)
        toolbar.addWidget(self.correction_toggle)
        toolbar.addWidget(self.aim_point_toggle)
        toolbar.addStretch(1)

        layout.addWidget(title)
        layout.addWidget(self.image_preview, 1)
        layout.addLayout(toolbar)

        return panel

    def _build_analysis_side_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(330)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(theme.SPACE_L, theme.SPACE_L, theme.SPACE_L, theme.SPACE_L)
        layout.setSpacing(0)

        title = QLabel("Wyniki analizy")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        layout.addSpacing(theme.SPACE_S)

        unit_row = QHBoxLayout()
        unit_label = QLabel("Jednostka")
        unit_label.setObjectName("statLabel")
        self.unit_selector = UnitSelector(LengthUnit.PX)
        self.unit_selector.set_context(UnitContext())
        self.unit_selector.unit_changed.connect(self._on_unit_selected)
        unit_row.addWidget(unit_label)
        unit_row.addStretch(1)
        unit_row.addWidget(self.unit_selector)
        layout.addLayout(unit_row)
        layout.addSpacing(theme.SPACE_XS)

        stats = [
            ("Liczba trafień", "-"),
            ("Celność", "-"),
            ("Precyzja", "-"),
            ("CEP 50%", "-"),
        ]

        for label, value in stats:
            row, result = self._build_stat_row(label, value)
            self.stat_value_labels[label] = result
            layout.addWidget(row)

        layout.addSpacing(theme.SPACE_M)

        # Explains why results are in px / why MOA and mrad are unavailable.
        self.unit_hint = QLabel()
        self.unit_hint.setObjectName("mutedText")
        self.unit_hint.setWordWrap(True)
        self.unit_hint.hide()
        layout.addWidget(self.unit_hint)

        self.goto_settings_button = QPushButton("Przejdź do Ustawień")
        self.goto_settings_button.setObjectName("linkButton")
        self.goto_settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.goto_settings_button.clicked.connect(lambda: self._select_page(3))
        self.goto_settings_button.hide()
        layout.addWidget(self.goto_settings_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addSpacing(theme.SPACE_S)

        self.analysis_note = QLabel("Wczytaj zdjęcie, a potem uruchom analizę.")
        self.analysis_note.setObjectName("mutedText")
        self.analysis_note.setWordWrap(True)
        layout.addWidget(self.analysis_note)

        layout.addStretch(1)

        layout.addWidget(self._build_save_section())
        layout.addSpacing(theme.SPACE_S)

        # Quick-link to statistics page
        goto_stats = QPushButton("Szczegółowe statystyki")
        goto_stats.setObjectName("secondaryButton")
        goto_stats.setCursor(Qt.CursorShape.PointingHandCursor)
        goto_stats.clicked.connect(lambda: self._select_page(1))
        layout.addWidget(goto_stats)

        return panel

    def _build_save_section(self):
        section = QWidget()

        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_S)

        separator = QFrame()
        separator.setObjectName("separator")
        layout.addWidget(separator)
        layout.addSpacing(theme.SPACE_XS)

        title = QLabel("Zapis sesji")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.weapon_type_input = QLineEdit()
        self.weapon_type_input.setPlaceholderText("opcjonalnie")

        # Plain numeric field; 0 is the "not given" state, shown as an em dash
        # and saved as NULL (see _save_current_session).
        self.distance_input = QDoubleSpinBox()
        self.distance_input.setRange(0.0, 1000.0)
        self.distance_input.setDecimals(1)
        self.distance_input.setSuffix(" m")
        self.distance_input.setSpecialValueText("—")
        self.distance_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.distance_input.setToolTip("Odległość strzelania w metrach – potrzebna do wyników w MOA i mrad.")
        self.distance_input.valueChanged.connect(lambda _value: self._refresh_timer.start())

        self.save_session_button = QPushButton("Zapisz sesję")
        self.save_session_button.setObjectName("secondaryButton")
        self.save_session_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_session_button.setEnabled(False)
        self.save_session_button.clicked.connect(self._save_current_session)

        self.export_csv_button = QPushButton("Eksportuj CSV")
        self.export_csv_button.setObjectName("secondaryButton")
        self.export_csv_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_csv_button.setEnabled(False)
        self.export_csv_button.clicked.connect(self._export_current_session_csv)

        buttons = QHBoxLayout()
        buttons.setSpacing(theme.SPACE_S)
        buttons.addWidget(self.save_session_button, 1)
        buttons.addWidget(self.export_csv_button, 1)

        fields = QGridLayout()
        fields.setHorizontalSpacing(theme.SPACE_S)
        fields.setVerticalSpacing(theme.SPACE_XS)
        weapon_label = QLabel("Rodzaj broni")
        weapon_label.setObjectName("statLabel")
        distance_label = QLabel("Odległość strzelania")
        distance_label.setObjectName("statLabel")
        fields.addWidget(weapon_label, 0, 0)
        fields.addWidget(distance_label, 0, 1)
        fields.addWidget(self.weapon_type_input, 1, 0)
        fields.addWidget(self.distance_input, 1, 1)
        fields.setColumnStretch(0, 1)
        fields.setColumnStretch(1, 1)

        layout.addLayout(fields)
        layout.addSpacing(theme.SPACE_XS)
        layout.addLayout(buttons)

        return section

    def _build_stat_row(self, label, value):
        row = QFrame()
        row.setObjectName("statRow")

        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 9, 0, 9)

        name = QLabel(label)
        name.setObjectName("mutedText")

        result = QLabel(value)
        result.setObjectName("statValue")
        result.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(name, 1)
        layout.addWidget(result)

        return row, result

    def _select_page(self, index):
        if index != 0:
            # Leaving the analysis page mid-calibration would otherwise leave
            # a stale first click waiting - the next click after coming back
            # would be silently treated as the second calibration point.
            self._cancel_calibration_if_active()

        self.pages.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)

        title, hint = self.nav_items[index]
        self.page_title.setText(title)
        self.page_hint.setText(hint)

    def _cancel_calibration_if_active(self):
        if not self.calibration_mode_enabled:
            return

        self.calibration_mode_enabled = False
        self.calibration_points = []
        if self.current_session is not None:
            self.analysis_note.setText(self._build_analysis_summary(self.current_session))

    def _choose_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz zdjęcie tarczy",
            "",
            "Obrazy (*.jpg *.jpeg *.png *.bmp *.tiff);;Wszystkie pliki (*)",
        )

        if not file_path:
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.image_preview.setPixmap(QPixmap())
            self.image_preview.setText("Nie udało się wczytać wybranego obrazu.")
            self.selected_image_path = None
            self.original_preview_pixmap = None
            self.analyze_button.setEnabled(False)
            return

        self.selected_image_path = file_path
        self.original_preview_pixmap = pixmap
        self.current_session = None
        self._session_is_loaded = False
        self._loaded_scale = None
        self.analyze_button.setEnabled(True)
        self._set_correction_mode(False)
        self._set_aim_point_mode(False)
        self._cancel_calibration_if_active()
        self.correction_toggle.setEnabled(False)
        self.aim_point_toggle.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.export_csv_button.setEnabled(False)
        self.weapon_type_input.setText("")
        self.distance_input.setValue(0.0)
        self.status_label.setText("Zdjęcie wczytane")
        self.analysis_note.setText("Zdjęcie jest gotowe do analizy.")
        self._refresh_results()
        self._update_preview_pixmap()

    def _update_preview_pixmap(self):
        if self.original_preview_pixmap is None:
            return

        target_size = self.image_preview.size()
        scaled = self.original_preview_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.image_preview.setText("")
        self.image_preview.setPixmap(scaled)

    def _analyze_selected_image(self):
        if not self.selected_image_path:
            return

        if self._analysis_in_progress:
            return  # guard against a second overlapping analysis

        self._analysis_in_progress = True
        self._set_controls_locked_during_analysis(True)
        self.status_label.setText("Analizuję…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        metadata = self.settings_page.get_metadata()
        self._start_analysis_worker(self.selected_image_path, metadata)

    def _start_analysis_worker(self, path: str, metadata: dict):
        self._analysis_worker = _AnalysisWorker(path, metadata)
        self._analysis_thread = QThread(self)
        self._analysis_worker.moveToThread(self._analysis_thread)

        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.analysis_error.connect(self._on_analysis_error)
        self._analysis_worker.unexpected_error.connect(self._on_analysis_unexpected_error)

        # Whichever outcome fires, stop the thread; _on_analysis_thread_finished
        # (connected below) does the common cleanup regardless of outcome.
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.analysis_error.connect(self._analysis_thread.quit)
        self._analysis_worker.unexpected_error.connect(self._analysis_thread.quit)
        self._analysis_thread.finished.connect(self._on_analysis_thread_finished)

        self._analysis_thread.start()

    def _set_controls_locked_during_analysis(self, locked: bool):
        """
        Controls that must not be touched while a background analysis owns
        self.current_session (or, for load_image_button/analyze_button, that
        would start a conflicting second operation). Re-enabled unconditionally
        once the worker thread finishes, regardless of success/failure.

        correction_toggle/aim_point_toggle/save_session_button/export_csv_button
        are NOT handled here - they depend on the analysis OUTCOME, so the
        three result slots (_on_analysis_finished/_on_analysis_error/
        _on_analysis_unexpected_error) set them explicitly, exactly like the
        old synchronous code did.
        """

        enabled = not locked
        self.load_image_button.setEnabled(enabled)
        self.analyze_button.setEnabled(enabled and self.selected_image_path is not None)
        self.settings_page.set_measure_button_enabled(enabled)
        self.sessions_page.set_analysis_lock(locked)

    def _on_analysis_finished(self, session):
        overlay = draw_analysis_overlay(session)
        self.current_session = session

        # Update image preview
        self.original_preview_pixmap = self._pixmap_from_bgr_image(overlay)
        self._update_preview_pixmap()

        # Update settings page with detected radius for scale calibration.
        # Done BEFORE reading the scale, so the results use the auto-filled
        # px value instead of the previous one.
        if session.target and session.target.radius:
            self.settings_page.set_detected_radius(session.target.radius)

        # Fresh analysis: scale from Ustawienia, distance from the form.
        self._last_settings_scale = self.settings_page.get_scale_mm_per_px()
        self._session_is_loaded = False
        self._loaded_scale = None
        self._refresh_results()

        self.status_label.setText("Analiza gotowa")
        self.analysis_note.setText(self._build_analysis_summary(session))
        self.correction_toggle.setEnabled(True)
        self.aim_point_toggle.setEnabled(True)
        self.save_session_button.setEnabled(True)
        self.export_csv_button.setEnabled(True)

    def _on_analysis_error(self, message: str):
        self.status_label.setText("Wymagana kalibracja")
        self.analysis_note.setText("Nie udało się automatycznie wykryć tarczy.")
        self.correction_toggle.setEnabled(False)
        self.aim_point_toggle.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.export_csv_button.setEnabled(False)
        QMessageBox.warning(self, "Analiza przerwana", message)

    def _on_analysis_unexpected_error(self, message: str):
        self.status_label.setText("Błąd analizy")
        self.analysis_note.setText("Analiza nie mogła zostać zakończona.")
        self.correction_toggle.setEnabled(False)
        self.aim_point_toggle.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.export_csv_button.setEnabled(False)
        QMessageBox.critical(self, "Błąd analizy", message)

    def _on_analysis_thread_finished(self):
        """Runs once per analysis, regardless of success/error - common
        cleanup only. Outcome-specific UI state is handled by the three
        _on_analysis_* slots above, which fire before this one."""

        QApplication.restoreOverrideCursor()
        self._set_controls_locked_during_analysis(False)
        self._analysis_in_progress = False

        if self._analysis_worker is not None:
            self._analysis_worker.deleteLater()
            self._analysis_worker = None
        if self._analysis_thread is not None:
            self._analysis_thread.deleteLater()
            self._analysis_thread = None

    def closeEvent(self, event):
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self._analysis_thread.quit()
            self._analysis_thread.wait()
        super().closeEvent(event)

    def _toggle_correction_mode(self):
        self._set_correction_mode(self.correction_toggle.isChecked())

    def _set_correction_mode(self, enabled: bool):
        self.correction_mode_enabled = enabled
        self.correction_toggle.setChecked(enabled)
        if enabled:
            self.calibration_mode_enabled = False  # the modes don't mix on the same click
            self._set_aim_point_mode(False)

        if enabled:
            self.analysis_note.setText(
                "Tryb korekty: kliknij na tarczy, aby dodać trafienie, "
                "lub kliknij istniejące, aby je usunąć."
            )
        elif self.current_session is not None:
            self.analysis_note.setText(self._build_analysis_summary(self.current_session))

    def _toggle_aim_point_mode(self):
        self._set_aim_point_mode(self.aim_point_toggle.isChecked())

    def _set_aim_point_mode(self, enabled: bool):
        self.aim_point_mode_enabled = enabled
        self.aim_point_toggle.setChecked(enabled)
        if enabled:
            self.calibration_mode_enabled = False
            self._set_correction_mode(False)

        if enabled:
            self.analysis_note.setText(
                "Tryb punktu celowania: kliknij miejsce, w które faktycznie celowano "
                "(np. środek strefy A na sylwetce) – celność będzie liczona względem niego."
            )
        elif self.current_session is not None:
            self.analysis_note.setText(self._build_analysis_summary(self.current_session))

    def _start_two_point_calibration(self):
        """Called by SettingsPage's 'Zmierz na zdjęciu' button."""
        if self.current_session is None:
            QMessageBox.information(
                self,
                "Najpierw analiza",
                "Uruchom najpierw analizę zdjęcia, aby móc zmierzyć na nim odległość.",
            )
            return

        self._set_correction_mode(False)
        self._set_aim_point_mode(False)
        self.calibration_mode_enabled = True
        self.calibration_points = []
        self._select_page(0)
        self.analysis_note.setText(
            "Kalibracja skali: kliknij dwa punkty na zdjęciu, między którymi znasz "
            "rzeczywistą odległość."
        )

    def _handle_image_click(self, label_x: float, label_y: float):
        if self.current_session is None:
            return

        pixmap = self.image_preview.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        image = self.current_session.image
        image_point = map_display_point_to_image(
            display_point=(label_x, label_y),
            label_size=(self.image_preview.width(), self.image_preview.height()),
            pixmap_size=(pixmap.width(), pixmap.height()),
            image_size=(image.width, image.height),
        )

        if image_point is None:
            return

        if self.calibration_mode_enabled:
            self._handle_calibration_click(image_point)
            return

        if self.aim_point_mode_enabled:
            self._handle_aim_point_click(image_point)
            return

        if not self.correction_mode_enabled:
            return

        session = self.current_session
        session.hits = toggle_hit_at(session.hits, session.target, image_point)
        session.statistics = recompute_statistics(session)
        self._after_hits_changed(session)

    def _handle_aim_point_click(self, image_point):
        session = self.current_session
        session.target = set_aim_point_at(session.target, image_point)
        session.statistics = recompute_statistics(session)

        self._set_aim_point_mode(False)
        self._after_hits_changed(session)

    def _handle_calibration_click(self, image_point):
        self.calibration_points.append(image_point)
        self._redraw_with_calibration_markers()

        if len(self.calibration_points) < 2:
            self.analysis_note.setText("Kalibracja skali: kliknij drugi punkt.")
            return

        (x1, y1), (x2, y2) = self.calibration_points
        distance_px = math.hypot(x2 - x1, y2 - y1)

        self.calibration_mode_enabled = False
        self.calibration_points = []

        self.settings_page.set_measured_pixel_distance(distance_px)
        self._after_hits_changed(self.current_session)  # redraws without the calibration crosses
        self.analysis_note.setText(self._build_analysis_summary(self.current_session))

        QMessageBox.information(
            self,
            "Zmierzono odległość",
            f"Zmierzona odległość: {fmt_num(distance_px)} px. Wpisz odpowiadającą jej "
            f"odległość rzeczywistą w mm w zakładce Ustawienia.",
        )
        self._select_page(3)

    def _redraw_with_calibration_markers(self):
        overlay = draw_analysis_overlay(self.current_session)
        for x, y in self.calibration_points:
            center = (int(round(x)), int(round(y)))
            cv2.drawMarker(
                overlay, center, (0, 140, 255),
                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2,
            )
        self.original_preview_pixmap = self._pixmap_from_bgr_image(overlay)
        self._update_preview_pixmap()

    def _after_hits_changed(self, session):
        overlay = draw_analysis_overlay(session)
        self.original_preview_pixmap = self._pixmap_from_bgr_image(overlay)
        self._update_preview_pixmap()
        self._refresh_results()

    def _save_current_session(self):
        if self.current_session is None:
            return

        weapon_type = self.weapon_type_input.text().strip() or None
        distance_m = self.distance_input.value() or None
        scale = self._current_unit_context().scale_mm_per_px  # the scale the results are shown with

        try:
            saved = save_session(
                self.current_session,
                weapon_type=weapon_type,
                distance_m=distance_m,
                scale_mm_per_px=scale,
            )
            QMessageBox.information(
                self,
                "Sesja zapisana",
                f"Sesja treningowa została zapisana (id {saved.id}).",
            )
        except Exception as error:
            QMessageBox.critical(self, "Błąd zapisu", str(error))

    def _export_current_session_csv(self):
        if self.current_session is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Eksportuj sesję do CSV", "", "CSV (*.csv)",
        )
        if not file_path:
            return

        scale = self._current_unit_context().scale_mm_per_px

        try:
            export_session_to_csv(self.current_session, file_path, scale_mm_per_px=scale)
        except Exception as error:
            QMessageBox.critical(self, "Błąd eksportu", str(error))

    def _load_session_into_analysis(self, session):
        """Called by SessionsPage when the user loads a past session."""
        metadata = session.metadata or {}
        self.current_session = session
        # The session's OWN scale only - never the unrelated global one.
        self._session_is_loaded = True
        self._loaded_scale = metadata.get("scale_mm_per_px")
        self.selected_image_path = session.image.path
        self.analyze_button.setEnabled(True)
        self.correction_toggle.setEnabled(True)
        self.aim_point_toggle.setEnabled(True)
        self.save_session_button.setEnabled(True)
        self.export_csv_button.setEnabled(True)
        self._set_correction_mode(False)
        self._set_aim_point_mode(False)
        self._cancel_calibration_if_active()

        overlay = draw_analysis_overlay(session)
        self.original_preview_pixmap = self._pixmap_from_bgr_image(overlay)
        self._update_preview_pixmap()

        self.weapon_type_input.setText(metadata.get("weapon_type") or "")
        # Reset to the empty state when the stored distance is NULL, so a
        # previous session's distance is never reused.
        self.distance_input.setValue(metadata.get("distance_m") or 0.0)

        self._refresh_results()

        self.status_label.setText("Sesja wczytana")
        self.analysis_note.setText(self._build_analysis_summary(session))
        self._select_page(0)

    def _pixmap_from_bgr_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width

        qimage = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

        return QPixmap.fromImage(qimage)

    # ------------------------------------------------------------------
    # Display units
    # ------------------------------------------------------------------

    def _current_unit_context(self) -> UnitContext:
        if self._session_is_loaded:
            scale = self._loaded_scale
        else:
            scale = self.settings_page.get_scale_mm_per_px()
        distance = self.distance_input.value() or None
        return UnitContext(scale_mm_per_px=scale, distance_m=distance)

    def _on_unit_selected(self, unit):
        self._unit_user_choice = LengthUnit(unit)
        self._refresh_results()

    def _on_settings_changed(self):
        if not hasattr(self, "_last_settings_scale") or self._analysis_in_progress:
            return  # still constructing, or the analysis will refresh itself
        scale = self.settings_page.get_scale_mm_per_px()
        if scale == self._last_settings_scale:
            return  # e.g. the manual-calibration checkbox - not a scale change
        self._last_settings_scale = scale
        # An explicit scale change in Ustawienia applies to the session shown now.
        self._session_is_loaded = False
        self._loaded_scale = None
        self._refresh_timer.start()

    def _refresh_results(self):
        """Single place that shows the current session's lengths in the chosen unit."""
        ctx = self._current_unit_context()
        choice = self._unit_user_choice
        unit = choice if choice is not None and is_available(choice, ctx) else default_unit(ctx)

        self.unit_selector.set_context(ctx)
        self.unit_selector.set_unit(unit)

        session = self.current_session
        if session is None:
            self._reset_statistics()
            self.unit_hint.hide()
            self.goto_settings_button.hide()
            return

        self._update_statistics(session.statistics, unit, ctx)
        self._update_unit_hint(ctx)
        self.statistics_page.set_session(
            session, scale_mm_per_px=ctx.scale_mm_per_px, distance_m=ctx.distance_m, unit=unit,
        )

    def _update_unit_hint(self, ctx: UnitContext):
        if not ctx.has_scale:
            text = (
                "Skala nie jest ustawiona – wyniki w pikselach. Ustaw skalę w Ustawieniach."
            )
            show_link = True
        elif not ctx.has_distance:
            text = "Podaj odległość strzelania, aby zobaczyć wyniki w MOA i mrad."
            show_link = False
        else:
            text, show_link = None, False

        self.unit_hint.setText(text or "")
        self.unit_hint.setVisible(text is not None)
        self.goto_settings_button.setVisible(show_link)

    def _update_statistics(self, statistics, unit=LengthUnit.PX, ctx=None):
        converted = convert_statistics(statistics, unit, ctx or UnitContext())

        def length(key):
            return format_length(converted.get(key, 0.0), unit)

        values = {
            "Liczba trafień": str(statistics.get("count", 0)),
            "Celność": length("accuracy_radius"),
            "Precyzja": length("precision_radius"),
            "CEP 50%": length("cep_50"),
        }

        for label, value in values.items():
            self.stat_value_labels[label].setText(value)

    def _reset_statistics(self):
        for value_label in self.stat_value_labels.values():
            value_label.setText("-")

    def _build_analysis_summary(self, session):
        target = session.target
        count = session.statistics.get("count", 0)

        if target.type != "circular":
            return (
                f"Liczba trafień: {count}. "
                f"Obszar tarczy: typ nieregularny, "
                f"środek analizy: ({target.center_x:.0f}, {target.center_y:.0f})."
            )

        return (
            f"Liczba trafień: {count}. "
            f"Środek tarczy: ({target.center_x:.0f}, {target.center_y:.0f}), "
            f"promień: {target.radius:.0f} px."
        )



    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview_pixmap()

    def _apply_styles(self):
        self.setStyleSheet(theme.build_stylesheet())
