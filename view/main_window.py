from PyQt6.QtCore import QSize, Qt
import cv2

from PyQt6.QtGui import QAction, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
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
from view.statistics_page import StatisticsPage
from view.settings_page import SettingsPage


class _NavItem(QFrame):
    """Custom nav item: QFrame + two QLabels. No QPushButton height constraints."""

    def __init__(self, label: str, description: str, parent=None):
        super().__init__(parent)
        self._checked = False
        self._callback = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(3)

        self._title = QLabel(label)
        self._title.setStyleSheet(
            "color: #ffffff; font-size: 13px; font-weight: 700; background: transparent;"
        )

        self._caption = QLabel(description)
        self._caption.setWordWrap(True)
        self._caption.setStyleSheet(
            "color: #b9c6d3; font-size: 11px; background: transparent;"
        )

        layout.addWidget(self._title)
        layout.addWidget(self._caption)

    # -- public API mimicking QPushButton.setChecked --
    def setChecked(self, checked: bool):
        self._checked = checked
        self._apply_style()

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

    def enterEvent(self, event):
        if not self._checked:
            self.setStyleSheet(
                "background: #223040; border: 1px solid #2d4054; border-radius: 8px;"
            )
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_style()
        super().leaveEvent(event)

    def _apply_style(self):
        if self._checked:
            self.setStyleSheet(
                "background: #2f6f73; border: 1px solid #3f8c91; border-radius: 8px;"
            )
        else:
            self.setStyleSheet(
                "background: transparent; border: 1px solid transparent; border-radius: 8px;"
            )


class MainWindow(QMainWindow):
    """Main application shell with navigation prepared for future modules."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Inz Range - analiza treningu strzeleckiego")
        self.setMinimumSize(QSize(1180, 760))

        self.nav_buttons = []
        self.pages = QStackedWidget()
        self.selected_image_path = None
        self.original_preview_pixmap = None
        self.current_session = None
        self.stat_value_labels = {}

        self._create_actions()
        self._build_layout()
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
        sidebar.setFixedWidth(220)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(8)

        title = QLabel("Inz Range")
        title.setObjectName("appTitle")

        subtitle = QLabel("Analiza trafien")
        subtitle.setObjectName("appSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(18)

        nav_items = [
            ("Analiza tarczy", "Wczytaj zdjecie i uruchom detekcje."),
            ("Statystyki", "Wyniki, wykresy i rozrzut trafien."),
            ("Sesje", "Historia treningow i porownania."),
            ("Ustawienia", "Kalibracja, skala i parametry analizy."),
        ]

        for index, (label, description) in enumerate(nav_items):
            button = self._create_nav_button(label, description, index)
            self.nav_buttons.append(button)
            layout.addWidget(button)

        layout.addStretch(1)

        footer = QLabel("Projekt inzynierski\nPython + PyQt + OpenCV")
        footer.setObjectName("sidebarFooter")
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
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(22)

        header = self._build_header()
        layout.addWidget(header)

        self.pages.addWidget(self._build_analysis_page())

        # --- Real Statistics page ---
        self.statistics_page = StatisticsPage()
        self.pages.addWidget(self.statistics_page)

        self.pages.addWidget(self._build_placeholder_page(
            "Sesje treningowe",
            "Tutaj bedzie lista zapisanych analiz oraz porownywanie wynikow z roznych dni.",
        ))
        self.settings_page = SettingsPage()
        self.pages.addWidget(self.settings_page)

        layout.addWidget(self.pages, 1)

        return content

    def _build_header(self):
        header = QFrame()
        header.setObjectName("header")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)

        text_area = QVBoxLayout()
        text_area.setSpacing(4)

        eyebrow = QLabel("Panel treningowy")
        eyebrow.setObjectName("eyebrow")

        heading = QLabel("Analiza trafien podczas treningu strzeleckiego")
        heading.setObjectName("heading")

        text_area.addWidget(eyebrow)
        text_area.addWidget(heading)

        self.status_label = QLabel("Gotowe do pracy")
        self.status_label.setObjectName("statusBadge")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(text_area, 1)
        layout.addWidget(self.status_label)

        return header

    def _build_analysis_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(22)

        image_panel = self._build_image_panel()
        side_panel = self._build_analysis_side_panel()

        layout.addWidget(image_panel, 1)
        layout.addWidget(side_panel)

        return page

    def _build_image_panel(self):
        panel = QFrame()
        panel.setObjectName("imagePanel")
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        title = QLabel("Podglad tarczy")
        title.setObjectName("sectionTitle")

        self.image_preview = QLabel("Wczytaj zdjecie tarczy, aby zobaczyc podglad analizy.")
        self.image_preview.setObjectName("imagePreview")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setWordWrap(True)
        self.image_preview.setMinimumHeight(420)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)

        load_button = QPushButton("Wczytaj zdjecie")
        load_button.setObjectName("primaryButton")
        load_button.setCursor(Qt.CursorShape.PointingHandCursor)
        load_button.clicked.connect(self._choose_image)

        self.analyze_button = QPushButton("Analizuj")
        self.analyze_button.setObjectName("secondaryButton")
        self.analyze_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._analyze_selected_image)

        toolbar.addWidget(load_button)
        toolbar.addWidget(self.analyze_button)
        toolbar.addStretch(1)

        layout.addWidget(title)
        layout.addWidget(self.image_preview, 1)
        layout.addLayout(toolbar)

        return panel

    def _build_analysis_side_panel(self):
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(330)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(16)

        title = QLabel("Wyniki analizy")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        stats = [
            ("Liczba trafien", "-"),
            ("Sredni promien", "-"),
            ("Rozrzut maks.", "-"),
            ("CEP 50%", "-"),
        ]

        for label, value in stats:
            row, result = self._build_stat_row(label, value)
            self.stat_value_labels[label] = result
            layout.addWidget(row)

        layout.addSpacing(8)

        self.analysis_note = QLabel("Wczytaj zdjecie, a potem uruchom analize.")
        self.analysis_note.setObjectName("mutedText")
        self.analysis_note.setWordWrap(True)
        layout.addWidget(self.analysis_note)

        layout.addStretch(1)

        # Quick-link to statistics page
        goto_stats = QPushButton("Zobacz szczegółowe statystyki →")
        goto_stats.setObjectName("secondaryButton")
        goto_stats.setCursor(Qt.CursorShape.PointingHandCursor)
        goto_stats.clicked.connect(lambda: self._select_page(1))
        layout.addWidget(goto_stats)

        return panel

    def _build_stat_row(self, label, value):
        row = QFrame()
        row.setObjectName("statRow")

        layout = QHBoxLayout(row)
        layout.setContentsMargins(14, 12, 14, 12)

        name = QLabel(label)
        name.setObjectName("statLabel")

        result = QLabel(value)
        result.setObjectName("statValue")
        result.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(name, 1)
        layout.addWidget(result)

        return row, result

    def _build_placeholder_page(self, title, text):
        page = QFrame()
        page.setObjectName("placeholderPage")

        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(10)

        label = QLabel(title)
        label.setObjectName("placeholderTitle")

        description = QLabel(text)
        description.setObjectName("placeholderText")
        description.setWordWrap(True)

        layout.addWidget(label)
        layout.addWidget(description)
        layout.addStretch(1)

        return page

    def _select_page(self, index):
        self.pages.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)

    def _choose_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz zdjecie tarczy",
            "",
            "Obrazy (*.jpg *.jpeg *.png *.bmp *.tiff);;Wszystkie pliki (*)",
        )

        if not file_path:
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.image_preview.setPixmap(QPixmap())
            self.image_preview.setText("Nie udalo sie wczytac wybranego obrazu.")
            self.selected_image_path = None
            self.original_preview_pixmap = None
            self.analyze_button.setEnabled(False)
            return

        self.selected_image_path = file_path
        self.original_preview_pixmap = pixmap
        self.current_session = None
        self.analyze_button.setEnabled(True)
        self.status_label.setText("Zdjecie wczytane")
        self.analysis_note.setText("Zdjecie jest gotowe do analizy.")
        self._reset_statistics()
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

        self.status_label.setText("Analiza...")
        self.analyze_button.setEnabled(False)

        try:
            metadata = self.settings_page.get_metadata()
            session = analyze_image(self.selected_image_path, metadata=metadata)
            overlay = draw_analysis_overlay(session)
            self.current_session = session

            # Update image preview
            self.original_preview_pixmap = self._pixmap_from_bgr_image(overlay)
            self._update_preview_pixmap()

            # Update quick stats on analysis page
            self._update_statistics(session.statistics)

            # Push session to statistics page
            scale = self.settings_page.get_scale_mm_per_px()
            self.statistics_page.set_session(session, scale_mm_per_px=scale)

            # Update settings page with detected radius for scale calibration
            if session.target and session.target.radius:
                self.settings_page.set_detected_radius(session.target.radius)

            self.status_label.setText("Analiza gotowa")
            self.analysis_note.setText(self._build_analysis_summary(session))

        except AnalysisError as error:
            self.status_label.setText("Wymagana kalibracja")
            self.analysis_note.setText("Nie udalo sie automatycznie wykryc tarczy.")
            QMessageBox.warning(self, "Analiza przerwana", str(error))
        except Exception as error:
            self.status_label.setText("Blad analizy")
            self.analysis_note.setText("Analiza nie mogla zostac zakonczona.")
            QMessageBox.critical(self, "Blad analizy", str(error))
        finally:
            self.analyze_button.setEnabled(self.selected_image_path is not None)

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

    def _update_statistics(self, statistics):
        values = {
            "Liczba trafien": str(statistics.get("count", 0)),
            "Sredni promien": self._format_px(statistics.get("mean_radius", 0.0)),
            "Rozrzut maks.": self._format_px(statistics.get("extreme_spread", 0.0)),
            "CEP 50%": self._format_px(statistics.get("cep_50", 0.0)),
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
                f"Wykryto {count} trafien. "
                f"Obszar tarczy: typ nieregularny, "
                f"srodek analizy: ({target.center_x:.0f}, {target.center_y:.0f})."
            )

        return (
            f"Wykryto {count} trafien. "
            f"Srodek tarczy: ({target.center_x:.0f}, {target.center_y:.0f}), "
            f"promien: {target.radius:.0f} px."
        )

    def _format_px(self, value):
        return f"{float(value):.1f} px"

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview_pixmap()

    def _apply_styles(self):
        font = QFont("Arial", 10)
        self.setFont(font)

        self.setStyleSheet("""
            QMainWindow {
                background: #f3f5f7;
            }

            QMenuBar {
                background: #ffffff;
                border-bottom: 1px solid #d9dee5;
                padding: 4px;
            }

            QMenuBar::item {
                padding: 6px 10px;
                border-radius: 4px;
            }

            QMenuBar::item:selected {
                background: #e9eef5;
            }

            #sidebar {
                background: #17202a;
                border-right: 1px solid #101820;
            }

            #appTitle {
                color: #ffffff;
                font-size: 25px;
                font-weight: 700;
            }

            #appSubtitle {
                color: #9fb0c2;
                font-size: 12px;
                text-transform: uppercase;
            }

            /* nav item styles handled in _NavItem class */

            #sidebarFooter {
                color: #90a1b3;
                font-size: 11px;
                line-height: 1.4;
            }

            #content {
                background: #f3f5f7;
            }

            #eyebrow {
                color: #607080;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
            }

            #heading {
                color: #15202b;
                font-size: 25px;
                font-weight: 700;
            }

            #statusBadge {
                color: #24635f;
                background: #dcefed;
                border: 1px solid #bddbd8;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 700;
            }

            #imagePanel,
            #sidePanel,
            #placeholderPage {
                background: #ffffff;
                border: 1px solid #dce2e8;
                border-radius: 8px;
            }

            #sectionTitle {
                color: #1d2935;
                font-size: 16px;
                font-weight: 700;
            }

            #imagePreview {
                color: #6b7886;
                background: #eef2f5;
                border: 1px dashed #b9c4cf;
                border-radius: 8px;
                font-size: 15px;
                padding: 24px;
            }

            #primaryButton,
            #secondaryButton {
                border-radius: 7px;
                padding: 11px 16px;
                font-size: 13px;
                font-weight: 700;
            }

            #primaryButton {
                color: #ffffff;
                background: #2f6f73;
                border: 1px solid #285f63;
            }

            #primaryButton:hover {
                background: #398287;
            }

            #secondaryButton {
                color: #1f3335;
                background: #edf5f5;
                border: 1px solid #c5dada;
            }

            #secondaryButton:hover {
                background: #dfeeee;
            }

            #secondaryButton:disabled {
                color: #8d9aa6;
                background: #f0f3f5;
                border: 1px solid #d8dee4;
            }

            #statRow {
                background: #f8fafb;
                border: 1px solid #e1e6eb;
                border-radius: 7px;
            }

            #statLabel {
                color: #657282;
                font-size: 12px;
            }

            #statValue {
                color: #1d2935;
                font-size: 16px;
                font-weight: 700;
            }

            #mutedText,
            #placeholderText {
                color: #657282;
                font-size: 13px;
            }

            #placeholderTitle {
                color: #1d2935;
                font-size: 22px;
                font-weight: 700;
            }

            QScrollArea {
                background: transparent;
                border: none;
            }
        """)