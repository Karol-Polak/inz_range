"""
Single source of visual design tokens for the whole view layer.

Every colour, font size and spacing value used by the Qt stylesheet and by
the matplotlib charts (view/chart_style.py) is defined here, so the two can
never drift apart. Widgets only set object names / dynamic properties;
all styling comes from build_stylesheet().
"""

from string import Template

from PyQt6.QtGui import QColor, QFont, QPalette

# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------
BG_APP        = "#f4f5f7"   # window background
SURFACE       = "#ffffff"   # panels, inputs
SURFACE_ALT   = "#eef0f3"   # sidebar, image well, disabled fill
SURFACE_HOVER = "#e4e7eb"   # hover on neutral surfaces
BORDER        = "#d5d9df"   # 1 px hairlines between regions
BORDER_INPUT  = "#b9c0c9"   # input / secondary button borders

TEXT          = "#1b1f24"   # body text (16.6:1 on white)
TEXT_MUTED    = "#5a6370"   # labels, hints (>= 5.3:1 on every surface)
TEXT_DISABLED = "#737b86"   # disabled text (3.8:1, still legible)

ACCENT         = "#2b4c7e"  # muted dark blue: primary action, selection, focus
ACCENT_HOVER   = "#233f69"
ACCENT_PRESSED = "#1c3357"
ACCENT_SOFT    = "#e3e9f2"  # selected nav item, checked toggle, list selection
ON_ACCENT      = "#ffffff"

DANGER  = "#b42318"
WARNING = "#9a6700"

# Chart-only tokens
CHART_AXIS      = "#c3c8cf"
CHART_GRID      = "#e6e8eb"
CHART_SECONDARY = "#b4532a"  # second data series (rust)
CHART_AIM       = "#d98a00"  # aim point, matches the orange marker on the image overlay

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_FAMILIES = ["Segoe UI", "Helvetica Neue", "Inter", "Arial"]

FS_SMALL = 12   # captions, field labels, hints
FS_BODY  = 13   # body, inputs, buttons, section titles
FS_TITLE = 18   # page title, KPI values
FW_BOLD  = 600

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
SPACE_XS, SPACE_S, SPACE_M, SPACE_L, SPACE_XL = 4, 8, 12, 16, 24
RADIUS_CONTROL = 4
RADIUS_PANEL   = 6
CONTROL_HEIGHT = 30
SIDEBAR_WIDTH  = 220


def app_font() -> QFont:
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPixelSize(FS_BODY)
    return font


def light_palette() -> QPalette:
    """Explicit light palette so the app ignores the OS dark mode."""
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: BG_APP,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: SURFACE,
        QPalette.ColorRole.AlternateBase: SURFACE_ALT,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.PlaceholderText: TEXT_MUTED,
        QPalette.ColorRole.Button: SURFACE,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.BrightText: ON_ACCENT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: ON_ACCENT,
        QPalette.ColorRole.ToolTipBase: SURFACE,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.Link: ACCENT,
        QPalette.ColorRole.Light: SURFACE,
        QPalette.ColorRole.Midlight: SURFACE_ALT,
        QPalette.ColorRole.Mid: BORDER,
        QPalette.ColorRole.Dark: BORDER_INPUT,
        QPalette.ColorRole.Shadow: BORDER_INPUT,
    }
    for role, colour in roles.items():
        palette.setColor(role, QColor(colour))

    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(disabled, role, QColor(TEXT_DISABLED))
    palette.setColor(disabled, QPalette.ColorRole.Base, QColor(SURFACE_ALT))
    palette.setColor(disabled, QPalette.ColorRole.Button, QColor(SURFACE_ALT))
    return palette


def apply_theme(app) -> None:
    """Fusion style + light palette + app font: identical look on macOS/Windows/Linux."""
    app.setStyle("Fusion")
    app.setPalette(light_palette())
    app.setFont(app_font())


_QSS = Template("""
QWidget {
    color: $TEXT;
    font-size: ${FS_BODY}px;
}

QMainWindow, #content {
    background: $BG_APP;
}

QToolTip {
    color: $TEXT;
    background: $SURFACE;
    border: 1px solid $BORDER;
    padding: 4px 6px;
}

/* ---- menu & status bar ------------------------------------------------ */
QMenuBar {
    background: $SURFACE;
    border-bottom: 1px solid $BORDER;
}
QMenuBar::item {
    padding: 4px 10px;
    background: transparent;
}
QMenuBar::item:selected {
    background: $SURFACE_ALT;
}
QMenu {
    background: $SURFACE;
    border: 1px solid $BORDER;
}
QMenu::item {
    padding: 5px 20px;
}
QMenu::item:selected {
    background: $ACCENT_SOFT;
    color: $TEXT;
}

QStatusBar {
    background: $SURFACE;
    border-top: 1px solid $BORDER;
}
QStatusBar::item {
    border: none;
}
#statusText {
    color: $TEXT_MUTED;
    font-size: ${FS_SMALL}px;
    padding: 2px 8px;
}

/* ---- sidebar ---------------------------------------------------------- */
#sidebar {
    background: $SURFACE_ALT;
    border-right: 1px solid $BORDER;
}
#appTitle {
    font-weight: $FW_BOLD;
}
#appSubtitle, #sidebarFooter {
    color: $TEXT_MUTED;
    font-size: ${FS_SMALL}px;
}
#navItem {
    background: transparent;
    border: none;
    border-left: 3px solid transparent;
}
#navItem:hover {
    background: $SURFACE_HOVER;
}
#navItem[selected="true"] {
    background: $ACCENT_SOFT;
    border-left: 3px solid $ACCENT;
}
#navTitle {
    font-weight: $FW_BOLD;
    background: transparent;
}
#navItem[selected="true"] #navTitle {
    color: $ACCENT;
}
#navHint {
    color: $TEXT_MUTED;
    font-size: ${FS_SMALL}px;
    background: transparent;
}

/* ---- page header & text roles ---------------------------------------- */
#pageTitle {
    font-size: ${FS_TITLE}px;
    font-weight: $FW_BOLD;
}
#pageHint, #mutedText, #statLabel {
    color: $TEXT_MUTED;
}
#statLabel {
    font-size: ${FS_SMALL}px;
}
#sectionTitle {
    font-weight: $FW_BOLD;
}
#statValue {
    font-weight: $FW_BOLD;
}
#kpiValue {
    font-size: ${FS_TITLE}px;
    font-weight: $FW_BOLD;
}
#warningText {
    color: $WARNING;
}

/* ---- surfaces --------------------------------------------------------- */
#panel {
    background: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: ${RADIUS_PANEL}px;
}
#statRow {
    background: transparent;
    border: none;
    border-bottom: 1px solid $BORDER;
}
#separator {
    background: $BORDER;
    border: none;
    max-height: 1px;
    min-height: 1px;
}
#imagePreview {
    color: $TEXT_MUTED;
    background: $SURFACE_ALT;
    border: 1px solid $BORDER;
    border-radius: ${RADIUS_CONTROL}px;
}
QScrollArea {
    background: transparent;
    border: none;
}

/* ---- buttons ---------------------------------------------------------- */
QPushButton {
    color: $TEXT;
    background: $SURFACE;
    border: 1px solid $BORDER_INPUT;
    border-radius: ${RADIUS_CONTROL}px;
    min-height: ${CONTROL_INNER}px;
    padding: 0px 14px;
}
QPushButton:hover {
    background: $SURFACE_ALT;
}
QPushButton:pressed {
    background: $SURFACE_HOVER;
}
QPushButton:focus {
    border-color: $ACCENT;
}
QPushButton:checked {
    color: $ACCENT;
    background: $ACCENT_SOFT;
    border-color: $ACCENT;
}
#primaryButton {
    color: $ON_ACCENT;
    background: $ACCENT;
    border-color: $ACCENT;
    font-weight: $FW_BOLD;
}
#primaryButton:hover {
    background: $ACCENT_HOVER;
    border-color: $ACCENT_HOVER;
}
#primaryButton:pressed {
    background: $ACCENT_PRESSED;
    border-color: $ACCENT_PRESSED;
}
#primaryButton:focus {
    border-color: $ACCENT_PRESSED;
}
QPushButton:disabled, #primaryButton:disabled {
    color: $TEXT_DISABLED;
    background: $SURFACE_ALT;
    border-color: $BORDER;
    font-weight: normal;
}

/* segmented control (unit selector) */
#segment {
    min-width: 34px;
    padding: 0px 10px;
    border-radius: 0px;
}
#segment[position="first"] {
    border-top-left-radius: ${RADIUS_CONTROL}px;
    border-bottom-left-radius: ${RADIUS_CONTROL}px;
}
#segment[position="middle"], #segment[position="last"] {
    border-left: none;
}
#segment[position="last"] {
    border-top-right-radius: ${RADIUS_CONTROL}px;
    border-bottom-right-radius: ${RADIUS_CONTROL}px;
}
#segment:checked {
    font-weight: $FW_BOLD;
}

/* link-style button (navigation inside text) */
#linkButton {
    color: $ACCENT;
    background: transparent;
    border: none;
    min-height: 0px;
    padding: 0px;
    text-align: left;
}
#linkButton:hover {
    color: $ACCENT_HOVER;
    text-decoration: underline;
}

/* ---- inputs ----------------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    color: $TEXT;
    background: $SURFACE;
    border: 1px solid $BORDER_INPUT;
    border-radius: ${RADIUS_CONTROL}px;
    min-height: ${CONTROL_INNER}px;
    padding: 0px 8px;
    selection-background-color: $ACCENT;
    selection-color: $ON_ACCENT;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: $ACCENT;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    color: $TEXT_DISABLED;
    background: $SURFACE_ALT;
    border-color: $BORDER;
}

QCheckBox {
    spacing: 8px;
    font-weight: $FW_BOLD;
}
QCheckBox:disabled {
    color: $TEXT_DISABLED;
}

QListWidget {
    background: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: ${RADIUS_CONTROL}px;
    outline: none;
}
QListWidget:focus {
    border-color: $ACCENT;
}
QListWidget::item {
    color: $TEXT;
    padding: 7px 8px;
    border-bottom: 1px solid $SURFACE_ALT;
}
QListWidget::item:hover {
    background: $SURFACE_ALT;
}
QListWidget::item:selected {
    color: $TEXT;
    background: $ACCENT_SOFT;
}
""")


def build_stylesheet() -> str:
    tokens = {name: value for name, value in globals().items() if name.isupper()}
    tokens["CONTROL_INNER"] = CONTROL_HEIGHT - 2  # minus the 1 px top/bottom border
    return _QSS.substitute(tokens)
