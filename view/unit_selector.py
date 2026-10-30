from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget

from services.units import ALL_UNITS, LengthUnit, UnitContext, missing_reason_text


class UnitSelector(QWidget):
    """
    Segmented control [ px | mm | MOA | mrad ]. Only displays and reports
    the choice - availability rules live in services.units.

    With a context set (set_context), units that cannot be computed are
    disabled with a tooltip saying what is missing. Without a context
    (Sesje page) every unit stays enabled; callers fall back per block.
    """

    unit_changed = pyqtSignal(object)  # LengthUnit, emitted on user clicks only

    def __init__(self, unit: LengthUnit = LengthUnit.MM, parent=None):
        super().__init__(parent)
        self._buttons = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        last = len(ALL_UNITS) - 1
        for index, option in enumerate(ALL_UNITS):
            button = QPushButton(option.value)
            button.setObjectName("segment")
            button.setProperty(
                "position", "first" if index == 0 else "last" if index == last else "middle"
            )
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked, u=option: self._on_clicked(u))
            self._group.addButton(button)
            self._buttons[option] = button
            layout.addWidget(button)

        self.set_unit(unit)

    def unit(self) -> LengthUnit:
        for option, button in self._buttons.items():
            if button.isChecked():
                return option
        return LengthUnit.PX

    def set_unit(self, unit: LengthUnit) -> None:
        """Programmatic change - does not emit unit_changed."""
        self._buttons[LengthUnit(unit)].setChecked(True)

    def set_context(self, ctx: Optional[UnitContext]) -> None:
        for option, button in self._buttons.items():
            reason = missing_reason_text(option, ctx) if ctx is not None else None
            button.setEnabled(reason is None)
            button.setToolTip(reason or "")

    def _on_clicked(self, unit: LengthUnit) -> None:
        self.unit_changed.emit(unit)
