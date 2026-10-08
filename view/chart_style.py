import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QSizePolicy

# ---------------------------------------------------------------------------
# Shared colour tokens - must match main_window.py stylesheet
# ---------------------------------------------------------------------------
BG       = "#ffffff"
CARD_BG  = "#f8fafb"
BORDER   = "#e1e6eb"
TEAL     = "#2f6f73"
TEAL2    = "#3f8c91"
ACCENT   = "#e05c2a"
TEXT     = "#1d2935"
MUTED    = "#657282"
GRID     = "#e8edf2"


def base_fig(rows: int = 1, cols: int = 1, h: float = 3.6):
    """Return a Figure + axes array styled to match the app palette."""
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, h))
    fig.patch.set_facecolor(BG)
    for ax in (np.array(axes).flat if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=GRID, linewidth=0.6, linestyle="--")
    fig.tight_layout(pad=1.6)
    return fig, axes


class Canvas(FigureCanvas):
    def __init__(self, fig: Figure):
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(260)
