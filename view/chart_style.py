import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from PyQt6.QtWidgets import QSizePolicy

from view import theme

# ---------------------------------------------------------------------------
# Chart colour tokens - all taken from view/theme.py (single source)
# ---------------------------------------------------------------------------
BG        = theme.SURFACE
AXIS      = theme.CHART_AXIS
GRID      = theme.CHART_GRID
TEXT      = theme.TEXT
MUTED     = theme.TEXT_MUTED
PRIMARY   = theme.ACCENT
SECONDARY = theme.CHART_SECONDARY
AIM       = theme.CHART_AIM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": theme.FONT_FAMILIES + ["DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})


class CommaFormatter(ScalarFormatter):
    """Default tick labels with a Polish decimal comma (0.5 -> 0,5)."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def base_fig(rows: int = 1, cols: int = 1, h: float = 3.6):
    """Return a Figure + axes array styled to match the app palette."""
    # A plain Figure (not plt.subplots): pyplot would keep every figure alive
    # in its global registry, leaking memory on each chart rebuild.
    fig = Figure(figsize=(5.2 * cols, h))
    axes = fig.subplots(rows, cols)
    fig.patch.set_facecolor(BG)
    for ax in (np.array(axes).flat if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_edgecolor(AXIS)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=0.6, linestyle="-")
        ax.xaxis.set_major_formatter(CommaFormatter())
        ax.yaxis.set_major_formatter(CommaFormatter())
    fig.tight_layout(pad=1.6)
    return fig, axes


class Canvas(FigureCanvas):
    def __init__(self, fig: Figure):
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(260)
