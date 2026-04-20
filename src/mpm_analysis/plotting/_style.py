"""
_style.py
---------
Shared plotting style constants and helpers.

All plotting functions in this package should import their constants from here.
"""
from __future__ import annotations

import numpy as np

_HAS_ZENO_PLOTTING = False
_PFS = None
_TTFC = None

def _mm_to_inch(mm: float) -> float:
    return mm / 25.4


# ---------------------------------------------------------------------------
# Figure dimensions
# ---------------------------------------------------------------------------

def mm_to_inch(mm: float) -> float:
    return _mm_to_inch(mm)


if _HAS_ZENO_PLOTTING:
    ONE_COLUMN_WIDTH = float(_PFS.ONE_COLUMN_FIGURE_WIDTH)
    TWO_COLUMN_WIDTH = float(_PFS.TWO_COLUMNS_FIGURE_WIDTH)
    DPI = int(_PFS.DPI)
    LABELS_FONT_SIZE = int(_PFS.LABELS_FONT_SIZE)
    MARKER_SIZE = float(_PFS.MARKER_SIZE)
    LINE_WIDTH = float(_PFS.LINE_WIDTH)
    AXES_ARROW_WIDTH = float(_PFS.AXES_ARROW_WIDTH)
    LAMBDA_OBS_COLOR = str(_PFS.LAMBDA_OBS_COLOR)
    SURVIVAL_PROBABILITY_LABEL = str(_PFS.SURVIVAL_PROBABILITY_LABEL)
else:
    ONE_COLUMN_WIDTH = mm_to_inch(88)   # 88 mm — Nature one-column
    TWO_COLUMN_WIDTH = mm_to_inch(180)  # 180 mm — Nature two-column
    DPI = 600
    LABELS_FONT_SIZE = 10
    MARKER_SIZE = 1.5
    LINE_WIDTH = 1.5
    AXES_ARROW_WIDTH = 0.03
    LAMBDA_OBS_COLOR = "#d02500"
    SURVIVAL_PROBABILITY_LABEL = r"$P_{|1\rangle}^{\mathrm{ens}}$"

# Fit line style (shared across all panels)
FIT_LINEWIDTH = LINE_WIDTH * 0.5
FIT_LINESTYLE = (0, (7, 2))  # long dash, short gap

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

if _HAS_ZENO_PLOTTING:
    PURPLE_MAGENTA = str(_TTFC.PURPLE_MAGENTA)
    TEAL           = str(_TTFC.TEAL)
    MUSTARD        = str(_TTFC.MUSTARD)
    AZURE_BLUE     = str(_TTFC.AZURE_BLUE)
    LAVENDER_BLUE  = str(_TTFC.LAVENDER_BLUE)
else:
    PURPLE_MAGENTA = "#A13D97"
    TEAL           = "#1ABC9C"
    MUSTARD        = "#D4AC0D"
    AZURE_BLUE     = "#3498DB"
    LAVENDER_BLUE  = "#8787de"

# Default linecut colors (for panel A)
LINECUT_COLORS = [PURPLE_MAGENTA, TEAL, MUSTARD]

# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def make_ax_lines_wider(ax, *, labelsize: int = LABELS_FONT_SIZE, linewidth: float = LINE_WIDTH) -> None:
    """Apply consistent tick and spine widths to a matplotlib Axes."""
    ax.tick_params(axis="both", which="both", labelsize=labelsize, width=linewidth)
    for sp in ax.spines.values():
        sp.set_linewidth(linewidth)


def apply_paper_style(fig=None, axes=None) -> None:
    """Apply paper-ready rc defaults to the current figure.

    Call once before plotting, or pass ``fig``/``axes`` directly.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": False,
        "font.size": LABELS_FONT_SIZE,
        "axes.linewidth": LINE_WIDTH,
        "xtick.major.width": LINE_WIDTH,
        "ytick.major.width": LINE_WIDTH,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "savefig.dpi": DPI,
        "figure.dpi": 100,
    })

    if axes is not None:
        axes_list = [axes] if hasattr(axes, "tick_params") else list(axes)
        for ax in axes_list:
            make_ax_lines_wider(ax)
