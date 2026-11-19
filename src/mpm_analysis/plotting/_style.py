"""
_style.py
---------
Shared plotting style constants and helpers.

The style values (DPI, font size, line and marker widths, column widths,
colours) are stored in ``mpm_analysis/config/plotting_params.json`` and read
here, once, into the constants below.  Change a value in the JSON, not here.

All plotting functions in this package import their constants from this
module; none of them types a value that is in the JSON.
"""
from __future__ import annotations

import decimal

from mpm_analysis.config import load_config

_PARAMS = load_config("plotting_params")


# ---------------------------------------------------------------------------
# Figure dimensions
# ---------------------------------------------------------------------------

def mm_to_inch(mm: float) -> float:
    return mm / 25.4


ONE_COLUMN_WIDTH = mm_to_inch(_PARAMS["one_column_width_mm"])   # 88 mm — Nature one-column
TWO_COLUMN_WIDTH = mm_to_inch(_PARAMS["two_column_width_mm"])   # 180 mm — Nature two-column
DPI = int(_PARAMS["dpi"])
LABELS_FONT_SIZE = int(_PARAMS["labels_font_size"])
MARKER_SIZE = float(_PARAMS["marker_size"])
LINE_WIDTH = float(_PARAMS["line_width"])
AXES_ARROW_WIDTH = float(_PARAMS["axes_arrow_width"])
LAMBDA_OBS_COLOR = str(_PARAMS["lambda_obs_color"])
SURVIVAL_PROBABILITY_LABEL = str(_PARAMS["survival_probability_label"])

# Fit line style (shared across all panels)
FIT_LINEWIDTH = LINE_WIDTH * 0.5
FIT_LINESTYLE = (0, (7, 2))  # long dash, short gap

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

PURPLE_MAGENTA = str(_PARAMS["colors"]["purple_magenta"])
TEAL           = str(_PARAMS["colors"]["teal"])
MUSTARD        = str(_PARAMS["colors"]["mustard"])
AZURE_BLUE     = str(_PARAMS["colors"]["azure_blue"])
LAVENDER_BLUE  = str(_PARAMS["colors"]["lavender_blue"])

# Default linecut colors (for panel A)
LINECUT_COLORS = [PURPLE_MAGENTA, TEAL, MUSTARD]


# ---------------------------------------------------------------------------
# Quoted values
# ---------------------------------------------------------------------------

def round_half_up(value: float, decimals: int) -> float:
    """``value`` rounded to ``decimals`` digits as a person quotes it.

    Half up on the decimal digits (1.0845 → 1.085, 1.0855 → 1.09), not
    Python's ``round()`` on the binary float (1.0845 → 1.084).  The critical
    lines of the paper figures are drawn at this value.
    """
    return float(decimal.Decimal(f"{float(value):.10f}").quantize(
        decimal.Decimal(1).scaleb(-decimals), rounding=decimal.ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def make_ax_lines_wider(ax, *, labelsize: int = LABELS_FONT_SIZE, linewidth: float = LINE_WIDTH) -> None:
    """Apply consistent tick and spine widths to a matplotlib Axes."""
    ax.tick_params(axis="both", which="both", labelsize=labelsize, width=linewidth)
    for sp in ax.spines.values():
        sp.set_linewidth(linewidth)


def panel_letter_text(letter: str) -> str:
    """The panel letter as text for the *active* renderer.

    Under ``text.usetex`` a bold sans letter needs LaTeX markup — matplotlib's
    ``fontweight="bold"`` is ignored by the LaTeX text pipeline.
    """
    import matplotlib.pyplot as plt

    if plt.rcParams.get("text.usetex", False):
        return rf"\textsf{{\textbf{{{letter}}}}}"
    return letter


def add_panel_letter(ax, letter: str, *, fontsize: float | None = None,
                     x: float = 0.06, y: float = 1.0,
                     dx_mm: float = -4.0, dy_mm: float = 4.0,
                     offset_mm: bool = True) -> None:
    """Place a bold panel letter near an axes' top-left corner.

    Two placements, both in axes coordinates for ``x``/``y``:

    * ``offset_mm=True`` (default) — the main-text convention: additionally
      nudged from the corner in *millimetres* via a ScaledTranslation, so it
      sits identically regardless of axes size or dpi.
    * ``offset_mm=False`` — plain axes-fraction placement, used by the
      appendix figures (``x=-0.2, y=1.05``: clear of the y-axis label).

    Either way the letter renders through :func:`panel_letter_text`, so the
    typography follows the active renderer.
    """
    transform = ax.transAxes
    if offset_mm:
        from matplotlib.transforms import ScaledTranslation

        transform = transform + ScaledTranslation(
            xt=mm_to_inch(dx_mm),
            yt=mm_to_inch(dy_mm),
            scale_trans=ax.figure.dpi_scale_trans,
        )
    ax.text(
        x=x, y=y, s=panel_letter_text(letter),
        transform=transform,
        fontsize=LABELS_FONT_SIZE if fontsize is None else fontsize,
        fontweight="bold",
        ha="left", va="top",
    )


def latex_available() -> bool:
    """Whether a LaTeX toolchain usable by matplotlib's usetex is installed.

    ``text.usetex`` needs ``latex`` and ``dvipng`` on PATH; without them every
    draw raises.  Cached so the check runs once per process.
    """
    global _LATEX_OK
    if _LATEX_OK is None:
        from shutil import which
        _LATEX_OK = bool(which("latex") and which("dvipng"))
    return _LATEX_OK


_LATEX_OK: bool | None = None


def apply_paper_style(fig=None, axes=None, *, usetex: bool | None = None) -> None:
    """Apply paper-ready rc defaults to the current figure.

    Typography follows the published convention of
    ``ZenoPlotting/base.py`` — LaTeX text rendering with the Computer Modern
    math fontset on a sans-serif family — so figures produced here match the
    paper's other panels.  ``usetex=None`` (default) enables it whenever a
    LaTeX toolchain is installed and silently falls back to mathtext with the
    same fontset otherwise, which keeps headless machines working.

    Call once before plotting, or pass ``fig``/``axes`` directly.
    """
    import matplotlib.pyplot as plt

    if usetex is None:
        usetex = latex_available()

    plt.rcParams.update({
        # --- typography: match ZenoPlotting/base.py ---
        "text.usetex": bool(usetex),
        "font.family": "sans-serif",
        "mathtext.fontset": "cm",
        "font.size": LABELS_FONT_SIZE,
        # --- line/marker weights ---
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
