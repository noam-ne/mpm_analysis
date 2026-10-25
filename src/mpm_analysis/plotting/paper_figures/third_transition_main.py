"""
third_transition_main.py
------------------------
Main-text figure for the third transition.

Panel A: P_1 probability linecuts at 3 selected λ values.
Panel B: decay rate (left axis) + oscillation frequency (right axis) vs λ.
Partially uses the stale naming convention of survival for P_1 in the code. Figures all follow the new naming.

"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def make_third_transition_figure(
    panel_b_result,
    panel_a_records,
    *,
    fit_dir: Path | None = None,
    panel_a_fit_result=None,
    panel_a_observable: str = "survival",
    target_lambdas: np.ndarray | None = None,
    lambda_max_b: float | None = 1.55,
    downsample_a: int = 2,
    downsample_b: int = 2,
    keep_all_within_b: float | None = None,
    keep_first_b: int = 0,
    transition_label: str | None = "Quantum\nZeno",
    save_dir: Path | None = None,
    figure_stem: str = "third_transition_main_text_figure",
    yticks_freq: list | None = None,
    ylim_decay_b: tuple[float, float] | None = None,
) -> "plt.Figure":
    """Build the two-panel main-text figure.

    Parameters
    ----------
    panel_b_result:
        ``ZenoAnalysisResult`` for panel B (decay / frequency vs λ).
    panel_a_records:
        ``list[ObservableRecord]`` for panel A (p1 linecuts).
    fit_dir:
        If given, overlay fit curves in panel A from this directory.
    panel_a_fit_result:
        ``ZenoAnalysisResult`` whose poles draw the panel-A overlays instead
        of ``fit_dir`` (pass ``panel_b_result`` for the analysis's own fit).
    panel_a_observable:
        Observable key of the panel-A records (``"survival"`` and ``"D"`` are
        both ``D_probability`` for the ensemble-average loader).
    target_lambdas:
        Lambda values to select for panel A linecuts.
        Default: ``[0.4, 1.1, 1.5]``.
    lambda_max_b:
        Upper λ cutoff for panel B.
    downsample_a, downsample_b:
        Plot every n-th time point of the panel-A traces / every n-th λ
        point of panel B (1 → all).
    keep_all_within_b:
        Panel B: λ half-width around the transition line (λ_c) inside which
        every point is plotted; ``downsample_b`` thins only the rest.
        ``None`` → ``downsample_b`` everywhere.
    keep_first_b:
        Panel B: the first ``keep_first_b`` λ points are always plotted.
    transition_label:
        Text written right of the panel-B transition line (λ_c of the sqrt
        fit) behind a short arrow, near the top; ``None`` → no label.
    save_dir:
        If given, save SVG + PDF to this directory.
    figure_stem:
        Filename stem (no extension).
    yticks_freq:
        Y-tick positions for the right frequency axis in panel B.
    ylim_decay_b:
        Fixed (bottom, top) of panel B's decay-rate axis (error bars beyond
        it are clipped); ``None`` → autoscale.

    Returns
    -------
    plt.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    from mpm_analysis.plotting._style import (
        ONE_COLUMN_WIDTH, LINECUT_COLORS, LABELS_FONT_SIZE, LINE_WIDTH,
        LAMBDA_OBS_COLOR, mm_to_inch, apply_paper_style,
    )
    from mpm_analysis.plotting.survival_curves import plot_survival_curves_panel
    from mpm_analysis.plotting.pole_spectrum import plot_decay_and_freq_panel

    if target_lambdas is None:
        target_lambdas = np.array([0.4, 1.1, 1.5])

    apply_paper_style()

    fig_w = float(ONE_COLUMN_WIDTH)
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_w * 1.0),
        layout="constrained",
    )

    # ---- Panel A ----
    _, plotted_lambdas = plot_survival_curves_panel(
        panel_a_records,
        ax=ax_a,
        target_lambdas=target_lambdas,
        colors=LINECUT_COLORS,
        fit_dir=fit_dir,
        fit_result=panel_a_fit_result,
        downsample=downsample_a,
        observable_key=panel_a_observable,
    )

    # ---- Panel B ----
    ax_b_decay, ax_b_freq = plot_decay_and_freq_panel(
        panel_b_result,
        ax=ax_b,
        lambda_max=lambda_max_b,
        pole_selection=1,                 # slot 1 = "Pole 2" (orange in the spectrum plot)
                                          #   on both sides of the EP, with its own error bars
        downsample=downsample_b,
        keep_all_within=keep_all_within_b,
        keep_first=keep_first_b,
        show_sqrt_fit=True,
        show_critical_line=True,
        yticks_freq=yticks_freq or [0.0, 0.4, 0.8],
        ylim_decay=ylim_decay_b,
    )

    # ---- "→ Quantum Zeno" beside the transition line ----
    sqrt_fit = panel_b_result.critical_point_sqrt_fit
    if transition_label and sqrt_fit is not None:
        _add_transition_label(
            ax_b, sqrt_fit.lambda_c, transition_label,
            LAMBDA_OBS_COLOR, LABELS_FONT_SIZE, LINE_WIDTH,
        )

    # ---- Lambda arrows below panel B ----
    _add_lambda_arrows(ax_b, plotted_lambdas, LINECUT_COLORS, LINE_WIDTH)

    # ---- Panel labels ----
    _add_panel_letter(ax_a, "a", mm_to_inch, LABELS_FONT_SIZE)
    _add_panel_letter(ax_b, "b", mm_to_inch, LABELS_FONT_SIZE)

    # ---- Save ----
    if save_dir is not None:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        save_figure_pdf_svg(fig, figure_stem, save_dir)

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_lambda_arrows(ax, lambdas: list[float], colors: list, arrow_lw: float) -> None:
    """Add coloured upward arrows below the x-axis at each plotted lambda."""
    ymin, ymax = ax.get_ylim()
    y_arrow = ymin - 0.04 * (ymax - ymin)
    for lam, color in zip(lambdas, colors):
        ax.annotate(
            "",
            xy=(lam, ymin),
            xytext=(lam, y_arrow),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=arrow_lw),
            zorder=20,
            clip_on=False,
        )


def _add_transition_label(ax, lam: float, label: str, color, fontsize: float, lw: float) -> None:
    """Arrow from the transition line at *lam* to *label*, near the top of *ax*."""
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    y, arrow_pt = 0.86, 10.0          # height (axes fraction), arrow length (pt)
    ax.annotate(
        "", xy=(lam, y), xycoords=trans,
        xytext=(arrow_pt, 0.0), textcoords="offset points",
        arrowprops=dict(arrowstyle="<-", color=color, lw=lw * 0.7,
                        shrinkA=0, shrinkB=0),
    )
    # va="top" raised by ~0.6 em puts the FIRST line's centre on the arrow.
    ax.annotate(
        label, xy=(lam, y), xycoords=trans,
        xytext=(arrow_pt + 1.5, 0.6 * fontsize), textcoords="offset points",
        ha="left", va="top", color=color, fontsize=fontsize, linespacing=1.0,
    )


def _add_panel_letter(ax, letter: str, mm_to_inch_fn, fontsize: int) -> None:
    """Thin wrapper kept for this module's call sites.

    The placement/typography convention lives in
    :func:`mpm_analysis.plotting._style.add_panel_letter`, shared with the
    appendix figures so every paper panel letter renders identically.
    """
    from mpm_analysis.plotting._style import add_panel_letter

    add_panel_letter(ax, letter, fontsize=fontsize)
