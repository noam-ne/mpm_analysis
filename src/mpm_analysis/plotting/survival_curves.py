"""
survival_curves.py
------------------
Panel-A style: survival probability linecuts with optional fit overlays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_survival_curves_panel(
    records,
    *,
    ax=None,
    target_lambdas: np.ndarray | None = None,
    colors: list | None = None,
    fit_dir: Path | None = None,
    downsample: int = 1,
    markersize: float | None = None,
    linewidth: float | None = None,
    label_fmt: str = r"$\lambda$={:.1f}",
    show_legend: bool = True,
    observable_key: str = "survival",
) -> tuple["plt.Axes", list[float]]:
    """Plot selected lambda linecuts as panel A.

    Parameters
    ----------
    records:
        ``list[ObservableRecord]``.
    ax:
        Target axes.  Created if ``None``.
    target_lambdas:
        Lambda values to plot.  Default: 3 evenly spaced values.
    colors:
        One colour per target lambda.  Defaults to ``LINECUT_COLORS``.
    fit_dir:
        If given, overlay per-lambda fit curves from this directory (matched
        by ``<N>_kHz`` in filename).
    downsample:
        Plot every ``downsample``-th time point.
    markersize, linewidth:
        Override style defaults.
    show_legend:
        Whether to add a legend.
    observable_key:
        Which observable to plot.

    Returns
    -------
    (ax, plotted_lambdas)
        The axes and the list of actual λ values plotted.
    """
    import matplotlib.pyplot as plt
    from mpm_analysis.plotting._style import (
        LINECUT_COLORS, LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH,
        FIT_LINESTYLE, FIT_LINEWIDTH, SURVIVAL_PROBABILITY_LABEL,
        make_ax_lines_wider,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    ms = markersize if markersize is not None else MARKER_SIZE * 1.5
    lw = linewidth if linewidth is not None else LINE_WIDTH / 2.0

    slices = sorted(
        [r for r in records if r.observable_key == observable_key],
        key=lambda r: r.lambda_val,
    )
    if not slices:
        ax.set_xlabel(r"Time ($\mu$s)", fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel(SURVIVAL_PROBABILITY_LABEL, fontsize=LABELS_FONT_SIZE)
        return ax, []

    all_lambdas = np.array([r.lambda_val for r in slices])

    if target_lambdas is None:
        # Pick 3 evenly spaced
        indices = np.round(np.linspace(0, len(slices) - 1, 3)).astype(int)
        target_lambdas = all_lambdas[indices]

    if colors is None:
        colors = LINECUT_COLORS

    # Build fit index if fit_dir provided
    fit_index = _build_fit_index(fit_dir) if fit_dir is not None else {}

    plotted_lambdas = []
    for lam_target, color in zip(target_lambdas, colors):
        idx = int(np.argmin(np.abs(all_lambdas - lam_target)))
        rec = slices[idx]
        plotted_lambdas.append(rec.lambda_val)

        t_plot = rec.t[::downsample]
        s_plot = rec.signal[::downsample]

        ax.errorbar(
            t_plot, s_plot,
            yerr=None,
            color=color,
            marker="o",
            markersize=ms,
            ls="none",
            linewidth=lw,
            label=label_fmt.format(rec.lambda_val),
            zorder=10,
        )

        # Overlay fit curve
        if fit_index:
            fit_jd = _load_fit_for_record(rec, fit_index)
            if fit_jd is not None:
                try:
                    t_fit, y_fit = _extract_fit_arrays(fit_jd)
                    ax.plot(
                        t_fit, y_fit,
                        color=color,
                        ls=FIT_LINESTYLE,
                        linewidth=FIT_LINEWIDTH,
                        alpha=0.95,
                        zorder=25,
                    )
                except Exception:
                    pass

    ax.set_ylabel(SURVIVAL_PROBABILITY_LABEL, fontsize=LABELS_FONT_SIZE)
    ax.set_xlabel(r"Time ($\mu$s)", fontsize=LABELS_FONT_SIZE, labelpad=1.0)
    if show_legend:
        ax.legend(fontsize=LABELS_FONT_SIZE)
    ax.grid(False)
    make_ax_lines_wider(ax)

    return ax, plotted_lambdas


# ---------------------------------------------------------------------------
# Fit-loading helpers (re-implemented without polluting the public namespace)
# ---------------------------------------------------------------------------

import re as _re
_KHZ_RE = _re.compile(r"OBG0_(\d+)_kHz|(\d+)_kHz", _re.IGNORECASE)


def _khz_from_name(fp: Path) -> int:
    m = _KHZ_RE.search(fp.name)
    if not m:
        raise ValueError(fp.name)
    return int(m.group(1) or m.group(2))


def _build_fit_index(fit_dir: Path) -> dict[int, Path]:
    idx: dict[int, Path] = {}
    if not fit_dir or not Path(fit_dir).exists():
        return idx
    for fp in Path(fit_dir).glob("*.json"):
        try:
            idx[_khz_from_name(fp)] = fp
        except ValueError:
            continue
    return idx


def _load_fit_for_record(rec, fit_index: dict) -> dict | None:
    import json
    from mpm_analysis.utils.windows_paths import win_long_path
    try:
        khz = _khz_from_name(Path(rec.source_file))
    except ValueError:
        return None
    fp = fit_index.get(khz)
    if fp is None:
        return None
    with open(win_long_path(fp), "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_fit_arrays(fit_jd: dict) -> tuple[np.ndarray, np.ndarray]:
    t = (
        fit_jd.get("time smooth")
        or fit_jd.get("time_smooth")
        or fit_jd.get("times_smooth")
        or fit_jd.get("times smooth")
    )
    y = fit_jd.get("fit_curve") or fit_jd.get("fit curve")
    if t is None or y is None:
        raise KeyError("Fit JSON missing 'time smooth' or 'fit_curve'.")
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.nanmax(t) > 1_000.0:
        t = t / 1e3
    return t, y
