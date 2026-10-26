"""
ksi_plot.py
-----------
Plotting helpers for the two transitions read off the pole strands:

* ``plot_ksi_curve``          — ξ(λ) = (2·Re(e3) − Re(e2))/(Re(e2) − Re(e3)).
  Its zero is the **state-freezing point** λ(ξ=0), *not* the exceptional point.
* ``plot_discriminant_curve`` — Δ(λ) = c1² − 4c0 of the pole pair.  Its zero is
  the **exceptional point** λ_EP.
* ``plot_discriminant_window_sweep`` — Δ(λ) and its line fit for several fit
  windows on one figure, λ_EP ± error per window in the legend.

Both use the paper style of the sqrt-fit figure: measured points with error
bars, the fit as a black dashed line, the transition as a vertical dashed line
with its bootstrap band.
"""
from __future__ import annotations

import numpy as np


def _paper_axes(ax, figsize=(4.4, 3.2)):
    """Create/return an axes with the paper style applied, plus its kwargs."""
    import matplotlib.pyplot as plt
    from mpm_analysis.plotting._style import (
        LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH, apply_paper_style,
    )

    apply_paper_style()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    err_kws = dict(
        elinewidth=LINE_WIDTH * 0.6,
        capsize=MARKER_SIZE * 1.2,
        capthick=LINE_WIDTH * 0.6,
        zorder=10,
    )
    return ax, err_kws, LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH


def _asym_yerr(y, lower, upper):
    """(2, n) asymmetric error bars from 16/84 curves around the plotted y."""
    if lower is None or upper is None:
        return None
    y = np.asarray(y, dtype=float)
    lo = np.clip(y - np.asarray(lower, dtype=float), 0, None)
    hi = np.clip(np.asarray(upper, dtype=float) - y, 0, None)
    return np.vstack([lo, hi])


def plot_ksi_curve(
    ksi_result,
    *,
    ax=None,
    color: str = "steelblue",
    crossing_color: str | None = None,
    show_zero_line: bool = True,
    show_band: bool = False,
    show_crossing_quantiles: bool = True,
    title: str | None = None,
) -> "plt.Axes":
    """Plot ξ(λ) with error bars and the state-freezing crossing λ(ξ=0).

    Parameters
    ----------
    ksi_result:
        A ``KsiCrossingResult`` from ``KsiCrossingPipeline.run()``.
    ax:
        Target axes.  Created if ``None``.
    color:
        Colour of the measured points.
    crossing_color:
        Colour of the vertical crossing line (default: the paper's λ colour).
    show_zero_line:
        Draw the horizontal ξ = 0 line.
    show_band:
        Also shade the 16/84 band (off by default — the error bars carry it).
    show_crossing_quantiles:
        Mark the bootstrap crossing median + 16/84 span when available.
    title:
        Axes title; a default naming the transition is used when ``None``.

    Returns
    -------
    plt.Axes
    """
    from mpm_analysis.plotting._style import LAMBDA_OBS_COLOR

    ax, err_kws, fs, ms, lw = _paper_axes(ax)
    crossing_color = crossing_color or LAMBDA_OBS_COLOR

    lam = np.asarray(ksi_result.lambda_values, dtype=float)
    ksi = np.asarray(ksi_result.ksi_values, dtype=float)
    yerr = _asym_yerr(ksi, ksi_result.ksi_lower, ksi_result.ksi_upper)

    if show_band and ksi_result.ksi_lower is not None:
        ax.fill_between(lam, ksi_result.ksi_lower, ksi_result.ksi_upper,
                        color=color, alpha=0.2, linewidth=0)
    ax.errorbar(lam, ksi, yerr=yerr, color=color, marker="o", linestyle="none",
                markersize=ms, label=r"$\xi(\lambda)$", **err_kws)

    if show_zero_line:
        ax.axhline(0, color="k", ls="--", lw=lw * 0.6, alpha=0.7)

    quantiles = ksi_result.crossing_quantiles
    if show_crossing_quantiles and quantiles is not None:
        q16, q50, q84 = quantiles
        ax.axvspan(q16, q84, color=crossing_color, alpha=0.15, linewidth=0, zorder=0)
        ax.axvline(q50, color=crossing_color, ls="--", lw=lw, alpha=0.8, zorder=0,
                   label=(rf"$\lambda_{{\xi=0}} = {q50:.3f}"
                          rf"^{{+{q84 - q50:.3f}}}_{{-{q50 - q16:.3f}}}$"))
    else:
        for lc in ksi_result.zero_crossings:
            ax.axvline(lc, color=crossing_color, ls="--", lw=lw, alpha=0.8, zorder=0,
                       label=rf"$\lambda_{{\xi=0}} = {lc:.3f}$")

    ax.set_xlabel(r"$\lambda$", fontsize=fs)
    # Pole route: ξ from e2/e3; spiral route (SpiralKsiResult): ξ = γ0/γa − 1
    # from the fitted rates — the result says which through ``ksi_label``.
    if hasattr(ksi_result, "pole_idx1"):
        ylabel = r"$\xi = (2e_3 - \mathrm{Re}\,e_2)\,/\,(\mathrm{Re}\,e_2 - e_3)$"
        default_title = ("State freezing: zero of "
                         rf"$\xi$ (poles {ksi_result.pole_idx1}, {ksi_result.pole_idx2})")
    else:
        ylabel = getattr(ksi_result, "ksi_label", r"$\xi = \gamma_0/\gamma_a - 1$")
        default_title = r"State freezing: zero of $\xi$ (spiral approach rate)"
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_title(title or default_title, fontsize=fs)
    ax.legend(fontsize=fs * 0.8)
    ax.grid(False)
    return ax


def plot_discriminant_curve(
    lambdas,
    delta,
    *,
    fit=None,
    lambda_ep=None,
    lambda_ep_sigma=None,
    first_crossing=None,
    crossing_quantiles=None,
    delta_sigma=None,
    delta_lower=None,
    delta_upper=None,
    ax=None,
    color: str = "steelblue",
    crossing_color: str | None = None,
    title: str | None = None,
) -> "plt.Axes":
    """Plot the EP-pair discriminant with the straight line fitted through zero.

    ``Delta(lambda) = c1**2 - 4*c0`` is negative while the pair is complex and
    positive once it has split into two real poles, and it is linear in λ near
    a square-root exceptional point — so the fitted zero of that line is the EP.

    Parameters
    ----------
    lambdas, delta:
        The measured curve (normally from the central pole set).
    fit:
        The ``dict`` returned by
        :func:`mpm_analysis.analysis.ksi_crossing.linfit_zero_crossing` with
        ``return_fit=True``; its window and fitted line are drawn.
    lambda_ep, first_crossing:
        The reported λ_EP (dashed vertical) and the raw first sign change
        (dotted).  The line's own zero is in its legend entry.
    lambda_ep_sigma:
        1-σ error of λ_EP (run.py: from the line fit's covariance): the
        label reads λ_EP ± σ and λ_EP ± σ is shaded.
    crossing_quantiles:
        ``(16, 50, 84)`` percentiles of per-realization crossings, shaded;
        the λ_EP label gives +(q84 − λ_EP) / −(λ_EP − q16).  Ignored when
        ``lambda_ep_sigma`` is given.
    delta_sigma:
        1-σ error bar of every point — the SAME σ the line fit was weighted
        with (``1/σ²``), drawn symmetric about the point, so the plot shows
        exactly what was fitted.
    delta_lower, delta_upper:
        Legacy alternative to ``delta_sigma``: 16/84 percentiles of the
        per-realization Delta curves, drawn as (asymmetric) bars around the
        points.  Ignored when ``delta_sigma`` is given.
    ax, color, crossing_color, title:
        Cosmetics; ``ax`` is created when ``None``.

    Returns
    -------
    plt.Axes
    """
    from mpm_analysis.plotting._style import LAMBDA_OBS_COLOR

    ax, err_kws, fs, ms, lw = _paper_axes(ax)
    crossing_color = crossing_color or LAMBDA_OBS_COLOR

    lam = np.asarray(lambdas, dtype=float)
    d = np.asarray(delta, dtype=float)
    if delta_sigma is not None:
        yerr = np.asarray(delta_sigma, dtype=float)
    else:
        yerr = _asym_yerr(d, delta_lower, delta_upper)

    ax.axhline(0, color="k", ls="--", lw=lw * 0.6, alpha=0.7)
    fin = np.isfinite(d)
    ax.plot(lam[fin], d[fin], color=color, lw=lw * 0.9, alpha=0.75, zorder=9)
    ax.errorbar(lam, d, yerr=yerr, color=color, marker="o", linestyle="none",
                markersize=ms, label=r"$\Delta = c_1^2 - 4c_0$", **err_kws)

    if fit is not None and fit.get("coef") is not None:
        w = fit["window"]
        xs = np.linspace(lam[w].min(), lam[w].max(), 50)
        if fit.get("n_below") is not None or fit.get("n_above") is not None:
            where = (f"{fit.get('n_below') or 0} pts before + {fit.get('n_above') or 0} "
                     r"after $\Delta$ turns positive")
        elif fit.get("widened"):
            where = rf"{int(w.sum())} nearest pts, $\pm${fit['half_width']:.2f} had fewer"
        else:
            where = rf"{int(w.sum())} pts in $\pm${fit['half_width']:.2f}"
        a, b = fit["coef"][0], fit["coef"][1]
        if a != 0:
            where += f"; zero at {-b / a:.3f}"
        ax.plot(xs, np.polyval(fit["coef"], xs), color="k", linestyle="--",
                linewidth=lw, zorder=11,
                label=("weighted " if fit.get("weighted") else "") + f"line fit ({where})")

    has_sigma = (lambda_ep_sigma is not None and lambda_ep is not None
                 and np.isfinite(lambda_ep) and np.isfinite(lambda_ep_sigma))
    if has_sigma:
        crossing_quantiles = None
        ax.axvspan(lambda_ep - lambda_ep_sigma, lambda_ep + lambda_ep_sigma,
                   color=crossing_color, alpha=0.15, linewidth=0, zorder=0)
    if crossing_quantiles is not None:
        q16, _, q84 = crossing_quantiles
        ax.axvspan(q16, q84, color=crossing_color, alpha=0.15, linewidth=0, zorder=0)
    if lambda_ep is not None and np.isfinite(lambda_ep):
        label = rf"$\lambda_{{EP}} = {lambda_ep:.3f}$"
        if has_sigma:
            label = rf"$\lambda_{{EP}} = {lambda_ep:.3f} \pm {lambda_ep_sigma:.3f}$"
        if crossing_quantiles is not None:
            # errors to the 84th / 16th percentile, about the value shown
            q16, _, q84 = crossing_quantiles
            label = (rf"$\lambda_{{EP}} = {lambda_ep:.3f}"
                     rf"^{{+{q84 - lambda_ep:.3f}}}_{{-{lambda_ep - q16:.3f}}}$")
        ax.axvline(lambda_ep, color=crossing_color, ls="--", lw=lw, alpha=0.8,
                   zorder=0, label=label)
    if first_crossing is not None and np.isfinite(first_crossing):
        ax.axvline(first_crossing, color=crossing_color, ls=":", lw=lw * 0.8,
                   alpha=0.6, zorder=0,
                   label=rf"first crossing $= {first_crossing:.3f}$")

    finite = d[np.isfinite(d)]
    if finite.size:
        span = np.percentile(np.abs(finite), 90)
        ax.set_ylim(-1.4 * span, 1.4 * span)
    ax.set_xlabel(r"$\lambda$", fontsize=fs)
    ax.set_ylabel(r"$\Delta(\lambda)$", fontsize=fs)
    ax.set_title(title or "Exceptional point: zero of the pair discriminant",
                 fontsize=fs)
    ax.legend(fontsize=fs * 0.8)
    ax.grid(False)
    return ax


# High-contrast series for the window sweep, fixed order (never cycled), each
# with its own marker so a window is never told apart by colour alone.
_SWEEP_STYLES = [("#2a78d6", "o"), ("#e34948", "s"), ("#008300", "^"), ("#6250d6", "D"),
                 ("#eb6834", "v"), ("#1baf7a", "P"), ("#d55181", "X"), ("#333333", "*")]


def plot_discriminant_window_sweep(curves, *, title: str | None = None):
    """Delta(λ) and its line fit for several fit windows on one figure.

    ``curves`` is a list of dicts (one per window, at most 8): ``label``
    (the window), ``lam_d``, ``d_central``, ``d_sigma`` (or None), ``fit``
    (``linfit_zero_crossing(..., return_fit=True)``), ``lam_ep`` (zero of the
    weighted line fit to the measured Delta) and ``lam_ep_err`` (its 1-σ
    error from the fit, or None).

    One panel: the points of every window with their error bars and a line
    through them, each window's λ_EP as a dashed vertical; the y range
    follows the complex side and the fit windows (far points with huge
    error bars run off it).  Legend: the window and λ_EP ± its fit error
    (the number step 3 prints).

    Returns the figure.
    """
    import matplotlib.pyplot as plt
    from mpm_analysis.plotting._style import apply_paper_style

    if len(curves) > len(_SWEEP_STYLES):
        raise ValueError(f"at most {len(_SWEEP_STYLES)} windows per figure, got {len(curves)}.")
    apply_paper_style()
    ink = "#222222"
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.axhline(0, color=ink, ls="--", lw=0.8, alpha=0.7)
    y_lo, y_hi = [], []
    for k, c in enumerate(curves):
        col, mk = _SWEEP_STYLES[k]
        lam = np.asarray(c["lam_d"], dtype=float)
        d = np.asarray(c["d_central"], dtype=float)
        lam_ep, err = c.get("lam_ep"), c.get("lam_ep_err")
        has_ep = lam_ep is not None and np.isfinite(lam_ep)
        if not has_ep:
            ep_txt = "no crossing"
        elif err is None:
            ep_txt = rf"$\lambda_{{EP}} = \mathbf{{{lam_ep:.3f}}}$ (no error bar)"
        else:
            ep_txt = rf"$\lambda_{{EP}} = \mathbf{{{lam_ep:.3f} \pm {err:.3f}}}$"
        fin = np.isfinite(d)
        # the curve through the points, then the points with their error bars
        ax.plot(lam[fin], d[fin], color=col, lw=1.6, alpha=0.8, zorder=4)
        ax.errorbar(lam, d, yerr=c.get("d_sigma"), color=col, marker=mk, ms=5,
                    linestyle="none", elinewidth=0.9, capsize=2, alpha=0.9, zorder=5,
                    label=f"{c['label']}:  {ep_txt}")
        if fin.any():
            y_lo.append(np.min(d[fin]))
        fit = c.get("fit") or {}
        if fit.get("coef") is not None:
            # the fit (not drawn) sets the range, not single far points of its window
            w = fit["window"]
            xs = np.linspace(lam[w].min(), lam[w].max(), 50)
            y_hi.extend([np.polyval(fit["coef"], xs).max(), np.nanpercentile(d[w], 75)])
        if has_ep:
            ax.axvline(lam_ep, color=col, ls="--", lw=1.1, alpha=0.8, zorder=1)

    lo = min(y_lo) if y_lo else -1.0
    hi = max(y_hi) if y_hi else abs(lo)
    lo, hi = min(lo, -0.05 * abs(hi)), max(hi, 0.05 * abs(lo))
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
    ax.set_xlabel(r"$\lambda$", color=ink)
    ax.set_ylabel(r"$\Delta(\lambda) = c_1^2 - 4c_0$", color=ink)
    ax.set_title(title or "Exceptional point vs fit window", color=ink)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, lw=0.5)
    return fig
