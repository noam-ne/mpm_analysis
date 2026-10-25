"""
appendix_bootstrap.py
---------------------
Appendix figure: full pole spectrum (decay rates) + imaginary splitting +
real splitting for a single transition.

Replaces ``ZenoPlotting/Noams_code/first_transition_plotting.py::plot_paper_style_bootstrap()``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def draw_full_spectrum_panel(
    ax,
    lambdas: np.ndarray,
    r_med: np.ndarray,
    r_lower: np.ndarray,
    r_upper: np.ndarray,
    *,
    real_axis_scale: str = "linear",
    real_axis_ticks: list | None = None,
    err_kws: dict | None = None,
) -> None:
    """Draw the full-pole-spectrum panel: -Re(e_k) / Ω_S strands vs λ.

    The shared panel A of the paper bootstrap figures (this module's 3-panel
    sqrt layout and ``first_transition_main``'s 2-panel discriminant layout).
    Decay rates arrive already normalized by Ω_S; the critical-λ line, the
    panel tag and the x-label are the caller's business.
    """
    import matplotlib.ticker as ticker

    from mpm_analysis.plotting._style import (
        LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH, LINECUT_COLORS,
    )

    # Slot 0 (EP pair; the faster branch after the split) teal, slot 1 (the
    # slower branch) purple, slot 2 mustard — the paper's style.
    colors = [LINECUT_COLORS[1], LINECUT_COLORS[0], LINECUT_COLORS[2], "gray"]
    if err_kws is None:
        err_kws = dict(
            elinewidth=LINE_WIDTH * 0.6,
            capsize=MARKER_SIZE * 1.2,
            capthick=LINE_WIDTH * 0.6,
            zorder=10,
        )

    for k in range(r_med.shape[1]):
        c = colors[k % len(colors)]
        kws = dict(err_kws)
        if k == 0:
            # Before the EP slots 0 and 1 coincide: slot 0 on top, so the
            # merged pair shows teal; after the split teal = faster branch.
            kws["zorder"] = kws.get("zorder", 10) + 1
        # No abs(): a decay rate <= 0 (a growing pole) must show, not fold over.
        ax.errorbar(
            lambdas, r_med[:, k],
            yerr=[r_lower[:, k], r_upper[:, k]],
            color=c, marker="o",
            markersize=MARKER_SIZE,
            linestyle="none",
            label=rf"$-\mathrm{{Re}}(e_{{{k+1}}})$",
            **kws,
        )
    ax.set_yscale(real_axis_scale)
    ax.set_ylabel(
        r"$-\mathrm{Re}(e_k)\,/\,\Omega_\mathrm{S}$",
        fontsize=LABELS_FONT_SIZE, labelpad=0,
    )
    ax.legend(loc="upper left", fontsize=LABELS_FONT_SIZE)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    if real_axis_scale == "log":
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=2))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    elif real_axis_ticks is not None:
        ax.set_yticks(real_axis_ticks)
    ax.tick_params(
        axis="both", which="major",
        width=LINE_WIDTH, labelsize=LABELS_FONT_SIZE,
    )


def discriminant_line_fit(
    result, *, lambda_max: float, half_width: float, fit: dict | None = None,
):
    """Δ12 / Ω_S² of the EP pair and the straight-line fit through its zero.

    Shared by ``first_transition_main``'s discriminant layout (Δ12 and the
    line) and the sqrt layout's ``splitting_fit="discriminant"`` overlay (√ of
    the line).  Δ12 comes from the central poles, followed along λ past the
    EP; its σ is half the 16/84 spread of the bootstrap draws when the result
    kept them (``None`` otherwise), and the same σ weights the line fit.

    ``fit`` — a line fit already made elsewhere (``run.py``'s step-3 product
    ``ep["delta_fit"]``: ``lambda_ep``, ``coef`` in the poles' units,
    ``lambda_window``) is used as it is, not redone; ``half_width`` is then
    unused.

    Returns
    -------
    (lambdas, delta, sigma, lambda_ep, line)
        λ ≤ ``lambda_max``, Δ12/Ω_S² and its σ there, the zero of the line
        and ``dict(coef=[slope, intercept] in Ω_S² units or None,
        lambda_window=(lo, hi) or None)``.
    """
    from mpm_analysis.analysis.ksi_crossing import (
        discriminant_crossing_samples,
        discriminant_from_poles,
        linfit_zero_crossing,
    )

    omega_s = result.omega_s
    mask = result.lambda_values <= lambda_max
    lambdas = result.lambda_values[mask]
    poles = result.poles[mask]

    # Poles enter Δ quadratically → normalize by Ω_S².
    central = -poles.decay_rates + 1j * poles.frequencies
    delta = discriminant_from_poles(central) / omega_s**2

    sigma = None
    raw = result.poles.raw_samples
    if raw is not None:
        _, d_med, d_lo, d_hi = discriminant_crossing_samples(
            raw, lambda_max=lambda_max, half_width=half_width,
        )
        if d_med is not None and len(d_med) == len(lambdas):
            d_lo = np.asarray(d_lo, dtype=float) / omega_s**2
            d_hi = np.asarray(d_hi, dtype=float) / omega_s**2
            sigma = (d_hi - d_lo) / 2
    if fit is not None:
        lam_ep = float(fit["lambda_ep"])
        coef = (None if fit["coef"] is None
                else np.asarray(fit["coef"], dtype=float) / omega_s**2)
        window = fit["lambda_window"]
    else:
        lam_ep, own = linfit_zero_crossing(
            lambdas, delta, sigma=sigma, half_width=half_width, return_fit=True,
        )
        coef = own["coef"]
        window = (None if coef is None else
                  (float(lambdas[own["window"]].min()), float(lambdas[own["window"]].max())))
    return lambdas, delta, sigma, lam_ep, dict(coef=coef, lambda_window=window)


def plot_appendix_bootstrap(
    result,
    *,
    parameters: dict | None = None,
    lambda_max: float = 1.5,
    fit_validity_limit: float | None = None,
    splitting_fit: str = "sqrt",
    discriminant_half_width: float = 0.15,
    discriminant_fit: dict | None = None,
    real_axis_scale: str = "linear",
    real_axis_ticks: list | None = None,
    real_split_axis_ticks: list | None = None,
    save_dir: Path | None = None,
    figure_stem: str = "appendix_bootstrap",
) -> "plt.Figure":
    """Build the 3-panel appendix bootstrap figure.

    Panel A (top): all decay rates vs λ (full pole spectrum).
    Panel B (mid): imaginary splitting Im(Δe₁₂) / Ω_S vs λ with sqrt fit.
    Panel C (bot): real splitting Re(Δe₁₂) / Ω_S vs λ with fit.

    The overlay in B/C and the vertical line come from ``splitting_fit``:

    * ``"sqrt"`` — the critical-point sqrt fit (``result.critical_point_sqrt_fit``),
      line at its λ_c.
    * ``"discriminant"`` — the straight-line fit of Δ12 = (e₁ − e₂)² through
      its zero (:func:`discriminant_line_fit`), taken as a square root.  The
      plotted splittings are half the pole difference, so the curves are
      √(−Δ12)/2 (panel B, below the EP) and √Δ12/2 (panel C, above), drawn
      over the line-fit window; line at its λ_EP.  Needs no sqrt fit.

    Parameters
    ----------
    result:
        ``ZenoAnalysisResult`` (must have ``poles.has_splitting`` True).
    parameters:
        Override ``result.parameters`` for display (optional).
    lambda_max:
        Upper λ cutoff for all panels.
    fit_validity_limit:
        Upper λ cutoff for the sqrt-fit overlay.  Default: ``lambda_max``.
        (The ``"discriminant"`` overlay spans the line-fit window instead.)
    splitting_fit:
        ``"sqrt"`` or ``"discriminant"`` — source of the B/C overlay (above).
    discriminant_half_width:
        Half-window in λ of the Δ12 line fit (``"discriminant"`` only).
    discriminant_fit:
        A Δ12 line fit made elsewhere, drawn as it is instead of refitting
        (see :func:`discriminant_line_fit`); ``discriminant_half_width`` is
        then unused.
    real_axis_scale:
        ``"linear"`` or ``"log"`` for panel A.
    real_axis_ticks:
        Manual y-tick positions for panel A.
    real_split_axis_ticks:
        Manual y-tick positions for panel C.
    save_dir:
        If given, save SVG + PDF here.
    figure_stem:
        Filename stem.

    Returns
    -------
    plt.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    from mpm_analysis.plotting._style import (
        TWO_COLUMN_WIDTH, LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH,
        LAMBDA_OBS_COLOR, add_panel_letter, apply_paper_style,
        make_ax_lines_wider,
    )

    if fit_validity_limit is None:
        fit_validity_limit = lambda_max

    omega_s = result.omega_s

    apply_paper_style()

    # --- Data prep ---
    lambdas = result.lambda_values
    mask = lambdas <= lambda_max
    lambdas = lambdas[mask]
    poles = result.poles[mask]

    # Panel A: full decay rates
    r_med   = poles.decay_rates / omega_s
    r_lower = poles.decay_rates_lower / omega_s
    r_upper = poles.decay_rates_upper / omega_s

    # Panels B, C: splitting
    if not poles.has_splitting:
        raise ValueError(
            "result.poles.has_splitting is False — re-run the analysis to compute splittings."
        )

    # Asymmetric 16/84 bars: a splitting is >= 0 in every draw, so its band
    # never reaches below zero.  A symmetric bar would wherever the central
    # value is 0 while some draws already sit on the other side of the EP.
    imag_split = (poles.imag_splitting / omega_s).flatten()
    imag_err   = np.vstack([np.ravel(poles.imag_splitting_lower),
                            np.ravel(poles.imag_splitting_upper)]) / omega_s

    real_split = (poles.real_splitting / omega_s).flatten()
    real_err   = np.vstack([np.ravel(poles.real_splitting_lower),
                            np.ravel(poles.real_splitting_upper)]) / omega_s

    # Fit curves (if available)
    lam_fit = imag_fit = real_fit = None
    lc = None
    if splitting_fit == "sqrt":
        fit = result.critical_point_sqrt_fit
        if fit is not None and fit.lambda_plot is not None:
            fv_mask = fit.lambda_plot <= fit_validity_limit
            lam_fit  = fit.lambda_plot[fv_mask]
            imag_fit = fit.imag_fit_curve[fv_mask] if fit.imag_fit_curve is not None else None
            real_fit = fit.real_fit_curve[fv_mask] if fit.real_fit_curve is not None else None
            lc = fit.lambda_c
    elif splitting_fit == "discriminant":
        _, _, _, lam_ep, dfit = discriminant_line_fit(
            result, lambda_max=lambda_max, half_width=discriminant_half_width,
            fit=discriminant_fit,
        )
        if dfit["coef"] is not None and np.isfinite(lam_ep):
            # √ of the line over its window (cut at lambda_max), λ_EP on the
            # grid so both branches reach zero; the other side is clipped to
            # 0 as in the sqrt fit.
            lo, hi = dfit["lambda_window"]
            hi = min(hi, lambda_max)
            lam_fit = np.union1d(np.linspace(lo, hi, 500), np.clip([lam_ep], lo, hi))
            d_fit = np.polyval(dfit["coef"], lam_fit)
            imag_fit = np.sqrt(np.clip(-d_fit, 0.0, None)) / 2.0
            real_fit = np.sqrt(np.clip(d_fit, 0.0, None)) / 2.0
            lc = lam_ep
    else:
        raise ValueError(
            f"splitting_fit must be 'sqrt' or 'discriminant', got {splitting_fit!r}."
        )

    # --- Figure layout ---
    fig_w = float(TWO_COLUMN_WIDTH) * 0.67
    fig = plt.figure(figsize=(fig_w, fig_w * 1.2))
    fig.set_layout_engine("constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    ax_real_full  = fig.add_subplot(gs[0])
    ax_imag       = fig.add_subplot(gs[1])
    ax_real_split = fig.add_subplot(gs[2], sharex=ax_imag)

    err_kws = dict(
        elinewidth=LINE_WIDTH * 0.6,
        capsize=MARKER_SIZE * 1.2,
        capthick=LINE_WIDTH * 0.6,
        zorder=10,
    )

    def _add_crit_line(ax):
        if lc is not None:
            ax.axvline(
                lc, color=LAMBDA_OBS_COLOR, linestyle="--",
                linewidth=LINE_WIDTH, alpha=0.8, zorder=0,
            )
        make_ax_lines_wider(ax)

    # --- Panel A: full decay rates (shared with the discriminant layout) ---
    draw_full_spectrum_panel(
        ax_real_full, lambdas, r_med, r_lower, r_upper,
        real_axis_scale=real_axis_scale, real_axis_ticks=real_axis_ticks,
        err_kws=err_kws,
    )
    ax_real_full.set_xlabel(r"$\lambda$", fontsize=LABELS_FONT_SIZE)
    _add_crit_line(ax_real_full)

    add_panel_letter(
        ax_real_full, "a",
        x=-0.2, y=1.05, offset_mm=False,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )

    # --- Panel B: imaginary splitting ---
    ax_imag.errorbar(
        lambdas, imag_split, yerr=imag_err,
        color="steelblue", marker="o",
        linestyle="none", markersize=MARKER_SIZE,
        **err_kws,
    )
    if lam_fit is not None and imag_fit is not None:
        ax_imag.plot(lam_fit, imag_fit, color="k", linestyle="--", linewidth=LINE_WIDTH)
    ax_imag.set_ylabel(
        r"$\mathrm{Im}(\Delta e_{12})\,/\,\Omega_\mathrm{S}$",
        fontsize=LABELS_FONT_SIZE,
    )
    plt.setp(ax_imag.get_xticklabels(), visible=False)
    ax_imag.set_yticks([0.0, 0.4, 0.8])
    _add_crit_line(ax_imag)
    add_panel_letter(
        ax_imag, "b",
        x=-0.2, y=1.05, offset_mm=False,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )

    # --- Panel C: real splitting ---
    ax_real_split.errorbar(
        lambdas, real_split, yerr=real_err,
        color="steelblue", marker="o",
        linestyle="none", markersize=MARKER_SIZE,
        **err_kws,
    )
    if lam_fit is not None and real_fit is not None:
        ax_real_split.plot(lam_fit, real_fit, color="k", linestyle="--", linewidth=LINE_WIDTH)
    ax_real_split.set_ylabel(
        r"$\mathrm{Re}(\Delta e_{12})\,/\,\Omega_\mathrm{S}$",
        fontsize=LABELS_FONT_SIZE,
    )
    ax_real_split.set_xlabel(r"$\lambda$", fontsize=LABELS_FONT_SIZE)
    if real_split_axis_ticks is not None:
        ax_real_split.set_yticks(real_split_axis_ticks)
    _add_crit_line(ax_real_split)
    add_panel_letter(
        ax_real_split, "c",
        x=-0.2, y=1.05, offset_mm=False,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )

    # x-axis tick formatting
    for _ax in [ax_real_split, ax_real_full]:
        _ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # y labels of all three panels in one column, whatever the tick widths
    fig.align_ylabels([ax_real_full, ax_imag, ax_real_split])

    if save_dir is not None:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        save_figure_pdf_svg(fig, figure_stem, save_dir)

    return fig
