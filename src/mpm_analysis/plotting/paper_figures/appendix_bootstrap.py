"""
appendix_bootstrap.py
---------------------
Appendix figure: full pole spectrum (decay rates) + imaginary splitting +
real splitting — or the pair discriminant — for a single transition.

Replaces ``ZenoPlotting/Noams_code/first_transition_plotting.py::plot_paper_style_bootstrap()``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Panel letters: x in axes fraction (the panels share one width), the top a
# fixed distance above every frame whatever the panel's height — where panel
# A's letter sat when each letter was placed at 5 % of its own panel's height.
PANEL_LETTER_X = -0.2
PANEL_LETTER_DY_MM = 2.78
# Every y label's BASELINE sits this far (pt) left of its frame, in every
# figure drawn here — so the post-selected and the ens-avg figure match — and
# the labels of all panels share one line (a label's box reaches below its
# baseline by its descent, which differs between labels).  34 pt clears tick
# labels as wide as "−0.5" by YLABEL_PAD.
YLABEL_OFFSET_PT = 34.0
# Least gap (pt) from the widest tick labels to the baseline; wider tick labels
# than YLABEL_OFFSET_PT allows move every label of the figure out.
YLABEL_PAD = 6.5
# Data points and error bars (bar and cap width, cap length) of the 3-panel
# appendix figure, relative to the shared paper style.
APPENDIX_MARKER_SCALE = 1.5


def place_ylabels(axes) -> None:
    """Put the y labels of ``axes`` on one line ``YLABEL_OFFSET_PT`` left of the frames.

    The baseline is the anchor (``va="baseline"``), a fixed distance from the
    left spine whatever the tick labels — unless the widest of them would come
    within ``YLABEL_PAD`` of it, then all move out together.  Call once the
    ticks are final (it draws the figure once to measure them).
    """
    from matplotlib.transforms import ScaledTranslation

    fig = axes[0].figure
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    widest = 0.0
    for ax in axes:
        lo, hi = sorted(ax.get_ylim())
        spine = ax.get_window_extent(r).x0
        for tick in ax.yaxis.get_major_ticks():
            if lo <= tick.get_loc() <= hi and tick.label1.get_visible() and tick.label1.get_text():
                widest = max(widest, (spine - tick.label1.get_window_extent(r).x0) * 72.0 / fig.dpi)
    offset = max(YLABEL_OFFSET_PT, widest + YLABEL_PAD)
    for ax in axes:
        ax.yaxis.label.set_verticalalignment("baseline")
        ax.yaxis.set_label_coords(0.0, 0.5, transform=ax.transAxes + ScaledTranslation(
            -offset / 72.0, 0.0, fig.dpi_scale_trans))


def fade_error_bars(container, alpha: float) -> None:
    """Draw the bars and caps of an ``errorbar`` container at opacity ``alpha``.

    The markers stay solid (``alpha=`` on the ``errorbar`` call would fade
    them too).  ``alpha = 1`` leaves the artists untouched.
    """
    if alpha >= 1.0:
        return
    _, caps, bars = container.lines
    for artist in (*caps, *bars):
        artist.set_alpha(alpha)


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
    error_alpha: float = 1.0,
    markersize: float | None = None,
) -> None:
    """Draw the full-pole-spectrum panel: -Re(e_k) / Ω_S strands vs λ.

    The shared panel A of the paper bootstrap figures (this module's 3-panel
    sqrt layout and ``first_transition_main``'s 2-panel discriminant layout).
    Decay rates arrive already normalized by Ω_S; the critical-λ line, the
    panel tag and the x-label are the caller's business.  ``error_alpha`` is
    the opacity of the bars and caps (:func:`fade_error_bars`);
    ``markersize`` None → the paper style's MARKER_SIZE.
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
        fade_error_bars(ax.errorbar(
            lambdas, r_med[:, k],
            yerr=[r_lower[:, k], r_upper[:, k]],
            color=c, marker="o",
            markersize=MARKER_SIZE if markersize is None else markersize,
            linestyle="none",
            label=rf"$-\mathrm{{Re}}(e_{{{k+1}}})$",
            **kws,
        ), error_alpha)
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
    panel_c: str = "real",
    split_lambda_ticks: list | None = None,
    error_alpha: float = 1.0,
    crit_line_decimals: int | None = None,
    save_dir: Path | None = None,
    figure_stem: str = "appendix_bootstrap",
) -> "plt.Figure":
    """Build the 3-panel appendix bootstrap figure.

    Panel A (top): all decay rates vs λ (full pole spectrum).
    Panel B (mid): imaginary splitting Im(Δe₁₂) / Ω_S vs λ with sqrt fit.
    Panel C (bot): ``panel_c="real"`` — real splitting Re(Δe₁₂) / Ω_S vs λ
    with fit; ``"discriminant"`` — (Δe₁₂)² / Ω_S² = Δ12 / (4 Ω_S²) at every λ,
    the square of what B plots (−Im² before the EP, +Re² after it): the
    Δ12 points and σ of :func:`discriminant_line_fit`, its line over the
    line's λ window and Δ = 0.

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
    panel_c:
        ``"real"`` or ``"discriminant"`` — what panel C shows (above).
    split_lambda_ticks:
        λ ticks of the one λ axis shared by panels A, B and C (labelled on C
        only); ``None`` → every 0.5.
    error_alpha:
        Opacity of the error bars and caps in all panels; markers stay solid.
    crit_line_decimals:
        Draw the vertical critical-λ line at the critical point ROUNDED to
        this many decimals (the value as it is quoted, e.g. 2 → 1.30), in all
        three panels; ``None`` → the unrounded value.  Only the drawn line
        moves: the dashed curves still end at the fit's own zero.
    save_dir:
        If given, save SVG + PDF here.
    figure_stem:
        Filename stem.

    The y labels of the three panels share one baseline ``YLABEL_OFFSET_PT``
    left of the frames (:func:`place_ylabels`) and every panel letter sits
    ``PANEL_LETTER_DY_MM`` above its frame.

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
    if panel_c not in ("real", "discriminant"):
        raise ValueError(f"panel_c must be 'real' or 'discriminant', got {panel_c!r}.")

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
    # In each draw the splitting is a monotonic function of its Δ12 (Im =
    # √(−Δ12)/2, Re = √Δ12/2), so these bars ARE the draws' 16/84 range of Δ12,
    # not symmetrized, converted.  Only the Δ12 line fit (and the Δ12 points drawn
    # with it) uses the symmetrized σ.
    imag_split = (poles.imag_splitting / omega_s).flatten()
    imag_err   = np.vstack([np.ravel(poles.imag_splitting_lower),
                            np.ravel(poles.imag_splitting_upper)]) / omega_s

    real_split = (poles.real_splitting / omega_s).flatten()
    real_err   = np.vstack([np.ravel(poles.real_splitting_lower),
                            np.ravel(poles.real_splitting_upper)]) / omega_s

    # Δ12 / Ω_S² of the EP pair, its σ and line: the "discriminant" overlay
    # of B/C and panel C's discriminant view
    disc = None
    if splitting_fit == "discriminant" or panel_c == "discriminant":
        disc = discriminant_line_fit(
            result, lambda_max=lambda_max, half_width=discriminant_half_width,
            fit=discriminant_fit,
        )

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
        _, _, _, lam_ep, dfit = disc
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
    # One λ axis for all three panels (the ticks of panels B/C); only C
    # carries the tick labels and the λ label, so A–B and B–C are spaced alike.
    ax_real_full  = fig.add_subplot(gs[0])
    ax_imag       = fig.add_subplot(gs[1], sharex=ax_real_full)
    ax_real_split = fig.add_subplot(gs[2], sharex=ax_real_full)

    ms = MARKER_SIZE * APPENDIX_MARKER_SCALE
    err_kws = dict(
        elinewidth=LINE_WIDTH * 0.6 * APPENDIX_MARKER_SCALE,
        capsize=ms * 1.2,
        capthick=LINE_WIDTH * 0.6 * APPENDIX_MARKER_SCALE,
        zorder=10,
    )

    # The line is drawn at the quoted (rounded) value when asked for — rounded
    # half up on the decimal digits, as a person quotes it (1.0845 → 1.085),
    # not Python's round() on the binary float (→ 1.084).
    if lc is not None and crit_line_decimals is not None:
        from decimal import Decimal, ROUND_HALF_UP
        lc_line = float(Decimal(f"{float(lc):.10f}").quantize(
            Decimal(1).scaleb(-crit_line_decimals), rounding=ROUND_HALF_UP))
    else:
        lc_line = lc

    def _add_crit_line(ax):
        if lc_line is not None:
            ax.axvline(
                lc_line, color=LAMBDA_OBS_COLOR, linestyle="--",
                linewidth=LINE_WIDTH, alpha=0.8, zorder=0,
            )
        make_ax_lines_wider(ax)

    # --- Panel A: full decay rates (shared with the discriminant layout) ---
    draw_full_spectrum_panel(
        ax_real_full, lambdas, r_med, r_lower, r_upper,
        real_axis_scale=real_axis_scale, real_axis_ticks=real_axis_ticks,
        err_kws=err_kws, error_alpha=error_alpha, markersize=ms,
    )
    ax_real_full.tick_params(axis="x", labelbottom=False)   # C's λ labels name the shared axis
    _add_crit_line(ax_real_full)

    add_panel_letter(
        ax_real_full, "a",
        x=PANEL_LETTER_X, y=1.0, dx_mm=0.0, dy_mm=PANEL_LETTER_DY_MM,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )

    # --- Panel B: imaginary splitting ---
    fade_error_bars(ax_imag.errorbar(
        lambdas, imag_split, yerr=imag_err,
        color="steelblue", marker="o",
        linestyle="none", markersize=ms,
        **err_kws,
    ), error_alpha)
    if lam_fit is not None and imag_fit is not None:
        ax_imag.plot(lam_fit, imag_fit, color="k", linestyle="--", linewidth=LINE_WIDTH)
    ax_imag.set_ylabel(
        r"$\mathrm{Im}(\Delta e_{12})\,/\,\Omega_\mathrm{S}$",
        fontsize=LABELS_FONT_SIZE,
    )
    ax_imag.tick_params(axis="x", labelbottom=False)   # C's λ labels name the shared axis
    ax_imag.set_yticks([0.0, 0.4, 0.8])
    _add_crit_line(ax_imag)
    add_panel_letter(
        ax_imag, "b",
        x=PANEL_LETTER_X, y=1.0, dx_mm=0.0, dy_mm=PANEL_LETTER_DY_MM,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )

    # --- Panel C: real splitting, or the discriminant ---
    if panel_c == "discriminant":
        # (Δe12)²/Ω_S² = Δ12/(4 Ω_S²): B and C plot HALF the pole difference,
        # Δ12 = (e1 − e2)² is the whole one.  The σ is the one that weights
        # the line, drawn symmetric, so the panel shows what was fitted.
        lam_d, delta, delta_sigma, _, dfit = disc
        fade_error_bars(ax_real_split.errorbar(
            lam_d, delta / 4.0, yerr=None if delta_sigma is None else delta_sigma / 4.0,
            color="steelblue", marker="o",
            linestyle="none", markersize=ms,
            **err_kws,
        ), error_alpha)
        if dfit["coef"] is not None:
            lo, hi = dfit["lambda_window"]
            lam_line = np.linspace(lo, min(hi, lambda_max), 50)
            ax_real_split.plot(lam_line, np.polyval(dfit["coef"], lam_line) / 4.0,
                               color="k", linestyle="--", linewidth=LINE_WIDTH)
        ax_real_split.axhline(0.0, color="0.5", linewidth=LINE_WIDTH * 0.5, zorder=0)
        ax_real_split.set_ylabel(
            r"$\Delta e_{12}^2\,/\,\Omega_\mathrm{S}^2$",
            fontsize=LABELS_FONT_SIZE,
        )
    else:
        fade_error_bars(ax_real_split.errorbar(
            lambdas, real_split, yerr=real_err,
            color="steelblue", marker="o",
            linestyle="none", markersize=ms,
            **err_kws,
        ), error_alpha)
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
        x=PANEL_LETTER_X, y=1.0, dx_mm=0.0, dy_mm=PANEL_LETTER_DY_MM,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )

    # x-axis tick formatting (A and B share C's λ axis)
    if split_lambda_ticks is None:
        ax_real_split.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    else:
        ax_real_split.set_xticks(split_lambda_ticks)

    # y labels of all three panels on one line, the same distance from the
    # frames in every figure, whatever the tick widths
    place_ylabels([ax_real_full, ax_imag, ax_real_split])

    if save_dir is not None:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        save_figure_pdf_svg(fig, figure_stem, save_dir)

    return fig
