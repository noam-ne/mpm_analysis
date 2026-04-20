"""
pole_spectrum.py
----------------
Panel-B style: decay rates (left axis) + oscillation frequency (right axis)
vs lambda, with optional critical-point vertical line and sqrt-fit curve.
"""
from __future__ import annotations

import numpy as np

def plot_decay_and_freq_panel(
    result,
    *,
    ax=None,
    lambda_max: float | None = None,
    pole_selection: str = "second_smallest_decay",
    downsample: int = 1,
    show_sqrt_fit: bool = True,
    show_critical_line: bool = True,
    annotate_lambda_c: bool = False,
    decay_color: str | None = None,
    freq_color: str | None = None,
    markersize: float | None = None,
    linewidth: float | None = None,
    yticks_freq: list | None = None,
) -> tuple["plt.Axes", "plt.Axes"]:
    """Plot decay rate (left axis) and oscillation frequency (right axis) vs λ.

    Parameters
    ----------
    result:
        ``ZenoAnalysisResult``.
    ax:
        Left-axis axes.  If ``None``, a new figure is created.
    lambda_max:
        Upper λ cutoff.  Overrides ``PANEL_B_LAMBDA_MAX`` in the original script.
    pole_selection:
        Which pole to plot.  Options:
        ``"second_smallest_decay"`` (default, matches original paper),
        ``"smallest_decay"``,
        ``int`` (0-indexed).
    downsample:
        Keep every ``downsample``-th lambda point.
    show_sqrt_fit:
        If ``True`` and ``result.critical_point_sqrt_fit`` is present, overlay
        the fit curve on the frequency axis.
    show_critical_line:
        If ``True`` and a fit is present, draw a vertical line at λ_c.
    annotate_lambda_c:
        If ``True``, add a text annotation for λ_c.
    yticks_freq:
        Override tick positions on the right (frequency) axis.

    Returns
    -------
    (ax_decay, ax_freq)
        The left-axis and right-axis ``Axes`` objects.
    """
    import matplotlib.pyplot as plt
    from zeno_analysis.plotting._style import (
        AZURE_BLUE, LAVENDER_BLUE, LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH,
        FIT_LINESTYLE, FIT_LINEWIDTH, LAMBDA_OBS_COLOR, make_ax_lines_wider,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    decay_c = decay_color or AZURE_BLUE
    freq_c  = freq_color  or LAVENDER_BLUE
    ms = markersize or MARKER_SIZE * 1.5
    lw = linewidth  or LINE_WIDTH

    # -- Prepare data --
    lambdas = result.lambda_values
    poles   = result.poles
    omega_s = result.omega_s

    mask = np.ones(len(lambdas), dtype=bool)
    if lambda_max is not None:
        mask &= lambdas <= lambda_max

    lam = lambdas[mask]
    poles_m = poles[mask]

    # Pole selection
    if pole_selection == "second_smallest_decay":
        idx_pol = np.argsort(poles_m.decay_rates, axis=1)[:, 1]
    elif pole_selection == "smallest_decay":
        idx_pol = np.argsort(poles_m.decay_rates, axis=1)[:, 0]
    elif isinstance(pole_selection, int):
        idx_pol = np.full(len(lam), pole_selection, dtype=int)
    else:
        raise ValueError(f"Unknown pole_selection '{pole_selection}'.")

    rows = np.arange(len(lam))
    r2      = poles_m.decay_rates[rows, idx_pol] / omega_s
    r2_err  = poles_m.decay_rates_upper[rows, idx_pol] / omega_s
    f2      = poles_m.frequencies[rows, idx_pol] / omega_s
    f2_err  = poles_m.frequencies_upper[rows, idx_pol] / omega_s

    # Downsample
    s = slice(None, None, downsample)
    lam_ds = lam[s]; r2 = r2[s]; r2_err = r2_err[s]
    f2 = f2[s]; f2_err = f2_err[s]

    err_kw = dict(
        ls="none", marker="o", markersize=ms,
        elinewidth=lw * 0.6,
        capsize=ms * 1.2,
        capthick=lw * 0.6,
        zorder=10,
    )

    # -- Left axis: decay rate --
    ax.errorbar(lam_ds, r2, yerr=r2_err, color=decay_c, alpha=0.85, **err_kw)
    ax.set_xlabel(r"$\lambda$", fontsize=LABELS_FONT_SIZE, labelpad=1.0)
    ax.set_ylabel(r"Decay rate / $\Omega_\mathrm{S}$", fontsize=LABELS_FONT_SIZE)
    ax.tick_params(axis="y", colors=decay_c)
    make_ax_lines_wider(ax)

    # -- Right axis: frequency / imaginary splitting --
    ax_freq = ax.twinx()
    ax_freq.set_ylabel(
        r"($2\pi\times$osc. freq.) / $\Omega_\mathrm{S}$",
        fontsize=LABELS_FONT_SIZE,
    )
    ax_freq.tick_params(axis="y", colors=freq_c)
    make_ax_lines_wider(ax_freq)

    if yticks_freq is not None:
        ax_freq.set_yticks(yticks_freq)

    fit = result.critical_point_sqrt_fit if show_sqrt_fit else None

    if fit is not None and fit.lambda_data is not None and len(fit.lambda_data) > 0:
        # PRIMARY: imaginary splitting data from the sqrt-fit window.
        # imag_split_data and imag_split_err are already normalised by omega_s
        # inside critical_point.fit_sqrt_to_eigenvalues().
        lam_split = fit.lambda_data[::downsample]
        freq_split = fit.imag_split_data[::downsample]
        err_split = fit.imag_split_err[::downsample]

        ax_freq.errorbar(
            lam_split, freq_split, yerr=err_split,
            color=freq_c, alpha=0.85, **err_kw,
        )

        # FALLBACK: pole frequencies for lambda points NOT covered by the fit window.
        # (matches reference: "fill missing" logic using pole-frequency fallback)
        tol = 1e-6
        for lp, fp, ep in zip(lam_ds, f2, f2_err):
            if np.isnan(fp):
                continue
            if not np.any(np.isclose(fit.lambda_data, lp, atol=tol, rtol=0.0)):
                ax_freq.errorbar(
                    [lp], [fp], yerr=[ep],
                    color=freq_c, alpha=0.85, **err_kw,
                )

        # Fit curve overlay
        if fit.lambda_plot is not None and fit.imag_fit_curve is not None:
            ax_freq.plot(
                fit.lambda_plot, fit.imag_fit_curve,
                color=freq_c,
                ls=FIT_LINESTYLE,
                linewidth=FIT_LINEWIDTH,
                alpha=1.0,
                zorder=25,
            )
    else:
        # No fit available: fall back to raw pole frequencies
        ax_freq.errorbar(lam_ds, f2, yerr=f2_err, color=freq_c, alpha=0.85, **err_kw)

    # -- Critical-point line --
    if show_critical_line and result.critical_point_sqrt_fit is not None:
        lc = result.critical_point_sqrt_fit.lambda_c
        for _ax in (ax, ax_freq):
            _ax.axvline(
                lc,
                color=LAMBDA_OBS_COLOR,
                linestyle="--",
                linewidth=lw,
                zorder=0,
            )
        if annotate_lambda_c:
            ax.annotate(
                rf"$\lambda_c={lc:.2f}$",
                xy=(lc, ax.get_ylim()[1]),
                fontsize=LABELS_FONT_SIZE - 1,
                color=LAMBDA_OBS_COLOR,
                ha="left",
                va="top",
            )

    return ax, ax_freq
