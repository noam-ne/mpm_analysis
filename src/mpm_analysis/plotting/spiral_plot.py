"""
spiral_plot.py
--------------
Interactive browser of the spiral-approach fits (``analysis/spiral_approach``):
per λ the conditional trajectory in the XZ-plane with the fitted eq. (4)
curve and its fixed point, the X(t) / Z(t) traces with the fit over the
late-time window, and ln S(t) with the γ0 line.
"""
from __future__ import annotations

import numpy as np


def plot_spiral_fits_slider(
    tomo_records,
    spiral_result,
    *,
    survival_key: str = "S",
    cmap: str = "viridis",
) -> "Slider":
    """λ slider over the spiral fits.

    Parameters
    ----------
    tomo_records:
        The records the pipeline ran on (``survival_key`` + ``spiral_result.keys``).
    spiral_result:
        ``SpiralKsiResult`` from ``SpiralKsiPipeline.run()``; its ``fits`` are
        drawn at the matching λ.

    Returns
    -------
    matplotlib.widgets.Slider
        Keep the reference alive (matplotlib weak-references callbacks).
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from mpm_analysis.data_types.observables import group_records_by_lambda

    keys = [survival_key, *spiral_result.keys]
    groups = group_records_by_lambda(tomo_records, keys)
    lam_res = np.asarray(spiral_result.lambda_values, dtype=float)
    # keep only λ the result knows, in result order
    by_lam = {}
    for lam, t, Y, _ in groups:
        i = int(np.argmin(np.abs(lam_res - lam)))
        if abs(lam_res[i] - lam) <= 1e-6:
            by_lam[i] = (t, Y)
    idx = sorted(by_lam)
    if not idx:
        raise ValueError("plot_spiral_fits_slider: no record λ matches the result.")

    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], left=0.06, right=0.98,
                          top=0.9, bottom=0.2, hspace=0.35, wspace=0.25)
    ax_xz = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1], sharex=ax_t)
    ax_slider = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    # mathtext only: the paper style (apply_paper_style) may have usetex on
    slider = Slider(ax_slider, r"$\lambda$ index", 0, len(idx) - 1, valinit=0, valstep=1)
    kx, kz = spiral_result.keys

    def _draw(pos):
        i = idx[int(pos)]
        t, Y = by_lam[i]
        S, X, Z = Y[0], Y[1], Y[2]
        fit = spiral_result.fits[i] if i < len(spiral_result.fits) else None
        lam = lam_res[i]
        ga, g0, ksi = spiral_result.gamma_a[i], spiral_result.gamma0[i], spiral_result.ksi_values[i]
        for ax in (ax_xz, ax_t, ax_s):
            ax.cla()

        # --- XZ plane --------------------------------------------------------
        th = np.linspace(0, 2 * np.pi, 200)
        ax_xz.plot(np.cos(th), np.sin(th), color="0.8", lw=1)
        ax_xz.axhline(0, color="0.9", lw=0.8); ax_xz.axvline(0, color="0.9", lw=0.8)
        sc = ax_xz.scatter(X, Z, c=t, cmap=cmap, s=18, zorder=3)
        if fit is not None:
            m = (t >= fit.t0) & np.isfinite(S) & (S >= spiral_result.min_survival)
            tt = np.linspace(fit.t0, t[m].max() if m.any() else t[-1], 300)
            C = fit.curve(tt)
            ax_xz.plot(C[0], C[1], "k-", lw=1.4, label="eq. (4) fit", zorder=4)
            ax_xz.plot(X[m], Z[m], "o", mfc="none", mec="k", ms=7, lw=0.8,
                       label="fitted window", zorder=5)
            ax_xz.plot(*fit.r_inf, marker="*", color="crimson", ms=14,
                       label=rf"$R_\infty$=({fit.r_inf[0]:.2f}, {fit.r_inf[1]:.2f})", zorder=6)
        ax_xz.set_aspect("equal"); ax_xz.set_xlim(-1.1, 1.1); ax_xz.set_ylim(-1.1, 1.1)
        ax_xz.set_xlabel(r"$\langle X\rangle$"); ax_xz.set_ylabel(r"$\langle Z\rangle$")
        ax_xz.legend(fontsize=8, loc="lower left")
        if not hasattr(_draw, "cbar"):
            _draw.cbar = fig.colorbar(sc, ax=ax_xz, pad=0.02, fraction=0.046,
                                      label=r"$t$ ($\mu$s)")
        else:
            _draw.cbar.update_normal(sc)

        # --- X(t), Z(t) -------------------------------------------------------
        ax_t.plot(t, X, "o", ms=3, color="C0", label=kx.replace("_", r"\_"))
        ax_t.plot(t, Z, "o", ms=3, color="C1", label=kz.replace("_", r"\_"))
        if fit is not None:
            tt = np.linspace(fit.t0, t[-1], 300)
            C = fit.curve(tt)
            ax_t.plot(tt, C[0], "-", color="C0", lw=1.2)
            ax_t.plot(tt, C[1], "-", color="C1", lw=1.2)
            ax_t.axvspan(fit.t0, t[-1], color="0.92", zorder=0)
            ax_t.set_title(
                rf"$\gamma_a = {ga:.3f} \pm {spiral_result.gamma_a_err[i]:.3f}$ /$\mu$s, "
                rf"$\omega = {fit.omega:.3f}$, $\chi^2_r = {fit.chi2_red:.2f}$", fontsize=9)
        ax_t.set_ylabel("Bloch coordinate"); ax_t.legend(fontsize=8, loc="upper right")

        # --- ln S(t) ----------------------------------------------------------
        pos_ok = np.isfinite(S) & (S > 0)
        ax_s.semilogy(t[pos_ok], S[pos_ok], "o", ms=3, color="0.3", label=survival_key)
        if np.isfinite(g0):
            tail = pos_ok & (S >= spiral_result.min_survival)
            if tail.any():
                t_last = t[tail][-1]
                t_line = np.linspace(max(t_last - 10.0, t[0]), t_last, 50)
                src = spiral_result.gamma0_source.replace("_", r"\_")
                ax_s.semilogy(t_line, S[tail][-1] * np.exp(-g0 * (t_line - t_last)),
                              "r-", lw=1.4,
                              label=rf"$\gamma_0 = {g0:.3f}$ /$\mu$s ({src})")
        ax_s.axhline(spiral_result.min_survival, color="0.7", ls=":", lw=1)
        ax_s.set_xlabel(r"$t$ ($\mu$s)"); ax_s.set_ylabel("survival"); ax_s.legend(fontsize=8)

        fig.suptitle(rf"$\lambda = {lam:.3f}$    $\xi = \gamma_0/\gamma_a - 1 = {ksi:+.3f}$",
                     fontsize=11)
        fig.canvas.draw_idle()

    slider.on_changed(_draw)
    _draw(0)
    fig._spiral_slider = slider      # strong reference on the figure
    return slider
