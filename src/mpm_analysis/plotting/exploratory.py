"""
exploratory.py
--------------
Quick sanity-check plots for raw data inspection.

All functions return a matplotlib ``Axes`` or ``Figure`` and take an optional
``ax`` / ``fig`` argument so they can be embedded into larger layouts.
They use minimal styling (no paper-quality formatting) for speed.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

def plot_raw_survival(
    records,
    *,
    ax=None,
    observable_key: str = "survival",
    downsample: int = 1,
    colormap: str = "viridis",
    title: str = "Raw curves",
) -> "plt.Axes":
    """Plot all lambda linecuts on one axis, colour-coded by λ.

    Parameters
    ----------
    records:
        ``list[ObservableRecord]``.
    ax:
        Existing axes to plot into.  ``None`` → create new figure.
    observable_key:
        Which observable to plot.
    downsample:
        Plot every ``downsample``-th time point.
    colormap:
        Matplotlib colormap name for λ colour-coding.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    slices = [r for r in records if r.observable_key == observable_key]
    slices.sort(key=lambda r: r.lambda_val)

    if not slices:
        raise ValueError(f"No records found with observable_key='{observable_key}'.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    lam_vals = np.array([r.lambda_val for r in slices])
    cmap = cm.get_cmap(colormap)
    lam_norm = (lam_vals - lam_vals.min()) / max(lam_vals.ptp(), 1e-9)

    for r, lnorm in zip(slices, lam_norm):
        ax.plot(
            r.t[::downsample],
            r.signal[::downsample],
            color=cmap(lnorm),
            linewidth=0.8,
            alpha=0.7,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lam_vals.min(), lam_vals.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=r"$\lambda$")

    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel(observable_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    return ax

def plot_svd_singular_values(
    lambdas,
    sv_list,
    *,
    n_sv: int = 10,
    ax=None,
    title: str = "Hankel SVD singular values vs λ",
) -> "plt.Axes":
    """Plot Hankel SVD singular value magnitudes vs λ for model-order selection.

    Each line corresponds to one singular value index; a clear gap ('knee')
    between lines indicates the appropriate model order.

    Parameters
    ----------
    lambdas:
        1-D array of λ values, shape ``(n_lambda,)``.
    sv_list:
        List of 1-D arrays, one per λ slice, containing the singular values
        in descending order.
    n_sv:
        Number of singular values (lines) to display.
    ax:
        Existing axes to plot into.  ``None`` → new figure.
    title:
        Axes title.

    Returns
    -------
    plt.Axes
    """
    import matplotlib.cm as cm

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    lambdas = np.asarray(lambdas)
    n_lambda = len(lambdas)

    # Truncate every sv array to the common minimum length, then to n_sv
    min_len = min(len(s) for s in sv_list)
    n_show = min(n_sv, min_len)

    sv_matrix = np.array([s[:n_show] for s in sv_list])  # (n_lambda, n_show)

    colors = cm.tab10(np.linspace(0, 1, n_show))
    for k in range(n_show):
        ax.semilogy(lambdas, sv_matrix[:, k], color=colors[k],
                    linewidth=1.2, label=f"SV {k + 1}")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Singular value magnitude")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    return ax


def plot_eigenvalue_spectrum(
    result,
    *,
    ax=None,
    normalized: bool = True,
    mode: str = "decay",
    title: str | None = None,
) -> "plt.Axes":
    """Plot decay rates or frequencies vs lambda from a ``ZenoAnalysisResult``.

    Parameters
    ----------
    result:
        ``ZenoAnalysisResult``.
    normalized:
        If ``True``, divide by ``result.omega_s``.
    mode:
        ``"decay"`` → plot -Re(poles), ``"frequency"`` → plot Im(poles).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    lambdas = result.lambda_values
    omega_s = result.omega_s if normalized else 1.0

    if mode == "decay":
        data = result.poles.decay_rates
        y_label = r"Decay rate / $\Omega_S$"
    elif mode == "frequency":
        data = result.poles.frequencies
        y_label = r"Frequency / $\Omega_S$"
    else:
        raise ValueError(f"mode must be 'decay' or 'frequency', got '{mode}'.")

    order = data.shape[1]
    if mode == "decay":
        lower = result.poles.decay_rates_lower
        upper = result.poles.decay_rates_upper
    else:
        lower = result.poles.frequencies_lower
        upper = result.poles.frequencies_upper
    for k in range(order):
        med = data[:, k] / omega_s
        err_lo = lower[:, k] / omega_s
        err_hi = upper[:, k] / omega_s
        ax.errorbar(
            lambdas, med,
            yerr=[err_lo, err_hi],
            fmt="o--", markersize=3,
            linewidth=0.8,
            capsize=2,
            label=f"Pole {k+1}",
        )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)
    ax.set_title(title or mode.capitalize() + " spectrum")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    return ax


def plot_time_traces_slider(
    records,
    *,
    observable_key: str = "survival",
    boot_results=None,
    log_scale: bool = True,
    mpm_order: int = 3,
    show_fit: bool = True,
) -> None:
    """Interactive slider to browse per-lambda time traces with optional MPM fit overlay.

    Parameters
    ----------
    records:
        ``list[ObservableRecord]`` — the raw experimental records.
    observable_key:
        Which observable to browse.
    boot_results:
        Unused; kept for API compatibility.
    log_scale:
        Use logarithmic y-axis (useful for survival probability curves).
    mpm_order:
        MPM model order used for the fit overlay.
    show_fit:
        If ``True`` (default), overlay the MPM-reconstructed fit on each trace.
        Set to ``False`` to show raw data only.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from mpm_analysis.analysis.matrix_pencil import (
        matrix_pencil_method,
        solve_linear_amplitudes,
        reconstruct_signal,
    )

    slices = sorted(
        [r for r in records if r.observable_key == observable_key],
        key=lambda r: r.lambda_val,
    )
    if not slices:
        raise ValueError(f"No records with observable_key='{observable_key}'.")

    def _get_fit_curve(rec):
        try:
            poles, _, _ = matrix_pencil_method(rec.t, rec.signal, order=mpm_order)
            amps = solve_linear_amplitudes(rec.t, rec.signal, poles)
            return reconstruct_signal(rec.t, poles, amps)
        except Exception:
            return None

    def _ylim_for(sig):
        finite = sig[np.isfinite(sig)]
        if finite.size == 0:
            return None
        if log_scale:
            pos = finite[finite > 0]
            if pos.size == 0:
                return None
            return pos.min() * 0.5, pos.max() * 2.0
        span = finite.max() - finite.min()
        margin = 0.12 * span if span > 0 else 0.1
        return finite.min() - margin, finite.max() + margin

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.22)

    rec0 = slices[0]
    line_raw = ax.scatter(
    rec0.t,
    rec0.signal,
    color="#2166ac",
    s=20,
    marker="o",
    linewidths=0,
    label="data",
    zorder=3,
)

    fit0 = _get_fit_curve(rec0) if show_fit else None
    fit_y0 = fit0 if fit0 is not None else []
    fit_x0 = rec0.t if fit0 is not None else []
    line_fit, = ax.plot(fit_x0, fit_y0,
                        color="#d6604d", lw=1.5, label=f"MPM fit (order {mpm_order})",
                        zorder=5, linestyle="--")
    line_fit.set_visible(show_fit)

    title = ax.set_title(f"λ = {rec0.lambda_val:.4f}  (1 / {len(slices)})", fontsize=12)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(r"Time ($\mu$s)", fontsize=11)
    ax.set_ylabel(observable_key, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(rec0.t.min(), rec0.t.max())
    lim = _ylim_for(rec0.signal)
    if lim:
        ax.set_ylim(*lim)

    ax_sl = plt.axes([0.18, 0.08, 0.64, 0.03])
    slider = Slider(ax_sl, "λ index", 0, len(slices) - 1, valinit=0, valstep=1)

    def _update(val):
        idx = int(slider.val)
        rec = slices[idx]
        line_raw.set_offsets(np.column_stack((rec.t, rec.signal)))
        if show_fit:
            fit = _get_fit_curve(rec)
            if fit is not None:
                line_fit.set_data(rec.t, fit)
            else:
                line_fit.set_data([], [])
        ax.set_xlim(rec.t.min(), rec.t.max())
        lim = _ylim_for(rec.signal)
        if lim:
            ax.set_ylim(*lim)
        title.set_text(f"λ = {rec.lambda_val:.4f}  ({idx + 1} / {len(slices)})")
        fig.canvas.draw_idle()

    slider.on_changed(_update)
    plt.show()


def plot_bootstrap_distribution(
    boot_results,
    lambda_idx: int,
    pole_idx: int = 0,
    *,
    ax=None,
    component: str = "real",
) -> "plt.Axes":
    """Histogram of bootstrap pole values at a specific lambda slice.

    Parameters
    ----------
    boot_results:
        Output of ``analysis.bootstrap.run_bootstrap``.
    lambda_idx:
        Index into ``boot_results`` list.
    pole_idx:
        Which pole to show.
    component:
        ``"real"`` (decay) or ``"imag"`` (frequency).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))

    br = boot_results[lambda_idx]
    data = br.boot_real[:, pole_idx] if component == "real" else br.boot_imag[:, pole_idx]
    data = data[np.isfinite(data)]

    ax.hist(data, bins=40, edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(data), color="red", linestyle="--", linewidth=1.2, label="median")
    ax.set_xlabel(f"Pole {pole_idx+1} {'Re' if component == 'real' else 'Im'}")
    ax.set_ylabel("Count")
    ax.set_title(f"Bootstrap distribution at λ={br.lambda_val:.3f}")
    ax.legend()
    return ax

