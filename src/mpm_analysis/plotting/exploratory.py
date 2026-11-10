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
    logscale_y: bool = False,
    take_abs: bool = False,
    omega_DG: float | None = None,
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
        if take_abs:
            sig = np.abs(r.signal)
        else:
            sig = r.signal
        ax.plot(
            r.t[::downsample],
            sig[::downsample],
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
    if logscale_y:
        ax.set_yscale("log")
    
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


# One name and colour per exponent (canonical pole slot k) in every pole plot —
# the spectrum panels and the slider's terms — so e_1 is the same pole everywhere.
def exponent_color(k: int) -> str:
    """Colour of exponent slot ``k``: matplotlib's C0, C1, … ."""
    return f"C{k % 10}"


def exponent_label(k: int) -> str:
    """Legend name of exponent slot ``k``: e_{k+1}."""
    return rf"$e_{{{k + 1}}}$"


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
            color=exponent_color(k),
            label=exponent_label(k),
        )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)
    ax.set_title(title or mode.capitalize() + " spectrum")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    return ax


def plot_spectrum_panels(result, *, title=None, axes=None, figsize=(15, 6)):
    """The standard two-panel pole diagnostic: all decay rates (left) and all
    frequencies (right) vs λ, one errorbar series per pole.

    This is the figure ``run.py`` shows after every MPM run; other callers
    (e.g. the dead-time compression sweep, one figure per ``c``) reuse it so
    every pole spectrum in the project looks the same.

    Parameters
    ----------
    result:
        ``ZenoAnalysisResult`` (or anything with ``lambda_values``, ``poles``
        and ``omega_s``).
    title:
        Figure suptitle.
    axes:
        Two existing axes to draw into; created when ``None``.

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = axes[0].get_figure()
    if title:
        fig.suptitle(title, fontsize=14)
    plot_eigenvalue_spectrum(result, ax=axes[0], mode="decay")
    axes[0].set_title("Decay rates", fontsize=12)
    plot_eigenvalue_spectrum(result, ax=axes[1], mode="frequency")
    axes[1].set_title("Frequencies", fontsize=12)
    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    return fig, axes


# Built with LaTeX off: apply_paper_style() (called by plot_ksi_curve and the
# paper figures earlier in a run) turns text.usetex on globally, and LaTeX
# cannot compile the unicode λ in the title/slider label — the draw fails and
# no traces appear.  Each Text keeps the usetex it was created with, so this
# also holds for every later redraw.
@plt.rc_context({"text.usetex": False})
def plot_time_traces_slider(
    records,
    *,
    observable_key: str | list[str] = "survival",
    scales: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    boot_results=None,
    take_abs: bool = False,
    log_scale: bool = True,
    mpm_order: int = 3,
    show_fit: bool = True,
    components: bool = False,
    fixed_poles: dict[float, np.ndarray] | None = None,
    colors: dict[str, str] | None = None,
) -> "Slider":
    """Interactive slider to browse per-lambda time traces with optional MPM fit overlay.

    Supports multiple observables on shared or separate y-axes: observables
    with the same scale share one axis; each distinct scale gets its own
    (twinned) axis.  Each axis owns a color family — its label, ticks, and
    spine are tinted to match — and observables sharing an axis get
    distinguishable shades within that family.

    Parameters
    ----------
    records:
        ``list[ObservableRecord]`` — the raw experimental records.
    observable_key:
        Which observable(s) to browse.  A list of keys shows all of them,
        with the MPM fit computed jointly (common poles, per-observable
        amplitudes).
    scales:
        Mapping ``key -> "log" | "linear"``.  Unlisted keys default to
        ``"linear"``.  ``None`` with a single key falls back to the legacy
        ``log_scale`` flag.
    labels:
        Mapping ``key -> display name`` for the legend and axis labels.
        Unlisted keys use the key itself.
    boot_results:
        Unused; kept for API compatibility.
    take_abs:
        Plot (and fit) ``|signal|``.  Note: abs is applied *before* fitting,
        so it changes the fitted poles; on a log axis non-positive values
        vanish silently.
    log_scale:
        Legacy flag, honoured only for a single observable when ``scales``
        is ``None``.
    mpm_order:
        MPM model order used for the fit overlay.
    show_fit:
        If ``True`` (default), overlay the MPM-reconstructed fit on each trace.
        Set to ``False`` to show raw data only.
    components:
        If ``True``, draw the fit's exponential terms over each trace instead
        of their sum, ``Re[A_k exp(p_k t)]`` with the amplitudes solved on that
        trace (the terms add up to the fit), as curves named and coloured like
        the spectrum panels: e_1, e_2, … = canonical slots 0, 1, … in C0, C1, …
        (``exponent_color``); a conjugate pair is one curve, its real sum (a
        damped cosine), "e_1+e_2" in e_1's colour; the data turn grey.  On a
        log axis a term is drawn as its magnitude, dashed where it is negative.
        The y range covers the data and the terms (on a log axis down to at
        most 3 decades below the data).
    fixed_poles:
        Optional mapping ``lambda -> complex poles``.  When a slider λ is
        present here, the overlay uses these poles (amplitudes re-solved
        linearly) instead of re-running MPM — e.g. the refined/smoothed
        poles from an analysis result.
    colors:
        Optional mapping ``key -> matplotlib color`` overriding the default
        per-axis family shades — use it when observables sharing one axis
        need clearly distinct hues.

    Returns
    -------
    matplotlib.widgets.Slider
        The slider widget.  Callers running outside an interactive session
        should keep this reference alive (it is also stored on the figure);
        matplotlib only weak-references widget callbacks, so a dropped
        Slider silently stops responding.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.widgets import Slider
    from mpm_analysis.analysis.matrix_pencil import (
        matrix_pencil_method,
        solve_linear_amplitudes,
        reconstruct_signal,
    )
    from mpm_analysis.analysis.pole_sorting import sort_poles_canonical
    from mpm_analysis.data_types.observables import group_records_by_lambda

    keys = [observable_key] if isinstance(observable_key, str) else list(observable_key)
    labels = labels or {}
    if scales is None and len(keys) == 1:
        scales = {keys[0]: "log" if log_scale else "linear"}
    scales = scales or {}
    scale_of = {k: scales.get(k, "linear") for k in keys}

    groups = group_records_by_lambda(records, keys)
    if not groups:
        raise ValueError(f"No records with observable_key(s) {keys}.")

    def _signals(idx):
        _, t, Y, _ = groups[idx]
        return t, (np.abs(Y) if take_abs else Y)

    fit_cache: dict[int, tuple | None] = {}

    def _get_fit(idx):
        """Joint MPM fit for slider index: (poles, amplitudes (k, M)) — common
        poles, per-key amplitudes."""
        if idx not in fit_cache:
            lam, t, Y, _ = groups[idx]
            if take_abs:
                Y = np.abs(Y)
            try:
                fixed = fixed_poles.get(lam) if fixed_poles is not None else None
                if fixed is not None and np.all(np.isfinite(fixed)):
                    poles = np.asarray(fixed, dtype=complex)      # the result's slot order
                else:
                    poles, _, _ = matrix_pencil_method(t, Y, order=mpm_order)
                    poles = sort_poles_canonical(poles)            # slots as in the result
                fit_cache[idx] = (poles, np.atleast_2d(solve_linear_amplitudes(t, Y, poles)))
            except Exception:
                fit_cache[idx] = None
        return fit_cache[idx]

    def _get_fit_curves(idx):
        """The summed fit per key, shape (k, N), or None."""
        fit = _get_fit(idx)
        return None if fit is None else np.atleast_2d(reconstruct_signal(groups[idx][1], *fit))

    def _get_components(idx):
        """The fit's terms per key: [(slot, legend label, Re[A_k exp(p_k t)])],
        a conjugate pair as one real term ("e_1+e_2", slot of its first pole);
        None when the fit failed."""
        fit = _get_fit(idx)
        if fit is None:
            return None
        poles, amps = fit
        phi = np.exp(np.outer(groups[idx][1], poles))           # (N, M)
        terms, used = [], set()
        for a in range(len(poles)):
            if a in used or not np.isfinite(poles[a]):
                continue
            b = next((b for b in range(a + 1, len(poles)) if b not in used
                      and abs(poles[a].imag) > 1e-9
                      and abs(poles[b] - np.conj(poles[a])) <= 2e-2 * (1 + abs(poles[a]))), None)
            terms.append([a] if b is None else [a, b])
            used.update(terms[-1])
        return [[(m[0], "+".join(exponent_label(s) for s in m).replace("$+$", "+"),
                  np.real(phi[:, m] @ amp[m])) for m in terms]
                for amp in amps]

    def _ylim_for(sig, scale):
        finite = sig[np.isfinite(sig)]
        if finite.size == 0:
            return None
        if scale == "log":
            pos = finite[finite > 0]
            if pos.size == 0:
                return None
            return pos.min() * 0.5, pos.max() * 2.0
        span = finite.max() - finite.min()
        margin = 0.12 * span if span > 0 else 0.1
        return finite.min() - margin, finite.max() + margin

    # --- Axes: one per distinct scale, in first-appearance order -----------
    axis_scales = list(dict.fromkeys(scale_of[k] for k in keys))
    axis_keys = {sc: [k for k in keys if scale_of[k] == sc] for sc in axis_scales}

    fig, ax_host = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.22)

    axes: dict[str, "plt.Axes"] = {}
    for j, sc in enumerate(axis_scales):
        if j == 0:
            axis = ax_host
        else:
            axis = ax_host.twinx()
            if j >= 2:                       # offset a 3rd+ axis outward
                axis.spines["right"].set_position(("outward", 55 * (j - 1)))
        axis.set_yscale(sc)
        axes[sc] = axis

    # --- Colors: one sequential family per axis, shades per observable ----
    _FAMILIES = ["Blues", "Oranges", "Greens", "Purples"]
    key_color = {}
    for j, sc in enumerate(axis_scales):
        cmap = plt.get_cmap(_FAMILIES[j % len(_FAMILIES)])
        ks = axis_keys[sc]
        shades = cmap(np.linspace(0.55, 0.9, len(ks))) if len(ks) > 1 else [cmap(0.75)]
        for k, shade in zip(ks, shades):
            key_color[k] = shade
        if components:      # the terms take C0, C1, … (exponent_color): data in greys
            for n, k in enumerate(ks):
                key_color[k] = str(0.15 + 0.3 * (n % 3))
        if colors:
            for k in ks:
                if k in colors:
                    key_color[k] = colors[k]
        axis = axes[sc]
        axis_color = cmap(0.8)
        axis.set_ylabel(", ".join(labels.get(k, k) for k in ks), fontsize=11,
                        color=axis_color)
        axis.tick_params(axis="y", colors=axis_color)
        spine = "left" if j == 0 else "right"
        axis.spines[spine].set_color(axis_color)

    # --- Initial draw ------------------------------------------------------
    draw_fit = show_fit and not components      # components replace the summed fit
    lam0, _, _, _ = groups[0]
    t0, Y0 = _signals(0)
    fits0 = _get_fit_curves(0) if draw_fit else None

    scatter_artists = {}
    fit_lines = {}
    for i, k in enumerate(keys):
        axis = axes[scale_of[k]]
        scatter_artists[k] = axis.scatter(
            t0, Y0[i],
            color=key_color[k], s=20, marker="o", linewidths=0,
            label=labels.get(k, k), zorder=3,
        )
        fit_x = t0 if fits0 is not None else []
        fit_y = fits0[i] if fits0 is not None else []
        fit_lines[k], = axis.plot(
            fit_x, fit_y,
            color=key_color[k], lw=1.5, linestyle="--", zorder=5,
        )
        fit_lines[k].set_visible(draw_fit)

    # Component artists per key, made on first use: (solid line, dashed line where < 0)
    comp_lines: dict[str, list] = {k: [] for k in keys}

    def _comp_artists(k, c):
        while len(comp_lines[k]) <= c:
            line, = axes[scale_of[k]].plot([], [], lw=1.6, zorder=4.5)
            neg, = axes[scale_of[k]].plot([], [], lw=1.6, ls="--", zorder=4)
            comp_lines[k].append((line, neg))
        return comp_lines[k][c]

    def _runs(mask):
        """``mask`` widened by one sample on each side, so the solid and dashed
        stretches of a term meet instead of leaving gaps."""
        return mask | np.r_[mask[1:], False] | np.r_[False, mask[:-1]]

    def _draw_components(idx):
        """Set the component curves of slider index idx; returns the components."""
        comps = _get_components(idx) if components else None
        t = groups[idx][1]
        for i, k in enumerate(keys):
            items = comps[i] if comps else []
            for c, (slot, _, y) in enumerate(items):
                line, neg = _comp_artists(k, c)
                line.set_color(exponent_color(slot))
                neg.set_color(exponent_color(slot))
                if scale_of[k] == "log":     # magnitude, dashed where the term is negative
                    line.set_data(t, np.where(_runs(y >= 0), np.abs(y), np.nan))
                    neg.set_data(t, np.where(_runs(y < 0), np.abs(y), np.nan))
                else:
                    line.set_data(t, y)
                    neg.set_data([], [])
                line.set_visible(True)
                neg.set_visible(True)
            for line, neg in comp_lines[k][len(items):]:
                line.set_visible(False)
                neg.set_visible(False)
        return comps

    title = ax_host.set_title(f"λ = {lam0:.4f}  (1 / {len(groups)})", fontsize=12)
    ax_host.set_xlabel(r"Time ($\mu$s)", fontsize=11)
    ax_host.grid(True, alpha=0.3, linewidth=0.5)
    ax_host.set_xlim(t0.min(), t0.max())

    def _rescale_axes(t, Y, comps=None):
        for sc in axis_scales:
            idxs = [keys.index(k) for k in axis_keys[sc]]
            data = np.concatenate([Y[i] for i in idxs])
            lim = _ylim_for(data, sc)
            if lim and comps:
                # the range covers the terms too: above (large, cancelling
                # amplitudes near an EP) and below — on a log axis at most 3
                # decades below the data, or a fast term would flatten the plot
                ext = np.concatenate([y for i in idxs for *_, y in comps[i]])
                ext = np.abs(ext[np.isfinite(ext)]) if sc == "log" else ext[np.isfinite(ext)]
                if sc == "log":
                    ext = ext[ext > 0]
                if ext.size and sc == "log":
                    lim = (max(min(lim[0], ext.min() * 0.5), lim[0] * 1e-3),
                           max(lim[1], ext.max() * 2.0))
                elif ext.size:
                    lim = _ylim_for(np.concatenate([data, ext]), sc)
            if lim:
                axes[sc].set_ylim(*lim)

    def _legend(comps=None):
        # Single legend on the host axis; twin-axis artists are legal there.
        handles = [scatter_artists[k] for k in keys]
        legend_labels = [labels.get(k, k) for k in keys]
        if draw_fit:
            handles.append(Line2D([], [], color="0.4", lw=1.5, linestyle="--"))
            legend_labels.append(
                f"LSQ fit (order {mpm_order})" if fixed_poles
                else f"MPM fit (order {mpm_order})"
            )
        if comps:
            for i, k in enumerate(keys):
                for c, (slot, label, _) in enumerate(comps[i]):
                    handles.append(Line2D([], [], color=exponent_color(slot), lw=1.6))
                    legend_labels.append(label if len(keys) == 1 else f"{labels.get(k, k)}: {label}")
            if any(scale_of[k] == "log" and (y < 0).any()
                   for i, k in enumerate(keys) for *_, y in comps[i]):
                handles.append(Line2D([], [], color="0.4", lw=1.6, ls="--"))
                legend_labels.append("< 0")
        ax_host.legend(handles, legend_labels, fontsize=10)

    comps0 = _draw_components(0)
    _rescale_axes(t0, Y0, comps0)
    _legend(comps0)

    # fig.add_axes, not plt.axes: never depends on which figure is "current"
    ax_sl = fig.add_axes([0.18, 0.08, 0.64, 0.03])
    slider = Slider(ax_sl, "λ index", 0, len(groups) - 1, valinit=0, valstep=1)

    def _update(val):
        idx = int(slider.val)
        lam = groups[idx][0]
        t, Y = _signals(idx)
        fits = _get_fit_curves(idx) if draw_fit else None
        for i, k in enumerate(keys):
            scatter_artists[k].set_offsets(np.column_stack((t, Y[i])))
            if draw_fit:
                if fits is not None:
                    fit_lines[k].set_data(t, fits[i])
                else:
                    fit_lines[k].set_data([], [])
        comps = _draw_components(idx)
        ax_host.set_xlim(t.min(), t.max())
        _rescale_axes(t, Y, comps)
        if components:
            _legend(comps)
        title.set_text(f"λ = {lam:.4f}  ({idx + 1} / {len(groups)})")
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    # Matplotlib's CallbackRegistry keeps only *weak* references to widget
    # callbacks, so a Slider that nobody holds is garbage-collected as soon as
    # this function returns — it still draws, but dragging it does nothing.
    # Anchor it to the figure (alive as long as the window) and return it.
    fig._mpm_slider = slider

    plt.show(block=False)
    return slider


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

