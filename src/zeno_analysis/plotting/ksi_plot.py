"""
ksi_plot.py
-----------
Plotting helper for ksi (ξ) crossing results.
"""
from __future__ import annotations

import numpy as np


def plot_ksi_curve(
    ksi_result,
    *,
    ax=None,
    color: str = "steelblue",
    crossing_color: str = "crimson",
    show_zero_line: bool = True,
    markersize: float = 4.0,
) -> "plt.Axes":
    """Plot ξ(λ) with zero crossings highlighted.

    Parameters
    ----------
    ksi_result:
        A ``KsiCrossingResult`` from ``KsiCrossingPipeline.run()``.
    ax:
        Target axes.  Created if ``None``.
    color:
        Colour for the ξ curve.
    crossing_color:
        Colour for the vertical lines at zero crossings.
    show_zero_line:
        Whether to draw a horizontal dashed line at ξ = 0.
    markersize:
        Marker size for data points.

    Returns
    -------
    plt.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))

    lam = ksi_result.lambda_values
    ksi = ksi_result.ksi_values

    ax.plot(
        lam, ksi,
        "o-",
        color=color,
        markersize=markersize,
        linewidth=1.2,
        label=r"$\xi(\lambda)$",
    )

    if show_zero_line:
        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)

    for lc in ksi_result.zero_crossings:
        ax.axvline(
            lc,
            color=crossing_color,
            ls=":",
            lw=1.2,
            alpha=0.9,
            label=rf"$\lambda_{{EP}} \approx {lc:.3f}$",
        )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\xi$")
    ax.set_title(
        rf"Ksi crossing  (poles {ksi_result.pole_idx1}, {ksi_result.pole_idx2})"
    )
    ax.legend(fontsize=8)
    ax.grid(False)

    return ax
