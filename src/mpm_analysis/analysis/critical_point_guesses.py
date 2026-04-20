"""
critical_point_guesses.py
-------------------------
Pre-calibrated initial guesses for the sqrt critical-point fit.

Usage
-----
    from mpm_analysis.analysis.critical_point_guesses import get_guess

    guess = get_guess("ens_avg_experimental")
    fit = pipeline.fit(
        lambda_c_guess=guess["lambda_c"],
        a_guess=guess["a"],
        b_guess=guess["b"],
    )

Or pass the preset key directly when pipelines support it::

    fit = pipeline.fit(preset="ens_avg_experimental")

Available presets
-----------------
``"post_selected_experimental"``
    Post-selected tomography data from the lab (HMM-calibrated λ scale).
    λ_c ≈ 1.023  (from ``ex_critical_point_fit.py`` "first" config).

``"ens_avg_experimental"``
    Ensemble-average data, scan_experiment_results_50.
    λ_c ≈ 1.1    (from ``ex_critical_point_fit.py`` "third" config).

``"post_selected_simulation"``
    Non-noisy ``PostSelectedDynamicsSimulator`` output (λ_c set by constructor).
    λ_c ≈ 1.3    (default simulator range midpoint).
"""
from __future__ import annotations

_PRESETS: dict[str, dict[str, float]] = {
    # First transition (post-selected tomography, HMM-calibrated λ scale)
    # a / b from critical_points_analysis.py: FIRST_TRANSITION_A_GUESS = 1.12, _B = 0.5
    "post_selected_experimental": dict(lambda_c=1.0,  a=1.12,  b=0.5),
    # Third transition (ensemble-average, scan_experiment_results_50)
    # a / b from object_oriented_poles.py: THIRD_TRANSITION_A_GUESS = 0.62, _B = -0.15
    "ens_avg_experimental":       dict(lambda_c=1.15,  a=0.54,  b=0.09),
    "post_selected_simulation":   dict(lambda_c=1.3,  a=1.0,   b=0.0),
}


def get_guess(preset: str) -> dict[str, float]:
    """Return the initial-guess dict for a named preset.

    Parameters
    ----------
    preset:
        One of ``"post_selected_experimental"``, ``"ens_avg_experimental"``,
        ``"post_selected_simulation"``.

    Returns
    -------
    dict with keys ``"lambda_c"``, ``"a"``, ``"b"``.

    Raises
    ------
    KeyError
        If ``preset`` is not a known key.
    """
    if preset not in _PRESETS:
        raise KeyError(
            f"Unknown critical-point preset '{preset}'. "
            f"Available: {list(_PRESETS)}."
        )
    return dict(_PRESETS[preset])  # return a copy so callers can't mutate the registry


def list_presets() -> list[str]:
    """Return the list of available preset names."""
    return list(_PRESETS)
