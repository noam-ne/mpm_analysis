"""
physics.py
----------
Pure-Python / numpy physics helper functions for the Zeno experiment.

These were previously duplicated across ``ZenoPlotting/base.py`` and various
analysis scripts.  No heavy imports — numpy only.
"""
from __future__ import annotations

import numpy as np


def estimate_lambda_zeno(
    omega_bg: float,
    omega_s: float,
) -> float:
    """Estimate the dimensionless Zeno parameter λ = α / (2 Ω_S).

    Here ``alpha = omega_bg`` is the bare coupling rate (in the same units as
    ``omega_s``).

    Parameters
    ----------
    omega_bg:
        Bare coupling / drive amplitude α (e.g. in MHz · 2π or rad/µs).
    omega_s:
        Rabi frequency of the sensor Ω_S (same units).

    Returns
    -------
    float
        λ = omega_bg / (2 * omega_s)
    """
    return float(omega_bg / (2.0 * omega_s))


def gamma_m(
    T_D: float,
    T1: float,
) -> float:
    """Effective measurement-induced dephasing rate Γ_m.

    Γ_m = 1/T_D + 1/(2 T₁)

    Parameters
    ----------
    T_D:
        Pure dephasing time (µs or same units as T₁).
    T1:
        Energy relaxation time T₁ (µs or same units as T_D).

    Returns
    -------
    float
        Γ_m in units of (1 / time unit used for T_D and T₁).
    """
    return 1.0 / T_D + 1.0 / (2.0 * T1)


def get_theta_plus(lambda_val: float) -> float:
    """Theoretical steady-state Bloch-sphere polar angle θ₊(λ).

    For λ < 1 (Zeno regime):  θ₊ = arccos(1 − 2λ²)  (normalised)
    For λ ≥ 1:                θ₊ = π  (fully mixed / no Zeno protection)

    This is an approximation valid for the idealised three-state model.
    """
    if lambda_val >= 1.0:
        return float(np.pi)
    return float(np.arccos(1.0 - 2.0 * lambda_val**2))


def p_infinity(lambda_val: float) -> float:
    """Theoretical long-time survival probability P_∞(λ).

    P_∞ = cos²(θ₊ / 2)

    This saturates to 0.5 at λ = 1 and drops below for λ > 1 in the
    three-state effective model.
    """
    theta = get_theta_plus(lambda_val)
    return float(np.cos(theta / 2.0) ** 2)


def convert_omega_bg_to_alpha(
    omega_bg_khz: float,
    *,
    units: str = "rad_us",
) -> float:
    """Convert a drive amplitude from kHz to the requested angular frequency units.

    Parameters
    ----------
    omega_bg_khz:
        Drive amplitude in kHz (as stored in experiment filenames).
    units:
        Output units — ``"rad_us"`` (rad/µs, default) or ``"mhz_2pi"`` (MHz·2π).

    Returns
    -------
    float
    """
    # kHz → rad/µs:  1 kHz = 2π × 10³ rad/s = 2π × 10⁻³ rad/µs
    alpha_rad_us = 2.0 * np.pi * omega_bg_khz * 1e-3
    if units == "rad_us":
        return float(alpha_rad_us)
    if units == "mhz_2pi":
        return float(omega_bg_khz * 1e-3)  # kHz → MHz
    raise ValueError(f"Unknown units '{units}'. Use 'rad_us' or 'mhz_2pi'.")
