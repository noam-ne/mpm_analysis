"""
bootstrap.py
------------
Bootstrap uncertainty quantification for the MPM pole pipeline.

Strategy: for each lambda slice, fit a clean model to the data, resample
residuals, and re-extract poles.  Collecting n_boot pole estimates gives an
empirical distribution from which we compute median ± asymmetric uncertainties.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.data_types.poles import PolesResult
from mpm_analysis.data_types.pole_samples import PoleSamples
from mpm_analysis.analysis.pole_sorting import sort_poles_canonical, enforce_pole_count


# ---------------------------------------------------------------------------
# Internal result type (per-lambda bootstrap draws)
# ---------------------------------------------------------------------------

@dataclass
class _LambdaBootstrapResult:
    """Bootstrap distribution for a single lambda slice."""
    lambda_val: float
    boot_real: np.ndarray   # (n_boot, order)
    boot_imag: np.ndarray   # (n_boot, order)


# ---------------------------------------------------------------------------
# Core bootstrap function
# ---------------------------------------------------------------------------

def run_bootstrap(
    records: list[ObservableRecord],
    order: int = 3,
    n_boot: int = 1000,
    observable_key: str | None = None,
    *,
    L: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[_LambdaBootstrapResult]:
    """Run bootstrap pole extraction for each lambda slice.

    Parameters
    ----------
    records:
        List of ``ObservableRecord`` objects.
    order:
        Number of poles to extract.
    n_boot:
        Number of bootstrap resamples per lambda slice.
    observable_key:
        Which observable to analyse.  If ``None``, derived from first record.
    L:
        Pencil parameter override.
    rng:
        Numpy random generator (``None`` → fresh default RNG).

    Returns
    -------
    list[_LambdaBootstrapResult]
        One result per lambda slice, sorted by lambda.
    """
    from mpm_analysis.analysis.matrix_pencil import (
        matrix_pencil_method,
        solve_linear_amplitudes,
        reconstruct_signal,
    )

    if rng is None:
        rng = np.random.default_rng()

    if observable_key is None:
        if not records:
            raise ValueError("records is empty; cannot derive observable_key.")
        observable_key = records[0].observable_key

    slices = [r for r in records if r.observable_key == observable_key]
    slices.sort(key=lambda r: r.lambda_val)

    results: list[_LambdaBootstrapResult] = []

    for rec in slices:
        t = rec.t
        y = rec.signal

        poles_clean, _, L_used = matrix_pencil_method(t, y, order, L=L)
        amps_clean = solve_linear_amplitudes(t, y, poles_clean)
        y_clean = reconstruct_signal(t, poles_clean, amps_clean)
        residuals = y - y_clean

        boot_real = np.zeros((n_boot, order), dtype=float)
        boot_imag = np.zeros((n_boot, order), dtype=float)

        for b in range(n_boot):
            y_boot = y_clean + rng.choice(residuals, size=len(residuals), replace=True)
            poles_b, _, _ = matrix_pencil_method(t, y_boot, order, L=L_used)
            poles_b = sort_poles_canonical(enforce_pole_count(poles_b, order))
            boot_real[b] = poles_b.real
            boot_imag[b] = poles_b.imag

        results.append(
            _LambdaBootstrapResult(
                lambda_val=rec.lambda_val,
                boot_real=boot_real,
                boot_imag=boot_imag,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_bootstrap(
    boot_results: list[_LambdaBootstrapResult],
    *,
    percentile_lo: float = 16.0,
    percentile_hi: float = 84.0,
    observable_key: str = "survival",
) -> PolesResult:
    """Aggregate bootstrap distributions into a ``PolesResult``.

    Parameters
    ----------
    boot_results:
        Output of ``run_bootstrap``.
    percentile_lo, percentile_hi:
        Percentiles for lower/upper uncertainty bounds.  Default 16/84 ≈ ±1σ.

    Returns
    -------
    PolesResult
    """
    n_lam = len(boot_results)
    order = boot_results[0].boot_real.shape[1] if boot_results else 0

    decay_med = np.zeros((n_lam, order))
    decay_lo  = np.zeros((n_lam, order))
    decay_hi  = np.zeros((n_lam, order))
    freq_med  = np.zeros((n_lam, order))
    freq_lo   = np.zeros((n_lam, order))
    freq_hi   = np.zeros((n_lam, order))

    for i, br in enumerate(boot_results):
        valid_real = br.boot_real[~np.isnan(br.boot_real).any(axis=1)]
        valid_imag = br.boot_imag[~np.isnan(br.boot_imag).any(axis=1)]

        if len(valid_real) == 0:
            decay_med[i] = decay_lo[i] = decay_hi[i] = np.nan
            freq_med[i]  = freq_lo[i]  = freq_hi[i]  = np.nan
            continue

        # Decay rates: -Re(pole)  (positive = decaying)
        dr = -valid_real
        decay_med[i] = np.median(dr, axis=0)
        decay_lo[i]  = decay_med[i] - np.percentile(dr, percentile_lo, axis=0)
        decay_hi[i]  = np.percentile(dr, percentile_hi, axis=0) - decay_med[i]

        # Frequencies: Im(pole) — keep signed so conjugate pairs appear at ±ω
        fr = valid_imag
        freq_med[i] = np.median(fr, axis=0)
        freq_lo[i]  = freq_med[i] - np.percentile(fr, percentile_lo, axis=0)
        freq_hi[i]  = np.percentile(fr, percentile_hi, axis=0) - freq_med[i]

    # Compute splittings between poles 0 and 1
    (rs_med, rs_lo, rs_hi,
     is_med, is_lo, is_hi) = _compute_splitting(boot_results, percentile_lo, percentile_hi)

    # Raw samples container: (n_lam, n_boot, order) complex
    lambda_values = np.array([br.lambda_val for br in boot_results])
    raw_samples_arr = np.stack(
        [br.boot_real + 1j * br.boot_imag for br in boot_results], axis=0
    )
    raw_samples = PoleSamples(
        samples=raw_samples_arr,
        observable_key=observable_key,
        lambda_values=lambda_values,
    )

    return PolesResult(
        decay_rates=decay_med,
        decay_rates_lower=np.abs(decay_lo),
        decay_rates_upper=np.abs(decay_hi),
        frequencies=freq_med,
        frequencies_lower=np.abs(freq_lo),
        frequencies_upper=np.abs(freq_hi),
        imag_splitting=is_med,
        imag_splitting_lower=np.abs(is_lo),
        imag_splitting_upper=np.abs(is_hi),
        real_splitting=rs_med,
        real_splitting_lower=np.abs(rs_lo),
        real_splitting_upper=np.abs(rs_hi),
        raw_samples=raw_samples,
    )


def _compute_splitting(
    boot_results: list[_LambdaBootstrapResult],
    plo: float,
    phi: float,
) -> tuple[np.ndarray, ...]:
    """Compute real and imaginary pole splittings between poles 0 and 1.

    Poles are sorted by real-part ascending (most negative first = fastest
    decay), so the two EP poles occupy indices 0 and 1.  Splittings are
    absolute values of the difference — absorbs sign ambiguity for conjugate
    pairs and across the EP where pole identities swap.

    Returns
    -------
    (rs_med, rs_lo, rs_hi, is_med, is_lo, is_hi)
        Six 1-D arrays of shape (n_lambda,).
    """
    n = len(boot_results)
    rs_med = np.zeros(n); rs_lo = np.zeros(n); rs_hi = np.zeros(n)
    is_med = np.zeros(n); is_lo = np.zeros(n); is_hi = np.zeros(n)

    for i, br in enumerate(boot_results):
        valid_real = br.boot_real[~np.isnan(br.boot_real).any(axis=1)]
        valid_imag = br.boot_imag[~np.isnan(br.boot_imag).any(axis=1)]

        if valid_real.shape[1] < 2 or len(valid_real) == 0:
            rs_med[i] = rs_lo[i] = rs_hi[i] = np.nan
            is_med[i] = is_lo[i] = is_hi[i] = np.nan
            continue

        # |Re(pole₁ − pole₀)| / 2  — poles 0 and 1 (EP pair)
        rs = np.abs(valid_real[:, 1] - valid_real[:, 0]) / 2.0
        rs_med[i] = np.median(rs)
        rs_lo[i]  = rs_med[i] - np.percentile(rs, plo)
        rs_hi[i]  = np.percentile(rs, phi) - rs_med[i]

        # |Im(pole₁ − pole₀)| / 2  — same pair
        imag_s = np.abs(valid_imag[:, 1] - valid_imag[:, 0]) / 2.0
        is_med[i] = np.median(imag_s)
        is_lo[i]  = is_med[i] - np.percentile(imag_s, plo)
        is_hi[i]  = np.percentile(imag_s, phi) - is_med[i]

    return rs_med, rs_lo, rs_hi, is_med, is_lo, is_hi
