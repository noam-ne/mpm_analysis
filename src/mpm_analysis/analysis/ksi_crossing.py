"""
ksi_crossing.py
---------------
Ksi (ξ) crossing-point detection from sorted MPM poles.

Background
~~~~~~~~~~
Near the exceptional point (EP), three poles are involved.  After canonical
sorting (real part ascending), poles at indices 0, 1, 2 correspond to:
    index 0  →  most negative real part  (fast decay, one EP pole)
    index 1  →  middle real part          (other EP pole)
    index 2  →  least negative real part  (slow / non-EP pole)

The ksi function computes a ratio between the real parts of poles 1 and 2:

    ξ(p₁, p₂) = (2·Re(p₂) − Re(p₁)) / (Re(p₁) − Re(p₂))

This quantity changes sign at the EP: negative on one side (coalescence), zero
at the EP, positive on the other side.  The zero-crossing λ_ξ is an
alternative estimate of the critical point, complementing the sqrt fit.

Pole indexing note
~~~~~~~~~~~~~~~~~~
Old code used ``pole_2`` / ``pole_3`` for indices 1 / 2 (1-indexed), matching
the notation in the literature.  This module uses 0-indexed arrays throughout.

References
~~~~~~~~~~
Literal port of ``get_ksi``, ``ksi_zero_crossings`` logic from
``object_oriented_poles.py`` lines 2053–2060.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def ksi(p1: complex, p2: complex) -> float:
    """Compute the ξ ratio from two sorted complex poles.

    Parameters
    ----------
    p1:
        Pole at index 1 (second-most decaying after canonical sort).
    p2:
        Pole at index 2 (least decaying).

    Returns
    -------
    float
        ξ = (2·Re(p₂) − Re(p₁)) / (Re(p₁) − Re(p₂))

    Notes
    -----
    Sign convention: ξ < 0 means p₁ and p₂ are separated (before EP);
    ξ → ∞ at coalescence; ξ > 0 after EP.  A zero crossing from negative to
    positive marks the EP.
    """
    denom = p1.real - p2.real + 1e-12  # guard against exact equality
    return float((2.0 * p2.real - p1.real) / denom)


# ---------------------------------------------------------------------------
# Array-level helpers
# ---------------------------------------------------------------------------

def compute_ksi_curve(
    poles_result,
    *,
    pole_idx1: int = 1,
    pole_idx2: int = 2,
) -> np.ndarray:
    """Compute median ξ(λ) from a ``PolesResult``.

    Parameters
    ----------
    poles_result:
        A ``PolesResult`` with ``decay_rates`` of shape (n_lambda, order).
    pole_idx1, pole_idx2:
        Column indices for the two poles (default 1 and 2, matching old code).

    Returns
    -------
    np.ndarray
        ξ values, shape (n_lambda,).  Entries are NaN where data is missing.
    """
    # decay_rates = -Re(pole); Re(pole) = -decay_rates
    dr = poles_result.decay_rates  # (n_lambda, order)
    if dr.shape[1] <= pole_idx2:
        raise ValueError(
            f"poles_result has order={dr.shape[1]}; pole indices "
            f"{pole_idx1} and {pole_idx2} are out of range."
        )
    # Reconstruct real parts (poles are sorted real-ascending, so most negative first)
    real_parts = -dr  # decay_rates = -Re, so Re = -decay_rates

    ksi_vals = np.full(dr.shape[0], np.nan)
    for i in range(dr.shape[0]):
        p1_re = real_parts[i, pole_idx1]
        p2_re = real_parts[i, pole_idx2]
        if np.isnan(p1_re) or np.isnan(p2_re):
            continue
        denom = p1_re - p2_re + 1e-12
        ksi_vals[i] = float((2.0 * p2_re - p1_re) / denom)
    return ksi_vals


def find_zero_crossings(
    lambdas: np.ndarray,
    ksi_values: np.ndarray,
) -> list[float]:
    """Find λ values where ξ crosses zero from negative to positive.

    Uses linear interpolation between adjacent sign changes.

    Parameters
    ----------
    lambdas:
        1-D array of λ values.
    ksi_values:
        1-D array of ξ values (same length, may contain NaN).

    Returns
    -------
    list[float]
        Interpolated λ values of zero crossings.  Empty if none found.
    """
    crossings: list[float] = []
    lam = np.asarray(lambdas, dtype=float)
    ksi = np.asarray(ksi_values, dtype=float)

    for i in range(len(ksi) - 1):
        k0, k1 = ksi[i], ksi[i + 1]
        if np.isnan(k0) or np.isnan(k1):
            continue
        if k0 < 0 < k1:
            # Linear interpolation
            lam_cross = lam[i] + (lam[i + 1] - lam[i]) * (-k0) / (k1 - k0)
            crossings.append(float(lam_cross))
    return crossings


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KsiCrossingResult:
    """Result of the ksi crossing analysis.

    Attributes
    ----------
    lambda_values:
        Lambda axis, shape (n_lambda,).
    ksi_values:
        ξ(λ), shape (n_lambda,).
    zero_crossings:
        List of λ values where ξ crosses zero (EP candidates).
    pole_idx1, pole_idx2:
        Pole indices used for the ksi computation.
    """
    lambda_values: np.ndarray
    ksi_values: np.ndarray
    zero_crossings: list[float]
    pole_idx1: int = 1
    pole_idx2: int = 2

    @property
    def lambda_ep(self) -> float | None:
        """First zero crossing, or ``None`` if no crossing was found."""
        return self.zero_crossings[0] if self.zero_crossings else None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class KsiCrossingPipeline:
    """Compute the ξ crossing from a ``ZenoAnalysisResult``.

    Parameters
    ----------
    result:
        A ``ZenoAnalysisResult`` with at least 3 poles.
    lambda_max:
        If given, restrict the analysis to λ ≤ lambda_max.
    pole_idx1, pole_idx2:
        Pole indices for the ksi ratio (default 1, 2 — matching old code).
    """

    def __init__(
        self,
        result,
        *,
        lambda_max: float | None = None,
        pole_idx1: int = 1,
        pole_idx2: int = 2,
    ) -> None:
        self._result = result
        self._lambda_max = lambda_max
        self._pole_idx1 = pole_idx1
        self._pole_idx2 = pole_idx2

    @classmethod
    def from_result(
        cls,
        result,
        **kwargs,
    ) -> "KsiCrossingPipeline":
        """Convenience factory."""
        return cls(result, **kwargs)

    def run(self) -> KsiCrossingResult:
        """Compute ξ(λ) and find zero crossings.

        Returns
        -------
        KsiCrossingResult
        """
        lam = self._result.lambda_values.copy()
        poles = self._result.poles

        if self._lambda_max is not None:
            mask = lam <= self._lambda_max
            lam = lam[mask]
            poles = poles[mask]

        ksi_vals = compute_ksi_curve(
            poles,
            pole_idx1=self._pole_idx1,
            pole_idx2=self._pole_idx2,
        )
        crossings = find_zero_crossings(lam, ksi_vals)

        if crossings:
            print(
                f"KsiCrossingPipeline: found {len(crossings)} zero crossing(s): "
                + ", ".join(f"λ_EP ≈ {c:.4f}" for c in crossings)
            )
        else:
            print("KsiCrossingPipeline: no zero crossings found.")

        return KsiCrossingResult(
            lambda_values=lam,
            ksi_values=ksi_vals,
            zero_crossings=crossings,
            pole_idx1=self._pole_idx1,
            pole_idx2=self._pole_idx2,
        )
