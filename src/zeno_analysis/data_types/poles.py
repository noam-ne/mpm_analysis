"""
poles.py
--------
Data container for pole-analysis results.

All arrays follow the convention:
    axis 0  →  lambda index  (n_lambda,)
    axis 1  →  pole index    (order,)   where applicable

Splitting arrays are 1-D (n_lambda,) — one value per lambda point,
computed from the dominant conjugate pole pair (indices 0 and 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from zeno_analysis.data_types.pole_samples import PoleSamples


@dataclass
class PolesResult:
    """Bootstrap-aggregated pole spectrum across a lambda scan.

    All *_lower / *_upper fields are **half-widths** (distance from median to
    the 16th / 84th percentile), not the absolute percentile values.

    Attributes
    ----------
    decay_rates, decay_rates_lower, decay_rates_upper:
        Decay rates −Re(pole).  Positive = decaying.  Shape (n_lambda, order).
    frequencies, frequencies_lower, frequencies_upper:
        Oscillation frequencies Im(pole).  Shape (n_lambda, order).
    imag_splitting, imag_splitting_lower, imag_splitting_upper:
        |Im(pole₁ − pole₀)| / 2 for the dominant conjugate pair.
        Shape (n_lambda,).  None if bootstrap was not run.
    real_splitting, real_splitting_lower, real_splitting_upper:
        |Re(pole₁ − pole₀)| / 2.  Shape (n_lambda,).  None if not computed.
    raw_samples:
        Full bootstrap draws.  Shape (n_lambda, n_boot, order) complex.
    """

    # Required fields — decay rates and frequencies
    decay_rates:       np.ndarray   # (n_lambda, order)
    decay_rates_lower: np.ndarray
    decay_rates_upper: np.ndarray
    frequencies:       np.ndarray   # (n_lambda, order)
    frequencies_lower: np.ndarray
    frequencies_upper: np.ndarray

    # Optional — pole splittings
    imag_splitting:       np.ndarray | None = None   # (n_lambda,)
    imag_splitting_lower: np.ndarray | None = None
    imag_splitting_upper: np.ndarray | None = None
    real_splitting:       np.ndarray | None = None   # (n_lambda,)
    real_splitting_lower: np.ndarray | None = None
    real_splitting_upper: np.ndarray | None = None

    # Optional — raw bootstrap samples
    raw_samples: "PoleSamples | None" = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_lambda(self) -> int:
        return self.decay_rates.shape[0]

    @property
    def order(self) -> int:
        return self.decay_rates.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.decay_rates.shape

    @property
    def has_splitting(self) -> bool:
        return self.imag_splitting is not None

    def __getitem__(self, idx) -> "PolesResult":
        """Mask / slice all arrays simultaneously.

        ``raw_samples`` is dropped on sliced views — it's a PoleSamples object
        (not a plain array) and is only needed on the full unsliced result.
        """
        def _s(a):
            return a[idx] if a is not None else None

        return PolesResult(
            decay_rates=self.decay_rates[idx],
            decay_rates_lower=self.decay_rates_lower[idx],
            decay_rates_upper=self.decay_rates_upper[idx],
            frequencies=self.frequencies[idx],
            frequencies_lower=self.frequencies_lower[idx],
            frequencies_upper=self.frequencies_upper[idx],
            imag_splitting=_s(self.imag_splitting),
            imag_splitting_lower=_s(self.imag_splitting_lower),
            imag_splitting_upper=_s(self.imag_splitting_upper),
            real_splitting=_s(self.real_splitting),
            real_splitting_lower=_s(self.real_splitting_lower),
            real_splitting_upper=_s(self.real_splitting_upper),
            raw_samples=None,
        )
