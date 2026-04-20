"""
pole_samples.py
---------------
Container for raw bootstrap pole samples, exposing the full
(n_lambda, n_boot, order) complex array for downstream analysis.

The median/uncertainty view lives in ``PolesResult``; this container stores
the raw draws so callers can apply any aggregation they like (e.g. ksi
crossing, histogram inspection, custom percentiles).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoleSamples:
    """Raw complex pole samples from the bootstrap.

    Attributes
    ----------
    samples:
        Complex array of shape ``(n_lambda, n_boot, order)``.
        Poles are sorted by real part ascending within each bootstrap draw
        (canonical order matching ``sort_poles_canonical``).
    observable_key:
        The observable key used to produce these samples.
    lambda_values:
        1-D array of λ values, shape ``(n_lambda,)``.
    """

    samples: np.ndarray        # complex (n_lambda, n_boot, order)
    observable_key: str
    lambda_values: np.ndarray  # float  (n_lambda,)

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=complex)
        self.lambda_values = np.asarray(self.lambda_values, dtype=float)
        if self.samples.ndim != 3:
            raise ValueError(
                f"PoleSamples.samples must be 3-D (n_lambda, n_boot, order), "
                f"got shape {self.samples.shape}."
            )
        if len(self.lambda_values) != self.samples.shape[0]:
            raise ValueError(
                f"lambda_values length {len(self.lambda_values)} does not match "
                f"samples.shape[0] = {self.samples.shape[0]}."
            )

    @property
    def n_lambda(self) -> int:
        return self.samples.shape[0]

    @property
    def n_boot(self) -> int:
        return self.samples.shape[1]

    @property
    def order(self) -> int:
        return self.samples.shape[2]

    def get_lambda(self, idx: int) -> np.ndarray:
        """Return the ``(n_boot, order)`` sample slice for lambda index ``idx``."""
        return self.samples[idx]
