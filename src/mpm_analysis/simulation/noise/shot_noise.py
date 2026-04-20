"""shot_noise.py — Binomial shot noise for survival-probability signals."""
from __future__ import annotations

import numpy as np

from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.simulation.noise.base import NoiseModel


class ShotNoise(NoiseModel):
    """Binomial shot noise for a survival-probability signal.

    Each time point is treated as an independent Bernoulli experiment with
    success probability ``s(t)``.  For ``n_shots`` repetitions the standard
    error of the estimated probability is::

        sigma_i = sqrt(s_i * (1 - s_i)) / sqrt(n_shots)

    Parameters
    ----------
    n_shots:
        Number of repeated measurements per time point.  Default: 1000.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(self, n_shots: int = 30000, seed: int | None = None) -> None:
        self.n_shots = int(n_shots)
        self.seed = seed

    def apply(self, records: list[ObservableRecord]) -> list[ObservableRecord]:
        rng = np.random.default_rng(self.seed)
        noisy = []
        for r in records:
            s = r.signal
            sigma = np.sqrt(s * (1.0 - s)) / np.sqrt(self.n_shots)
            noisy.append(
                ObservableRecord(
                    lambda_val=r.lambda_val,
                    t=r.t.copy(),
                    signal=r.signal + rng.normal(0.0, sigma, size=r.signal.shape),
                    observable_key=r.observable_key,
                    source_file=r.source_file,
                    metadata=r.metadata,
                )
            )
        return noisy

    def description(self) -> dict:
        return {"type": "ShotNoise", "n_shots": self.n_shots}
