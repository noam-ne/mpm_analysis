"""composite.py — Chain multiple noise models together."""
from __future__ import annotations

from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.simulation.noise.base import NoiseModel


class CompositeNoise(NoiseModel):
    """Apply a sequence of noise models in order.

    Parameters
    ----------
    models:
        Ordered list of ``NoiseModel`` instances to apply in sequence.

    Example
    -------
    >>> noise = CompositeNoise([
    ...     ReadoutError(false_positive=0.02, false_negative=0.03),
    ...     GaussianNoise(sigma=0.01),
    ... ])
    >>> noisy_records = noise.apply(clean_records)
    """

    def __init__(self, models: list[NoiseModel]) -> None:
        self.models = models

    def apply(self, records: list[ObservableRecord]) -> list[ObservableRecord]:
        result = records
        for model in self.models:
            result = model.apply(result)
        return result

    def description(self) -> dict:
        return {
            "type": "CompositeNoise",
            "models": [m.description() for m in self.models],
        }
