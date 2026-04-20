"""base.py — Abstract base for noise models."""
from __future__ import annotations

from abc import ABC, abstractmethod

from zeno_analysis.data_types.observables import ObservableRecord


class NoiseModel(ABC):
    """Abstract noise model.

    All concrete noise models are stateless with respect to data.
    Configuration (noise level etc.) lives in ``__init__``.
    """

    @abstractmethod
    def apply(self, records: list[ObservableRecord]) -> list[ObservableRecord]:
        """Return a new list of records with noise applied.

        The original records are not modified.
        """

    def description(self) -> dict:
        """Return a JSON-serialisable dict describing this noise model."""
        return {"type": self.__class__.__name__}
