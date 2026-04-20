"""
observables.py
--------------
Universal intermediate data type.  Every data source (experimental JSON folder,
analytical NPZ, Monte Carlo simulation) produces ``ObservableRecord`` objects.
Every analysis step consumes them.

Design goals
~~~~~~~~~~~~
* Only numpy as a dependency.
* A record captures one lambda slice of one observable.
* ``observable_key`` makes the type extensible: adding a new observable type
   requires only a new loader — not a new class.

Typed aliases
~~~~~~~~~~~~~
``SurvivalRecord`` and ``ZRecord`` are just ``ObservableRecord`` instances with
a specific ``observable_key``.  They are provided as factory functions so that
call sites read clearly:

    rec = SurvivalRecord(lambda_val=0.4, t=t_us, signal=P_survival)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ObservableRecord:
    """One lambda slice of one measured/simulated observable.

    Parameters
    ----------
    lambda_val:
        Dimensionless Zeno parameter λ for this slice.
    t:
        Time axis in **microseconds** (always µs — loaders must normalise).
    signal:
        Observable values, same shape as ``t``.  Normalised to [0, 1] for
        survival probability; arbitrary units for other observables.
    observable_key:
        String tag identifying the observable, e.g. ``"survival"``, ``"Z"``,
        ``"click_rate"``.  Used by analysis steps to select the right signal.
    source_file:
        Original filename (for traceability).  May be empty.
    metadata:
        Free-form dict for any additional per-record context (e.g. navg,
        false-positive rates, omega_DG).
    """

    lambda_val: float
    t: np.ndarray
    signal: np.ndarray
    observable_key: str
    source_file: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.signal = np.asarray(self.signal, dtype=float)
        


    @property
    def n_points(self) -> int:
        return len(self.t)

    def downsample(self, factor: int) -> "ObservableRecord":
        """Return a new record with every ``factor``-th point kept."""
        return ObservableRecord(
            lambda_val=self.lambda_val,
            t=self.t[::factor],
            signal=self.signal[::factor],
            observable_key=self.observable_key,
            source_file=self.source_file,
            metadata=self.metadata,
        )

    def time_slice(self, t_min: float | None = None, t_max: float | None = None) -> "ObservableRecord":
        """Return a new record restricted to ``[t_min, t_max]`` µs."""
        mask = np.ones(len(self.t), dtype=bool)
        if t_min is not None:
            mask &= self.t >= t_min
        if t_max is not None:
            mask &= self.t <= t_max
        return ObservableRecord(
            lambda_val=self.lambda_val,
            t=self.t[mask],
            signal=self.signal[mask],
            observable_key=self.observable_key,
            source_file=self.source_file,
            metadata=self.metadata,
        )


# ---------------------------------------------------------------------------
# Factory helpers — typed aliases with a clear call site
# ---------------------------------------------------------------------------

def SurvivalRecord(
    lambda_val: float,
    t: np.ndarray,
    signal: np.ndarray,
    source_file: str = "",
    metadata: dict | None = None,
) -> ObservableRecord:
    """Factory: an ObservableRecord with observable_key='survival'."""
    return ObservableRecord(
        lambda_val=lambda_val,
        t=t,
        signal=signal,
        observable_key="survival",
        source_file=source_file,
        metadata=metadata or {},
    )


def ZRecord(
    lambda_val: float,
    t: np.ndarray,
    signal: np.ndarray,
    source_file: str = "",
    metadata: dict | None = None,
) -> ObservableRecord:
    """Factory: an ObservableRecord with observable_key='Z'."""
    return ObservableRecord(
        lambda_val=lambda_val,
        t=t,
        signal=signal,
        observable_key="Z",
        source_file=source_file,
        metadata=metadata or {},
    )
