"""
base.py
-------
Abstract base class for analysis pipeline steps.

Each step receives a shared ``state`` dict and a list of ``ObservableRecord``
objects, modifies the state, and returns it.  The pipeline runs steps in
sequence, passing state from one to the next.

Conventions for the state dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After ``MPMStep``:
    state["poles_raw"]       : list of np.ndarray, one per lambda slice
    state["L_used"]          : list of int

After ``BootstrapStep``:
    state["boot_results"]    : list[_LambdaBootstrapResult]
    state["poles_result"]    : PolesResult  (aggregated)

After ``RefineStep``:
    state["poles_refined"]   : list of np.ndarray (refined per-lambda poles)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from zeno_analysis.data_types.observables import ObservableRecord


class AnalysisStep(ABC):
    """Base class for analysis pipeline steps.

    Each concrete step implements ``process()``, which transforms the pipeline
    state dict.  Steps should be stateless with respect to data — all
    configuration belongs in ``__init__``.
    """

    @abstractmethod
    def process(
        self,
        records: list[ObservableRecord],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Process ``records`` given current ``state``, return updated state.

        Parameters
        ----------
        records:
            Full list of observable records (all lambdas, all observables).
        state:
            Shared mutable state dict.  Read inputs left by previous steps,
            write outputs for subsequent steps.

        Returns
        -------
        dict
            Updated state dict (same object, returned for convenience).
        """

    def description(self) -> str:
        """Human-readable description for logging and result metadata."""
        return self.__class__.__name__
