"""
bootstrap_step.py
-----------------
Pipeline step: run bootstrap uncertainty quantification.
"""
from __future__ import annotations

from typing import Any

from zeno_analysis.analysis.steps.base import AnalysisStep
from zeno_analysis.data_types.observables import ObservableRecord


class BootstrapStep(AnalysisStep):
    """Run bootstrap and aggregate results into a ``PolesResult``.

    Reads ``state["observable_key"]`` set by ``MPMStep``.

    After this step, ``state`` contains:
        ``"boot_results"``  : list[_LambdaBootstrapResult]
        ``"poles_result"``  : PolesResult

    Parameters
    ----------
    n_boot:
        Number of bootstrap resamples per lambda slice.
    order:
        Model order.  If ``None``, uses ``state["poles_raw"][0].shape[0]``.
    seed:
        RNG seed for reproducibility.  ``None`` → random.
    """

    def __init__(
        self,
        n_boot: int = 2000,
        order: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_boot = n_boot
        self.order = order
        self.seed = seed

    def process(
        self,
        records: list[ObservableRecord],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        import numpy as np
        from zeno_analysis.analysis.bootstrap import run_bootstrap, aggregate_bootstrap

        obs_key = state.get("observable_key")
        if obs_key is None:
            if not records:
                raise ValueError(
                    "BootstrapStep: records is empty and state has no 'observable_key'."
                )
            obs_key = records[0].observable_key

        order = self.order
        if order is None:
            raw = state.get("poles_raw")
            order = int(raw[0].shape[0]) if raw else 3

        rng = np.random.default_rng(self.seed)
        L_list = state.get("L_used")
        # Use the first L as a representative (all should be similar)
        L_repr = int(L_list[0]) if L_list else None

        boot_results = run_bootstrap(
            records,
            order=order,
            n_boot=self.n_boot,
            observable_key=obs_key,
            L=L_repr,
            rng=rng,
        )
        poles_result = aggregate_bootstrap(boot_results, observable_key=obs_key)

        state["boot_results"] = boot_results
        state["poles_result"] = poles_result
        return state

    def description(self) -> str:
        return f"BootstrapStep(n_boot={self.n_boot}, seed={self.seed})"
