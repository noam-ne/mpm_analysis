"""
refine_step.py
--------------
Optional pipeline step: nonlinear refinement of MPM poles.

This step reads ``state["poles_raw"]`` (from ``MPMStep``) and refines
each set of poles using ``analysis.matrix_pencil.refine_poles``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from zeno_analysis.analysis.steps.base import AnalysisStep
from zeno_analysis.data_types.observables import ObservableRecord


class RefineStep(AnalysisStep):
    """Nonlinear refinement of raw MPM poles.

    Reads ``state["poles_raw"]`` and ``state["lambdas"]``, writes
    ``state["poles_raw"]`` in-place with refined values.

    Parameters
    ----------
    bounds_real:
        Optional ``(lower, upper)`` bounds on Re(pole).
    bounds_imag:
        Optional ``(lower, upper)`` bounds on Im(pole).
    observable_key:
        Observable to use for residuals.
    """

    def __init__(
        self,
        bounds_real: tuple[float, float] | None = (-np.inf, 0.0),
        bounds_imag: tuple[float, float] | None = None,
        observable_key: str = "survival",
    ) -> None:
        self.bounds_real = bounds_real
        self.bounds_imag = bounds_imag
        self.observable_key = observable_key

    def process(
        self,
        records: list[ObservableRecord],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        from zeno_analysis.analysis.matrix_pencil import refine_poles

        poles_raw: list = state.get("poles_raw", [])
        lambdas: list = state.get("lambdas", [])

        slices = {
            r.lambda_val: r
            for r in records
            if r.observable_key == self.observable_key
        }

        poles_refined = []
        for lam, poles in zip(lambdas, poles_raw):
            if np.any(np.isnan(poles)):
                poles_refined.append(poles)
                continue
            rec = slices.get(lam)
            if rec is None:
                poles_refined.append(poles)
                continue
            try:
                poles_r = refine_poles(
                    rec.t,
                    rec.signal,
                    poles,
                    bounds_real=self.bounds_real,
                    bounds_imag=self.bounds_imag,
                )
                poles_refined.append(poles_r)
            except Exception as exc:
                print(f"Warning: RefineStep failed for lam={lam:.3f}: {exc}")
                poles_refined.append(poles)

        state["poles_raw"] = poles_refined
        return state

    def description(self) -> str:
        return f"RefineStep(bounds_real={self.bounds_real}, bounds_imag={self.bounds_imag})"
