"""
mpm_step.py
-----------
Pipeline step: run the Matrix Pencil Method on each lambda slice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mpm_analysis.analysis.steps.base import AnalysisStep
from mpm_analysis.data_types.observables import ObservableRecord


class MPMStep(AnalysisStep):
    """Extract poles from each lambda slice using the Matrix Pencil Method.

    After this step, ``state`` contains:
        ``"poles_raw"``   : list[np.ndarray]   complex poles per slice
        ``"L_used"``      : list[int]           pencil parameter used per slice
        ``"lambdas"``     : list[float]         lambda values in order
        ``"observable_key"`` : str

    Parameters
    ----------
    order:
        Model order (number of poles).
    L:
        Pencil parameter override.  ``None`` → auto (N//3).
    observable_key:
        Which observable to analyse.
    plot_svd:
        If ``True``, plot the Hankel SVD singular values vs λ after processing.
        Use this as a calibration step to visually select the model order:
        look for a gap between line ``order`` and ``order+1``.
    n_sv:
        Number of singular values to display when ``plot_svd=True``.
    save_dir:
        Directory to save the SVD figure as PNG when ``plot_svd=True``.
        ``None`` → do not save.
    """

    def __init__(
        self,
        order: int = 3,
        L: int | None = None,
        observable_key: str | None = None,
        *,
        plot_svd: bool = False,
        n_sv: int = 10,
        save_dir: Path | str | None = None,
    ) -> None:
        self.order = order
        self.L = L
        self.observable_key = observable_key  # None → derived from first record
        self.plot_svd = plot_svd
        self.n_sv = n_sv
        self.save_dir = Path(save_dir) if save_dir is not None else None

    def process(
        self,
        records: list[ObservableRecord],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        from mpm_analysis.analysis.matrix_pencil import matrix_pencil_method
        from mpm_analysis.analysis.pole_sorting import sort_poles_canonical, enforce_pole_count

        # Derive observable_key from records if not set explicitly
        obs_key = self.observable_key
        if obs_key is None:
            if not records:
                raise ValueError("MPMStep: records is empty; cannot derive observable_key.")
            obs_key = records[0].observable_key

        slices = sorted(
            [r for r in records if r.observable_key == obs_key],
            key=lambda r: r.lambda_val,
        )

        if not slices:
            raise ValueError(
                f"MPMStep: no records found with observable_key='{obs_key}'. "
                f"Available keys: {sorted({r.observable_key for r in records})}."
            )

        poles_raw = []
        L_used = []
        lambdas = []
        sv_list = []

        for rec in slices:
            try:
                poles, sv, L_u = matrix_pencil_method(
                    rec.t, rec.signal, self.order, L=self.L
                )
                poles = sort_poles_canonical(enforce_pole_count(poles, self.order))
                poles_raw.append(poles)
                L_used.append(L_u)
                lambdas.append(rec.lambda_val)
                sv_list.append(sv)
            except Exception as exc:
                print(f"Warning: MPMStep failed for lam={rec.lambda_val:.3f}: {exc}")
                poles_raw.append(np.full(self.order, np.nan, dtype=complex))
                L_used.append(0)
                lambdas.append(rec.lambda_val)
                sv_list.append(np.array([np.nan]))

        if self.plot_svd and sv_list:
            import matplotlib.pyplot as plt
            from mpm_analysis.plotting.exploratory import plot_svd_singular_values

            valid = [(lam, sv) for lam, sv in zip(lambdas, sv_list)
                     if not np.all(np.isnan(sv))]
            if valid:
                lams_v, svs_v = zip(*valid)
                ax = plot_svd_singular_values(
                    np.array(lams_v), list(svs_v),
                    n_sv=self.n_sv,
                    title=f"MPM Hankel SVD — model order calibration (order={self.order})",
                )
                fig = ax.get_figure()
                fig.tight_layout()
                if self.save_dir is not None:
                    fname = self.save_dir / f"svd_{obs_key}.png"
                    fig.savefig(fname, bbox_inches="tight")
                    print(f"  SVD figure saved: {fname.name}")
                plt.show()

        state["poles_raw"] = poles_raw
        state["L_used"] = L_used
        state["lambdas"] = lambdas
        state["observable_key"] = obs_key
        return state

    def description(self) -> str:
        return (
            f"MPMStep(order={self.order}, L={self.L}, "
            f"obs='{self.observable_key}', plot_svd={self.plot_svd})"
        )
