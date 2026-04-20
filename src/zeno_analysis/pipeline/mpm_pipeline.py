"""
mpm_pipeline.py
---------------
High-level pipeline: load data → run analysis steps → return / save result.

Replaces the monolithic ``ZenoTimeDomainAnalyzer`` / ``object_oriented_poles.py``
entry points with a clean class that delegates to ``io/`` and ``analysis/``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from zeno_analysis.data_types.analysis_result import ZenoAnalysisResult
from zeno_analysis.data_types.observables import ObservableRecord
from zeno_analysis.analysis.steps.base import AnalysisStep
from zeno_analysis.analysis.steps.mpm_step import MPMStep
from zeno_analysis.analysis.steps.bootstrap_step import BootstrapStep


_DEFAULT_STEPS = lambda order, n_boot: [MPMStep(order=order), BootstrapStep(n_boot=n_boot)]


class MPMPipeline:
    """Matrix Pencil + Bootstrap analysis pipeline.

    Parameters
    ----------
    records:
        Pre-loaded list of ``ObservableRecord`` objects.
    steps:
        Ordered list of ``AnalysisStep`` objects.  Default:
        ``[MPMStep(order=3), BootstrapStep(n_boot=2000)]``.
    metadata:
        Extra key-value pairs to embed in the result metadata.

    Typical usage
    ~~~~~~~~~~~~~
    ::

        pipeline = MPMPipeline.from_ensemble_average(folder, order=3, n_boot=500)
        result = pipeline.run()
        pipeline.save(output_dir=Path("results/"))

    Or with custom steps::

        from zeno_analysis.analysis.steps import MPMStep, BootstrapStep, RefineStep
        steps = [MPMStep(order=3), RefineStep(), BootstrapStep(n_boot=2000)]
        pipeline = MPMPipeline.from_ensemble_average(folder, steps=steps)
    """

    def __init__(
        self,
        records: list[ObservableRecord],
        steps: list[AnalysisStep] | None = None,
        *,
        metadata: dict | None = None,
        order: int = 3,
        n_boot: int = 2000,
    ) -> None:
        self.records = records
        self.steps = steps if steps is not None else _DEFAULT_STEPS(order, n_boot)
        self.metadata = dict(metadata or {})
        self._result: ZenoAnalysisResult | None = None

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ensemble_average(
        cls,
        folder: str | Path,
        steps: list[AnalysisStep] | None = None,
        observables: list[str] | None = None,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        order: int = 3,
        n_boot: int = 2000,
        csv_path: str | Path | None = None,
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Load ensemble-average data and return a ready-to-run pipeline."""
        from zeno_analysis.io.experimental import EnsembleAverageLoader
        folder = Path(folder)
        loader = EnsembleAverageLoader(csv_path=csv_path)
        records = loader.load(
            folder,
            observables=observables or ["survival"],
            lambda_range=lambda_range,
            max_files=max_files,
        )
        meta = {
            "analysis_type": "mpm",
            "source_folder": str(folder),
            "source_tag": "ens_avg",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metadata_kwargs,
        }
        return cls(records, steps, metadata=meta, order=order, n_boot=n_boot)

    @classmethod
    def from_post_selected(
        cls,
        folder: str | Path,
        steps: list[AnalysisStep] | None = None,
        observables: list[str] | None = None,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        order: int = 3,
        n_boot: int = 2000,
        csv_path: str | Path | None = None,
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Load post-selected tomography data and return a ready-to-run pipeline.

        Parameters
        ----------
        csv_path:
            Optional path to HMM results CSV (``Gamma_up_3_states_hmm`` column).
            When given, lambda is computed from the CSV rather than the JSON
            ``lambda_zeno_estimation`` — this is the updated, HMM-calibrated scale.
        """
        from zeno_analysis.io.experimental import PostSelectedLoader
        folder = Path(folder)
        loader = PostSelectedLoader(csv_path=csv_path)
        records = loader.load(
            folder,
            observables=observables or ["survival"],
            lambda_range=lambda_range,
            max_files=max_files,
        )
        meta = {
            "analysis_type": "mpm",
            "source_folder": str(folder),
            "source_tag": "post_selected",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metadata_kwargs,
        }
        return cls(records, steps, metadata=meta, order=order, n_boot=n_boot)

    @classmethod
    def from_parity(
        cls,
        folder: str | Path,
        steps: list[AnalysisStep] | None = None,
        observables: list[str] | None = None,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        order: int = 3,
        n_boot: int = 2000,
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Load parity-scan data and return a ready-to-run pipeline."""
        from zeno_analysis.io.experimental import ParityLoader
        folder = Path(folder)
        loader = ParityLoader()
        records = loader.load(
            folder,
            observables=observables or ["parity_real"],
            lambda_range=lambda_range,
            max_files=max_files,
        )
        meta = {
            "analysis_type": "mpm",
            "source_folder": str(folder),
            "source_tag": "parity",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metadata_kwargs,
        }
        return cls(records, steps, metadata=meta, order=order, n_boot=n_boot)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        data_type: str = "analytical",
        steps: list[AnalysisStep] | None = None,
        observables: list[str] | None = None,
        lambda_range: tuple[float, float] | None = None,
        order: int = 3,
        n_boot: int = 2000,
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Load analytical / MC NPZ and return a ready-to-run pipeline.

        Parameters
        ----------
        data_type:
            ``"analytical"`` or ``"monte_carlo"``.
        """
        from zeno_analysis.io.simulation import load_analytical_npz, load_monte_carlo_npz
        path = Path(path)
        loader_fn = load_analytical_npz if data_type == "analytical" else load_monte_carlo_npz
        records = loader_fn(
            path,
            observables=observables or ["survival"],
            lambda_range=lambda_range,
        )
        meta = {
            "analysis_type": "mpm",
            "source_folder": str(path),
            "source_tag": data_type,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metadata_kwargs,
        }
        return cls(records, steps, metadata=meta, order=order, n_boot=n_boot)

    @classmethod
    def from_records(
        cls,
        records: list[ObservableRecord],
        steps: list[AnalysisStep] | None = None,
        order: int = 3,
        n_boot: int = 2000,
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Build a pipeline from an already-loaded list of records.

        Useful when records come from a simulator's ``to_observable_records()``.
        """
        meta = {
            "analysis_type": "mpm",
            "source_tag": "simulation",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metadata_kwargs,
        }
        return cls(records, steps, metadata=meta, order=order, n_boot=n_boot)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> ZenoAnalysisResult:
        """Execute all analysis steps and return a ``ZenoAnalysisResult``.

        The result is also stored in ``self._result`` for later access or saving.
        """
        state: dict[str, Any] = {}
        for step in self.steps:
            print(f"  > {step.description()}")
            state = step.process(self.records, state)

        poles_result = state.get("poles_result")
        if poles_result is None:
            raise RuntimeError(
                "No 'poles_result' in state after running steps. "
                "Make sure a BootstrapStep is included."
            )

        lambdas = np.asarray(state.get("lambdas", []))
        step_descriptions = [s.description() for s in self.steps]

        params = {
            "pole_order": state.get("poles_raw", [np.array([])])[0].shape[0]
                          if state.get("poles_raw") else 3,
            "step_descriptions": step_descriptions,
        }

        if "scan_number" not in self.metadata:
            scan_numbers = {
                int(r.metadata["scan_number"])
                for r in self.records
                if r.metadata.get("scan_number") is not None
            }
            if len(scan_numbers) == 1:
                self.metadata["scan_number"] = scan_numbers.pop()

        # Pull omega_DG from first record metadata if available
        if self.records:
            omega_dg = self.records[0].metadata.get("omega_DG")
            if omega_dg is not None:
                params["omega_DG"] = float(omega_dg)

        self._result = ZenoAnalysisResult(
            metadata=self.metadata,
            parameters=params,
            lambda_values=lambdas,
            poles=poles_result,
        )
        print(f"Pipeline complete: {len(lambdas)} lambda points, poles shape {poles_result.decay_rates.shape}")
        return self._result

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        output_dir: str | Path,
        stem: str | None = None,
    ) -> Path:
        """Save the most recent ``run()`` result to ``output_dir``.

        Parameters
        ----------
        output_dir:
            Directory to write into.
        stem:
            Filename stem (without ``.json``).  Auto-generated if ``None``.

        Returns
        -------
        Path
            Path to the saved file.
        """
        if self._result is None:
            raise RuntimeError("Call run() before save().")
        from zeno_analysis.io.file_naming import save_result
        return save_result(self._result, output_dir, stem=stem)
