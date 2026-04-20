"""
survival_probability.py
-----------------------
Concrete loaders for the two main experimental data formats.

PostSelectedLoader
    Folders of per-lambda JSON files from post-selected quantum-state-tomography.
    Files contain ``data.p_0_t_array`` (survival probability).

    Lambda source (two modes):
      - CSV mode (accurate):  pass ``csv_path`` pointing to a results CSV with a
        ``Gamma_up_3_states_hmm`` column.  Lambda is then computed as
        ``Gamma_up / (2 * Omega_DG * 1e-6)``, matching the original analysis in
        ``object_oriented_poles.py``.  This is the "updated lambda scale" that
        uses HMM-inferred drive rates rather than the stored estimate.
      - JSON mode (fallback):  if no CSV is given, reads
        ``parameters.lambda_zeno_estimation`` directly from each JSON file.

EnsembleAverageLoader
    Folders of per-lambda JSON files from ensemble-average measurements.
    Files contain ``data.D_probability`` (survival) and ``data.G_probability``
    which serves as the Z-like observable (matching ``object_oriented_poles.py``
    where ``z_arr = g_arr = jd["data"]["D_probability"] - jd["data"]["G_probability"]``).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from zeno_analysis.io.experimental.base import (
    ExperimentalLoader,
    _load_json,
    _get_nested,
    _khz_from_name,
)
from zeno_analysis.utils.windows_paths import win_long_path

_SCAN_NUMBER_RE = re.compile(r"_(\d+)$")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_lambda_from_json(jd: dict) -> float:
    """Extract lambda from a raw JSON dict (tries several key variants)."""
    params = jd.get("parameters", jd)
    lam = (
        params.get("lambda_zeno_estimation")
        or params.get("lambda")
        or params.get("lambda_zeno")
        or _get_nested(jd, ["parameters", "lambda_zeno_estimation"])
    )
    if lam is None:
        raise KeyError("Cannot find lambda_zeno_estimation in JSON parameters.")
    return float(lam)


def _read_omega_dg_mhz(jd: dict) -> float:
    """Read Omega_DG from JSON parameters and convert Hz -> MHz (rad/us)."""
    params = jd.get("parameters", jd)
    val = params.get("Omega_DG") or params.get("omega_DG")
    if val is None:
        raise KeyError("Cannot find Omega_DG in JSON parameters.")
    return float(val) * 1e-6  # Hz -> MHz (= rad/us)


def _read_time_us(jd: dict) -> np.ndarray:
    """Extract time array in µs from nested data.time_array (originally in ns)."""
    # Original code: times = np.array(jd["data"]["time_array"]) / 1000.0
    t = _get_nested(jd, ["data", "time_array"]) or _get_nested(jd, ["data", "time_array_ns"])
    if t is None:
        # Fallback: try top-level keys for non-standard files
        t = jd.get("time_array") or jd.get("time_array_ns")
    if t is None:
        raise KeyError("Cannot find time_array in JSON (checked data.time_array and top level).")
    t = np.asarray(t, dtype=float)
    # Convert ns -> µs (heuristic: max > 1000 means ns)
    if np.nanmax(t) > 1_000.0:
        t = t / 1e3
    return t


def _load_csv_gamma_ups(csv_path: Path) -> dict[str, float]:
    """Load HMM Gamma_up values from a results CSV, keyed by filename fragment.

    The CSV must have columns ``filename`` and ``Gamma_up_3_states_hmm``.
    The key stored is the fragment between ``OBG0`` and ``kHz`` in the filename,
    matching the convention used in ``object_oriented_poles.py``.
    """
    gamma_ups: dict[str, float] = {}
    before_string = "OBG0"
    after_string = "kHz"
    with open(win_long_path(csv_path), "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            key = fname.split(before_string)[-1].split(after_string)[0]
            gamma_ups[key] = float(row["Gamma_up_3_states_hmm"])
    return gamma_ups


# ---------------------------------------------------------------------------
# Post-selected loader
# ---------------------------------------------------------------------------

class PostSelectedLoader(ExperimentalLoader):
    """Load post-selected tomography survival-probability data.

    Parameters
    ----------
    csv_path:
        Path to the HMM results CSV file (columns: ``filename``,
        ``Gamma_up_3_states_hmm``).  When given, lambda is computed as
        ``Gamma_up / (2 * Omega_DG * 1e-6)`` — the updated, HMM-calibrated
        lambda scale.  When omitted, falls back to
        ``parameters.lambda_zeno_estimation`` in each JSON.

    Default observables registered:
        ``"survival"``      <- ``data.p_0_t_array``
        ``"Z"``             <- ``data.s_z``  (NaN if absent)
        ``"G_probability"`` <- ``data.G_probability``  (NaN if absent)
        ``"B_probability"`` <- ``data.B_probability``  (NaN if absent)
        ``"D_probability"`` <- ``data.D_probability``  (NaN if absent)
    """

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path is not None else None
        self._gamma_ups: dict[str, float] | None = None  # lazy-loaded
        super().__init__()

    def _register_default_extractors(self) -> None:
        self.add_extractor(
            "survival",
            lambda jd, t: np.asarray(jd["data"]["p_0_t_array"], dtype=float),
        )
        self.add_extractor(
            "Z",
            lambda jd, t: np.asarray(jd["data"]["s_z"], dtype=float)
            if "s_z" in jd.get("data", {})
            else np.full_like(t, float("nan")),
        )
        self.add_extractor(
            "G_probability",
            lambda jd, t: np.asarray(jd["data"]["G_probability"], dtype=float)
            if "G_probability" in jd.get("data", {})
            else np.full_like(t, float("nan")),
        )
        self.add_extractor(
            "B_probability",
            lambda jd, t: np.asarray(jd["data"]["B_probability"], dtype=float)
            if "B_probability" in jd.get("data", {})
            else np.full_like(t, float("nan")),
        )
        self.add_extractor(
            "D_probability",
            lambda jd, t: np.asarray(jd["data"]["D_probability"], dtype=float)
            if "D_probability" in jd.get("data", {})
            else np.full_like(t, float("nan")),
        )

    def _discover_files(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.json"), key=lambda fp: _safe_lambda(fp))

    def _extract_lambda_and_time(self, jd: dict, fp: Path) -> tuple[float, np.ndarray]:
        t = _read_time_us(jd)

        if self._csv_path is not None:
            # CSV (HMM) lambda: Gamma_up / (2 * Omega_DG)
            if self._gamma_ups is None:
                self._gamma_ups = _load_csv_gamma_ups(self._csv_path)
            before_string = "OBG0"
            after_string = "kHz"
            key = fp.name.split(before_string)[-1].split(after_string)[0]
            if key not in self._gamma_ups:
                raise KeyError(
                    f"Filename fragment '{key}' not found in CSV {self._csv_path.name}. "
                    f"Available keys: {list(self._gamma_ups)[:5]}..."
                )
            omega_dg = _read_omega_dg_mhz(jd)
            lam = self._gamma_ups[key] / (2.0 * omega_dg)
        else:
            # Fallback: use stored lambda estimate from JSON
            lam = _read_lambda_from_json(jd)

        return lam, t


# ---------------------------------------------------------------------------
# Ensemble-average loader
# ---------------------------------------------------------------------------

class EnsembleAverageLoader(ExperimentalLoader):
    """Load ensemble-average survival-probability data.

    Default observables registered:
        ``"survival"`` <- ``data.D_probability``
        ``"Z"``        <- ``data.D_probability - data.G_probability``  
    """
    def __init__(self, csv_path: str | Path | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path is not None else None
        self._gamma_ups: dict[str, float] | None = None  # lazy-loaded
        super().__init__()

    def _register_default_extractors(self) -> None:
        self.add_extractor(
            "survival",
            lambda jd, t: (
                np.asarray(jd["data"]["D_probability"], dtype=float)
            
            ),
        )
        self.add_extractor(
            "Z",
            lambda jd, t: (
                np.asarray(jd["data"]["D_probability"], dtype=float)
                - np.asarray(jd["data"]["G_probability"], dtype=float)
            ) if "G_probability" in jd.get("data", {})
            else np.full_like(t, float("nan")),
        )

    def _discover_files(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.json"), key=lambda fp: _safe_lambda(fp))

    def _extract_lambda_and_time(self, jd: dict, fp: Path) -> tuple[float, np.ndarray]:
        if self._csv_path is not None:
            # CSV (HMM) lambda: Gamma_up / (2 * Omega_DG)
            if self._gamma_ups is None:
                self._gamma_ups = _load_csv_gamma_ups(self._csv_path)
            before_string = "OBG0"
            after_string = "kHz"
            key = fp.name.split(before_string)[-1].split(after_string)[0]
            if key not in self._gamma_ups:
                raise KeyError(
                    f"Filename fragment '{key}' not found in CSV {self._csv_path.name}. "
                    f"Available keys: {list(self._gamma_ups)[:5]}..."
                )
            omega_dg = _read_omega_dg_mhz(jd)
            lam = self._gamma_ups[key] / (2.0 * omega_dg)
        else:           
            # Fallback: use stored lambda estimate from JSON
            lam = _read_lambda_from_json(jd)
        return lam, _read_time_us(jd)

class ParityLoader(ExperimentalLoader):
    """Loader for parity-scan experiments.

    Extracts from the raw JSON data the following observables:"""

    def _register_default_extractors(self) -> None:
        self.add_extractor(
            "parity_real",
            lambda jd, t: np.asarray(jd["data"]["fourier_transform_pk"]["real part"], dtype=float),
        )
        self.add_extractor(
            "parity_imag",
            lambda jd, t: np.asarray(jd["data"]["fourier_transform_pk"]["imaginary part"], dtype=float),
        )
    def _discover_files(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.json"))   # sort however fits your naming

    def _extract_lambda_and_time(self, jd: dict, fp: Path=None) -> tuple[float, np.ndarray]:
        t = _read_time_us(jd)
        lam = _read_lambda_from_json(jd)

        return lam, t
# ---------------------------------------------------------------------------
# Helper: safe lambda for sorting before the JSON is read
# ---------------------------------------------------------------------------

def _safe_lambda(fp: Path) -> float:
    """Return a sort key from the filename (kHz number), fallback to 0."""
    try:
        return float(_khz_from_name(fp))
    except ValueError:
        return 0.0

