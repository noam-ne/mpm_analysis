"""
simulation.py
-------------
Load simulation output (NPZ files) into ``list[ObservableRecord]``.

Supported formats
~~~~~~~~~~~~~~~~~
``load_analytical_npz``
    NPZ files produced by ``PostSelectedDynamicsSimulator.save_npz_for_analysis()``.
    Expected keys: ``lambda_values``, ``time_array_us``, ``D_probability``,
    optionally ``Z``, ``theta``.

``load_monte_carlo_npz``
    Same format, produced by MC simulation runs.

Both functions accept a ``lambda_range`` filter and an ``observables`` list.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.utils.windows_paths import win_long_path

# Keys tried in order when reading the time axis from an NPZ
_TIME_KEYS = ("time_array_us", "time_array", "t_us", "t")
# Keys tried for lambda values
_LAMBDA_KEYS = ("lambda_values", "lambdas", "lambda_arr", "alpha_over_2omega")
# Observable key → list of NPZ key candidates
_SIGNAL_KEYS: dict[str, list[str]] = {
    "survival": ["D_probability", "survival", "P_survival", "p0"],
    "Z": ["Z", "s_z", "z_t"],
    "theta": ["theta", "theta_t"],
}


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    return np.load(win_long_path(path), allow_pickle=True)


def _find_key(npz, candidates: list[str]):
    for k in candidates:
        if k in npz:
            return npz[k]
    return None


def _read_time(npz) -> np.ndarray:
    t = _find_key(npz, _TIME_KEYS)
    if t is None:
        raise KeyError(f"Cannot find time array. Tried: {_TIME_KEYS}")
    t = np.asarray(t, dtype=float)
    if np.nanmax(t) > 1_000.0:  # likely ns → convert to µs
        t = t / 1e3
    return t


def _read_lambdas(npz) -> np.ndarray:
    lam = _find_key(npz, _LAMBDA_KEYS)
    if lam is None:
        raise KeyError(f"Cannot find lambda values. Tried: {_LAMBDA_KEYS}")
    return np.asarray(lam, dtype=float)


def _build_records(
    npz,
    source_tag: str,
    observables: list[str],
    lambda_range: tuple[float, float] | None,
) -> list[ObservableRecord]:
    """Common implementation for both analytical and MC NPZ formats."""
    lambdas = _read_lambdas(npz)
    t_us = _read_time(npz)

    # Load all requested observable arrays — shape (n_lambda, n_time)
    signals: dict[str, np.ndarray] = {}
    for obs_key in observables:
        candidates = _SIGNAL_KEYS.get(obs_key, [obs_key])
        arr = _find_key(npz, candidates)
        if arr is None:
            raise KeyError(
                f"Cannot find observable '{obs_key}'. "
                f"Tried NPZ keys: {candidates}."
            )
        signals[obs_key] = np.asarray(arr, dtype=float)

    records: list[ObservableRecord] = []
    for i, lam in enumerate(lambdas):
        if lambda_range is not None and not (lambda_range[0] <= lam <= lambda_range[1]):
            continue
        for obs_key, arr in signals.items():
            records.append(
                ObservableRecord(
                    lambda_val=float(lam),
                    t=t_us,
                    signal=arr[i],
                    observable_key=obs_key,
                    source_file=source_tag,
                    metadata={"simulation_index": i},
                )
            )
    records.sort(key=lambda r: (r.lambda_val, r.observable_key))
    return records


def load_analytical_npz(
    path: str | Path,
    observables: list[str] | None = None,
    *,
    lambda_range: tuple[float, float] | None = None,
) -> list[ObservableRecord]:
    """Load an analytical simulation NPZ into ``ObservableRecord`` objects.

    Parameters
    ----------
    path:
        Path to the ``.npz`` file.
    observables:
        Observable keys to load.  Default ``["survival"]``.
        Available keys depend on what the simulator saved; common values are
        ``"survival"``, ``"Z"``, ``"theta"``.
    lambda_range:
        Optional ``(min, max)`` filter on λ.

    Returns
    -------
    list[ObservableRecord]
    """
    if observables is None:
        observables = ["survival"]
    path = Path(path)
    npz = _load_npz(path)
    return _build_records(npz, path.name, observables, lambda_range)


def load_monte_carlo_npz(
    path: str | Path,
    observables: list[str] | None = None,
    *,
    lambda_range: tuple[float, float] | None = None,
) -> list[ObservableRecord]:
    """Load a Monte Carlo simulation NPZ.  Same interface as ``load_analytical_npz``."""
    if observables is None:
        observables = ["survival"]
    path = Path(path)
    npz = _load_npz(path)
    return _build_records(npz, path.name, observables, lambda_range)
