"""
survival_probability.py
-----------------------
Concrete loaders for the two main experimental data formats.

PostSelectedLoader
    Folders of per-lambda JSON files from post-selected quantum-state-tomography.
    Files contain ``data.p_0_t_array`` (survival probability).

    Lambda source is resolved per folder by :func:`discover_hmm_table`, not
    configured per run:
      - an HMM table (``filename`` + ``Gamma_up_3_states_hmm``) whose runs
        match the folder's own files gives ``Gamma_up / (2 * Omega_DG)``;
      - otherwise ``parameters.lambda_zeno_estimation`` from each JSON, which
        for the post-selected scans is itself ``Gamma_BG_HMM / (2 * Omega_DG)``
        and so needs no rescale.
    Matching is on the full run id (frequency *and* timestamp), so a table
    belonging to a different scan is never applied.  ``csv_path=False`` forces
    the stored JSON lambda; an explicit path is provenance-checked like any
    other candidate.  An explicit per-drive table (``drive_khz`` +
    ``gamma_BG_per_s_mean``, see :func:`_drive_gamma_ups`) calibrates the drive
    rather than one run, so it is never discovered, only passed.

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

from mpm_analysis.io.experimental.base import (
    ExperimentalLoader,
    HMMTableMismatch,
    _load_json,
    _get_nested,
    _khz_from_name,
)
from mpm_analysis.utils.windows_paths import win_long_path

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


def _hmm_key(name: str) -> str:
    """Join key for HMM tables: everything after the ``data__`` prefix.

    ``click_record_data__OBG0_1020_kHz_07_12_2025__11_45_42`` and
    ``survival probability_data__OBG0_1020_kHz_07_12_2025__11_45_42`` share the
    key ``OBG0_1020_kHz_07_12_2025__11_45_42`` — same run, different product.

    Keeping the date and timestamp is the point.  The old key was just the
    ``OBG0`` frequency fragment (``_1020_``), which silently matched files from
    a *different* scan that happened to visit the same drive frequency, and
    rescaled lambda by ~10 % with no error.
    """
    stem = name[:-5] if name.endswith(".json") else name
    return stem.split("data__", 1)[1] if "data__" in stem else stem


def _load_csv_gamma_ups(csv_path: Path) -> dict[str, float]:
    """Load HMM Gamma_up values from a results CSV, keyed by :func:`_hmm_key`.

    The CSV must have columns ``filename`` and ``Gamma_up_3_states_hmm``.
    A duplicated key means two rows describe the same run, which the caller
    cannot silently resolve — so it is an error rather than a last-one-wins.
    """
    gamma_ups: dict[str, float] = {}
    with open(win_long_path(csv_path), "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _hmm_key(row["filename"])
            if key in gamma_ups:
                raise HMMTableMismatch(
                    f"{csv_path.name}: duplicate row for '{key}' — cannot pick "
                    f"a Gamma_up_3_states_hmm unambiguously."
                )
            gamma_ups[key] = float(row["Gamma_up_3_states_hmm"])
    return gamma_ups


_HMM_COLUMNS = ("filename", "Gamma_up_3_states_hmm")
_DRIVE_COLUMNS = ("drive_khz", "gamma_BG_per_s_mean")


def _is_drive_table(csv_path: Path) -> bool:
    """True for a per-drive HMM table (columns :data:`_DRIVE_COLUMNS`)."""
    try:
        with open(win_long_path(csv_path), "r", newline="") as f:
            header = next(csv.reader(f), [])
    except (OSError, csv.Error, UnicodeDecodeError):
        return False
    return all(c in header for c in _DRIVE_COLUMNS)


def _drive_gamma_ups(csv_path: Path, folder: Path) -> dict[str, float]:
    """Gamma_up (1/us) of every run in *folder* from a per-drive HMM table.

    ``hmm_rates_per_drive.csv`` (offline three-state HMM of the alpha click
    records) is keyed by drive amplitude, not by run: it calibrates the drive
    itself, so it applies to any scan measured on the same drive points.  Each
    file's drive (the kHz in its name) is looked up exactly — never
    interpolated or extrapolated; a drive missing from the table raises.
    """
    with open(win_long_path(csv_path), "r", newline="") as f:
        gamma = {int(float(r["drive_khz"])): float(r["gamma_BG_per_s_mean"]) * 1e-6
                 for r in csv.DictReader(f)}
    # Only data files carry a drive in their name; side files such as a
    # combined folder's provenance.json are skipped by the loader anyway.
    drives = {}
    for p in _long_glob(folder, "*.json"):
        try:
            drives[_hmm_key(p.name)] = _khz_from_name(p)
        except ValueError:
            continue
    missing = sorted({k for k in drives.values() if k not in gamma})
    if missing:
        raise HMMTableMismatch(
            f"{csv_path.name} has no row for drives {missing} kHz of "
            f"{folder.name} (table rows: {sorted(gamma)}); no interpolation."
        )
    return {key: gamma[k] for key, k in drives.items()}


def _long_glob(folder: Path, pattern: str) -> list[Path]:
    """``folder.glob(pattern)`` that survives Windows' 260-char listing limit.

    Listing a directory appends a wildcard to its path, so a folder whose path
    is only ~250 characters (e.g. ``scan_iteration/alpha_measurement`` deep in
    the Dropbox tree) raises ``WinError 3`` although it exists.  The listing
    goes through the extended-length form (:func:`win_long_path`); the
    returned paths stay in normal form.  A folder that is missing or
    unreadable yields no matches.
    """
    long = Path(win_long_path(folder))
    try:
        return sorted(folder / p.name for p in long.glob(pattern))
    except OSError:
        return []


def _is_long_dir(folder: Path) -> bool:
    """``folder.is_dir()`` that also works past the Windows path limit."""
    return Path(win_long_path(folder)).is_dir()


def _folder_runs(folder: Path) -> set[str]:
    """Run ids of every JSON in *folder* (see :func:`_hmm_key`)."""
    return {_hmm_key(p.name) for p in _long_glob(folder, "*.json")}


def _table_runs(csv_path: Path) -> set[str] | None:
    """Run ids in an HMM table, or ``None`` if the CSV is not one.

    A CSV without both ``filename`` and ``Gamma_up_3_states_hmm`` carries no
    lambda rescale (``results_df.csv`` holds only ``Omega_BG0``, the
    ``hmm_scan_results_*.csv`` tables are keyed by drive amplitude rather than
    by run), so it is never a candidate.
    """
    try:
        with open(win_long_path(csv_path), "r", newline="") as f:
            reader = csv.DictReader(f)
            if not all(c in (reader.fieldnames or []) for c in _HMM_COLUMNS):
                return None
            return {_hmm_key(row["filename"]) for row in reader}
    except (OSError, csv.Error, KeyError, UnicodeDecodeError):
        return None


def discover_hmm_table(folder: Path) -> Path | None:
    """Find the HMM lambda table that provably belongs to *folder*.

    Searched: the folder itself, its ``alpha_measurement`` subfolder, and the
    sibling scan folders — tables are routinely filed next to the wrong scan.
    A candidate qualifies only when its run ids (drive frequency *and*
    timestamp) intersect the folder's own files, so a neighbouring scan's
    table can never be picked up no matter where it sits.

    ``None`` means no table covers this folder, in which case the stored
    ``lambda_zeno_estimation`` is already the corrected value — that is the
    case for the post-selected scans, whose JSON lambda was written as
    ``Gamma_BG_HMM / (2 * Omega_DG)`` at acquisition time.
    """
    folder = Path(folder)
    runs = _folder_runs(folder)
    if not runs:
        return None

    places = [folder, folder / "alpha_measurement"]
    if _is_long_dir(folder.parent):
        for sib in _long_glob(folder.parent, "*"):
            if _is_long_dir(sib) and sib != folder:
                places += [sib, sib / "alpha_measurement"]

    scored: list[tuple[int, Path]] = []
    for place in places:
        if not _is_long_dir(place):
            continue
        for cand in _long_glob(place, "*.csv"):
            table = _table_runs(cand)
            if table and (n := len(table & runs)):
                scored.append((n, cand))
    if not scored:
        return None

    best = max(n for n, _ in scored)
    winners = [c for n, c in scored if n == best]
    if len(winners) > 1:
        raise HMMTableMismatch(
            f"{len(winners)} HMM tables each cover {best} runs of {folder}: "
            f"{[str(w) for w in winners]}. Pass the intended one explicitly."
        )
    return winners[0]


def _resolve_hmm_table(setting, folder: Path) -> Path | None:
    """Turn the ``csv_path`` setting into the table to use for *folder*.

    ``"auto"``/``None`` discover, ``False`` forces the stored JSON lambda, and
    an explicit path is checked against the folder like any other candidate —
    a table from another scan raises instead of silently shifting lambda.
    A per-drive table has no runs to check; its drives are checked when it
    is read (:func:`_drive_gamma_ups`).
    """
    if setting is False:
        return None
    if setting is None or setting == "auto":
        return discover_hmm_table(folder)

    csv_path = Path(setting)
    if _is_drive_table(csv_path):
        return csv_path
    runs, table = _folder_runs(folder), _table_runs(csv_path)
    if table is None:
        raise HMMTableMismatch(
            f"{csv_path} is not an HMM table: it needs columns "
            f"{list(_HMM_COLUMNS)}."
        )
    if not (table & runs):
        raise HMMTableMismatch(
            f"{csv_path} describes no run in {folder}. "
            f"CSV runs (first 3): {sorted(table)[:3]}; "
            f"folder runs (first 3): {sorted(runs)[:3]}. "
            "This HMM table belongs to a different scan."
        )
    return csv_path


def _resolve_lambda(loader, jd: dict, fp: Path) -> float:
    """Lambda for one file, resolving the folder's HMM table once per load.

    Either ``Gamma_up_3_states_hmm / (2 * Omega_DG)`` from a table that belongs
    to this folder, or the stored ``lambda_zeno_estimation``.  Which one it was
    is printed once and recorded in the result metadata by
    :meth:`ExperimentalLoader.lambda_source`, so a saved analysis always states
    the lambda axis it was produced on.
    """
    if not loader._resolved:
        loader._csv_path = _resolve_hmm_table(loader._csv_setting, fp.parent)
        if loader._csv_path is None:
            loader._gamma_ups = None
        elif _is_drive_table(loader._csv_path):
            loader._gamma_ups = _drive_gamma_ups(loader._csv_path, fp.parent)
        else:
            loader._gamma_ups = _load_csv_gamma_ups(loader._csv_path)
        loader._resolved = True
        if loader._csv_path is None:
            print(f"[lambda] {fp.parent.name}: stored lambda_zeno_estimation "
                  f"(no HMM table covers this scan)")
        else:
            covered = len(set(loader._gamma_ups) & _folder_runs(fp.parent))
            print(f"[lambda] {fp.parent.name}: HMM rescale from "
                  f"{loader._csv_path.name} (covers {covered} runs)")

    if loader._gamma_ups is None:
        return _read_lambda_from_json(jd)
    key = _hmm_key(fp.name)
    if key not in loader._gamma_ups:
        raise KeyError(
            f"no HMM row for run '{key}' in "
            f"{loader._csv_path.name} ({len(loader._gamma_ups)} rows)"
        )
    return loader._gamma_ups[key] / (2.0 * _read_omega_dg_mhz(jd))


# ---------------------------------------------------------------------------
# Post-selected loader
# ---------------------------------------------------------------------------

def _navg(jd: dict) -> int:
    """Repetitions per axis setting of THIS file (merged folders mix runs)."""
    params = jd.get("parameters", {})
    navg = (params.get("actual_navg_repeat_experiment")
            or params.get("navg_repeat_experiment"))
    if not navg:
        raise KeyError("navg_repeat_experiment missing from parameters — the "
                       "tomography channels cannot be normalised.")
    return int(navg)


def _z_survival(jd: dict) -> np.ndarray:
    """Survival of the z-axis tomography run: survivors of z/-z over 2·navg attempts."""
    return np.asarray(jd["data"]["z_axis_number_of_measurements"], dtype=float) / (2 * _navg(jd))


def _phase_survival(jd: dict) -> np.ndarray:
    """Survival of the phase-axis run: survivors per phase axis pair over 2·navg attempts."""
    return (np.asarray(jd["data"]["phase_axes_number_of_measurements"], dtype=float)
            / (2 * _navg(jd)))


# Survivals computed here from the first-click histogram (no JSON array of
# their own): one named segment per point, see click_record_npz.HISTOGRAM_SURVIVALS.
_NEXT_KEYS = ("S_next", "S_next_far", "S_next_x2")


def _first_click_histogram(data: dict) -> np.ndarray | None:
    """``data.p_0_t_pooled_meta.first_click_counts`` (ragged: axis → segment
    L → slots 0..L) padded to ``(n_axes, n_seg, n_seg + 1)``; None if absent."""
    hist = (data.get("p_0_t_pooled_meta") or {}).get("first_click_counts")
    if not hist:
        return None
    n_axes, n_seg = len(hist), len(hist[0])
    counts = np.zeros((n_axes, n_seg, n_seg + 1), dtype=np.int64)
    for a in range(n_axes):
        for j, row in enumerate(hist[a]):
            counts[a, j, :len(row)] = row
    return counts


def _histogram_survival(data: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    """``(fraction, attempts)`` of a histogram survival on the file's full time
    axis (index 0 = t=0 anchor; NaN where the curve has no segment)."""
    from mpm_analysis.io.experimental.click_record_npz import (
        HISTOGRAM_SURVIVALS, pooled_survival_from_counts,
    )
    counts = _first_click_histogram(data)
    if counts is None:
        raise KeyError("p_0_t_pooled_meta.first_click_counts missing (run "
                       "raw_data_tests/run_pooled_p0.py on this folder)")
    frac, _, n_att = pooled_survival_from_counts(
        counts, **HISTOGRAM_SURVIVALS[key](counts.shape[1]))
    return frac, n_att


class PostSelectedLoader(ExperimentalLoader):
    """Load post-selected tomography survival-probability data.

    Parameters
    ----------
    csv_path:
        Path to the HMM results CSV file (columns: ``filename``,
        ``Gamma_up_3_states_hmm``).  When given, lambda is computed as
        ``Gamma_up / (2 * Omega_DG * 1e-6)`` — the updated, HMM-calibrated
        lambda scale.  When omitted, falls back to
        ``parameters.lambda_zeno_estimation`` in each JSON.  A per-drive
        table (``drive_khz``, ``gamma_BG_per_s_mean``) is looked up at each
        file's drive instead (:func:`_drive_gamma_ups`).

    Default observables registered:
        ``"S"``             <- ``data.p_0_t_array``  (dedicated survival run)
        ``"S_prev_click"``  <- ``data.p_0_t_array_prev_click``
        ``"S_prev_noclick"``<- ``data.p_0_t_array_prev_noclick``
        ``"S_pooled"``      <- ``data.p_0_t_array_pooled``
        ``"S_pooled_strict"`` <- ``data.p_0_t_array_pooled_strict`` (no
                               segment's own last bin; one point shorter)
        ``"S_next"``        <- segment L = i + 1 only, from the first-click
                               histogram (no ring-down bin, shots of S)
        ``"S_next_far"``    <- segment L = i + 1 + n_seg//2 only
        ``"S_next_x2"``     <- both (2x shots, half the record)
        ``"SZ"`` / ``"SX"`` <- survival of the z / phase tomography runs
        ``"Z"``, ``"X"``, ``"Y"``  <- SZ·<Z>, SX·<X>, SX·<Y>  (unnormalised
                               Bloch components — the 3x3-generator variables
                               the joint pencil fits)
        ``"Z_cond"``, ``"X_cond"``, ``"Y_cond"`` <- the raw conditional Bloch
                               coordinates <Z>, <X>, <Y> on the surviving
                               trajectories (``z_axis_array``,
                               ``main_phase_axis_array``,
                               ``orthogonal_phase_axis_array``) — the spiral
                               trajectory R(t) = (X, Z) of the state-freezing
                               approach-rate fit (analysis/spiral_approach.py)

    The conditional pair are the raw-record survivals conditioned on the
    preceding segment's outcome, written into the scan JSONs by
    ``run_files/raw_data_tests/run_conditional_p0.py`` (the file's
    ``data.p_0_t_conditional_meta`` records which click rule built them and
    the per-point class counts).  Files that never ran it lack the keys and
    the observable is skipped with a warning.  The measurement-statistics
    (data) bootstrap resamples these channels binomially on their own
    per-point class counts from the meta, so any uncertainty source works.

    ``S_pooled`` is the survival rebuilt from the raw NPZ by pooling every
    segment that reaches ``t_i`` (``run_files/raw_data_tests/run_pooled_p0.py``):
    up to 59× the attempts of ``S`` at early times.  Its
    ``data.p_0_t_pooled_meta`` carries per-point attempt counts and sigmas
    and the per-axis, per-segment first-bright-bin histogram; the data
    bootstrap resamples that histogram multinomially (one draw per axis and
    segment), which keeps the correlation between pooled points that share
    shots.

    ``S_pooled_strict`` pools only the segments that run PAST ``t_i``
    (``L > i``), so no point contains a segment's own last bin — the bin
    the cavity ring-down reads bright too often (scans 3-5).  Its last point
    has no such segment and is stored as ``null``; the loader trims that
    trailing undefined sample, so the record ends one bin earlier.  Same
    histogram, same multinomial bootstrap (``min_extra_bins = 1``).

    ``S_next`` takes each point from ONE named segment instead of a pool:
    point ``i`` from the segment one bin longer (``L = i + 1``), so no point
    contains a record-ending bin while every point keeps the stored S's shots
    and the points stay independent (record ends at ``i = n_seg - 1``).
    ``S_next_far`` uses ``L = i + 1 + n_seg//2`` and ``S_next_x2`` both —
    disjoint segment sets, 2x shots, record ending at ``i = n_seg - 1 -
    n_seg//2``.  The halves follow previous records of different length
    (initialization history), so check ``S_next`` vs ``S_next_far`` before
    trusting ``S_next_x2`` (mpm_analysis_output/s_next_ringdown_20261026).
    All three are computed from ``first_click_counts`` (no JSON array);
    multinomial data bootstrap as above, and the ``errorbar`` bootstrap gets
    their binomial sigma (independent points).
    """

    def __init__(self, csv_path: str | Path | bool | None = "auto") -> None:
        self._csv_setting = csv_path        # "auto" | False | explicit path
        self._csv_path: Path | None = None  # table actually used, per folder
        self._gamma_ups: dict[str, float] | None = None
        self._resolved = False
        super().__init__()
    
    

    def _register_default_extractors(self) -> None:
        self.add_extractor(
            "S",
            lambda jd, t: np.asarray(jd["data"]["p_0_t_array"], dtype=float),
        )
        # Raw-record survival conditioned on the preceding segment's outcome —
        # written back into the scan JSONs by run_conditional_p0.py (see the
        # class docstring); missing in scans that never ran it.
        self.add_extractor(
            "S_prev_click",
            lambda jd, t: np.asarray(
                jd["data"]["p_0_t_array_prev_click"], dtype=float),
        )
        self.add_extractor(
            "S_prev_noclick",
            lambda jd, t: np.asarray(
                jd["data"]["p_0_t_array_prev_noclick"], dtype=float),
        )
        # Segment-pooled survival rebuilt from the raw NPZ (run_pooled_p0.py;
        # see the class docstring); missing in scans that never ran it.
        self.add_extractor(
            "S_pooled",
            lambda jd, t: np.asarray(
                jd["data"]["p_0_t_array_pooled"], dtype=float),
        )
        # Same pool without each segment's own last bin; the stored last
        # point is null (-> NaN), which load() trims off the record end.
        self.add_extractor(
            "S_pooled_strict",
            lambda jd, t: np.asarray(
                jd["data"]["p_0_t_array_pooled_strict"], dtype=float),
        )
        # One named segment per point, computed from the same histogram (no
        # JSON array); undefined tail points are NaN, which load() trims.
        for key in _NEXT_KEYS:
            self.add_extractor(
                key, lambda jd, t, key=key: _histogram_survival(jd["data"], key)[0])
        # Tomography channels.  The scan stores the survivor COUNTS per axis
        # pair (z: z/-z, phase: x/-x and y/-y) and the conditional Bloch
        # coordinates on those survivors.  Each axis pair is measured navg
        # times per setting, i.e. navg*2 attempts, so
        #   SZ = N_z / (2 navg),   SX = N_phase / (2 navg)
        # are that experiment's own survival and  Z = SZ·<Z>, X = SX·<X>,
        # Y = SX·<Y>  the unnormalised Bloch components p<Z>, p<X>, p<Y> —
        # the variables of the 3x3 generator.  navg comes from each file
        # (merged folders hold files of different runs), never a constant.
        self.add_extractor("SZ", lambda jd, t: _z_survival(jd))
        self.add_extractor("SX", lambda jd, t: _phase_survival(jd))
        self.add_extractor(
            "Z",
            lambda jd, t: np.asarray(jd["data"]["z_axis_array"], dtype=float) * _z_survival(jd),
        )
        self.add_extractor(
            "X",
            lambda jd, t: np.asarray(jd["data"]["main_phase_axis_array"], dtype=float) * _phase_survival(jd),
        )
        self.add_extractor(
            "Y",
            lambda jd, t: np.asarray(jd["data"]["orthogonal_phase_axis_array"], dtype=float)
            * _phase_survival(jd),
        )
        # Raw conditional Bloch coordinates on the survivors (no survival
        # factor): the trajectory R(t) = (X, Z) whose late-time approach to
        # the fixed point gives gamma_a (analysis/spiral_approach.py).  The
        # data bootstrap resamples them binomially on N_X / N_Z alone.
        self.add_extractor(
            "Z_cond", lambda jd, t: np.asarray(jd["data"]["z_axis_array"], dtype=float),
        )
        self.add_extractor(
            "X_cond",
            lambda jd, t: np.asarray(jd["data"]["main_phase_axis_array"], dtype=float),
        )
        self.add_extractor(
            "Y_cond",
            lambda jd, t: np.asarray(jd["data"]["orthogonal_phase_axis_array"], dtype=float),
        )

    def _point_sigma(self, jd: dict, key: str) -> np.ndarray | None:
        """Measured per-point error bar of ``S`` / ``S_pooled`` from the raw NPZ.

        ``run_files/raw_data_tests/run_pooled_p0.py`` (``WRITE_ERRORBARS``)
        splits the repetitions of each file into consecutive blocks, rebuilds
        both curves per block and writes the block standard error of the
        mean into ``data.p_0_t_errorbar_meta[key]["sigma"]`` (next to the
        binomial sigma for comparison).  Files without the block are loaded
        without error bars; the ``errorbar`` bootstrap then refuses to run.

        When ``data.p_0_t_bootstrap_meta`` holds the key (written by
        raw_data_tests/bootstrap_s_errorbars.py: std per point over a
        bootstrap of the file's repetitions, S only) that sigma is used
        instead.
        """
        boot = jd["data"].get("p_0_t_bootstrap_meta") or {}
        em = jd["data"].get("p_0_t_errorbar_meta") or {}
        blk = boot.get(key) or em.get(key)
        if (not blk or blk.get("sigma") is None) and key in _NEXT_KEYS:
            # one segment per point: independent binomial points (the shot
            # bootstrap of S equals binomial to 0.5 %); anchor sigma 0
            frac, n_att = _histogram_survival(jd["data"], key)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.sqrt(frac * (1.0 - frac) / n_att)
        if not blk or blk.get("sigma") is None:
            return None
        return np.asarray(blk["sigma"], dtype=float)

    def _bootstrap_stats(self, jd: dict, mask: np.ndarray | None) -> dict | None:
        """Binomial statistics for every post-selected channel.

        The scan runs ``navg`` repetitions of each of 6 axis settings
        (z, -z, x, -x, y, -y).  Each measurement therefore has its own number
        of attempts, and its own survival:

        ==========  ===============  ====================================
        channel     attempts         survival fraction
        ==========  ===============  ====================================
        ``S``       ``navg * 6``     ``p_0_t_array``   (dedicated run)
        ``SX``      ``navg * 4``     ``N_phase / (navg*2)``  (x,-x,y,-y)
        ``SZ``      ``navg * 2``     ``N_z / (navg*2)``      (z,-z)
        ==========  ===============  ====================================

        ``phase_axes_number_of_measurements`` counts survivors *per phase
        axis* (out of ``navg*2``), so the phase-axis run pools twice that many
        survivors over twice as many attempts — the fraction is unchanged,
        which is why ``SX`` and ``SZ`` share the ``navg*2`` denominator while
        their attempt counts differ.

        The conditional Bloch coordinates X / Y / Z are resampled on the
        surviving trajectories (``N_X``/``N_Y``/``N_Z``); X and Y share the
        one count array the files provide.
        """
        data = jd["data"]
        params = jd.get("parameters", {})

        def _arr(key, dtype=float, *, required=True):
            v = data.get(key)
            if v is None:
                if required:
                    raise KeyError(f"'{key}' missing from data")
                return None
            v = np.asarray(v, dtype=dtype)
            return v[mask] if mask is not None else v

        # Every file carries its own repetition count (merged folders hold
        # files of different runs with different navg) — never a default.
        navg = (params.get("actual_navg_repeat_experiment")
                or params.get("navg_repeat_experiment"))
        if not navg:
            return None    # no shot count → no data bootstrap for this file
        navg = int(navg)

        # Shared count array for both phase axes (survivors per phase axis)
        n_phase = _arr("phase_axes_number_of_measurements", int)
        n_z = _arr("z_axis_number_of_measurements", int)
        per_axis_attempts = navg * 2          # one axis pair = 2 settings

        out = {
            "data_mode": "post_selected",
            # --- survival channels: fraction + attempts behind it ---
            "survival": _arr("p_0_t_array"),
            "N_survival": navg * 6,
            "SX": n_phase / per_axis_attempts,
            "N_attempts_SX": navg * 4,        # x, -x, y, -y
            "SZ": n_z / per_axis_attempts,
            "N_attempts_SZ": navg * 2,        # z, -z
            # --- conditional Bloch coordinates + their survivor counts ---
            "X": _arr("main_phase_axis_array"),
            "Y": _arr("orthogonal_phase_axis_array", required=False),
            "Z": _arr("z_axis_array"),
            "N_X": n_phase,
            "N_Y": n_phase,
            "N_Z": n_z,
        }

        # Conditioned raw-record survivals (written back into the JSON by
        # run_conditional_p0.py): binomial per time point on their own class
        # counts.  The t=0 anchor resamples on the full row count (p = 1 —
        # deterministic, matching its by-construction value).
        cm = data.get("p_0_t_conditional_meta") or {}
        for tag in ("prev_click", "prev_noclick"):
            arr = data.get(f"p_0_t_array_{tag}")
            counts = cm.get(f"n_{tag}")
            if arr is None or counts is None:
                continue
            v = np.asarray(arr, dtype=float)
            n = np.concatenate(([int(cm.get("n_rows", 0))],
                                np.asarray(counts, dtype=np.int64)))
            if mask is not None:
                v, n = v[mask], n[mask]
            out[f"S_{tag}"] = v
            out[f"N_attempts_S_{tag}"] = n

        # Segment-pooled survival (run_pooled_p0.py write-back).  Beyond the
        # per-point attempts (binomial fallback) the meta holds the ragged
        # first-bright-bin histogram; padded to (n_axes, n_seg, n_seg + 1)
        # it is the multinomial resampling unit — see
        # bootstrap.bootstrap_post_selected_measurements.  The histogram is
        # kept whole and the curve it produces is masked afterwards, so the
        # pooled points always come from every segment of the file.
        # The strict pool (segments L > i only) shares the histogram; its
        # attempts sit in the meta's "strict" block (last point: 0 attempts).
        pm = data.get("p_0_t_pooled_meta") or {}
        pooled = (("S_pooled", data.get("p_0_t_array_pooled"), pm),
                  ("S_pooled_strict", data.get("p_0_t_array_pooled_strict"),
                   pm.get("strict") or {}))
        for key, arr, meta in pooled:
            if arr is None or meta.get("n_attempts") is None:
                continue
            v = np.asarray(arr, dtype=float)
            n = np.asarray(meta["n_attempts"], dtype=np.int64)
            if mask is not None:
                v, n = v[mask], n[mask]
            out[key] = v
            out[f"N_attempts_{key}"] = n
        counts = _first_click_histogram(data)
        if counts is not None:
            # the one-segment-per-point curves exist wherever the histogram does
            for key in _NEXT_KEYS:
                v, n = _histogram_survival(data, key)
                if mask is not None:
                    v, n = v[mask], n[mask]
                out[key] = v
                out[f"N_attempts_{key}"] = n
            out["pooled_counts"] = counts
            out["pooled_mask"] = None if mask is None else np.asarray(mask, bool)
        return out

    def _discover_files(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.json"), key=lambda fp: _safe_lambda(fp))

    def _extract_lambda_and_time(self, jd: dict, fp: Path) -> tuple[float, np.ndarray]:
        t = _read_time_us(jd)

        return _resolve_lambda(self, jd, fp), t


# ---------------------------------------------------------------------------
# Ensemble-average loader
# ---------------------------------------------------------------------------

class EnsembleAverageLoader(ExperimentalLoader):
    """Load ensemble-average survival-probability data.

    Default observables registered:
        ``"survival"`` <- ``data.D_probability``
        ``"Z"``        <- ``data.D_probability - data.G_probability``
        ``"click_probability"`` <- ``data.click_probability`` — the fraction
        of shots whose fast readout at that time bin returned bright, i.e.
        the ensemble probability of occupying |B> as seen by the click
        detector.  Not a fit target: it drives the detector dead-time
        compression of the time axis
        (``analysis.dead_time.compress_records``).
        ``"B_plus_error"`` <- ``data.B_probability + data.error_probability``
        — the projective readout's |B> occupation, an alternative p_click.
    """
    def __init__(self, csv_path: str | Path | bool | None = "auto") -> None:
        self._csv_setting = csv_path        # "auto" | False | explicit path
        self._csv_path: Path | None = None  # table actually used, per folder
        self._gamma_ups: dict[str, float] | None = None
        self._resolved = False
        super().__init__()

    def _register_default_extractors(self) -> None:
        self.add_extractor(
            "survival",
            lambda jd, t: (
                np.asarray(jd["data"]["D_probability"], dtype=float)
            )),
        self.add_extractor(
            "D",
            lambda jd, t: (
                np.asarray(jd["data"]["D_probability"], dtype=float)
            )),
        self.add_extractor(
                    "G",
                    lambda jd, t: (
                        np.asarray(jd["data"]["G_probability"], dtype=float)
                    )),
        self.add_extractor(
                    "B",
                    lambda jd, t: (
                        np.asarray(jd["data"]["B_probability"], dtype=float)
                    )),
        self.add_extractor(
                            "error",
                            lambda jd, t: (
                                np.asarray(jd["data"]["error_probability"], dtype=float)
                            )),
        self.add_extractor(
            "Z",
            lambda jd, t: (
                np.asarray(jd["data"]["D_probability"], dtype=float)
                - np.asarray(jd["data"]["G_probability"], dtype=float)
            ) if "G_probability" in jd.get("data", {})
            else np.full_like(t, float("nan")),
        )
        self.add_extractor(
            "click_probability",
            lambda jd, t: np.asarray(jd["data"]["click_probability"], dtype=float),
        )
        # The projective readout's own estimate of the |B> occupation: the
        # first projective measurement came out excited (B = excited then not,
        # error = excited twice).  An alternative p_click for the dead-time
        # compression (DT_CLICK_KEY = "B_plus_error"): sampled at t itself
        # by the slower, cleaner readout instead of the fast one (which sits
        # ~0.04 higher at late times from its per-bin false positives).
        self.add_extractor(
            "B_plus_error",
            lambda jd, t: (
                np.asarray(jd["data"]["B_probability"], dtype=float)
                + np.asarray(jd["data"]["error_probability"], dtype=float)
            ),
        )


    def _bootstrap_stats(self, jd: dict, mask: np.ndarray | None) -> dict | None:
        """Multinomial statistics: D/G/B/error share the same N_shots.

        The four outcomes are mutually exclusive results of the same shots
        (they sum to 1), so resampling is one multinomial draw per time
        point — never independent noise per channel.

        Every repetition runs the sequence twice, with the normal and the
        inverted readout (``inverse_*_stream`` in ``stream_tags``), and each
        channel is the mean of the two passes
        (``opx_experiments ... zeno_survival_probability.fetch_results``), so
        a point holds ``2 * navg`` shots — ``D * 2 navg`` is an integer in
        every file, ``D * navg`` is not.
        """
        data = jd["data"]
        params = jd.get("parameters", {})
        navg = (params.get("actual_navg_repeat_experiment")
                or params.get("navg_repeat_experiment"))
        if not navg:
            return None    # never invent shot counts
        passes = 2 if "inverse_D_stream" in (params.get("stream_tags") or ()) else 1

        P = np.vstack([
            np.asarray(data["D_probability"], dtype=float),
            np.asarray(data["G_probability"], dtype=float),
            np.asarray(data["B_probability"], dtype=float),
            np.asarray(data["error_probability"], dtype=float),
        ])
        if mask is not None:
            P = P[:, mask]
        return {
            "data_mode": "ens_avg",
            "P": P,
            "channels": ["D", "G", "B", "error"],
            "N_shots": passes * int(navg),
        }

    def _discover_files(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.json"), key=lambda fp: _safe_lambda(fp))

    def _extract_lambda_and_time(self, jd: dict, fp: Path) -> tuple[float, np.ndarray]:
        return _resolve_lambda(self, jd, fp), _read_time_us(jd)

class ParityLoader(ExperimentalLoader):
    """Loader for parity-scan experiments.

    Extracts ``parity_real`` and ``parity_imag`` observables.

    Parameters
    ----------
    k_idx:
        Index into ``data.fourier_k_array`` to use as the 1-D time trace.
        When set, slices ``data.fourier_transform_pk_array`` at that k
        index (new 2-D data format).  When ``None`` (default), falls back
        to the legacy ``data.fourier_transform_pk`` field (single-k format).
        Use ``ParityKSpaceLoader`` to load all k values simultaneously.
    """

    def __init__(self, k_idx: int | None = None) -> None:
        self._k_idx = k_idx
        # Set on the FIRST file that triggers the DFT fallback so we only
        # print the announcement banner once per ParityLoader instance.
        self._fallback_announced = False
        super().__init__()

    def _compute_pkt_column(self, jd: dict, k_idx: int, component: str) -> np.ndarray:
        """Fallback: compute one column of P(k,t) by DFT of P(n,t).

        Used when ``data.fourier_transform_pk_array`` is missing (older
        scans like scan_experiment_results_12).  Uses the same default
        k-grid (``np.linspace(0, 2*pi, 101)``) as the newer scans, so
        ``k_idx`` keeps the same physical meaning across scans.
        """
        # Local import avoids any chance of circulars at module load.
        from mpm_analysis.kspace.loader import compute_pkt_from_pnt

        P_nt_raw = jd["data"]["jump_probability_distribution"]
        P_nt = np.asarray(P_nt_raw, dtype=float)
        P_kt, _ = compute_pkt_from_pnt(P_nt, k_array=None)  # (n_time, n_k)

        if not self._fallback_announced:
            print(
                f"[ParityLoader] *** P(k,t) FALLBACK ACTIVE (k_idx={k_idx}) ***\n"
                "  'fourier_transform_pk_array' not in JSON; computing column via\n"
                "      P(k,t) = sum_n P(n,t) * exp(+i*k*n)\n"
                "  on default k-grid np.linspace(0, 2*pi, 101) "
                "from data.jump_probability_distribution."
            )
            self._fallback_announced = True

        col = P_kt[:, k_idx]  # (n_time,)
        return col.real if component == "real" else col.imag

    def _register_default_extractors(self) -> None:
        k_idx = self._k_idx

        if k_idx is not None:
            # New 2-D format: slice at the requested k index.
            # JSON layout: fourier_transform_pk_array.real/imag  shape (n_time, n_k)
            # Falls back to DFT of P(n,t) for older scans missing that array.
            def _extract_real(jd: dict, t: np.ndarray) -> np.ndarray:
                pk_arr = jd["data"].get("fourier_transform_pk_array")
                if pk_arr is None:
                    return self._compute_pkt_column(jd, k_idx, "real")
                arr = np.asarray(pk_arr["real part"], dtype=float)
                return arr[:, k_idx]  # (n_time,)

            def _extract_imag(jd: dict, t: np.ndarray) -> np.ndarray:
                pk_arr = jd["data"].get("fourier_transform_pk_array")
                if pk_arr is None:
                    return self._compute_pkt_column(jd, k_idx, "imag")
                arr = np.asarray(pk_arr["imaginary part"], dtype=float)
                return arr[:, k_idx]  # (n_time,)
        else:
            # Legacy single-k format
            def _extract_real(jd: dict, t: np.ndarray) -> np.ndarray:
                return np.asarray(
                    jd["data"]["fourier_transform_pk"]["real part"], dtype=float
                )

            def _extract_imag(jd: dict, t: np.ndarray) -> np.ndarray:
                return np.asarray(
                    jd["data"]["fourier_transform_pk"]["imaginary part"], dtype=float
                )

        self.add_extractor("parity_real", _extract_real)
        self.add_extractor("parity_imag", _extract_imag)

    def _discover_files(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.json"))

    def _extract_lambda_and_time(self, jd: dict, fp: Path = None) -> tuple[float, np.ndarray]:
        t = _read_time_us(jd)
        lam = _read_lambda_from_json(jd)
        return lam, t

    def load(self, folder, observables=None, **kwargs):
        if observables is None:
            observables = ["parity_real", "parity_imag"]
        return super().load(folder, observables=observables, **kwargs)
# ---------------------------------------------------------------------------
# Helper: safe lambda for sorting before the JSON is read
# ---------------------------------------------------------------------------

def _safe_lambda(fp: Path) -> float:
    """Return a sort key from the filename (kHz number), fallback to 0."""
    try:
        return float(_khz_from_name(fp))
    except ValueError:
        return 0.0

