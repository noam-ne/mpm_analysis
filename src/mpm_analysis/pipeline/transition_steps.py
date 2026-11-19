"""The steps the two paper transitions share, as plain functions.

``run_files/run.py`` runs both paper transitions step by step; the steps that
load the traces, run the matrix pencil with its bootstrap, find the exceptional
point and draw the appendix figure are defined here so that code outside
``run_files`` can call them too (ZenoPlotting's paper-figure scripts).

    load_records      step 1  the traces of one folder -> records
    run_mpm           step 2  matrix pencil + bootstrap -> (result, json_path)
    run_transitions   step 3  the exceptional point: zero of ONE weighted line
                              through Delta over a fixed lambda window -> ep
    run_figure        step 6  the appendix figure from (result, ep)

Nothing here reads a global setting or a machine-specific path: every folder,
the seed, the number of workers and the save / show switches are keyword
arguments.  ``run.py`` wraps these functions and passes its own GLOBAL SETTINGS.
"""
from __future__ import annotations

import fnmatch
import re
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline
from mpm_analysis.analysis.steps import MPMStep, BootstrapStep, RefineStep
from mpm_analysis.analysis.ksi_crossing import (
    discriminant_from_poles, linfit_zero_crossing, find_zero_crossings,
)
from mpm_analysis.data_types.analysis_result import ZenoAnalysisResult
from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.plotting.ksi_plot import plot_discriminant_curve
from mpm_analysis.io.analysis_json import augment_json_with_transitions
from mpm_analysis.utils.windows_paths import win_long_path
from mpm_analysis.plotting.exploratory import plot_spectrum_panels
from mpm_analysis.plotting._style import DPI
from mpm_analysis.io.file_naming import build_figure_filename


# Key under which load_records stamps the provenance into every record's
# metadata; run_mpm copies it into the result metadata (→ the saved JSON).
_PROVENANCE = "run_provenance"
# ... and, for a load from several folders, the folder each record's file sits in
_FILE_FOLDER = "source_folder_of_file"


def _show_and_save(fig, figure_tag: str, *, data_mode: str = "unknown",
                   scan_number: int | None = None,
                   debug: bool = False, output_dir: str | Path | None = None,
                   save_figures: bool = False, show_plots: bool = True) -> None:
    """Save the figure as png into ``output_dir`` (if enabled) and show it."""
    if save_figures and not debug and output_dir is not None:
        filename = build_figure_filename(
            figure_tag=figure_tag, data_mode=data_mode,
            scan_number=scan_number, extension="png",
        )
        path = Path(output_dir) / filename
        fig.savefig(path, bbox_inches="tight")
        print(f"  Figure saved: {path.name}")
    if show_plots:
        plt.show(block=False)


def _result_tags(result) -> tuple[str, int | None]:
    """(data_mode, scan_number) of a result, for figure file names."""
    data_mode = str(result.metadata.get("data_mode",
                                        result.metadata.get("source_tag", "unknown")))
    scan_number = result.metadata.get("scan_number")
    return data_mode, (int(scan_number) if scan_number is not None else None)


def _provenance(records: list[ObservableRecord]) -> dict:
    """The provenance load_records stamped into the records."""
    try:
        return records[0].metadata[_PROVENANCE]
    except (IndexError, KeyError):
        raise ValueError("records did not come from load_records() — "
                         "no provenance (data mode, folder, λ source) attached.") from None


def _observables(result: ZenoAnalysisResult) -> list[str]:
    """Observable keys the result was fitted on, in the order they were loaded."""
    return list(result.metadata["observables"])


def _central_poles(result: ZenoAnalysisResult) -> np.ndarray:
    """Central poles  -decay + i·frequency, shape (n_lambda, order)."""
    return -result.poles.decay_rates + 1j * result.poles.frequencies


def _check_pole_ordering(result: ZenoAnalysisResult) -> None:
    """Every bootstrap draw must label its poles the way the central fit does.

    Per λ: (1) no decay rate <= 0 (a growing pole), central or draw; (2) the
    EP pair the discriminant follows along λ is slots 0/1 in every draw —
    the splittings, the sqrt fit and ksi read slots 0/1/2 directly; (3) the
    stored central value (decay rate per slot, real and imaginary splitting)
    lies inside its draws' 16-84 band.  The aggregated half-widths do not show
    a central value outside the band (mislabelled draws, or a bias larger than
    the error bar): decay rates end their bar at the point, the others keep
    |c - p16|, |p84 - c|, so it would otherwise go unseen.
    Prints one line when everything passes, the offending λ otherwise.
    """
    raw = result.poles.raw_samples
    if raw is None:
        print("  ordering check: no bootstrap draws - skipped.")
        return
    from mpm_analysis.analysis.ksi_crossing import discriminant_from_poles

    lam = np.asarray(result.lambda_values, dtype=float)
    P = result.poles
    C = _central_poles(result)                  # (n_lambda, order)
    D = np.asarray(raw.samples)                 # (n_lambda, n_boot, order)
    problems = []

    n_c, n_d = int(np.sum(C.real >= 0)), int(np.sum(D.real >= 0))
    if n_c or n_d:
        problems.append(f"decay rate <= 0: {n_c} central, {n_d} draw poles")

    if C.shape[1] >= 2:
        d_track = discriminant_from_poles(D)
        d_slot = np.real((D[..., 0] - D[..., 1]) ** 2)
        ok = np.isfinite(d_track)
        off = ok & ~np.isclose(d_track, d_slot, rtol=1e-6, atol=1e-12)
        bad = np.flatnonzero(off.any(axis=1))
        if bad.size:
            problems.append("EP pair not in slots 0/1 at lam = " + ", ".join(
                f"{lam[i]:.3f} ({off[i].sum()}/{ok[i].sum()} draws)" for i in bad))

    outside = []
    for i in range(len(lam)):
        d = D[i][np.all(np.isfinite(D[i]), axis=1)]
        if len(d) < 2:
            continue
        checks = [(f"decay{k}", -d[:, k].real, P.decay_rates[i, k]) for k in range(d.shape[1])]
        if d.shape[1] >= 2 and P.has_splitting:
            checks += [("Re split", np.abs(d[:, 1].real - d[:, 0].real) / 2, P.real_splitting[i]),
                       ("Im split", np.abs(d[:, 1].imag - d[:, 0].imag) / 2, P.imag_splitting[i])]
        for name, x, c in checks:
            lo, hi = np.percentile(x, [16, 84])
            if np.isfinite(c) and not lo - 1e-12 <= c <= hi + 1e-12:
                outside.append(f"{lam[i]:.3f} {name}")
    if outside:
        problems.append("central value outside its draws' 16-84 band: " + ", ".join(outside))

    if problems:
        print("  ordering check: PROBLEMS\n    " + "\n    ".join(problems))
    else:
        print(f"  ordering check: OK - {D.shape[1]} draws x {len(lam)} lambdas: all decay "
              "rates > 0, EP pair = slots 0/1 in every draw, every central value inside "
              "its draws' 16-84 band")


def _resolve_uncertainty_source(source: str, data_mode: str) -> str:
    """Validate the uncertainty source and apply per-mode fallbacks."""
    if source not in ("data_bootstrap", "errorbar_bootstrap", "fit_covariance",
                      "residual_bootstrap"):
        raise ValueError(f"Unknown uncertainty_source '{source}'.")
    if source == "data_bootstrap" and data_mode not in ("ens_avg", "post_selected"):
        print(f"Warning: data bootstrap has no measurement statistics for "
              f"'{data_mode}' — falling back to residual_bootstrap.")
        source = "residual_bootstrap"
    return source


def load_records(
    loader, folder: str | Path, *,
    data_mode: str,
    observables: list[str],
    lambda_range: tuple[float, float] | None,
    t_range: tuple[float | None, float | None] | None,
    drop_first_point: bool,
    file_glob: str | tuple[str, ...] = "*.json",
    max_files: int | None = None,
) -> list[ObservableRecord]:
    """Step 1 — load the traces.  Returns the records.

    One ``ObservableRecord`` per (λ file × observable), sorted by λ then
    observable; ``record.metadata`` carries the bootstrap statistics the
    loader found, plus the provenance of this load (data mode, folder, file
    glob, λ source, observables, t_range, drop_first_point) that run_mpm
    writes into the result JSON.  ``t_range = (t_min, t_max)`` in µs keeps
    only ``t_min <= t <= t_max`` (either end ``None`` → open); ``None`` keeps
    the full record.  ``file_glob`` keeps only the files whose name matches
    (one source run of a merged folder); a tuple of globs / file names keeps
    their union.  ``max_files`` caps the number of files (None → all).  A ``folder`` that holds
    ``scan_exp_<scan>`` subfolders (one subfolder per scan), or a
    tuple / list of folders: each is loaded on its own and the records are merged in λ order;
    the provenance then lists every folder and each record keeps its own (``_FILE_FOLDER``).
    """
    if not isinstance(folder, (list, tuple)):
        # one subfolder per scan → load them side by side, in scan order (3, 14, 16)
        scans = [Path(folder) / p.name for p in Path(win_long_path(folder)).glob("scan_exp_*")
                 if p.is_dir()]
        if scans:
            folder = sorted(scans, key=lambda p: int(re.search(r"\d+", p.name).group()))
    if isinstance(folder, (list, tuple)):
        records = []
        for f in folder:
            loader._resolved = False      # the loader caches ONE folder's λ table: resolve this folder's
            part = load_records(loader, f, data_mode=data_mode, observables=observables,
                                lambda_range=lambda_range, t_range=t_range,
                                drop_first_point=drop_first_point, file_glob=file_glob,
                                max_files=max_files)
            if loader.lambda_source.startswith("json"):
                raise RuntimeError(f"[{data_mode}] {f} has no HMM λ table: its files would sit at "
                                   "the stored JSON λ next to HMM λ files.  Add the table.")
            for r in part:
                r.metadata[_FILE_FOLDER] = str(f)
            records += part
        records.sort(key=lambda r: float(r.lambda_val))   # stable: a file's observables keep their order
        provenance = {**records[0].metadata[_PROVENANCE], "source_folder": [str(f) for f in folder]}
        for r in records:
            r.metadata[_PROVENANCE] = provenance
        print(f"  merged {len(records)} records from {len(folder)} folders")
        return records
    folder = Path(folder)
    if not Path(win_long_path(folder)).exists():
        raise FileNotFoundError(f"[{data_mode}] folder not found: {folder}")
    names = [p.name for p in Path(win_long_path(folder)).glob("*.json")]
    patterns = (file_glob,) if isinstance(file_glob, str) else tuple(file_glob)
    if not any(fnmatch.fnmatch(n, p) for n in names for p in patterns):
        stamps = sorted({m.group(1) for n in names
                         if (m := re.search(r"_kHz_(.+)\.json$", n))})
        raise FileNotFoundError(
            f"[{data_mode}] no file in {folder} matches {patterns[:3]!r}; the file stamps "
            f"there are {stamps}.")
    observables = list(observables)
    if t_range is None:
        t_min = t_max = None
    else:
        t_min, t_max = t_range
        if t_min is not None and t_max is not None and not t_min < t_max:
            raise ValueError(f"[{data_mode}] t_range must be (t_min, t_max) with "
                             f"t_min < t_max, got {t_range}.")
    records = loader.load(
        folder,
        observables=observables,
        lambda_range=lambda_range,
        max_files=max_files,
        file_glob=file_glob,
        t_min=t_min,
        t_max=t_max,
        drop_first_point=drop_first_point,
    )
    if not records:
        raise RuntimeError(f"[{data_mode}] no records loaded from {folder} "
                           f"(files {file_glob}, observables={observables}, "
                           f"lambda_range={lambda_range}).")
    provenance = {
        "data_mode": data_mode,
        "source_folder": str(folder),
        "file_glob": file_glob if isinstance(file_glob, str) else list(file_glob),
        "lambda_source": loader.lambda_source,
        "observables": observables,
        "lambda_range": None if lambda_range is None else [float(x) for x in lambda_range],
        "t_range": None if t_range is None else [t_min, t_max],
        "drop_first_point": bool(drop_first_point),
    }
    for rec in records:
        rec.metadata[_PROVENANCE] = provenance
    lambdas = sorted({float(r.lambda_val) for r in records})
    t0, t1 = float(records[0].t[0]), float(records[0].t[-1])
    print(f"\n=== Load ({data_mode}) ===\n"
          f"  {len(records)} records, {len(lambdas)} lambda points in "
          f"[{lambdas[0]:.3f}, {lambdas[-1]:.3f}], observables={observables}, "
          f"lambda axis = {loader.lambda_source}\n"
          f"  time window t_range={t_range} -> {records[0].n_points} points, "
          f"t = [{t0:.3f}, {t1:.3f}] us\n  {folder}"
          + ("" if file_glob == "*.json" else f"  (files {file_glob})" if isinstance(file_glob, str)
             else f"  ({len(patterns)} files)"))
    return records


def run_mpm(
    records: list[ObservableRecord], *,
    order: int,
    uncertainty_source: str,
    n_boot: int,
    refine_poles: bool,
    conjugate_pairs: bool,
    plot_svd: bool,
    bootstrap_center: str = "measured",
    redraw_around: str = "measured",
    nyquist_check: bool = True,
    plot_spectrum: bool = True,
    seed: int | None = None,
    n_jobs: int = -1,
    debug: bool = False,
    output_dir: str | Path | None = None,
    save_full_analysis: bool = False,
    save_figures: bool = False,
    show_plots: bool = True,
) -> tuple[ZenoAnalysisResult, Path | None]:
    """Step 2 — matrix pencil + bootstrap on the records.

    Returns ``(result, json_path)``: the ``ZenoAnalysisResult`` (central
    poles, 16/84 bands, raw bootstrap draws in ``result.poles.raw_samples``)
    and the path of the result JSON saved into ``output_dir`` (``None`` with
    ``debug`` or without an ``output_dir``; ``save_full_analysis`` adds the raw
    pole draws to it).  Also draws the spectrum panels (``plot_spectrum``).
    ``seed`` / ``n_jobs``: RNG seed (None → OS entropy) and parallel workers
    (-1 → all cores) of the bootstrap.  ``nyquist_check=False`` keeps the pole
    sets (and draws) with a pole at the Nyquist frequency instead of blanking
    them; step 3's Delta then leaves that pole out of the pair.
    """
    prov = _provenance(records)
    print(f"\n=== MPM ({prov['data_mode']}, observables={prov['observables']}, order={order}) ===")
    pipeline, result = _mpm_pipeline(
        records, order=order, uncertainty_source=uncertainty_source, n_boot=n_boot,
        refine_poles=refine_poles, conjugate_pairs=conjugate_pairs, plot_svd=plot_svd,
        bootstrap_center=bootstrap_center, redraw_around=redraw_around,
        nyquist_check=nyquist_check,
        seed=seed, n_jobs=n_jobs, debug=debug, output_dir=output_dir,
    )
    _check_pole_ordering(result)

    if plot_spectrum:
        fig, _ = plot_spectrum_panels(result, title=prov["data_mode"])
        fig.tight_layout()
        tag_mode, scan_number = _result_tags(result)
        _show_and_save(fig, "spectrum", data_mode=tag_mode, scan_number=scan_number,
                       debug=debug, output_dir=output_dir, save_figures=save_figures,
                       show_plots=show_plots)

    json_path = None
    if not debug and output_dir is not None:
        json_path = pipeline.save(output_dir, full=save_full_analysis)
        print(f"Saved: {json_path.name}")
    return result, json_path


def _mpm_pipeline(
    records: list[ObservableRecord], *,
    order: int,
    uncertainty_source: str,
    n_boot: int,
    refine_poles: bool,
    conjugate_pairs: bool,
    plot_svd: bool,
    bootstrap_center: str,
    redraw_around: str = "measured",
    nyquist_check: bool = True,
    seed: int | None = None,
    n_jobs: int = -1,
    debug: bool = False,
    output_dir: str | Path | None = None,
) -> tuple[MPMPipeline, ZenoAnalysisResult]:
    """MPM → (Refine) → Bootstrap on the records, run (step 2 without its
    figure and JSON; the window sweep reruns it per window).  ``debug`` caps
    the bootstrap at 20 draws; ``output_dir`` is where the SVD figure goes.
    Returns ``(pipeline, result)``."""
    prov = _provenance(records)
    data_mode, observables = prov["data_mode"], prov["observables"]
    src = _resolve_uncertainty_source(uncertainty_source, data_mode)
    n_boot = min(n_boot, 20) if debug else n_boot
    steps = [
        MPMStep(order=order, observable_key=observables,
                plot_svd=plot_svd, n_sv=10, save_dir=output_dir,
                drop_aliased=nyquist_check),
    ]
    if refine_poles or src == "fit_covariance":
        # fit_covariance needs the refinement covariance → RefineStep forced on
        steps.append(RefineStep(observable_key=observables,
                                conjugate_pairs=conjugate_pairs))
    mode = {"data_bootstrap": "data", "errorbar_bootstrap": "errorbar",
            "fit_covariance": "parametric", "residual_bootstrap": "residual"}[src]
    steps.append(BootstrapStep(n_boot=n_boot, seed=seed, mode=mode, n_jobs=n_jobs,
                               center=bootstrap_center, redraw_around=redraw_around,
                               drop_aliased=nyquist_check))

    # Same metadata MPMPipeline.from_loader would have written.
    metadata = dict(
        source_folder=prov["source_folder"],
        data_mode=data_mode,
        source_tag=data_mode,          # legacy alias kept for old readers
        observables=observables,
        lambda_source=prov["lambda_source"],
        uncertainty_source=src,
        bootstrap_center=bootstrap_center,
        bootstrap_redraw=redraw_around,
    )
    if prov["drop_first_point"]:
        metadata["drop_first_point"] = True
    if not nyquist_check:
        metadata["nyquist_check"] = False   # step 3 reads it: Delta skips the aliased pole
    if prov.get("svd_cut") is not None:
        metadata["svd_cut"] = prov["svd_cut"]
    if prov.get("file_glob", "*.json") != "*.json":
        metadata["file_glob"] = prov["file_glob"]
    pipeline = MPMPipeline.from_records(records, steps=steps, **metadata)
    return pipeline, pipeline.run()


def run_transitions(
    result: ZenoAnalysisResult, *,
    json_path: Path | None,
    method: str,
    lambda_max: float | None,
    plot_discriminant: bool,
    half_width: float = 0.3,
    points_below: int | None = None,
    points_above: int | None = None,
    lambda_window: tuple[float, float] | None = None,
    debug: bool = False,
    output_dir: str | Path | None = None,
    save_figures: bool = False,
    show_plots: bool = True,
) -> dict:
    """Step 3 — the exceptional point on the pole strands, with its error bar.

    exceptional pt : lam_EP, zero of the pair discriminant Delta = c1² − 4·c0

    The bootstrap is used once: at every λ, half the 16-84 % spread of the
    draws' Delta is that point's error bar.  Then ONE straight line is
    fitted to the measured Delta around its sign change, weighted by those
    error bars; λ_EP is the line's zero and its 1-σ error comes from the
    fit's covariance (error bars taken as absolute; χ²/dof printed beside).
    No line is fitted to individual draws.  (State freezing is a separate
    step: ``run_pole_state_freezing`` / ``run_spiral_state_freezing`` — only
    the first transition has one.)

    Returns ``ep``, a dict with ``"lambda_ep"`` (NaN if none),
    ``"lambda_ep_error"`` (None without bootstrap draws), ``"chi2"``,
    ``"dof"``, ``"delta_crossing_method"``, ``"delta_half_width"``,
    ``"delta_lambda_max"`` and ``"delta_fit"`` (the line: ``lambda_ep``,
    ``coef``, ``lambda_window``) that the paper figure draws.  Appended to
    ``json_path`` (unless ``debug`` / None).
    """
    data_mode, scan_number = _result_tags(result)
    obs_tag = "+".join(_observables(result))
    print(f"\n=== Exceptional point ({data_mode}, {obs_tag}) ===")

    c = _ep_crossing(result, method=method, half_width=half_width, lambda_max=lambda_max,
                     points_below=points_below, points_above=points_above,
                     lambda_window=lambda_window)
    lam_d, d_central, d_sigma = c["lam_d"], c["d_central"], c["d_sigma"]
    lam_ep, err, fit, first = c["lam_ep"], c["lam_ep_err"], c["fit"], c["first"]
    # a fixed λ window: a zero outside it is an extrapolation, not a crossing
    outside = bool(lambda_window is not None and np.isfinite(lam_ep)
                   and not lambda_window[0] <= lam_ep <= lambda_window[1])
    ep: dict = {"delta_crossing_method": method, "delta_half_width": half_width,
                "delta_points_below": points_below, "delta_points_above": points_above,
                "delta_lambda_window": None if lambda_window is None else list(lambda_window),
                "delta_lambda_max": lambda_max, "lambda_ep": float(lam_ep),
                "lambda_ep_error": err, "chi2": c["chi2"], "dof": c["dof"],
                "lambda_ep_outside_window": outside}
    # The line itself (coef in the poles' units): the paper figure (step 6)
    # draws this fit, it does not refit.
    ep["delta_fit"] = dict(
        lambda_ep=float(lam_ep),
        coef=fit["coef"],
        lambda_window=(None if fit["coef"] is None else
                       (float(lam_d[fit["window"]].min()), float(lam_d[fit["window"]].max()))),
    )

    if not c["has_draws"]:
        print("  EXCEPTIONAL PT  no bootstrap draws (n_boot = 0): no error bars on Delta, "
              "unweighted fit, no error on lam_EP.")
    if not np.isfinite(lam_ep):
        print("  EXCEPTIONAL PT  no crossing: the measured Delta has no sign change.")
    elif err is None:
        print(f"  EXCEPTIONAL PT  lam_EP     = {lam_ep:.4f}  (no error bar)")
    else:
        red = c["chi2"] / c["dof"] if c["dof"] else np.nan
        print(f"  EXCEPTIONAL PT  lam_EP     = {lam_ep:.4f} +/- {err:.4f}   "
              f"[weighted line fit to the measured Delta, chi2/dof = {c['chi2']:.2f}/{c['dof']}"
              + (f" = {red:.2f}; error x sqrt(chi2/dof) = {err * np.sqrt(red):.4f}" if red > 1 else "")
              + "]")
    if outside:
        print(f"  NOTE  the line's zero {lam_ep:.3f} lies OUTSIDE the fit range lam "
              f"{lambda_window[0]:g}-{lambda_window[1]:g}: Delta does not cross zero inside it "
              "(extrapolated value, not an EP; not drawn).")
    if fit["window"] is not None:
        wl = lam_d[fit["window"]]
        n_lo = int(np.sum(wl < lam_ep))
        rule = (f"every point of lam {lambda_window[0]:g}-{lambda_window[1]:g}"
                if lambda_window is not None else
                f"fixed: {points_below} before / {points_above} after Delta turns positive"
                if points_below is not None or points_above is not None
                else f"+-{half_width}" + (" held fewer than 5" if fit.get("widened") else ""))
        print(f"                  fit points: lam {', '.join(f'{x:.3f}' for x in wl)}  "
              f"({n_lo} below / {wl.size - n_lo} above lam_EP; {rule}"
              + "); raw first sign change at " + (f"{first[0]:.4f}" if first else "none"))

    if plot_discriminant:
        ax_d = plot_discriminant_curve(
            lam_d, d_central, fit=fit, lambda_ep=None if outside else lam_ep,
            lambda_ep_sigma=None if outside else err,
            first_crossing=first[0] if first else None, delta_sigma=d_sigma,
            title=f"Exceptional point: Delta = 0  ({data_mode}, {obs_tag})",
        )
        _show_and_save(ax_d.get_figure(), "exceptional_point_discriminant",
                       data_mode=data_mode, scan_number=scan_number,
                       debug=debug, output_dir=output_dir, save_figures=save_figures,
                       show_plots=show_plots)

    if json_path is not None and not debug and np.isfinite(lam_ep):
        saved = {"delta_crossing_method": method, "delta_linfit_central": float(lam_ep)}
        if err is not None:
            saved.update(delta_linfit_sigma=float(err), delta_linfit_chi2=c["chi2"],
                         delta_linfit_dof=c["dof"])
        augment_json_with_transitions(json_path, saved)
    return ep


def _ep_crossing(
    result: ZenoAnalysisResult, *,
    method: str,
    half_width: float,
    lambda_max: float | None,
    points_below: int | None = None,
    points_above: int | None = None,
    lambda_window: tuple[float, float] | None = None,
) -> dict:
    """Delta(λ) of the EP pair, its error bars and the λ_EP of ONE weighted
    line fit (the numbers of step 3; the window sweep reuses them per window).

    1. Delta at every λ from the measured (unresampled) poles.  Delta follows
       the EP pair along λ: past the EP it stays on the two real poles that
       continue the pair, not on whichever two are closest.
    2. Its error bar from the bootstrap: half the 16-84 % spread of the
       draws' Delta at that λ.
    3. One straight line through the measured Delta around its sign change,
       weighted 1/σ²; λ_EP = its zero, with the 1-σ error propagated from the
       fit covariance (error bars absolute) and χ² of the fit.  The points:
       ``points_below`` / ``points_above`` around the first rise of Delta
       through zero when given (fixed count, no re-centring), else
       ±``half_width`` around the sign change (≥ 5 nearest); ``lambda_window``
       = (lo, hi) instead fits EVERY point in that λ range (no seed).

    ``method="first"``: λ_EP = the first sign change, no error.  Returns a
    dict: ``lam_d``, ``d_central``, ``d_sigma`` (None without draws),
    ``lam_ep``, ``lam_ep_err`` (None without draws), ``chi2``, ``dof``,
    ``fit`` (window, coef, …), ``first`` (raw sign changes), ``has_draws``.
    """
    import warnings

    lam_d = np.asarray(result.lambda_values, dtype=float)
    skip = not result.metadata.get("nyquist_check", True)   # Nyquist check off: kept aliased poles
    d_central = discriminant_from_poles(_central_poles(result), skip_aliased=skip)
    keep = np.ones(lam_d.shape, dtype=bool) if lambda_max is None else lam_d <= lambda_max
    lam_d, d_central = lam_d[keep], d_central[keep]

    d_sigma = None
    raw = result.poles.raw_samples
    if raw is not None:
        d_draws = discriminant_from_poles(np.asarray(raw.samples), skip_aliased=skip)[keep]   # (n_lam, n_boot)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # failed draws: NaN
            lo, hi = np.nanpercentile(d_draws, [16, 84], axis=1)
        d_sigma = (hi - lo) / 2

    lam_ep, fit = linfit_zero_crossing(lam_d, d_central, sigma=d_sigma,
                                       half_width=half_width, return_fit=True,
                                       n_below=points_below, n_above=points_above,
                                       window=None if lambda_window is None else tuple(lambda_window))
    first = find_zero_crossings(lam_d, d_central)
    lam_ep_err = fit.get("sigma_zero")
    if method == "first":
        lam_ep, lam_ep_err = (first[0] if first else np.nan), None
    return dict(lam_d=lam_d, d_central=d_central, d_sigma=d_sigma,
                lam_ep=float(lam_ep), lam_ep_err=lam_ep_err,
                chi2=fit.get("chi2"), dof=fit.get("dof"), fit=fit, first=first,
                has_draws=raw is not None)


def run_figure(
    result: ZenoAnalysisResult, ep: dict, *,
    json_path: Path | None,
    figure_name: str,
    layout: str,
    lambda_max: float,
    fit_validity: float | None,
    real_axis_scale: str,
    real_axis_ticks: list | None,
    real_split_axis_ticks: list | None,
    splitting_fit: str,
    panel_c: str,
    split_lambda_ticks: list | None,
    error_bar_alpha: float = 1.0,
    crit_line_decimals: int | None = None,
    debug: bool = False,
    save_figures: bool = False,
    plots_dir: str | Path | None = None,
    dpi: int = DPI,
    show_plots: bool = True,
):
    """Step 6 — paper transition figure from this run's result.

    The Δ line fit and λ_EP drawn are step 3's (``ep["delta_fit"]``), not a
    refit, so the figure shows exactly what step 3 reported.
    ``layout="sqrt"`` with ``splitting_fit="sqrt"`` needs
    ``result.critical_point_sqrt_fit`` (step 4); with ``"discriminant"`` the
    overlay is the square root of step 3's line and step 4 is not needed.
    ``panel_c`` ("sqrt" layout): panel C shows ``"real"`` Re(Δe12)/Ω_S, or
    ``"discriminant"`` (Δe12)²/Ω_S² = Delta/(4 Ω_S²) at every λ with step 3's
    line over its window; ``split_lambda_ticks``: the λ ticks of panels B/C
    (None → every 0.5).
    The Im / Re bars are the attempts' 16-84 % range of Delta (not symmetrized)
    converted to splittings; only the Delta line fit (and the Delta points drawn
    with it) uses the symmetrized bar.
    ``error_bar_alpha`` is the opacity of all bars.  ``crit_line_decimals``:
    the vertical λ_EP line is drawn at λ_EP rounded to that many decimals —
    the value as the paper quotes it (2 → 1.30) — in every panel; None →
    unrounded.  The dashed curves still end at the fit's own zero.
    Returns the figure, or ``None`` when the sqrt fit is missing.  With
    ``save_figures`` (and not ``debug``): ``<figure_name>.pdf`` / ``.svg`` /
    ``.png`` (``dpi``) into ``plots_dir``/appendix — the paper's file names, so a
    rerun overwrites them — and the data the figure shows beside it as
    ``<figure_name>_data.json``: a copy of ``json_path``, this run's result JSON
    with step 3's line appended.
    """
    from mpm_analysis.plotting.paper_figures.first_transition_main import (
        plot_first_transition_figure,
    )
    data_mode, observables = _result_tags(result)[0], _observables(result)
    print(f"\n=== Paper figure ({data_mode}, {'+'.join(observables)}, {layout}) ===")
    if (layout == "sqrt" and splitting_fit == "sqrt"
            and getattr(result, "critical_point_sqrt_fit", None) is None):
        print("[figure] result carries no sqrt fit — enable the critical point "
              "(or use splitting_fit='discriminant').")
        return None
    fig = plot_first_transition_figure(
        result,
        layout=layout,
        lambda_max=lambda_max,
        fit_validity_limit=fit_validity,
        splitting_fit=splitting_fit,
        real_axis_scale=real_axis_scale,
        real_axis_ticks=real_axis_ticks,
        real_split_axis_ticks=real_split_axis_ticks,
        panel_c=panel_c,
        split_lambda_ticks=split_lambda_ticks,
        discriminant_fit=ep["delta_fit"],
        error_alpha=error_bar_alpha,
        crit_line_decimals=crit_line_decimals,
    )
    # Paper figures: the shared plots folder, PDF + SVG + high-res PNG under the
    # paper's file name, and the result JSON the figure was drawn from beside it.
    if save_figures and not debug:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        if plots_dir is None:
            raise ValueError("run_figure: save_figures needs plots_dir (the paper's Plots folder).")
        out_dir = Path(plots_dir) / "appendix"
        save_figure_pdf_svg(fig, figure_name, out_dir, dpi=dpi, png=True)
        if json_path is not None:
            data_path = out_dir / f"{figure_name}_data.json"
            shutil.copy2(win_long_path(json_path), win_long_path(data_path))
            print(f"  Data: {data_path}  (= {Path(json_path).name})")
    if show_plots:
        plt.show(block=False)
    return fig
