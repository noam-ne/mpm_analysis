"""MPM analysis driver for the Zeno paper transitions.

How to run
----------
    python run_files/run.py

Everything is configured by the ``if (X := True/False):`` toggles inside
``main()`` — flip the boolean in the ``if`` line to switch a step on or off.

Each paper transition is run step by step.  Every step is one clause of the
shape

    if (FIRST_<STEP> := True):
        first_<product> = run_<step>(<products of earlier steps>, <settings>)

so a clause states what the step consumes (positional arguments = products
of earlier steps, keyword arguments = this step's own settings) and what it
produces (the name on the left).  A step that needs an earlier product is
nested under that product's clause, so it cannot run without it.  The step
functions take keyword-only settings: a misspelt or missing argument is a
``TypeError`` at the call, never a silently ignored key.

    RUN_FIRST_TRANSITION   post-selected survival  (scans 3, 14, 16) → first transition
    RUN_THIRD_TRANSITION   ensemble-average  D     (scans 0 + 1)     → third transition
    RUN_MC_TWIN            the first-transition steps on a simulated scan (MC_RUNS)
    RUN_SIM_GRID           the first-transition steps on a simulated (λ × T) survival grid NPZ

Steps of a transition and what they produce:

    1  load_records       → records       list[ObservableRecord], one per (λ file × observable)
    1b load_records       → tomo          (first only) raw conditional <X>, <Z> + S for step 3b
    1c run_svd_gap_cut    → records       (first only) drops the λ traces whose σ_order/σ_order+1
                                          Hankel singular-value ratio is near the noise floor
    2  run_mpm            → result, json  matrix-pencil poles + error bars; the saved result JSON
    3  run_transitions    → ep            exceptional point (Delta = 0 of the pole pair)
    3c run_window_sweep   → window_sweep  (first only) steps 1-3 redone for ~6 fit windows
                                          (t_min or t_max moved); one Delta figure, λ_EP ± error
                                          per window in the legend
    3a run_pole_state_freezing → ksi      (first only) state freezing from the pencil poles,
                                          ksi = (2·Re e3 − Re e2)/(Re e2 − Re e3) = 0
    3b run_spiral_state_freezing → spiral (first only) state freezing from a DIRECT fit of the
                                          tomography spiral: ksi = γ0/γa − 1 = 0, no pencil
    4  run_critical_point → sqrt_fit      sqrt fit of the splitting → lambda_c
    5  run_slider_plot    → slider        interactive per-λ trace browser with the fitted poles
                                          (the summed fit, or each term in its own colour)
    6  run_figure         → fig           appendix figure from this run's result; saved under its
                                          paper name, with its data, in the paper's Plots/appendix
    7  run_export_figure_data → folder    per-λ fit curves in a NEW fitting_results_<stamp> folder.
                                          third: + the result JSON, in the data folder's
                                          pole_analysis_data_for_main_text_figure — what
                                          make_third_transition_figure.py draws the main-text
                                          figure from.  first: the S_next curves of chosen drives
                                          in the paper data folder's format (PAPER_FIT_REFERENCE),
                                          from its own S_next load + measured fit
    8  run_pole_table    → pole_table   (first only) the three poles of every file with the
                                          bootstrap error bars as one CSV row per file (+ a text
                                          file describing the bootstrap), in POST_SELECTED_FOLDER

The third transition (ensemble average) has no state-freezing exponent — only
steps 3 and 4 run on it.

The stand-alone checks at the end of ``main()`` (simulation test, benchmark,
Liouvillian spectrum) need no experimental data.

The module-level settings (paths, MC twin, λ source) are importable without
running anything (run_files/write_shots_per_lambda.py reads
``POST_SELECTED_FOLDER`` that way); the steps only run under ``__main__``.
"""
from __future__ import annotations

import csv
import fnmatch
import functools
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mpm_analysis.utils.lab_paths import get_lab_path
from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline
from mpm_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline
from mpm_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline
from mpm_analysis.analysis.steps import MPMStep, BootstrapStep, RefineStep
from mpm_analysis.analysis.ksi_crossing import (
    KsiCrossingPipeline, KsiCrossingResult,
    discriminant_from_poles, linfit_zero_crossing, find_zero_crossings,
)
from mpm_analysis.analysis.matrix_pencil import (
    solve_linear_amplitudes, reconstruct_signal,
)
from mpm_analysis.analysis.critical_point import fit_sqrt_per_realization
from mpm_analysis.analysis.spiral_approach import SpiralKsiPipeline, SpiralKsiResult
from mpm_analysis.plotting.spiral_plot import plot_spiral_fits_slider
from mpm_analysis.data_types.analysis_result import ZenoAnalysisResult
from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.data_types.pole_samples import PoleSamples
from mpm_analysis.plotting.ksi_plot import (
    plot_ksi_curve, plot_discriminant_curve, plot_discriminant_window_sweep,
)
from mpm_analysis.io.analysis_json import (
    augment_json_with_ksi, augment_json_with_transitions,
)
from mpm_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator
from mpm_analysis.simulation.liouvillian_spectrum import LiouvillianSpectrum
from mpm_analysis.io.experimental import EnsembleAverageLoader, PostSelectedLoader
from mpm_analysis.utils.windows_paths import win_long_path
from mpm_analysis.io.simulation import load_monte_carlo_npz
from mpm_analysis.simulation.noise import ShotNoise
from mpm_analysis.plotting.exploratory import (
    plot_spectrum_panels, plot_time_traces_slider, plot_svd_singular_values,
)
from mpm_analysis.plotting.paper_figures.appendix_bootstrap import plot_appendix_bootstrap
from mpm_analysis.io.file_naming import build_figure_filename, save_result
from mpm_analysis.pipeline import transition_steps as _steps
from mpm_analysis.pipeline.transition_steps import (   # these read no setting of this file
    _PROVENANCE, _FILE_FOLDER, _result_tags, _provenance, _observables, _central_poles,
    _check_pole_ordering, _resolve_uncertainty_source, _ep_crossing,
)

# ===========================================================================
# GLOBAL SETTINGS  (apply to every step)
# ===========================================================================

DEBUG = False               # True → few files, small bootstrap, nothing saved
SHOW_PLOTS = True       # plt.show() at the end of the run
SAVE_FIGURES = False       # png into OUTPUT_DIR (paper figures, step 6: PDF+SVG+PNG + their data
                           #   into PAPER_PLOTS_DROPBOX/appendix, overwriting the paper's files)
SAVE_FULL_ANALYSIS = False  # also write raw pole samples + refine params/cov into the result JSON
SEED: int | None = None     # RNG seed for every bootstrap; None → OS entropy
N_JOBS = -1                 # parallel workers for the data bootstrap (-1 → all cores)
MAX_FILES = 10 if DEBUG else None   # load only the first N λ files (debugging)

# ===========================================================================
# PATHS
# ===========================================================================
# Dropbox root is auto-detected from the hostname (utils/lab_paths.py); add
# your machine there if you get a "hostname not recognised" ValueError.
# Both transitions read their data from the shared paper data folder.  Paths
# under it can pass Windows' 260 characters: every folder listing and file
# access goes through utils/windows_paths.win_long_path.

PAPER_DATA_DROPBOX = Path(get_lab_path(r"Common\QuantumZeno\Paper\zeno_paper_data"))

# First transition — post-selected survival, scans 3, 14 and 16.  One subfolder per scan
# (scan_exp_3, scan_exp_14, scan_exp_16: load_records loads them side by side and merges the
# records in λ order) and hmm_lambda_per_file.csv, the HMM λ of every file.  Scan 16 is a HALF
# record (62 segments, to 12.4 µs).  README.md in the folder describes the scans and the arrays.
POST_SELECTED_FOLDER = PAPER_DATA_DROPBOX / "qj_union_scan_exp_3_14_16"

# Third transition — ensemble-average populations, scans 0 and 1 merged per drive, with
# hmm_rates_per_drive.csv, the HMM rates of every drive.
ENS_AVG_FOLDER = PAPER_DATA_DROPBOX / "combined_results_0_and_1_merged"
# ... and third-transition step 7's fitting_results_<stamp> folders: the pole analysis behind the
# main-text figure, which run_files/make_third_transition_figure.py draws from
THIRD_POLE_DATA_FOLDER = ENS_AVG_FOLDER / "pole_analysis_data_for_main_text_figure"

# The paper's figure folder, next to the data folder.  Step 6 saves the appendix figures there
# under their paper names (a rerun overwrites them), each with the data it shows beside it.
PAPER_PLOTS_DROPBOX = PAPER_DATA_DROPBOX.parent / "Plots"

# The S fits whose file format first-transition step 7 copies (reference_format): one
# <scan file>_mpm_fit.json per drive, keys times_smooth / fit_curve / lambda / original_file
PAPER_FIT_REFERENCE = (PAPER_DATA_DROPBOX / "deprecated_first_submission_data"
                       / "scan_experiment_results_10" / "fitting_results_20251202_162520")

# First-transition critical points (step 3 Delta line, step 3c window sweep, step 4 sqrt fit, the MC
# twin) are fitted ONLY on the λ inside this window.  29 Oct 2026, scans 3+4+5+14+16, t_max 12.4 µs:
# (1.0, 1.5) is the most stable choice - narrower windows only enlarge the errors (the t_min = 0.2 µs
# row deviates for every window: the bins 1-2 start-up excess).
EP_LAMBDA_WINDOW = (1.1, 1.5)

# Monte Carlo twins live in their own clause (RUN_MC_TWIN in main()) and never
# replace the data folders above.  mpm_benchmarks/scan8/mc/run_mc.py writes
# runs/<variant>/post_selected/ (scan-8 twin) or ens_avg/ (scan-0 twin) in the
# scanners' file formats, plus run_summary.json with the simulated parameters.
MC_RUNS = Path(__file__).parent.parent.parent / "mpm_benchmarks" / "scan8" / "mc" / "runs"

# λ axis.  HMM_DRIVE_LAMBDA True: λ = Γ_BG / (2·Ω_DG) from the offline HMM table (converged
# three-state HMM of the alpha click records) stored in the data folder, looked up exactly —
# per file (first transition) / per drive (third) — never interpolated; a file or drive
# missing from its table is an error.  False: each file's stored
# parameters.lambda_zeno_estimation, the scanner's online estimate.
LAMBDA_SOURCE_POST: bool | str | Path = False   # PostSelectedLoader csv_path (first transition)
LAMBDA_SOURCE_ENS: bool | str | Path = False    # EnsembleAverageLoader csv_path (third transition)
if (HMM_DRIVE_LAMBDA := True):
    LAMBDA_SOURCE_POST = POST_SELECTED_FOLDER / "hmm_lambda_per_file.csv"
    LAMBDA_SOURCE_ENS = ENS_AVG_FOLDER / "hmm_rates_per_drive.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "mpm_analysis_output"
PAPER_FIGURES_DIR = Path(__file__).parent.parent / "paper_figures"   # local drafts (make_appendix_options.py)
PAPER_FIGURES_DPI = 600


# ===========================================================================
# MAIN — the transitions, step by step
# ===========================================================================
# The step functions are defined further down (STEP FUNCTIONS); Python looks
# them up when main() runs, so the toggles can stay up here.

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # FIRST TRANSITION — post-selected survival probability
    # -----------------------------------------------------------------------
    if (RUN_FIRST_TRANSITION := False):
        print("\n##### FIRST TRANSITION  (post_selected) #####")

        # Step 1 — load.
        #   in : loader (λ axis from LAMBDA_SOURCE_POST) + folder, observables
        #   out: first_records — one ObservableRecord per (λ file × observable);
        #        the measurement statistics for the data bootstrap ride along
        #        in record.metadata, and the provenance (data mode, folder, λ
        #        source, drop_first_point) is stamped in for the result JSON.
        first_records = load_records(
            PostSelectedLoader(csv_path=LAMBDA_SOURCE_POST), POST_SELECTED_FOLDER,
            data_mode="post_selected",      # tag written into the result JSON / file names
            observables= ["S_next_x2"],  # keys to load; 1 key → classic MPM, >1 → joint
                                            # keys to load; 1 key → classic MPM, >1 → joint
                                            #   MPM (common poles, per-observable amplitudes).
                                            #   post_selected: "S" (stored p_0_t_array),
                                            #   "S_pooled" (segment-pooled rebuild from the raw
                                            #   NPZ, multinomial data bootstrap — written by
                                            #   run_files/raw_data_tests/run_pooled_p0.py),
                                            #   "S_pooled_strict" (same pool from segments L > i
                                            #   only: no ring-down last bin; record ends 1 bin
                                            #   earlier; scan 3 combined has it since 25 Oct),
                                            #   "S_next" (each point from the segment ONE bin
                                            #   longer only: no ring-down bin, the shots of S,
                                            #   independent points; computed from the JSON's
                                            #   first-click histogram), "S_next_far" (segment
                                            #   i+1+N//2), "S_next_x2" (both: 2x shots, record
                                            #   ends at ~12.4 us; halves differ ~1e-3 early by
                                            #   init history — s_next_ringdown_20261026; on a
                                            #   half record, scan 16, it is that record's S_next),
                                            #   "S_prev_click" / "S_prev_noclick", X/Y/Z …
                                            #   The MC twin writes only the stored S (its NPZ
                                            #   holds `clicks`, which run_pooled_p0.py does not
                                            #   read), so S_pooled does not exist for it → "S".
            lambda_range=None,              # λ window to load; None → every file
            t_range = None,               # (t_min, t_max) µs window the fit sees; None → full
                                            #   record, (None, 20) / (2, None) → one-sided.
                                            #   Same mask on the traces AND the bootstrap
                                            #   statistics; poles are window-independent,
                                            #   amplitudes refer to the window start.
            drop_first_point=False,         # drop the by-construction t=0 anchor (S=1) from the
                                            #   data AND the bootstrap statistics.  The anchor is
                                            #   inconsistent with the first-window loss / false
                                            #   clicks (order-3 EP 1.01 vs 1.15 with it, but the EP
                                            #   is poorly resolved without it — see
                                            #   mpm_benchmarks/noise_robustness/v3_transient.py).
        )

        # Step 1b — conditional tomography for the spiral fit (step 3b): the
        # RAW conditional <X>, <Z> on the surviving trajectories (not the
        # S·<X> products the joint pencil fits) plus S for γ0 / the window.
        # Full time range — the step applies its own late-time window.
        #   out: first_tomo
        if (FIRST_TOMOGRAPHY := False):
            first_tomo = load_records(
                PostSelectedLoader(csv_path=LAMBDA_SOURCE_POST), POST_SELECTED_FOLDER,
                data_mode="post_selected",
                observables=["S", "X_cond", "Z_cond"],
                lambda_range=(0, 1.6),
                t_range=None,
                drop_first_point=False,
            )

        # Step 1c — SVD gap cut: per trace, σ3/σ4 of the pencil's Hankel SVD
        # (order 3: last signal singular value over the first noise one; from
        # σ4 on it is the noise floor, σ4/σ5 ≈ 1).  Near 1 the third mode is
        # buried in the noise and the pencil's poles are not reliable — those
        # traces are dropped before every later step.  Figure: the singular
        # values + σ3/σ4 vs λ, dropped points marked.  The cut is written into
        # the result JSON (metadata.svd_cut).
        #   in : first_records    out: first_records (the kept traces)
        if (FIRST_SVD_CUT := False):
            first_records = run_svd_gap_cut(
                first_records,
                order=3,                    # must match run_mpm's order (σ3/σ4)
                min_ratio=2,              # drop below this.  scan 4+5: drops 1.18, 1.38, 1.43,
                                            #   1.47 (11.7k-rep file), 1.53, 1.58 (ratios 1.1-1.6);
                                            #   the next kept is 1.32 at 1.87
                lambda_window=None,         # (λ_min, λ_max) the cut applies in; None → every λ
                plot=True,
            )

        # Step 2 — matrix pencil + bootstrap.
        #   in : first_records
        #   out: first_result — ZenoAnalysisResult: poles per λ (central values
        #                       + 16/84 bands + the raw draws); spectrum figure
        #        first_json   — path of the saved result JSON (None in DEBUG);
        #                       steps 3 and 4 append their numbers to this file
        if (FIRST_MPM := True):
            first_result, first_json = run_mpm(
                first_records,
                order=3,                            # MPM model order (3 for both scans)
                uncertainty_source="data_bootstrap",  # how to propagate the uncertainty into the poles:

                #   "data_bootstrap"     resample the raw measurement statistics (binomial /
                #                        multinomial shot counts), rerun the pencil
                #                        (+ refinement) per draw — the reported error bars
                #   "errorbar_bootstrap" same per-draw analysis, but every point is drawn as
                #                        mean + sigma·N(0,1) from its MEASURED error bar.
                #                        For S: the std of a bootstrap over the file's
                #                        repetitions (data.p_0_t_bootstrap_meta, written by
                #                        raw_data_tests/bootstrap_s_errorbars.py); else the
                #                        block standard error of S / S_pooled that
                #                        raw_data_tests/run_pooled_p0.py (WRITE_ERRORBARS)
                #                        computes from the raw NPZ and writes into the JSON
                #                        (data.p_0_t_errorbar_meta); needs that write-back
                #   "fit_covariance"     fast Monte-Carlo propagation of the LSQ covariance
                #                        (forces refine_poles on — it needs the covariance)
                #   "residual_bootstrap" legacy residual resampling
                n_boot=2000,                         # bootstrap draws / covariance samples; 0 → no
                                                    #   bootstrap: the measured fit, zero error bars,
                                                    #   no per-draw EP / λ_c quantiles
                bootstrap_center="measured",           # "draws": central poles = mean of the draws,
                                                    #   error bar = their std — the measured curve is
                                                    #   one more sample of the data (errorbar_bootstrap)
                                                    # "measured": central = fit of the measured curve,
                                                    #   16/84 of the draws around it
                redraw_around="svd_truncated",           # data_bootstrap: what each copy is drawn around
                                                    #   "measured"      the measured shots (redraw of the
                                                    #                   first-click histogram): a copy has the
                                                    #                   data's own noise + one more dose
                                                    #   "svd_truncated" the measured curve with its Hankel cut to
                                                    #                   `order` SVD components (the pencil's own
                                                    #                   truncation, measured L); every point
                                                    #                   Binomial(N, S3(t)) - one dose, like a repeat
                                                    #                   of the experiment.  S / S_next* only (one
                                                    #                   segment per point), not S_pooled; ens_avg:
                                                    #                   one population (D), its multinomial with
                                                    #                   the D row set to S3
                refine_poles=False,                 # weighted complex-exponential LSQ after the pencil
                conjugate_pairs=True,               # constrain the refinement to conjugate pairs
                plot_svd=True,                     # SVD-vs-λ model-order calibration figure
                                                    #   (step 1c draws it for every trace, cut marked)
            )

            # Step 3 — exceptional point on the pole strands: zero of the pair
            # discriminant Delta = c1² − 4·c0.  Delta is linear in λ through a
            # sqrt EP.  The bootstrap gives each Delta point its error bar (half
            # the 16-84 % spread of the draws); ONE line is fitted to the
            # measured Delta over ±half_width around its sign change ("linfit"),
            # weighted by those error bars → λ_EP = its zero ± the fit error.
            # "first" takes the bracketing pair (reads ~0.1 low, no error).
            #   in : first_result (+ first_json to append to)
            #   out: first_ep — dict: "lambda_ep", "lambda_ep_error" (from the fit),
            #                   "chi2", "dof", "delta_fit" (the line, for step 6)
            if (FIRST_TRANSITIONS := True):
                first_ep = run_transitions(
                    first_result, json_path=first_json,
                    method="linfit",        # "linfit" (recommended) | "first"
                    lambda_window=EP_LAMBDA_WINDOW,  # ONE line through every λ in this window
                    lambda_max=None,        # λ cutoff of the Delta plot (the fit uses the window)
                    plot_discriminant=True, # Delta(λ) with error bars + the fitted line
                )

                # Step 3c — the EP vs the fit window: steps 1-3 redone for ~6
                # windows (start t_min, or end t_max, moved), every other
                # setting taken from first_records / first_result / first_ep.
                # One figure: Delta(λ) per window, λ_EP ± its fit error in the
                # legend.
                #   in : first_records, first_result, first_ep
                #   out: first_window_sweep — one dict per window (lam_ep, lam_ep_err, …)
                if (FIRST_WINDOW_SWEEP := False):
                    first_window_sweep = run_window_sweep(
                        first_records, first_result, first_ep,
                        loader=PostSelectedLoader(csv_path=LAMBDA_SOURCE_POST),
                        vary="t_min",       # "t_min" (move the window start) | "t_max" (its end)
                        values=(0,0.2,0.4,0.6),        # µs, e.g. (0, 0.2, 0.4, 0.6, 0.8, 1.0); None → 6
                                            #   automatic: the first 6 time samples from the main
                                            #   window's start ("t_min"), or 6 ends from the record
                                            #   end down to its middle ("t_max"); at most 8
                        n_boot=0,         # bootstrap draws PER window (the λ_EP error bar);
                                            #   the run takes ~6x one step-2 bootstrap of this size
                    )

            # Step 3a — state freezing from the pencil poles (do not conflate
            # with the EP): zero of ksi = (2·Re(e3) − Re(e2)) / (Re(e2) − Re(e3)),
            # i.e. γ0/γa − 1 with γ0 = −Re e3 and γa = Re e2 − Re e3 read off
            # the same order-3 pencil.  Redone in every bootstrap draw.
            #   in : first_result (+ first_json to append to)
            #   out: first_ksi — KsiCrossingResult: ksi(λ), lam(ksi=0), per-draw crossings
            if (FIRST_POLE_FREEZING := False):
                first_ksi = run_pole_state_freezing(
                    first_result, json_path=first_json,
                    lambda_max=None,
                    plot_ksi=True,          # ksi(λ) with error bars + the ksi = 0 line
                )

            # Step 3b — state freezing from the SPIRAL (pencil-free route to
            # the same ksi): γa from a direct fit of the conditional trajectory
            # (X,Z)(t) to R∞ + e^{−γt}[a cosh(st) + b sinh(st)/s] (paper eq. 4),
            # γa = γ − Re s; γ0 = click rate at the fixed point.  The spiral
            # fit is redone in every data-bootstrap draw of <X>, <Z>.
            #   in : first_tomo (step 1b), first_result (γ0 = slow pole + fit seeds)
            #   out: first_spiral — SpiralKsiResult: ksi(λ), γa(λ), γ0(λ), R∞(λ),
            #        lam(ksi=0) + bootstrap quantiles; JSON block "ksi_spiral"
            if (FIRST_SPIRAL_FREEZING := False):
                first_spiral = run_spiral_state_freezing(
                    first_tomo, first_result, json_path=first_json,
                    gamma0_source="mpm_slow_pole",  # "mpm_slow_pole": γ0 = decay of first_result's
                                                    #   slot-2 pole at the same λ (paired draws)
                                                    # "lnS_tail": paper Methods — linear fit of
                                                    #   ln S over the last n_tail_points ≥ min_survival
                    t_fit_min=2.0,                  # µs; start of the late-time window the spiral
                                                    #   fit sees (None → from t = 0).  Trade-off:
                                                    #   eq. (4) is asymptotic (bias ~ e^{−γa t}), but
                                                    #   by t ≳ 4 µs the approach is below the shot
                                                    #   noise and γa collapses (scan 4+5: 2 µs tracks
                                                    #   the pencil's Re e2 − Re e3, χ²_red ≈ 1)
                    min_survival=0.05,              # points with S below this are dropped (paper)
                    n_tail_points=10,               # lnS_tail: number of points of the ln S fit
                    n_boot=100,                     # spiral refits per λ on resampled <X>, <Z>
                    crossing_method="quadfit",      # "quadfit" (paper: quadratic through the
                                                    #   n_points nearest the sign change) |
                                                    #   "linfit" (±half_width) | "first"
                    n_points=10,
                    half_width=0.2,
                    lambda_min=0.65,                # λ window of the fit + crossing search.  Below
                                                    #   ~0.65 the window is near-undamped Rabi, not a
                                                    #   decaying approach (χ²_red ≫ 1, ξ ≫ 0) and
                                                    #   hands the crossing a spurious first sign change
                    lambda_max=None,
                    plot_ksi=True,                  # ksi(λ) = γ0/γa − 1 with the crossing
                    plot_spirals=True,              # per-λ slider: XZ spiral + fit, X/Z(t), ln S
                    print_table=True,               # per-λ γa, γ0, ω, χ²_red, ξ
                )

            # Step 4 — sqrt fit of the pole splitting: only λ_c is bounded, a/b
            # are free.  λ_c is fitted on the central values (aggregate) and,
            # with per_realization, separately in every bootstrap draw (16/50/84).
            #   in : first_result (+ first_json to append to)
            #   out: first_sqrt_fit — SqrtFitResult (lambda_c, lambda_c_err, chi2,
            #        R²); also attached to first_result for the "sqrt" layout of
            #        step 6; 3-panel appendix figure
            if (FIRST_CRITICAL_POINT := False):
                first_sqrt_fit = run_critical_point(
                    first_result, json_path=first_json,
                    lambda_c_guess=sum(EP_LAMBDA_WINDOW) / 2,
                    lambda_c_bounds=EP_LAMBDA_WINDOW,
                    lambda_max=None,
                    lambda_window=EP_LAMBDA_WINDOW,   # fit only the λ of step 3's line
                    per_realization=True,
                )

            # Step 5 — interactive per-λ trace browser; the overlay is the modal
            # model with first_result's poles, amplitudes re-solved per trace.
            #   in : first_records, first_result
            #   out: first_slider (kept alive in _LIVE_WIDGETS)
            if (FIRST_SLIDER := False):
                first_slider = run_slider_plot(
                    first_records, first_result,
                    scales={"S": "log", "S_pooled": "log", "S_pooled_strict": "log",
                            "S_next": "log", "S_next_far": "log", "S_next_x2": "log"},   # key → "log"/"linear"
                    take_abs=False,         # plot/fit |signal| (e.g. for parity)
                    components=True,        # True: each term A_k·exp(p_k·t) of the fit over the
                                            #   data instead of the summed fit, named and coloured
                                            #   as in the spectrum plot (e_1 C0, e_2 C1, ...; a
                                            #   conjugate pair = one curve "e_1+e_2").  On a log
                                            #   axis |term|, dashed where < 0.  False: the sum
                )

            # Step 6 — paper figure from this run's result.
            #   "discriminant" → 2 panels: A spectrum, B Delta(λ) line fit
            #   "sqrt"         → 3 panels: A spectrum, B Im splitting + a sqrt
            #                    overlay chosen by splitting_fit, C by panel_c
            #   The Delta line and λ_EP drawn ARE step 3's fit (first_ep), not a refit.
            #   in : first_result, first_ep (+ first_json, copied beside the figure as its data)
            #   out: first_fig (with SAVE_FIGURES: PAPER_PLOTS_DROPBOX/appendix/<figure_name>
            #        .pdf/.svg/.png + <figure_name>_data.json, overwriting the paper's files)
            if (FIRST_FIGURE := True):
                first_fig = run_figure(
                    first_result, first_ep, json_path=first_json,
                    figure_name="first_transition_appendix",   # file name in Plots/appendix
                    layout="sqrt",                      # appendix: A decay rates, B Im splitting +
                                                        #   black √ of step 3's Δ line, C (panel_c)
                    panel_c="discriminant",             # C: "discriminant" → (Δe12)²/Ω_S² = Δ/(4 Ω_S²)
                                                        #   at every λ + step 3's line over its window
                                                        #   | "real" → Re(Δe12)/Ω_S + √ of the line
                    lambda_max=2,                     # λ cutoff of every panel
                    fit_validity=1.2,
                    real_axis_scale="linear",           # panel A y axis: "linear" | "log"
                    real_axis_ticks=[0, 2, 4],          # panel A y ticks (None → matplotlib's)
                    real_split_axis_ticks=[-1, 0, 1],   # panel C y ticks
                    split_lambda_ticks=[0.5, 0.9, 1.3, 1.7],   # panels B/C λ ticks (None → every
                                                        #   0.5); one axis shared by A/B/C
                    splitting_fit="discriminant",       # "sqrt" layout overlay:
                                                        #   "discriminant" → √ of step 3's Delta line
                                                        #     (√(−Delta)/2 below λ_EP, √Delta/2 above,
                                                        #     over step 3's window); no step 4
                                                        #   "sqrt" → step 4's sqrt fit (needs step 4)
                    error_bar_alpha=0.4,                # opacity of the error bars; points stay solid
                    crit_line_decimals=2,               # λ_EP line drawn at the quoted value (1.30)
                )

            # Step 8 — pole table: the three poles of every file (central value =
            # the measured fit, error bars = step 2's bootstrap 16/84 half-widths)
            # as ONE CSV row per file, written into the data folder next to the
            # scan subfolders, with a text file that describes the bootstrap in plain steps.
            # Per-slot bars (each pole's Re and Im on its own) AND pair-consistent
            # bars (pair centre, Re/Im splitting, Delta: one number per draw).
            # Run on the FULL S_next record, all 24.8 µs (t_range=None in step 1:
            # 125 samples, t = 0 … 24.8 µs, 0.2 µs bins), every file (lambda_range=None).
            #   in : first_records (file, scan, drive, shots), first_result (poles + bars)
            #   out: first_pole_table — path of the CSV (the text file sits beside it)
            if (FIRST_POLE_TABLE := False):
                first_pole_table = run_pole_table(
                    first_records, first_result,
                    out_dir=POST_SELECTED_FOLDER,   # the data folder (a rerun on the same scans
                                                    #   overwrites the table)
                    observable="S_next",            # the trace the poles describe (step 1's key)
                )

        # Step 7 — S_next fit curves for the paper data folder, in the format of
        # PAPER_FIT_REFERENCE (the first submission's
        # fitting_results_20251202_162520: one <scan file>_mpm_fit.json per drive
        # holding times_smooth / fit_curve / lambda / original_file, nothing else),
        # written into a NEW PAPER_DATA_DROPBOX/fitting_results_<UTC stamp>/ folder
        # (old folders are never touched).  Its own load and fit, since step 1 may
        # load another observable / λ window: S_next of step 1's files, the order-3
        # pencil of step 2 without the bootstrap (the files hold only the curve),
        # no result JSON or spectrum figure.  The figure: each file's fit_curve,
        # read back from the file, over its S_next trace.  Off by default: every
        # run adds a folder to the shared paper folder.
        #   in : POST_SELECTED_FOLDER (step 1's files)
        #   out: first_export_dir
        if (FIRST_EXPORT := False):
            export_records = load_records(
                PostSelectedLoader(csv_path=LAMBDA_SOURCE_POST), POST_SELECTED_FOLDER,
                data_mode="post_selected",
                observables=["S_next"],         # the trace the curves describe
                lambda_range=None,              # every λ; step 7 keeps drives_khz
                t_range=None,                   # the full record, as the reference files
                drop_first_point=False,
            )
            _, export_result = _mpm_pipeline(
                export_records,
                order=3,                        # as step 2
                uncertainty_source="data_bootstrap",
                n_boot=0,                       # the fit of the measured curve only
                refine_poles=False,
                conjugate_pairs=True,
                plot_svd=False,
                bootstrap_center="measured",
            )
            first_export_dir = run_export_figure_data(
                export_records, export_result, json_path=None,
                parent=PAPER_DATA_DROPBOX,      # → zeno_paper_data/fitting_results_<stamp>/
                observable="S_next",
                n_smooth=300,                   # points of times_smooth, as the reference files
                drives_khz=(1000, 2125),        # scan 3: λ 0.527 (nearest 0.5) and 2.064
                reference_format=True,          # the four keys only, checked against the reference
                plot=True,                      # each file's curve read back over its trace
            )

    # -----------------------------------------------------------------------
    # THIRD TRANSITION — ensemble-average D population
    # -----------------------------------------------------------------------
    # Same steps as FIRST.  D alone at order 3: the joint D/G/B/error pencil is
    # a B-dominated truncation bias (mpm_benchmarks/noise_robustness/v3_channelcheck.py).
    if (RUN_THIRD_TRANSITION := False):
        print("\n##### THIRD TRANSITION  (ens_avg) #####")

        # Step 1 — load.   out: third_records
        third_records = load_records(
            EnsembleAverageLoader(csv_path=LAMBDA_SOURCE_ENS), ENS_AVG_FOLDER,
            data_mode="ens_avg",
            observables=["D"],              # "D" | "G" | "B" | "error", or several for a joint fit
            lambda_range=(0, 1.7),
            t_range=None,                   # (t_min, t_max) µs; None → full record
            drop_first_point=False,         # the t=0 point is an inconsistent anchor here too
                                            #   (λ₃ 1.13 → 1.03 without it)
        )

        # Step 2 — matrix pencil + bootstrap.   out: third_result, third_json
        if (THIRD_MPM := True):
            third_result, third_json = run_mpm(
                third_records,
                order=3,
                uncertainty_source="data_bootstrap",
                n_boot=2000,
                bootstrap_center="measured",    # central = fit of the measured curve (as FIRST)
                redraw_around="svd_truncated",  # copies drawn around the measured D cut to `order`
                                                #   Hankel SVD components: per point ONE multinomial
                                                #   over D/G/B/error (2·navg shots) with the D row set
                                                #   to that curve, the others in their measured
                                                #   ratios; t = 0 like every point.  "measured" →
                                                #   around the measured D (data noise + one dose)
                refine_poles=False,
                conjugate_pairs=True,
                plot_svd=True,
            )

            # Step 3 — exceptional point.   out: third_ep
            # (no state-freezing step here: the third transition has no ksi)
            if (THIRD_TRANSITIONS := True):
                third_ep = run_transitions(
                    third_result, json_path=third_json,
                    method="linfit",
                    lambda_window=(0.9, 1.3),       # ONE line through every λ in this window
                    lambda_max=None,
                    plot_discriminant=True,
                )

            # Step 4 — sqrt fit → λ_c.   out: third_sqrt_fit (also on third_result)
            if (THIRD_CRITICAL_POINT := False):
                third_sqrt_fit = run_critical_point(
                    third_result, json_path=third_json,
                    lambda_c_guess=1.1,
                    lambda_c_bounds=(0.95, 1.25),   # guess ± 0.15
                    lambda_max=1.7,
                    per_realization=True,
                )

            # Step 5 — trace browser.   out: third_slider
            if (THIRD_SLIDER := False):
                third_slider = run_slider_plot(
                    third_records, third_result,
                    scales={},                      # linear axes
                    take_abs=False,
                    components=True,                # True: the fit's terms instead of their sum (as FIRST)
                )

            # Step 6 — paper figure (Delta line + λ_EP = step 3's).   out: third_fig
            if (THIRD_FIGURE := True):
                third_fig = run_figure(
                    third_result, third_ep, json_path=third_json,
                    figure_name="third_transition_appendix",   # file name in Plots/appendix
                    layout="sqrt",                  # as FIRST: A decay rates, B Im splitting, C (panel_c)
                    panel_c="discriminant",         # C: (Δe12)²/Ω_S² at every λ + step 3's line | "real"
                    lambda_max=1.35,
                    fit_validity=1.3,
                    real_axis_scale="linear",
                    real_axis_ticks=[0.0, 1.0, 2.0],
                    real_split_axis_ticks=[-0.5, 0.0, 0.5],   # panel C y ticks (data −0.93 … +0.63)
                    split_lambda_ticks=[0.5, 0.9, 1.3],       # panels B/C λ ticks: every 0.4 from 0.5,
                                                    #   as FIRST (None → every 0.5); shared by A/B/C
                    splitting_fit="discriminant",   # "sqrt" layout overlay: √ of step 3's Delta line
                                                    #   | "sqrt" (step 4's fit)
                    error_bar_alpha=0.4,            # opacity of the error bars, as FIRST
                    crit_line_decimals=3,           # λ_EP line drawn at the quoted value (1.085)
                )

            # Step 7 — the pole analysis behind the main-text figure, written into a NEW
            # <parent>/fitting_results_<UTC stamp>/  folder (old folders are never
            # touched).  run_files/make_third_transition_figure.py draws the figure from
            # the result JSON of the folder its RESULT_JSON names (None → the newest) and
            # reloads the traces from ENS_AVG_FOLDER.
            #   mpm_ens_avg_*.json               panel B — third_json as it stands now
            #                                    (with the sqrt fit if step 4 ran)
            #   <scan file>_mpm_fit.json         panel A — one per λ: the modal fit
            #                                    Σ A_k exp(p_k t) of `observable` on a
            #                                    smooth grid (times_smooth / fit_curve),
            #                                    poles = third_result's, amplitudes
            #                                    re-solved on that trace; matched by
            #                                    kHz in the name
            #   in : third_records (the traces), third_result (the poles), third_json
            #   out: third_export_dir
            if (THIRD_EXPORT := True):
                third_export_dir = run_export_figure_data(
                    third_records, third_result, json_path=third_json,
                    parent=THIRD_POLE_DATA_FOLDER,  # → …/pole_analysis_data_for_main_text_figure/
                                                #   fitting_results_<stamp>/, where the figure
                                                #   script reads the result JSON from
                    observable="D",             # trace the fit curve describes — must be
                                                #   what panel A plots: the loader's
                                                #   "survival" key = D_probability
                    n_smooth=300,               # points of the smooth time grid
                )

    # -----------------------------------------------------------------------
    # MC TWIN — the first-transition steps on a SIMULATED scan (never the data)
    # -----------------------------------------------------------------------
    # mpm_benchmarks/scan8/mc/run_mc.py simulates a whole scan with the zeno
    # engine in the experiment's acquisition protocol and writes it in the
    # scanner's file formats under MC_RUNS/<variant>/, so the same step
    # functions run on it unchanged.  The twin's JSON λ IS the simulated λ
    # (HMM axis for *_hmm variants), hence csv_path=False.  The twin writes
    # only the stored S (no S_pooled, no error-bar meta) → observables ["S"],
    # data_bootstrap.  Its parameters: MC_RUNS/<variant>/run_summary.json.
    if (RUN_MC_TWIN := False):
        MC_VARIANT = "s8_phys_tof"      # post_selected twins: s8_base s8_hmm s8_phys s8_phys_tof
                                    #   s8_T2_40 s8v2_hmm s8v2_phys_tof
                                    # ens_avg twins (s0_base s0_hmm s0_hmm_T2_8 s0_T2_12
                                    #   s0_T2_8): EnsembleAverageLoader(csv_path=False),
                                    #   sub-folder "ens_avg", observables ["D"]
        print(f"\n##### MC TWIN  {MC_VARIANT}  "
              f"(parameters: {MC_RUNS / MC_VARIANT / 'run_summary.json'}) #####")

        # Step 1 — load the twin.   out: mc_records
        mc_records = load_records(
            PostSelectedLoader(csv_path=False), MC_RUNS / MC_VARIANT / "post_selected",
            data_mode="post_selected",
            observables=["S"],
            lambda_range=(0, 4),
            t_range=None,
            drop_first_point=False,
        )

        # Step 2 — matrix pencil + bootstrap.   out: mc_result, mc_json
        if (MC_MPM := True):
            mc_result, mc_json = run_mpm(
                mc_records,
                order=3,
                uncertainty_source="data_bootstrap",
                n_boot=100,
                refine_poles=False,
                conjugate_pairs=True,
                plot_svd=False,
            )

            # Step 3 — exceptional point.   out: mc_ep
            if (MC_TRANSITIONS := True):
                mc_ep = run_transitions(
                    mc_result, json_path=mc_json,
                    method="linfit",
                    lambda_window=EP_LAMBDA_WINDOW,   # as the data (first transition)
                    lambda_max=None,
                    plot_discriminant=True,
                )

            # Step 3a — state freezing from the pencil poles.   out: mc_ksi
            # (no spiral step: the twin writes no tomography records)
            if (MC_POLE_FREEZING := False):
                mc_ksi = run_pole_state_freezing(
                    mc_result, json_path=mc_json,
                    lambda_max=None,
                    plot_ksi=True,
                )

            # Step 4 — sqrt fit → λ_c.   out: mc_sqrt_fit
            if (MC_CRITICAL_POINT := False):
                mc_sqrt_fit = run_critical_point(
                    mc_result, json_path=mc_json,
                    lambda_c_guess=sum(EP_LAMBDA_WINDOW) / 2,
                    lambda_c_bounds=EP_LAMBDA_WINDOW,
                    lambda_max=None,
                    lambda_window=EP_LAMBDA_WINDOW,
                    per_realization=True,
                )

            # Step 5 — trace browser.   out: mc_slider
            if (MC_SLIDER := False):
                mc_slider = run_slider_plot(
                    mc_records, mc_result,
                    scales={"S": "log"},
                    take_abs=False,
                    components=False,               # True: the fit's terms instead of their sum (as FIRST)
                )

    # -----------------------------------------------------------------------
    # SIMULATED GRID — the first-transition steps on ONE (λ × T) survival NPZ
    # -----------------------------------------------------------------------
    # A simulation that stores the survival map S(λ, T) on a grid (lambda_values
    # × T_values), e.g. the three-level tomography Monte Carlo
    # (zeno_theory_and_experiment/.../three_level_tomography.py: fraction_success_2D
    # = no-click trials / attempted trials).  io/simulation.load_monte_carlo_npz
    # reads it; the λ axis is the simulated λ.  n_shots = the attempted trials
    # behind every point: the data bootstrap draws Binomial(n_shots, S(t)) / n_shots.
    if (RUN_SIM_GRID := True):
        SIM_GRID_NPZ = Path(r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab\Noam\RT_tomography_scan_experiment_results_0_three_states_simulation.npz")
        print(f"\n##### SIMULATED GRID  {SIM_GRID_NPZ.name} #####")

        # Step 1 — load the grid.   out: sim_records (one per λ row)
        sim_records = load_grid_records(
            SIM_GRID_NPZ,
            data_mode="post_selected",      # survival curve → post_selected bootstrap / file names
            observables=["survival"],       # the NPZ's fraction_success_2D (io/simulation
                                            #   _SIGNAL_KEYS["survival"])
            n_shots=300_000_000,            # attempted trials per (λ, T) point (the NPZ's N_trials)
            lambda_range=None,              # every λ row
            t_range=(None, 12.4),           # µs; the record starts at T = 0.64 (no t = 0 point)
        )

        # Step 2 — matrix pencil + bootstrap.   out: sim_result, sim_json
        if (SIM_MPM := True):
            sim_result, sim_json = run_mpm(
                sim_records,
                order=4,
                uncertainty_source="data_bootstrap",
                n_boot=0,
                bootstrap_center="measured",    # as FIRST
                redraw_around="svd_truncated",  # as FIRST: every point Binomial(n_shots, S3(t))
                refine_poles=False,
                conjugate_pairs=False,
                plot_svd=True,
                nyquist_check=False,            # True: blank a λ whose pencil has a pole at the
                                                #   Nyquist frequency |Im| >= 0.9·π/dt (and drop such
                                                #   draws).  False: keep every pole (full picture);
                                                #   step 3's Delta leaves the aliased pole out
            )

            # Step 3 — exceptional point.   out: sim_ep
            if (SIM_TRANSITIONS := True):
                sim_ep = run_transitions(
                    sim_result, json_path=sim_json,
                    method="linfit",
                    lambda_window=EP_LAMBDA_WINDOW,   # as the data (first transition)
                    lambda_max=None,
                    plot_discriminant=True,
                )

            # Step 5 — trace browser.   out: sim_slider
            if (SIM_SLIDER := True):
                sim_slider = run_slider_plot(
                    sim_records, sim_result,
                    scales={"survival": "log"},
                    take_abs=False,
                    components=True,                # True: the fit's terms instead of their sum (as FIRST)
                )

    # -----------------------------------------------------------------------
    # Stand-alone checks (no experimental data, no settings of their own)
    # -----------------------------------------------------------------------
    if (RUN_SIMULATION_TEST := False):
        run_simulation_test()       # small synthetic simulation through MPM + bootstrap

    if (RUN_BENCHMARK := False):
        run_benchmark()             # robustness benchmark with injected shot noise

    if (RUN_LIOUVILLIAN_SPECTRUM := False):
        run_liouvillian_spectrum()  # 4×4 Liouvillian eigenvalues vs λ at k=π

    print("\nDone.")
    if SHOW_PLOTS:
        plt.show()


# ===========================================================================
# HELPERS
# ===========================================================================

# Strong references to interactive widgets: matplotlib only weak-references
# slider callbacks, so a dropped slider keeps drawing but stops responding.
_LIVE_WIDGETS: list = []

# ---------------------------------------------------------------------------
# The steps the two paper transitions share - load_records, run_mpm, _mpm_pipeline,
# run_transitions, _ep_crossing, run_figure and their helpers - live in the package,
# mpm_analysis/pipeline/transition_steps.py, so other code can call them without this
# file (ZenoPlotting's paper-figure scripts do).  The wrappers below keep the names
# here and hand over this file's GLOBAL SETTINGS at the call, so a setting changed
# after import (run.SEED = ...) is the one used.
# ---------------------------------------------------------------------------

@functools.wraps(_steps._show_and_save)
def _show_and_save(*args, **kwargs):
    return _steps._show_and_save(*args, debug=DEBUG, output_dir=OUTPUT_DIR,
                                 save_figures=SAVE_FIGURES, show_plots=SHOW_PLOTS, **kwargs)


def _files_tag(prov: dict) -> str:
    """Short name of the files a load used: '' (whole folder), the glob, or —
    for a file list from the union folder — its scans ("scans 3+14+16"); for the scan_exp
    subfolders the same from their names."""
    fg = prov.get("file_glob", "*.json")
    if isinstance(prov.get("source_folder"), list):
        names = [Path(f).name.removeprefix("scan_exp_").replace("_", "+") for f in prov["source_folder"]]
        return "scans " + "+".join(names)
    if isinstance(fg, str):
        return "" if fg == "*.json" else f"files {fg}"
    from mpm_analysis.utils.windows_paths import win_long_path
    try:
        with open(win_long_path(Path(prov["source_folder"]) / "hmm_lambda_per_file.csv"),
                  newline="", encoding="utf-8") as f:
            scan_of = {r["filename"]: r.get("scan") for r in csv.DictReader(f)}
    except OSError:
        return f"{len(fg)} files"
    scans = {scan_of.get(n) for n in fg} - {None}
    if not scans:
        return f"{len(fg)} files"
    # measurement order: "45" (scans 4+5) between 3 and 10, then 10, 14, 16, 17, ...
    scans = sorted(scans, key=lambda s: 4.5 if s == "45" else float(s) if s.isdigit() else 1e9)
    return "scans " + "+".join("4+5" if s == "45" else s for s in scans)


def _order(result: ZenoAnalysisResult) -> int:
    """MPM model order of the result."""
    return int(result.parameters["pole_order"])


# ===========================================================================
# STEP FUNCTIONS  (one per step of main(); settings are keyword-only)
# ===========================================================================

@functools.wraps(_steps.load_records)        # step 1; MAX_FILES caps the number of files
def load_records(*args, **kwargs):
    return _steps.load_records(*args, max_files=MAX_FILES, **kwargs)


def load_grid_records(
    npz_path: str | Path, *,
    data_mode: str,
    observables: list[str],
    n_shots: int,
    lambda_range: tuple[float, float] | None,
    t_range: tuple[float | None, float | None] | None,
) -> list[ObservableRecord]:
    """Step 1 (simulated grid) — load ONE (λ × T) simulation NPZ.  Returns the records.

    ``io/simulation.load_monte_carlo_npz``: one ``ObservableRecord`` per λ row,
    ``t_range`` = (t_min, t_max) µs on the traces AND the bootstrap statistics;
    ``n_shots`` → Binomial(n_shots, S(t)) / n_shots in the data bootstrap.  The
    same provenance as ``load_records`` is stamped in for run_mpm / the result JSON.
    """
    npz_path = Path(npz_path)
    records = load_monte_carlo_npz(npz_path, list(observables), lambda_range=lambda_range,
                                   t_range=t_range, n_shots=n_shots)
    if not records:
        raise RuntimeError(f"[{data_mode}] no records loaded from {npz_path} "
                           f"(observables={observables}, lambda_range={lambda_range}).")
    provenance = {
        "data_mode": data_mode,
        "source_folder": str(npz_path),
        "lambda_source": "npz lambda_values",
        "observables": list(observables),
        "lambda_range": None if lambda_range is None else [float(x) for x in lambda_range],
        "t_range": None if t_range is None else list(t_range),
        "drop_first_point": False,
        "n_shots": int(n_shots),
    }
    for rec in records:
        rec.metadata[_PROVENANCE] = provenance
    lambdas = sorted({float(r.lambda_val) for r in records})
    t0, t1 = float(records[0].t[0]), float(records[0].t[-1])
    print(f"\n=== Load ({data_mode}, simulated grid) ===\n"
          f"  {len(records)} records, {len(lambdas)} lambda points in "
          f"[{lambdas[0]:.3f}, {lambdas[-1]:.3f}], observables={observables}, "
          f"n_shots={n_shots:,} per point\n"
          f"  time window t_range={t_range} -> {records[0].n_points} points, "
          f"t = [{t0:.3f}, {t1:.3f}] us\n  {npz_path}")
    return records


def run_svd_gap_cut(
    records: list[ObservableRecord], *,
    order: int,
    min_ratio: float,
    lambda_window: tuple[float, float] | None,
    plot: bool,
) -> list[ObservableRecord]:
    """Step 1c — drop the λ traces whose model order is not resolved.

    Per trace (per λ group for a joint fit) the Hankel SVD of the pencil —
    the same matrix and L as MPMStep — gives σ_order / σ_order+1, the last
    signal singular value over the first noise one.  Traces with a ratio
    below ``min_ratio`` inside ``lambda_window`` (``None`` → every λ) are
    dropped; a trace whose SVD fails is kept (run_mpm reports it).

    Returns the kept records.  The cut (settings + dropped λ / ratio / files)
    is added to their provenance, so run_mpm writes it into the result JSON.
    """
    from mpm_analysis.analysis.matrix_pencil import matrix_pencil_method
    from mpm_analysis.data_types.observables import group_records_by_lambda

    prov = _provenance(records)
    keys = prov["observables"]
    if len(keys) == 1:
        groups = [(r.lambda_val, r.t, r.signal, [r])
                  for r in sorted(records, key=lambda r: r.lambda_val)
                  if r.observable_key == keys[0]]
    else:
        groups = group_records_by_lambda(records, keys)

    lams, ratios, svs, drop = [], [], [], []
    for lam, t, Y, _recs in groups:
        try:
            _, sv, _ = matrix_pencil_method(t, Y, order)
            ratio = float(sv[order - 1] / sv[order])
        except Exception:
            sv, ratio = None, np.nan
        in_window = lambda_window is None or lambda_window[0] <= lam <= lambda_window[1]
        lams.append(float(lam)); ratios.append(ratio); svs.append(sv)
        drop.append(bool(in_window and np.isfinite(ratio) and ratio < min_ratio))
    lams, ratios, drop = np.array(lams), np.array(ratios), np.array(drop)

    dropped = [dict(lambda_val=float(lam), ratio=float(ratio),
                    files=[r.source_file for r in recs])
               for (lam, _t, _y, recs), ratio, d in zip(groups, ratios, drop) if d]
    gone = {id(r) for (*_x, recs), d in zip(groups, drop) if d for r in recs}
    kept = [r for r in records if id(r) not in gone]

    tag = f"s{order}/s{order + 1}"
    print(f"\n=== SVD gap cut ({prov['data_mode']}, order={order}, keep {tag} >= {min_ratio}, "
          f"lambda window {lambda_window}) ===\n"
          f"  dropped {len(dropped)}/{len(groups)} traces")
    for d in dropped:
        print(f"    lam = {d['lambda_val']:.3f}  {tag} = {d['ratio']:.2f}  {', '.join(d['files'])}")
    if not kept:
        raise RuntimeError(f"[{prov['data_mode']}] the SVD gap cut dropped every trace "
                           f"({tag} < {min_ratio} everywhere).")

    cut = dict(order=order, min_ratio=float(min_ratio),
               lambda_window=None if lambda_window is None else [float(x) for x in lambda_window],
               dropped=dropped)
    new_prov = {**prov, "svd_cut": cut}
    for rec in kept:
        rec.metadata[_PROVENANCE] = new_prov

    if plot:
        fig, (ax_sv, ax_r) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        ok = [sv is not None for sv in svs]
        plot_svd_singular_values(
            lams[ok], [sv for sv in svs if sv is not None], n_sv=10, ax=ax_sv,
            title=f"MPM Hankel SVD — {'+'.join(keys)}  (order {order}; red = dropped)",
        )
        ax_sv.set_xlabel("")
        for lam in lams[drop]:
            ax_sv.axvline(lam, color="red", alpha=0.25, linewidth=3)
        keep = ~drop & np.isfinite(ratios)
        ax_r.semilogy(lams[keep], ratios[keep], "o", color="C0", label="kept")
        ax_r.semilogy(lams[drop], ratios[drop], "x", color="red", ms=8, mew=2, label="dropped")
        ax_r.axhline(min_ratio, color="red", linestyle="--", linewidth=1,
                     label=f"min ratio {min_ratio}")
        ax_r.axhline(1.0, color="0.5", linestyle=":", linewidth=1, label="noise floor (1)")
        if lambda_window is not None:
            ax_r.axvspan(*lambda_window, color="0.92", zorder=0, label="cut window")
        ax_r.set_xlabel(r"$\lambda$")
        ax_r.set_ylabel(rf"$\sigma_{order}\,/\,\sigma_{order + 1}$")
        ax_r.legend(fontsize=7)
        ax_r.grid(True, which="both", alpha=0.3, linewidth=0.5)
        fig.tight_layout()
        scan_number = records[0].metadata.get("scan_number")
        _show_and_save(fig, "svd_gap_cut", data_mode=prov["data_mode"],
                       scan_number=int(scan_number) if scan_number is not None else None)
    return kept


@functools.wraps(_steps.run_mpm)             # step 2
def run_mpm(*args, **kwargs):
    return _steps.run_mpm(*args, seed=SEED, n_jobs=N_JOBS, debug=DEBUG, output_dir=OUTPUT_DIR,
                          save_full_analysis=SAVE_FULL_ANALYSIS, save_figures=SAVE_FIGURES,
                          show_plots=SHOW_PLOTS, **kwargs)


@functools.wraps(_steps._mpm_pipeline)       # step 2 without its figure and JSON
def _mpm_pipeline(*args, **kwargs):
    return _steps._mpm_pipeline(*args, seed=SEED, n_jobs=N_JOBS, debug=DEBUG,
                                output_dir=OUTPUT_DIR, **kwargs)


@functools.wraps(_steps.run_transitions)     # step 3
def run_transitions(*args, **kwargs):
    return _steps.run_transitions(*args, debug=DEBUG, output_dir=OUTPUT_DIR,
                                  save_figures=SAVE_FIGURES, show_plots=SHOW_PLOTS, **kwargs)


def run_window_sweep(
    records: list[ObservableRecord],
    result: ZenoAnalysisResult,
    ep: dict, *,
    loader,
    vary: str,
    values: tuple[float, ...] | list[float] | None,
    n_boot: int,
) -> list[dict]:
    """Step 3c — the exceptional point vs the fit window.  Returns one dict per window.

    Steps 1-3 are redone for every window, with every setting taken from
    the main run: the load of ``records`` (folder, file glob, λ range,
    observables, drop_first_point; the SVD cut's dropped files stay
    dropped) with the window's t_range, the pencil + bootstrap of
    ``result`` (order, refinement, uncertainty source, centre; ``n_boot``
    draws per window) and the Delta line fit of ``ep`` (method, half-width,
    λ_max).  ``loader`` must be the loader the records came from.

    ``vary="t_min"`` moves the window START over ``values`` (µs) with the end
    of the main window kept; ``"t_max"`` moves its END with the start kept.
    ``values=None`` → 6 automatic values: the first 6 time samples from the
    main window's start (t_min), or 6 ends from the record end down to its
    middle (t_max).

    Each dict: ``vary``, ``value``, ``t_window`` (first/last sample used),
    ``n_points`` and the ``_ep_crossing`` numbers (``lam_ep``, ``lam_ep_err``,
    ``chi2``, ``dof``, ``lam_d``, ``d_central``, ``d_sigma``, ``fit`` …).
    One figure: Delta(λ) of every window, λ_EP ± its fit error in the legend.
    """
    if vary not in ("t_min", "t_max"):
        raise ValueError(f"vary must be 't_min' or 't_max', got {vary!r}.")
    prov = _provenance(records)
    t = np.asarray(records[0].t, dtype=float)
    t_lo, t_hi = prov["t_range"] or (None, None)
    if values is None:
        dt = float(np.median(np.diff(t)))
        values = (t[0] + dt * np.arange(6) if vary == "t_min"
                  else np.linspace(t[-1], t[0] + (t[-1] - t[0]) / 2, 6))
    values = [round(float(v), 6) for v in values]
    eps = 1e-6                            # µs: a window edge ON a sample keeps it

    # Everything else exactly as in the main run.
    extras = result.extras or {}
    settings = dict(
        order=_order(result),
        uncertainty_source=result.metadata["uncertainty_source"],
        refine_poles="refine_conjugate" in extras,
        conjugate_pairs=bool(extras.get("refine_conjugate", True)),
        bootstrap_center=result.metadata.get("bootstrap_center", "measured"),
        redraw_around=result.metadata.get("bootstrap_redraw", "measured"),
        nyquist_check=result.metadata.get("nyquist_check", True),
    )
    crossing = dict(method=ep["delta_crossing_method"], half_width=ep["delta_half_width"],
                    lambda_max=ep.get("delta_lambda_max"),
                    points_below=ep.get("delta_points_below"),
                    points_above=ep.get("delta_points_above"),
                    lambda_window=ep.get("delta_lambda_window"))
    svd_cut = prov.get("svd_cut")
    dropped = {f for d in (svd_cut or {}).get("dropped", []) for f in d["files"]}
    subset = prov.get("file_glob", "*.json")
    print(f"\n=== EP vs fit window ({vary} = {values} us; n_boot = {n_boot} per window; "
          f"{settings}, {crossing}) ===")

    rows = []
    for v in values:
        t_range = (v - eps, t_hi) if vary == "t_min" else (t_lo, v + eps)
        print(f"\n--- window sweep: {vary} = {v:g} us ---")
        recs = load_records(
            loader, prov["source_folder"], file_glob=subset,
            data_mode=prov["data_mode"], observables=prov["observables"],
            lambda_range=prov["lambda_range"], t_range=t_range,
            drop_first_point=prov["drop_first_point"],
        )
        if svd_cut is not None:
            recs = [r for r in recs if r.source_file not in dropped]
            for r in recs:
                r.metadata[_PROVENANCE] = {**r.metadata[_PROVENANCE], "svd_cut": svd_cut}
        _, res = _mpm_pipeline(recs, n_boot=n_boot, plot_svd=False, **settings)
        c = _ep_crossing(res, **crossing)
        tw = np.asarray(recs[0].t, dtype=float)
        rows.append(dict(
            vary=vary, value=v, t_window=(float(tw[0]), float(tw[-1])),
            n_points=int(recs[0].n_points), **c,
        ))

    print(f"\n=== EP vs fit window ({vary}): lam_EP = zero of one weighted line fit to "
          f"the measured Delta, error from the fit ===\n"
          f"  {'window (us)':>16} {'points':>6} {'lam_EP':>8} {'error':>7} {'chi2/dof':>10}  fit points")
    for r in rows:
        err = "" if r["lam_ep_err"] is None else f"{r['lam_ep_err']:7.4f}"
        chi = "" if r["chi2"] is None else f"{r['chi2']:5.1f}/{r['dof']}"
        pts = ("" if r["fit"]["window"] is None else
               ", ".join(f"{x:.3f}" for x in r["lam_d"][r["fit"]["window"]]))
        lw_ = crossing.get("lambda_window")
        flag = ("   <- zero OUTSIDE the fit range: no crossing in it"
                if lw_ and np.isfinite(r["lam_ep"]) and not lw_[0] <= r["lam_ep"] <= lw_[1] else "")
        print(f"  [{r['t_window'][0]:5.2f}, {r['t_window'][1]:6.2f}] {r['n_points']:6d} "
              f"{r['lam_ep']:8.4f} {err:>7} {chi:>10}  {pts}{flag}")

    for r in rows:
        a, b = r["t_window"]
        r["label"] = (rf"$t_{{\mathrm{{{vary[2:]}}}}}$ = {r['value']:g} $\mu$s   "
                      rf"(data $t \in [{a:.2f}, {b:.2f}]$ $\mu$s)")
    data_mode, scan_number = _result_tags(result)
    obs_tag = "+".join(_observables(result))
    sub_tag = _files_tag(prov)
    lw = crossing.get("lambda_window")
    fit_tag = rf"; line fit over $\lambda$ = {lw[0]:g}-{lw[1]:g}" if lw else ""
    fig = plot_discriminant_window_sweep(
        rows, title=(f"Exceptional point vs window {'start' if vary == 't_min' else 'end'} "
                     f"({data_mode}, {obs_tag}{', ' + sub_tag if sub_tag else ''}{fit_tag})"))
    _show_and_save(fig, f"discriminant_window_sweep_{vary}",
                   data_mode=data_mode, scan_number=scan_number)
    return rows


def _print_state_freezing(tag: str, ksi) -> None:
    """One line: central lam(ksi=0) with the bootstrap 16/84 error bar."""
    central = ksi.lambda_ksi_zero
    if ksi.crossing_quantiles is not None:
        q16, q50, q84 = ksi.crossing_quantiles
        centre = central if central is not None else q50
        print(f"  STATE FREEZING ({tag})  lam(ksi=0) = {centre:.4f} +/- {(q84 - q16) / 2:.4f}   "
              f"[bootstrap 16/50/84 = {q16:.4f}/{q50:.4f}/{q84:.4f}]")
    elif central is not None:
        print(f"  STATE FREEZING ({tag})  lam(ksi=0) = {central:.4f}  "
              f"(no bootstrap draws — no error bar)")
    else:
        print(f"  STATE FREEZING ({tag})  ksi has no zero crossing.")


def run_pole_state_freezing(
    result: ZenoAnalysisResult, *,
    json_path: Path | None,
    lambda_max: float | None,
    plot_ksi: bool,
) -> KsiCrossingResult:
    """Step 3a — state freezing from the pencil poles.

    lam(ksi=0) with ksi = (2·Re(e3) − Re(e2)) / (Re(e2) − Re(e3)), i.e.
    γ0/γa − 1 with γ0 = −Re e3 (click rate at the fixed point) and
    γa = Re e2 − Re e3 (approach rate) read off the order-3 pencil of the
    survival.  The crossing is redone inside every bootstrap draw (16/84
    error bar); the central value comes from the unresampled poles.

    Returns the ``KsiCrossingResult``; appended to ``json_path`` as the
    ``ksi_crossing`` block (unless DEBUG / None).
    """
    data_mode, scan_number = _result_tags(result)
    obs_tag = "+".join(_observables(result))
    print(f"\n=== State freezing from the poles ({data_mode}, {obs_tag}) ===")
    ksi = KsiCrossingPipeline(result, lambda_max=lambda_max).run()
    _print_state_freezing("poles", ksi)
    if plot_ksi:
        ax = plot_ksi_curve(ksi, title=f"State freezing: ksi = 0 (poles; {data_mode}, {obs_tag})")
        _show_and_save(ax.get_figure(), "state_freezing_ksi",
                       data_mode=data_mode, scan_number=scan_number)
    if json_path is not None and not DEBUG:
        augment_json_with_ksi(json_path, ksi, block="ksi_crossing")
    return ksi


def run_spiral_state_freezing(
    tomo_records: list[ObservableRecord], result: ZenoAnalysisResult | None, *,
    json_path: Path | None,
    gamma0_source: str,
    t_fit_min: float | None,
    min_survival: float,
    n_tail_points: int,
    n_boot: int,
    crossing_method: str,
    n_points: int,
    half_width: float,
    lambda_min: float | None,
    lambda_max: float | None,
    plot_ksi: bool,
    plot_spirals: bool,
    print_table: bool,
) -> SpiralKsiResult:
    """Step 3b — state freezing from a direct fit of the tomography spiral.

    Per λ the late-time conditional trajectory (``X_cond``, ``Z_cond``) is
    fitted to paper eq. (4), ``R∞ + e^{−γt}[a cosh(st) + b sinh(st)/s]``, which
    gives the approach rate γa = γ − Re s without a matrix pencil; the click
    rate at the fixed point γ0 is the decay of ``result``'s slow pole at the
    same λ (``"mpm_slow_pole"``) or a linear fit to ln S over the last points
    (``"lnS_tail"``).  ksi = γ0/γa − 1; its zero is located with
    ``crossing_method`` and redone in every data-bootstrap draw of the
    tomography (paired with the pencil's draws for γ0).

    Returns the ``SpiralKsiResult`` (ksi, γa, γ0, ω, R∞ per λ, the fits, the
    crossing + quantiles); appended to ``json_path`` as the ``ksi_spiral``
    block (unless DEBUG / None).  Figures: ksi(λ) and the per-λ spiral slider
    (kept in ``_LIVE_WIDGETS``).
    """
    prov = _provenance(tomo_records)
    data_mode = prov["data_mode"]
    scan_number = tomo_records[0].metadata.get("scan_number")
    scan_number = int(scan_number) if scan_number is not None else None
    n_boot = min(n_boot, 20) if DEBUG else n_boot
    print(f"\n=== State freezing from the spiral ({data_mode}, gamma0={gamma0_source}, "
          f"window t >= {t_fit_min} us, S >= {min_survival}, n_boot={n_boot}) ===")
    spiral = SpiralKsiPipeline(
        tomo_records, result,
        gamma0_source=gamma0_source, t_fit_min=t_fit_min, min_survival=min_survival,
        n_tail_points=n_tail_points, lambda_min=lambda_min, lambda_max=lambda_max,
        n_boot=n_boot, seed=SEED, n_jobs=N_JOBS,
        crossing_method=crossing_method, n_points=n_points, half_width=half_width,
    ).run()
    _print_state_freezing("spiral", spiral)
    ok = np.isfinite(spiral.gamma_a)
    if ok.any():
        print(f"  gamma_a = {np.nanmin(spiral.gamma_a):.3f} .. {np.nanmax(spiral.gamma_a):.3f} /us, "
              f"gamma0 = {np.nanmin(spiral.gamma0):.3f} .. {np.nanmax(spiral.gamma0):.3f} /us, "
              f"median chi2_red = {np.nanmedian(spiral.chi2_red):.2f}  "
              f"({int(ok.sum())}/{ok.size} lambdas fitted)")
    if print_table:
        print("  lambda  gamma_a  +-      gamma0  +-      omega   chi2_red  n_pts   ksi")
        for i, lam in enumerate(spiral.lambda_values):
            f = spiral.fits[i]
            print(f"  {lam:6.3f} {spiral.gamma_a[i]:8.3f} {spiral.gamma_a_err[i]:7.3f} "
                  f"{spiral.gamma0[i]:8.3f} {spiral.gamma0_err[i]:7.3f} {spiral.omega[i]:7.3f} "
                  f"{spiral.chi2_red[i]:9.2f} {f.n_points if f else 0:6d} {spiral.ksi_values[i]:+8.3f}")
    if plot_ksi:
        ax = plot_ksi_curve(spiral, title=f"State freezing: ksi = 0 (spiral fit; {data_mode})")
        _show_and_save(ax.get_figure(), "state_freezing_spiral_ksi",
                       data_mode=data_mode, scan_number=scan_number)
    if plot_spirals:
        _LIVE_WIDGETS.append(plot_spiral_fits_slider(tomo_records, spiral))
    if json_path is not None and not DEBUG:
        augment_json_with_ksi(json_path, spiral, block="ksi_spiral")
    return spiral


def run_critical_point(
    result: ZenoAnalysisResult, *,
    json_path: Path | None,
    lambda_c_guess: float,
    lambda_c_bounds: tuple[float, float],
    lambda_max: float | None,
    per_realization: bool,
    lambda_window: tuple[float, float] | None = None,
):
    """Step 4 — sqrt fit of the pole splitting → lambda_c (aggregate + per realization).

    ``lambda_window = (lo, hi)`` fits only the λ inside it (the λ of step 3's
    Delta line, so both critical points use the same points); it overrides
    ``lambda_max``.

    Returns the ``SqrtFitResult``; it is also attached to the result as
    ``result.critical_point_sqrt_fit`` (the "sqrt" figure layout reads it)
    and written into ``json_path`` with the per-realization quantiles.
    """
    data_mode, scan_number = _result_tags(result)
    print(f"\n=== Critical-point fit ({data_mode}, {'+'.join(_observables(result))}) ===")
    lam_lo, lam_hi = lambda_window if lambda_window is not None else (None, lambda_max)
    cp = CriticalPointPipeline(result, lambda_max=lam_hi, lambda_min=lam_lo)
    if lambda_window is not None:
        print(f"  fit window: lam {lam_lo:g}-{lam_hi:g}")
    fit = cp.fit(
        lambda_c_guess=lambda_c_guess, soft_bounds=True, lambda_c_bounds=lambda_c_bounds,
        window_left=50, window_right=15,
    )
    print(f"  lam_c (aggregate fit) = {fit.lambda_c:.4f} +/- {fit.lambda_c_err:.4f}   "
          f"[guess {lambda_c_guess}, bounds {lambda_c_bounds}]")
    print(f"  chi2_red = {fit.chi2_reduced:.3f}  R2 = {fit.r_squared:.4f}")

    if per_realization and result.poles.raw_samples is not None:
        raw = result.poles.raw_samples
        if lam_hi is not None or lam_lo is not None:
            m = np.ones(np.shape(raw.lambda_values), dtype=bool)
            if lam_hi is not None:
                m &= raw.lambda_values <= lam_hi
            if lam_lo is not None:
                m &= raw.lambda_values >= lam_lo
            raw = PoleSamples(
                samples=raw.samples[m],
                observable_key=raw.observable_key,
                lambda_values=raw.lambda_values[m],
                param_samples=(raw.param_samples[m]
                               if raw.param_samples is not None else None),
            )
        lam_c_samples = fit_sqrt_per_realization(
            raw, lambda_c_guess=lambda_c_guess, lambda_c_bounds=lambda_c_bounds,
            window_left=50, window_right=15, omega_s=result.omega_s,
        )
        if np.any(np.isfinite(lam_c_samples)):
            q16, q50, q84 = np.nanpercentile(lam_c_samples, [16, 50, 84])
            print(f"  lam_c (per realization) = {q50:.4f} "
                  f"+{q84 - q50:.4f} / -{q50 - q16:.4f}  "
                  f"({np.mean(np.isnan(lam_c_samples)):.1%} of fits failed)")
            if json_path is not None and not DEBUG:
                augment_json_with_transitions(json_path, {
                    "sqrt_lambda_c_samples": lam_c_samples,
                    "sqrt_lambda_c_quantiles": [q16, q50, q84],
                })
        else:
            print("  Per-realization sqrt fit: no realization converged.")

    # 3-panel appendix-style figure; the fit lands on the result so the
    # "sqrt" figure layout can reuse it.
    cp.result.critical_point_sqrt_fit = fit
    # the figure shows every λ up to lambda_max (all of them without one); the sqrt
    # curve is drawn only up to the top of the fitted range
    fig_lam_max = lambda_max if lambda_max is not None else float(np.nanmax(cp.result.lambda_values))
    fig = plot_appendix_bootstrap(
        cp.result, lambda_max=fig_lam_max,
        fit_validity_limit=lam_hi if lam_hi is not None else fig_lam_max,
        real_axis_ticks=[0.0, 1, 2], real_split_axis_ticks=[0.0, 0.6, 1.2],
    )
    _show_and_save(fig, "critical_point_appendix",
                   data_mode=data_mode, scan_number=scan_number)

    if json_path is not None and not DEBUG:
        path = cp.save(json_path)
        print(f"Saved: {path.name}")
    return fit


def run_slider_plot(
    records: list[ObservableRecord], result: ZenoAnalysisResult, *,
    scales: dict[str, str],
    take_abs: bool,
    components: bool,
):
    """Step 5 — interactive per-λ trace browser; the fit overlay uses the result's poles.

    ``components=True`` draws the fit's terms A_k·exp(p_k·t) over the data instead of
    their sum, named and coloured as in the spectrum plot (e_1, e_2, ...; a conjugate
    pair as one curve, its real sum).  Returns the slider (also kept in
    ``_LIVE_WIDGETS`` so it stays responsive).
    """
    central = _central_poles(result)
    fixed_poles = {float(lam): central[i] for i, lam in enumerate(result.lambda_values)}
    slider = plot_time_traces_slider(
        records,
        observable_key=_observables(result),
        scales=scales,
        take_abs=take_abs,
        mpm_order=_order(result),
        components=components,
        fixed_poles=fixed_poles,
    )
    _LIVE_WIDGETS.append(slider)
    return slider


@functools.wraps(_steps.run_figure)          # step 6; saves into PAPER_PLOTS_DROPBOX/appendix
def run_figure(*args, **kwargs):
    return _steps.run_figure(*args, debug=DEBUG, save_figures=SAVE_FIGURES,
                             plots_dir=PAPER_PLOTS_DROPBOX, dpi=PAPER_FIGURES_DPI,
                             show_plots=SHOW_PLOTS, **kwargs)


def run_export_figure_data(
    records: list[ObservableRecord], result: ZenoAnalysisResult, *,
    json_path: Path | None,
    parent: str | Path,
    observable: str,
    n_smooth: int,
    drives_khz: tuple[int, ...] | None = None,
    reference_format: bool = False,
    plot: bool = False,
) -> Path:
    """Step 7 — per-λ fit curves into a NEW ``<parent>/fitting_results_<UTC stamp>/``.

    One ``<scan file>_mpm_fit.json`` per λ: ``times_smooth`` / ``fit_curve``
    of the modal model with the result's poles at that λ and amplitudes
    re-solved on the ``observable`` trace of ``records`` (``n_smooth`` points
    from the trace's first to its last time) — the same construction as the
    slider overlay.  Default (the files run_files/make_third_transition_figure.py
    reads): each file also carries the poles, amplitudes and provenance, and the
    folder receives this run's result JSON (panel B; ``json_path`` copied as it
    stands, or a fresh save when there is none).  ``reference_format`` → the
    files of PAPER_FIT_REFERENCE exactly: the keys times_smooth, fit_curve,
    lambda, original_file and nothing else, indent 4, CRLF, no result JSON;
    every file is checked against a reference file.  ``drives_khz`` → only the
    drives with these kHz in the file name (an error if one is missing).
    ``plot`` → each written file's fit_curve, read back from the file, over its
    trace.  Returns the folder.
    """
    from mpm_analysis.utils.windows_paths import win_long_path

    prov = _provenance(records)
    lam_res = np.asarray(result.lambda_values, dtype=float)
    poles_res = _central_poles(result)
    chosen, n_skipped = [], 0                      # (record, its poles) to write
    for rec, i in _records_with_pole_index(records, result, observable):
        if drives_khz is not None and _drive_khz(rec.source_file) not in drives_khz:
            continue
        poles = poles_res[i]
        if abs(lam_res[i] - rec.lambda_val) > 1e-6 or not np.all(np.isfinite(poles)):
            n_skipped += 1
            continue
        chosen.append((rec, poles))
    if drives_khz is not None:
        missing = sorted(set(drives_khz) - {_drive_khz(r.source_file) for r, _ in chosen})
        if missing:
            raise ValueError(f"export: no {observable} trace with finite poles at {missing} kHz "
                             f"in {prov['source_folder']}.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(parent) / f"fitting_results_{stamp}"
    Path(win_long_path(out)).mkdir(parents=True, exist_ok=False)
    print(f"\n=== Step 7 export ({prov['data_mode']}, {observable}) ===")

    result_name = None
    if not reference_format:
        # Panel B: the result JSON (already augmented with the sqrt fit if the
        # critical point ran; DEBUG never saved one, so save a fresh copy here).
        if json_path is not None:
            result_name = Path(json_path).name
            shutil.copy2(win_long_path(json_path), win_long_path(out / result_name))
        else:
            result_name = save_result(result, out, full=False).name
        print(f"  {result_name}")

    # Panel A: per-λ fit curves of `observable` with the result's poles.
    source_folder = prov["source_folder"]          # a list for a load from several folders
    written = []
    for rec, poles in chosen:
        amps = solve_linear_amplitudes(rec.t, rec.signal, poles)
        t_smooth = np.linspace(float(rec.t[0]), float(rec.t[-1]), int(n_smooth))
        payload = {
            "times_smooth": t_smooth.tolist(),
            "fit_curve": reconstruct_signal(t_smooth, poles, amps).tolist(),
            "lambda": float(rec.lambda_val),
        }
        if not reference_format:
            payload.update({
                "observable": observable,
                "poles_re_1_per_us": np.real(poles).tolist(),
                "poles_im_rad_per_us": np.imag(poles).tolist(),
                "amplitudes_re": np.real(amps).tolist(),
                "amplitudes_im": np.imag(amps).tolist(),
                "mpm_order": _order(result),
                "lambda_source": prov["lambda_source"],
                "analysis_json": result_name,
            })
        payload["original_file"] = str(Path(rec.metadata.get(_FILE_FOLDER) or source_folder)
                                       / rec.source_file)
        path = out / f"{Path(rec.source_file).stem}_mpm_fit.json"
        with open(win_long_path(path), "w", encoding="utf-8",
                  newline="\r\n" if reference_format else None) as f:
            json.dump(payload, f, indent=4 if reference_format else 1)
        if reference_format:
            _check_reference_format(path)
        written.append((rec, path))
    print(f"  {len(written)} *_mpm_fit.json files ({observable}, {n_smooth} points)"
          + (f", {n_skipped} records without a matching λ / finite poles" if n_skipped else "")
          + (f"; format checked against {PAPER_FIT_REFERENCE.name}" if reference_format else ""))
    print(f"  Folder: {out}")
    if plot and written:
        _plot_exported_fits(written, observable, prov["data_mode"])
    return out


def _drive_khz(name: str) -> int | None:
    """The drive (kHz) in a scan file name ``…OBG0_<kHz>_kHz…``; None without one."""
    m = re.search(r"OBG0_(\d+)_kHz", name)
    return int(m.group(1)) if m else None


def _records_with_pole_index(
    records: list[ObservableRecord], result: ZenoAnalysisResult, observable: str,
) -> list[tuple[ObservableRecord, int]]:
    """The ``observable`` records in λ order, each with the index of its λ slice in
    ``result`` — the pipeline's slice order (records sorted by λ, stable), so two
    files at one λ (scans side by side at one drive) each meet their own slice."""
    prov = _provenance(records)
    traces = sorted((r for r in records if r.observable_key == observable),
                    key=lambda r: r.lambda_val)
    if not traces:
        raise ValueError(f"observable '{observable}' is not among the loaded "
                         f"observables {prov['observables']}.")
    lam_res = np.asarray(result.lambda_values, dtype=float)
    taken: dict[float, int] = {}
    out = []
    for rec in traces:
        same = np.flatnonzero(np.abs(lam_res - rec.lambda_val) <= 1e-12)
        k = taken.get(rec.lambda_val, 0)
        taken[rec.lambda_val] = k + 1
        i = int(same[k]) if k < same.size else int(np.argmin(np.abs(lam_res - rec.lambda_val)))
        out.append((rec, i))
    return out


# the common prefix of the post-selected scan file names, dropped for the table's `name`
_SCAN_FILE_PREFIX = "click_record_with_post_selected_tomography_data__"


def run_pole_table(
    records: list[ObservableRecord], result: ZenoAnalysisResult, *,
    out_dir: str | Path,
    observable: str,
    plot: bool = True,
) -> Path:
    """Step 8 — the poles of every file with their error bars, one CSV row per file.

    Written into ``out_dir`` as ``poles_<observable>_scans_<…>.csv`` (no stamp: a rerun on
    the same scans overwrites) with ``…_bootstrap_procedure.txt`` beside it (the bootstrap
    of step 2 in numbered plain steps + the column glossary).  Rows are sorted by λ.  Columns:

    ``name`` (file name without the common prefix), ``scan`` (from the folder's
    hmm_lambda_per_file.csv: 3, 4+5, 14, …; else the records' scan_number),
    ``drive_kHz``, ``lambda``, ``shots_per_point`` (attempts behind every point
    t > 0 of ``observable``), ``n_draws_kept`` (bootstrap draws with three finite poles);
    then per pole k = 1..3 (slots 1, 2 = the EP pair, 3 = the slow real pole):
    ``pole{k}_re`` = Re (1/µs, = −decay rate) with ``_err_minus`` / ``_err_plus`` and
    ``pole{k}_im`` = Im (rad/µs) with the same — the per-slot 16/84 half-widths of
    ``result.poles``; then the pair-consistent numbers, one value per draw:
    ``pair_centre_re`` = (Re p1 + Re p2)/2, ``pair_re_splitting`` = |Re p2 − Re p1|/2,
    ``pair_im_splitting`` = |Im p2 − Im p1|/2 (``result.poles.*_splitting``),
    ``pair_delta`` = Re[(p1 − p2)²] (the EP discriminant of step 3), each with
    ``_err_minus`` / ``_err_plus`` = central − P16 / P84 − central of the draws;
    last ``file``.  A missing / non-finite value is an empty cell.

    ``plot`` → a check figure read back from the CSV: Re and Im of the three poles vs λ
    with their bars, and the pair columns.  Returns the CSV path.
    """
    from mpm_analysis.utils.windows_paths import win_long_path

    prov = _provenance(records)
    order = _order(result)
    lam_res = np.asarray(result.lambda_values, dtype=float)
    P = result.poles
    C = _central_poles(result)                                   # (n_lambda, order)
    raw = P.raw_samples
    D = None if raw is None else np.asarray(raw.samples)         # (n_lambda, n_boot, order)
    n_boot = 0 if D is None else D.shape[1]

    # scan tag per file: the union folder's per-file λ table (3 / 45 / 14 …)
    scan_of: dict[str, str] = {}
    folders = prov["source_folder"]
    for folder in (folders if isinstance(folders, list) else [folders]):
        table = Path(folder) / "hmm_lambda_per_file.csv"
        try:
            with open(win_long_path(table), newline="", encoding="utf-8") as f:
                scan_of.update({r["filename"]: r.get("scan", "") for r in csv.DictReader(f)})
        except OSError:
            pass
    def _scan_tag(rec) -> str:
        s = scan_of.get(rec.source_file)
        if s is None:
            s = rec.metadata.get("scan_number")
        s = "" if s is None else str(s)
        return "4+5" if s == "45" else s

    # pair-consistent draws: one number per draw, bars around the measured value
    def _bars(central: np.ndarray, draws: np.ndarray | None):
        """(err_minus, err_plus) = central − P16, P84 − central over the finite draws."""
        if draws is None:
            return np.full(central.shape, np.nan), np.full(central.shape, np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(draws, [16, 84], axis=1)
        return central - lo, hi - central
    centre_c = C[:, :2].real.mean(axis=1)
    delta_c = discriminant_from_poles(C)
    if D is not None:
        kept = np.all(np.isfinite(D), axis=2)                     # (n_lambda, n_boot)
        n_kept = kept.sum(axis=1)
        Dk = np.where(kept[..., None], D, np.nan)
        centre_m, centre_p = _bars(centre_c, Dk[:, :, :2].real.mean(axis=2))
        delta_m, delta_p = _bars(delta_c, discriminant_from_poles(Dk))
    else:
        n_kept = np.zeros(len(lam_res), dtype=int)
        centre_m = centre_p = delta_m = delta_p = np.full(len(lam_res), np.nan)
    def _split(name):
        v, lo, hi = (getattr(P, name), getattr(P, name + "_lower"), getattr(P, name + "_upper"))
        nan = np.full(len(lam_res), np.nan)
        return (nan if v is None else v, nan if lo is None else lo, nan if hi is None else hi)

    def _shots(rec) -> int | None:
        n = rec.metadata.get("bootstrap_stats", {}).get(f"N_attempts_{observable}")
        if n is None:
            return None
        n = np.asarray(n, dtype=float).ravel()
        body = n[1:] if n.size > 1 else n
        body = body[np.isfinite(body)]
        if body.size and not np.all(body == body[0]):
            raise ValueError(f"{rec.source_file}: N_attempts_{observable} is not constant for t > 0 "
                             f"({body.min():g} … {body.max():g}).")
        return int(body[0]) if body.size else None

    pole_cols = [f"pole{k}_{q}" for k in range(1, order + 1)
                 for q in ("re", "re_err_minus", "re_err_plus", "im", "im_err_minus", "im_err_plus")]
    pair_cols = [f"{p}{q}" for p in ("pair_centre_re", "pair_re_splitting", "pair_im_splitting",
                                     "pair_delta")
                 for q in ("", "_err_minus", "_err_plus")]
    header = (["name", "scan", "drive_kHz", "lambda", "shots_per_point", "n_draws_kept"]
              + pole_cols + pair_cols + ["file"])
    rsplit, isplit = _split("real_splitting"), _split("imag_splitting")

    def _cell(x):
        return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else x

    rows, scans_seen = [], []
    for rec, i in _records_with_pole_index(records, result, observable):
        if abs(lam_res[i] - rec.lambda_val) > 1e-6:
            raise RuntimeError(f"{rec.source_file}: λ {rec.lambda_val:.6f} has no slice in the result.")
        tag = _scan_tag(rec)
        if tag not in scans_seen:
            scans_seen.append(tag)
        row = [Path(rec.source_file).stem.removeprefix(_SCAN_FILE_PREFIX), tag,
               _drive_khz(rec.source_file), float(lam_res[i]), _shots(rec), int(n_kept[i])]
        for k in range(order):
            # Re = −decay: the decay rate's upper half-width is Re's lower one
            row += [-P.decay_rates[i, k], P.decay_rates_upper[i, k], P.decay_rates_lower[i, k],
                    P.frequencies[i, k], P.frequencies_lower[i, k], P.frequencies_upper[i, k]]
        row += [centre_c[i], centre_m[i], centre_p[i],
                rsplit[0][i], rsplit[1][i], rsplit[2][i],
                isplit[0][i], isplit[1][i], isplit[2][i],
                delta_c[i], delta_m[i], delta_p[i]]
        row.append(rec.source_file)
        # floats to 8 significant digits (readable; far below any error bar)
        rows.append([_cell(float(f"{x:.8g}") if isinstance(x, (float, np.floating)) else
                           int(x) if isinstance(x, np.integer) else x) for x in row])

    scans_tag = "_".join(s for s in scans_seen if s) or "unknown"
    # no stamp: a rerun on the same scans overwrites (order / n_boot are in the text file)
    stem = f"poles_{observable}_scans_{scans_tag}"
    out_dir = Path(out_dir)
    if not Path(win_long_path(out_dir)).is_dir():
        raise FileNotFoundError(f"pole table: out_dir does not exist: {out_dir}")
    csv_path = out_dir / f"{stem}.csv"
    txt_path = out_dir / f"{stem}_bootstrap_procedure.txt"
    with open(win_long_path(csv_path), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    t = np.asarray(records[0].t, dtype=float)
    dt = float(np.median(np.diff(t)))
    shots_by_scan = {}
    for rec, _i in _records_with_pole_index(records, result, observable):
        shots_by_scan.setdefault(_scan_tag(rec), set()).add(_shots(rec))
    shots_txt = "; ".join(f"scan {s or '?'}: {', '.join(f'{n:,}' for n in sorted(x for x in ns if x))}"
                          for s, ns in shots_by_scan.items())
    desc = next((s for s in result.parameters.get("step_descriptions", [])
                 if str(s).startswith("BootstrapStep")), "")
    with open(win_long_path(txt_path), "w", encoding="utf-8") as f:
        f.write(_pole_table_readme(
            csv_name=csv_path.name, observable=observable, order=order, n_boot=n_boot,
            scans=[s for s in scans_seen if s], n_files=len(rows), n_points=t.size,
            t_end=float(t[-1]), dt=dt, shots_txt=shots_txt, step_desc=desc,
            uncertainty_source=result.metadata.get("uncertainty_source"),
            redraw=result.metadata.get("bootstrap_redraw"), lambda_source=prov["lambda_source"],
        ))
    print(f"\n=== Step 8 pole table ({prov['data_mode']}, {observable}, order {order}, "
          f"{n_boot} draws) ===\n  {len(rows)} rows (scans {', '.join(scans_seen)}), "
          f"t = 0 ... {t[-1]:.1f} us ({t.size} points)\n  CSV : {csv_path}\n  text: {txt_path}")
    if plot:
        _plot_pole_table(csv_path, order, prov["data_mode"])
    return csv_path


def _pole_table_readme(*, csv_name, observable, order, n_boot, scans, n_files, n_points,
                       t_end, dt, shots_txt, step_desc, uncertainty_source, redraw,
                       lambda_source) -> str:
    """The text that sits beside the pole-table CSV: the bootstrap in numbered steps."""
    scans_s = ", ".join(scans) if scans else "?"
    return f"""POLES AND ERROR BARS IN {csv_name}
=================================================================================

Data: {observable} of scans {scans_s}, {n_files} files (one per drive), full record:
{n_points} points, t = 0 to {t_end:.1f} us, {dt:.1f} us bins.  Lambda: {lambda_source}.
Pencil order {order}.  Bootstrap: {n_boot} draws ({uncertainty_source}, around the {redraw} curve).

1. Data.  {observable}(t) = probability of no click up to t, from the segment one bin
   longer than t (the ring-down bin is never used).  Each point = count / N with
   N = shots_per_point (repetitions x 6 axes), the same N at every t > 0.
   N per scan: {shots_txt}.
   The t = 0 point is 1 by construction.

2. Measured fit = the central values.  Order-{order} matrix pencil on the measured curve
   (Hankel matrix; pencil parameter L at the largest gap between singular values
   {order} and {order + 1}).  Poles p = Re + i Im, Re in 1/us (decay rate = -Re), Im in rad/us.
   NOT normalized: raw physical units, nothing is divided by Omega_S (Omega_DG).
   Slots: the conjugate pair first (pole1 = -Im, pole2 = +Im), then the real poles by
   increasing Re.  Above the EP all three are real: pole1 / pole2 continue the pair,
   pole3 is the slow pole.

3. Smooth curve.  The measured curve with its Hankel matrix cut to the {order} largest
   SVD components (same L): S3(t).

4. Copies.  {n_boot} times, every point t > 0 is redrawn as Binomial(N, S3(t)) / N
   with the file's N; t = 0 stays 1.  One copy = one repeat with shot noise only.

5. Refit.  The same pencil (same L) on every copy, poles into the same slots.
   Copies with an aliased pole (|Im| >= 0.9 pi / dt) or a failed fit are dropped;
   n_draws_kept = copies left.

6. Per-slot bars (pole{{k}}_*_err_minus / _plus).  Per slot and per drive, the 16th and
   84th percentile of the kept copies' Re, and separately of their Im:
       err_minus = central - P16,   err_plus = P84 - central
   with central = the measured fit (not the mean of the copies).  Decay bars are
   clipped at 0, Im bars taken as absolute values.  Each bar is a percentile of its
   own list, so near the EP (copies with a pair mixed with copies with two real
   poles) the pole1 and pole2 bars do not come from the same copies.

7. The pair_* columns (12 columns: 4 numbers, each with err_minus / err_plus).

   * WHAT THE FOUR NUMBERS ARE.  pole1 and pole2 are a pair.  Instead of looking at
     each of them alone, these four numbers describe the pair as a whole:
         pair_centre_re    = (Re pole1 + Re pole2) / 2   the pair's position on the decay axis
         pair_re_splitting = |Re pole1 - Re pole2| / 2   half their distance in Re
         pair_im_splitting = |Im pole1 - Im pole2| / 2   half their distance in Im
         pair_delta        = Re[(pole1 - pole2)^2]       the discriminant Delta
     Below the EP the pair is complex conjugate (same Re, opposite Im), so
     pair_centre_re = Re of both, pair_re_splitting = 0, pair_im_splitting = the
     oscillation frequency, and pair_delta = -4 (pair_im_splitting)^2 < 0.
     Above the EP both poles are real, so pair_im_splitting = 0,
     pair_re_splitting = half their distance, and pair_delta = +4 (pair_re_splitting)^2 > 0.
     pair_delta = 0 exactly at the EP; it is the quantity the EP line fit uses.
     Example (lambda = 0.91): pole1 = -0.755 - 0.360 i, pole2 = -0.755 + 0.360 i
     gives pair_centre_re = -0.755, pair_re_splitting = 0, pair_im_splitting = 0.360,
     pair_delta = -0.517.

   * HOW THEIR BARS ARE MADE, AND WHY THEY ARE SEPARATE FROM THE pole1 / pole2 BARS.
     Each of the four numbers is recomputed in every one of the {n_boot} copies, giving
     a list of {n_boot} values per number; its bar is err_minus = value - P16 and
     err_plus = P84 - value, with P16 / P84 the 16th / 84th percentile of that list
     and "value" the measured fit.  The difference to step 6: there, the bar of pole1
     comes from pole1's list and the bar of pole2 from pole2's list, so near the EP,
     where some copies give a conjugate pair and other copies give two real poles,
     the two bars cannot tell you how the PAIR scatters (open or closed).  A pair
     number is one value per copy, so its bar does.  These bars are not clipped: a
     negative err_minus or err_plus means the measured value lies outside the copies'
     16-84 band on that side, by that amount (the step-6 bars show 0 there instead).

8. Columns.
   name             file name without the common prefix
   scan             3, 4+5, 14, ...
   drive_kHz        Omega_BG0 drive
   lambda           lambda of the file
   shots_per_point  N of step 1
   n_draws_kept     copies left of {n_boot} (step 5)
   pole{{k}}_re / _im   measured pole k (1/us, rad/us, not normalized) + per-slot bars (step 6)
   pair_*           pair values + bars (step 7)
   file             the data file
   Empty cell = no finite value.
"""


def _plot_pole_table(csv_path: Path, order: int, data_mode: str) -> None:
    """Step 8's check figure, read back from the CSV: Re and Im of every pole vs λ with
    the per-slot bars, and the pair centre / Im splitting / Δ with the pair bars."""
    from mpm_analysis.utils.windows_paths import win_long_path

    with open(win_long_path(csv_path), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    def col(name):
        x = np.array([float(r[name]) if r[name] != "" else np.nan for r in rows])
        # a negative pair bar = measured value outside the draws' band: drawn as 0
        return np.clip(x, 0, None) if "_err_" in name else x
    lam = col("lambda")
    scans = [r["scan"] for r in rows]
    markers = {s: m for s, m in zip(dict.fromkeys(scans), "osD^v<>")}
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    (ax_re, ax_im), (ax_c, ax_d) = axs
    colors = ["C0", "C1", "C2", "C3"]
    for k in range(1, order + 1):
        for s in markers:
            m = np.array([sc == s for sc in scans])
            ax_re.errorbar(lam[m], col(f"pole{k}_re")[m],
                           yerr=[col(f"pole{k}_re_err_minus")[m], col(f"pole{k}_re_err_plus")[m]],
                           fmt=markers[s], ms=4, color=colors[k - 1], capsize=2, lw=1,
                           label=f"pole {k}, scan {s}")
            ax_im.errorbar(lam[m], col(f"pole{k}_im")[m],
                           yerr=[col(f"pole{k}_im_err_minus")[m], col(f"pole{k}_im_err_plus")[m]],
                           fmt=markers[s], ms=4, color=colors[k - 1], capsize=2, lw=1)
    for s in markers:
        m = np.array([sc == s for sc in scans])
        ax_c.errorbar(lam[m], col("pair_centre_re")[m],
                      yerr=[col("pair_centre_re_err_minus")[m], col("pair_centre_re_err_plus")[m]],
                      fmt=markers[s], ms=4, color="C4", capsize=2, lw=1,
                      label=f"pair centre Re, scan {s}")
        ax_c.errorbar(lam[m], col("pair_im_splitting")[m],
                      yerr=[col("pair_im_splitting_err_minus")[m], col("pair_im_splitting_err_plus")[m]],
                      fmt=markers[s], ms=4, color="C5", capsize=2, lw=1,
                      label=f"pair Im splitting, scan {s}")
        ax_d.errorbar(lam[m], col("pair_delta")[m],
                      yerr=[col("pair_delta_err_minus")[m], col("pair_delta_err_plus")[m]],
                      fmt=markers[s], ms=4, color="C6", capsize=2, lw=1, label=f"scan {s}")
    ax_d.axhline(0, color="0.4", lw=0.8)
    ax_re.set_ylabel("Re pole (1/µs)"); ax_im.set_ylabel("Im pole (rad/µs)")
    ax_c.set_ylabel("pair centre Re (1/µs), Im splitting (rad/µs)")
    ax_d.set_ylabel(r"pair $\Delta$ = Re[(p1 − p2)²]")
    for ax in (ax_c, ax_d):
        ax.set_xlabel(r"$\lambda$")
    for ax in axs.flat:
        ax.grid(True, alpha=0.3, lw=0.5)
    ax_re.legend(fontsize=7, ncol=2); ax_c.legend(fontsize=7); ax_d.legend(fontsize=7)
    fig.suptitle(f"Pole table read back from {csv_path.name}  ({len(rows)} rows; per-slot bars "
                 "top, pair-consistent bars bottom)", fontsize=10)
    fig.tight_layout()
    _show_and_save(fig, "pole_table", data_mode=data_mode)


def _check_reference_format(path: Path) -> None:
    """Raise unless ``path`` is laid out like the PAPER_FIT_REFERENCE files: the same
    keys in the same order, value types, curve lengths, indent 4 + CRLF, file ending."""
    from mpm_analysis.utils.windows_paths import win_long_path

    ref_path = next(iter(sorted(PAPER_FIT_REFERENCE.glob("*_mpm_fit.json"))), None)
    if ref_path is None:
        raise FileNotFoundError(f"no *_mpm_fit.json in PAPER_FIT_REFERENCE = {PAPER_FIT_REFERENCE}")
    with open(win_long_path(ref_path), "rb") as f:
        ref = f.read()
    with open(win_long_path(path), "rb") as f:
        new = f.read()
    ref_json, new_json = json.loads(ref), json.loads(new)
    if list(new_json) != list(ref_json):
        raise RuntimeError(f"{path.name}: keys {list(new_json)}, {ref_path.name} has {list(ref_json)}.")
    head = b'{\r\n    "times_smooth": [\r\n        '
    failed = [name for name, ok in (
        ("value types", all(type(new_json[k]) is type(ref_json[k]) for k in ref_json)),
        ("curve lengths", all(len(new_json[k]) == len(ref_json[k])
                              for k in ("times_smooth", "fit_curve"))),
        ("indent 4 + CRLF", new.startswith(head) and new.count(b"\n") == new.count(b"\r\n")),
        ("file ending", new[-4:] == ref[-4:]),
    ) if not ok]
    if failed:
        raise RuntimeError(f"{path.name} differs from {ref_path.name} in: {', '.join(failed)}.")


def _plot_exported_fits(written: list, observable: str, data_mode: str) -> None:
    """Step 7's check figure: each written file's fit_curve, read back from the
    file, over the trace it was fitted to (log y for a positive trace)."""
    from mpm_analysis.utils.windows_paths import win_long_path

    ncols = min(len(written), 4)
    nrows = -(-len(written) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.4 * nrows), squeeze=False)
    for ax, (rec, path) in zip(axs.flat, written):
        with open(win_long_path(path), encoding="utf-8") as f:
            saved = json.load(f)
        ax.plot(rec.t, rec.signal, "o", ms=3, color="#2a78d6", label=f"{observable} data")
        ax.plot(saved["times_smooth"], saved["fit_curve"], color="k", lw=1.5,
                label="fit_curve read from the file")
        if np.all(np.asarray(rec.signal) > 0):
            ax.set_yscale("log")
        ax.set_title(f"{_drive_khz(rec.source_file)} kHz,  λ = {saved['lambda']:.3f}")
        ax.set_xlabel("t (µs)")
        ax.set_ylabel(observable)
        ax.legend(frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axs.flat)[len(written):]:
        ax.set_visible(False)
    fig.tight_layout()
    _show_and_save(fig, f"export_{observable}", data_mode=data_mode)


# ---- stand-alone checks ---------------------------------------------------

def run_simulation_test():
    """Quick end-to-end test on synthetic data (no real data needed)."""
    print("\n=== Simulation test ===")
    sim = PostSelectedDynamicsSimulator(
        num_lambdas=30, simulation_duration=25.0, min_alpha=0.1, max_alpha=2.0,
    )
    records = sim.to_observable_records()
    print(f"  Simulated {len(records)} records")
    steps = [MPMStep(order=3),
             BootstrapStep(n_boot=20 if DEBUG else 100, seed=SEED, mode="residual")]
    result = MPMPipeline.from_records(records, steps=steps).run()
    print(f"  Poles shape: {result.poles.decay_rates.shape}")
    fig, _ = plot_spectrum_panels(result, title="simulation")
    fig.tight_layout()
    _show_and_save(fig, "spectrum", data_mode="simulation")
    return result


def run_benchmark():
    """Analysis robustness with injected shot noise."""
    print("\n=== Benchmark ===")
    sim = PostSelectedDynamicsSimulator(
        num_lambdas=20, simulation_duration=40.0, min_alpha=0.1, max_alpha=2.0,
    )
    bench = BenchmarkPipeline(
        clean_records=sim.to_observable_records(),
        noise_models=[ShotNoise(n_shots=6 * 3e4, seed=0)],
        analysis_steps=[MPMStep(order=3),
                        BootstrapStep(n_boot=20 if DEBUG else 100, seed=SEED)],
        n_trials=5 if DEBUG else 20,
    )
    result = bench.run(lambda_c_guess=1.25)
    print(f"  lam_c bias = {result.lambda_c_bias:.4f}  std = {result.lambda_c_std:.4f}")
    return result


def run_liouvillian_spectrum():
    """Eigenvalues of the 4×4 Liouvillian vs λ at k=π."""
    print("\n=== Liouvillian spectrum (k=π) ===")
    fig, _ = LiouvillianSpectrum().plot()
    fig.suptitle("4×4 Liouvillian eigenvalues vs λ  (k=π)", fontsize=13)
    _show_and_save(fig, "liouvillian_spectrum", data_mode="theory")


if __name__ == "__main__":
    main()
