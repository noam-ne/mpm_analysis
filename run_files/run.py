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

    RUN_FIRST_TRANSITION   post-selected survival  (scan 3-8 / 10)   → first transition
    RUN_THIRD_TRANSITION   ensemble-average  D     (scan 0 / 1 / 50)  → third transition
    RUN_MC_TWIN            the first-transition steps on a simulated scan (MC_RUNS)

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
    6  run_figure         → fig           paper figure from this run's result
    7  run_export_figure_data → folder    (third only) result JSON + per-λ fit curves, a
                                          self-contained snapshot of the figure inputs
                                          (make_third_transition_figure.py reads third_json
                                          and its data folder directly; it does not need this)

The third transition (ensemble average) has no state-freezing exponent — only
steps 3 and 4 run on it.

The stand-alone checks at the end of ``main()`` (simulation test, benchmark,
Liouvillian spectrum) need no experimental data.

The module-level settings (paths, MC twin, λ source) are importable without
running anything (run_files/write_shots_per_lambda.py reads
``POST_SELECTED_FOLDER`` that way); the steps only run under ``__main__``.
"""
from __future__ import annotations

import fnmatch
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
from mpm_analysis.io.experimental import EnsembleAverageLoader, PostSelectedLoader, ParityLoader  # noqa: F401  (ParityLoader: PARITY_FOLDER, data_mode="parity")
from mpm_analysis.simulation.noise import ShotNoise
from mpm_analysis.plotting.exploratory import (
    plot_spectrum_panels, plot_time_traces_slider, plot_svd_singular_values,
)
from mpm_analysis.plotting.paper_figures.appendix_bootstrap import plot_appendix_bootstrap
from mpm_analysis.io.file_naming import build_figure_filename, save_result

# ===========================================================================
# GLOBAL SETTINGS  (apply to every step)
# ===========================================================================

DEBUG = False               # True → few files, small bootstrap, nothing saved
SHOW_PLOTS = True       # plt.show() at the end of the run
SAVE_FIGURES = False       # png into OUTPUT_DIR (paper figures: PDF+SVG+PNG into PAPER_FIGURES_DIR)
SAVE_FULL_ANALYSIS = False  # also write raw pole samples + refine params/cov into the result JSON
SEED: int | None = None     # RNG seed for every bootstrap; None → OS entropy
N_JOBS = -1                 # parallel workers for the data bootstrap (-1 → all cores)
MAX_FILES = 10 if DEBUG else None   # load only the first N λ files (debugging)

# ===========================================================================
# PATHS
# ===========================================================================
# Dropbox root is auto-detected from the hostname (utils/lab_paths.py); add
# your machine there if you get a "hostname not recognised" ValueError.

_ZT = r"Common\QuantumZeno\Paper\zeno_paper_data"
_NEW_SCAN_FOLDER = r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab\Experiments\Zeno\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons\click_record_with_post_selected_tomography\Omega_BG0_scan"
POST_SELECTED_DROPBOX = Path(get_lab_path(_ZT + r"\scan_experiment_results_10"))
ENS_AVG_DROPBOX = Path(get_lab_path(_ZT + r"\scan_experiment_results_50"))
PARITY_FOLDER = Path(get_lab_path(
    _ZT + r"\click_record_repeated\Omega_BG0_scan\scan_experiment_results_12"
))
# Data folders actually loaded.  Local mirrors hold the same files, BUT the
# ens_avg HMM λ table (scan-10's misfiled results.csv) is only discovered
# inside the Dropbox tree — from the local mirror the ens_avg λ silently
# falls back to the stored JSON value (~11 % lower), and would no longer
# match what make_third_transition_figure.py loads for panel A.
# All iterations of scan 8 merged per point (raw_data_tests/combine_scan8_iterations.py;
# rerun it after more scan_iteration_* points land, it refreshes the folder)
# First-transition dataset: name -> (folder, file-name glob).  ONE choice sets
# both, so the subset can never point into a folder that does not hold it.
# combined_results_4_and_5… holds scan 3 (copied in) and scans 4+5, told apart
# by the scanner's time stamp in the file name; its own HMM table covers all
# 41 drives, so "scan3" / "scan45" / "scan345" share one λ axis.  The λ table
# follows the folder (LAMBDA_SOURCE_POST below).
_COMBINED_345 = Path(_NEW_SCAN_FOLDER) / "combined_results_4_and_5__lambda_0.3-1.7__navg_11700"
POST_SELECTED_DATASETS = {
    "scan3":   (_COMBINED_345, "*_15_10_2026__16_57_51.json"),   # 18 drives
    "scan45":  (_COMBINED_345, "*_15_10_2026__10_08_55.json"),   # 23 drives
    "scan345": (_COMBINED_345, "*.json"),                        # all 41
    # scan 3 in its own folder: its own 18-drive HMM table, + S_pooled_strict
    "scan3_iterations": (Path(_NEW_SCAN_FOLDER) / "scan_experiment_results_3" / "combined_iterations",
                         "*.json"),
    # 3+4+5 with the scan-10 boost merged in (4 boosted drives, 22 Oct stamp)
    "scan345_boost": (Path(_NEW_SCAN_FOLDER) / "combined_results_3_4_5_and_10_boost", "*.json"),
    # all iterations of scan 8 merged per point (raw_data_tests/combine_scan8_iterations.py)
    "scan8": (Path(_NEW_SCAN_FOLDER) / "scan_experiment_results_8" / "combined_iterations", "*.json"),
}
POST_SELECTED_DATASET = "scan3"
POST_SELECTED_FOLDER, POST_SELECTED_FILES = POST_SELECTED_DATASETS[POST_SELECTED_DATASET]

ENS_AVG_FOLDER = Path(r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab\Experiments\Zeno\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons\survival_probability\Omega_BG0_scan\combined_results_0_and_1_merged")
# ENS_AVG_FOLDER = Path(r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code\old_survival_data\scan_experiment_results_50")

# Monte Carlo twins live in their own clause (RUN_MC_TWIN in main()) and never
# replace the data folders above.  mpm_benchmarks/scan8/mc/run_mc.py writes
# runs/<variant>/post_selected/ (scan-8 twin) or ens_avg/ (scan-0 twin) in the
# scanners' file formats, plus run_summary.json with the simulated parameters.
MC_RUNS = Path(__file__).parent.parent.parent / "mpm_benchmarks" / "scan8" / "mc" / "runs"

# λ axis.  Default: each file's stored parameters.lambda_zeno_estimation.
# Alternatively a per-drive HMM table (hmm_rates_per_drive.csv: converged
# offline HMM of the alpha click records, one row per drive):
# λ = gamma_BG_per_s_mean(drive) / (2·Ω_DG), looked up exactly at each file's
# drive — never interpolated; a drive missing from the table is an error.
# ("auto" would search the scan tree for a run-keyed table — scan-50 only.)
# One table per data mode; each must list every drive of the folder it is
# applied to (exact match), so the table follows the folder choice above:
#   scan 8 (combined_iterations / scan_iteration_*)  → scan_experiment_results_8/hmm_rates_per_drive.csv
#       (= alpha_measurement/hmm_rates_per_drive_records01_merged.csv: alpha records 0+1 of
#        every iteration, 71 drives; the full run is alpha_measurement/hmm_results_18_10_2026__21_33_27)
#   scan 3 / scan 4+5 combined                        → scan_experiment_results_3/hmm_rates_per_drive.csv
#                                                       (scan 5 drives are NOT in it → stored λ only)
#   ens_avg scan_experiment_results_1 / _0           → newest alpha_measurement/hmm_results_*/hmm_rates_per_drive.csv
#   (combined_results_0_and_1_merged/hmm_rates_per_drive.csv covers scans 0 and 1)
LAMBDA_SOURCE_POST: bool | str | Path = False   # PostSelectedLoader csv_path (first transition)
LAMBDA_SOURCE_ENS: bool | str | Path = False    # EnsembleAverageLoader csv_path (third transition)
if (HMM_DRIVE_LAMBDA := True):
    # the folder's own table (combined folders: merged over their source scans,
    # raw_data_tests/merge_hmm_drive_tables.py) or the one beside it (scan folders)
    LAMBDA_SOURCE_POST = next(
        (p for p in (Path(POST_SELECTED_FOLDER) / "hmm_rates_per_drive.csv",
                     Path(POST_SELECTED_FOLDER).parent / "hmm_rates_per_drive.csv")
         if p.exists()),
        Path(POST_SELECTED_FOLDER).parent / "hmm_rates_per_drive.csv",   # missing → loader error names it
    )
    # ens_avg scans keep the HMM run beside the alpha records: newest stamp wins
    LAMBDA_SOURCE_ENS = next(
        (p for p in [Path(ENS_AVG_FOLDER) / "hmm_rates_per_drive.csv"]
         + sorted((Path(ENS_AVG_FOLDER) / "alpha_measurement").glob("hmm_results_*/hmm_rates_per_drive.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
         if p.exists()),
        Path(ENS_AVG_FOLDER) / "hmm_rates_per_drive.csv",                 # missing → loader error names it
    )
OUTPUT_DIR = Path(__file__).parent.parent / "mpm_analysis_output"
PAPER_FIGURES_DIR = Path(__file__).parent.parent / "paper_figures"
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
    if (RUN_FIRST_TRANSITION := True):
        print("\n##### FIRST TRANSITION  (post_selected) #####")

        # Step 1 — load.
        #   in : loader (λ axis from LAMBDA_SOURCE_POST) + folder, observables
        #   out: first_records — one ObservableRecord per (λ file × observable);
        #        the measurement statistics for the data bootstrap ride along
        #        in record.metadata, and the provenance (data mode, folder, λ
        #        source, drop_first_point) is stamped in for the result JSON.
        first_records = load_records(
            PostSelectedLoader(csv_path=LAMBDA_SOURCE_POST), POST_SELECTED_FOLDER,
            file_glob=POST_SELECTED_FILES,  # files of the folder (POST_SELECTED_DATASET above)
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
                                            #   init history — s_next_ringdown_20261026),
                                            #   "S_prev_click" / "S_prev_noclick", X/Y/Z …
                                            #   The MC twin writes only the stored S (its NPZ
                                            #   holds `clicks`, which run_pooled_p0.py does not
                                            #   read), so S_pooled does not exist for it → "S".
            lambda_range=(0, 2),            # λ window to load
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
                file_glob=POST_SELECTED_FILES,
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
                n_boot=0,                         # bootstrap draws / covariance samples; 0 → no
                                                    #   bootstrap: the measured fit, zero error bars,
                                                    #   no per-draw EP / λ_c quantiles
                bootstrap_center="measured",           # "draws": central poles = mean of the draws,
                                                    #   error bar = their std — the measured curve is
                                                    #   one more sample of the data (errorbar_bootstrap)
                                                    # "measured": central = fit of the measured curve,
                                                    #   16/84 of the draws around it
                refine_poles=True,                 # weighted complex-exponential LSQ after the pencil
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
                    points_below=3,         # fit points: the 3 λ before Delta turns positive
                    points_above=2,         #   and the 2 after (None, None → ±half_width)
                    half_width=0.3,         # λ half-window, used only without points_*
                    lambda_max=None,        # e.g. 1.55 to match the critical-point window
                    plot_discriminant=True, # Delta(λ) with error bars + the fitted line
                )

                # Step 3c — the EP vs the fit window: steps 1-3 redone for ~6
                # windows (start t_min, or end t_max, moved), every other
                # setting taken from first_records / first_result / first_ep.
                # One figure: Delta(λ) per window, λ_EP ± its fit error in the
                # legend.
                #   in : first_records, first_result, first_ep
                #   out: first_window_sweep — one dict per window (lam_ep, lam_ep_err, …)
                if (FIRST_WINDOW_SWEEP := True):
                    first_window_sweep = run_window_sweep(
                        first_records, first_result, first_ep,
                        loader=PostSelectedLoader(csv_path=LAMBDA_SOURCE_POST),
                        vary="t_max",       # "t_min" (move the window start) | "t_max" (its end)
                        values=None,        # µs, e.g. (0, 0.2, 0.4, 0.6, 0.8, 1.0); None → 6
                                            #   automatic: the first 6 time samples from the main
                                            #   window's start ("t_min"), or 6 ends from the record
                                            #   end down to its middle ("t_max"); at most 8
                        n_boot=20,         # bootstrap draws PER window (the λ_EP error bar);
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
            if (FIRST_CRITICAL_POINT := True):
                first_sqrt_fit = run_critical_point(
                    first_result, json_path=first_json,
                    lambda_c_guess=1.4,
                    lambda_c_bounds=(1, 1.6),    # guess ± 0.15
                    lambda_max=2,                 # exclude λ above this from the fit
                    per_realization=True,
                )

            # Step 5 — interactive per-λ trace browser; the overlay is the modal
            # model with first_result's poles, amplitudes re-solved per trace.
            #   in : first_records, first_result
            #   out: first_slider (kept alive in _LIVE_WIDGETS)
            if (FIRST_SLIDER := True):
                first_slider = run_slider_plot(
                    first_records, first_result,
                    scales={"S": "log", "S_pooled": "log", "S_pooled_strict": "log",
                            "S_next": "log", "S_next_far": "log", "S_next_x2": "log"},   # key → "log"/"linear"
                    take_abs=False,         # plot/fit |signal| (e.g. for parity)
                )

            # Step 6 — paper figure from this run's result.
            #   "discriminant" → 2 panels: A spectrum, B Delta(λ) line fit
            #   "sqrt"         → 3 panels: A spectrum, B/C splittings + a sqrt
            #                    overlay chosen by splitting_fit
            #   The Delta line and λ_EP drawn ARE step 3's fit (first_ep), not a refit.
            #   in : first_result, first_ep
            #   out: first_fig (PDF+SVG+PNG into PAPER_FIGURES_DIR if SAVE_FIGURES)
            if (FIRST_FIGURE := True):
                first_fig = run_figure(
                    first_result, first_ep,
                    layout="sqrt",                      # appendix: A decay rates, B Im / C Re
                                                        #   splitting + black √ of step 3's Δ line
                    lambda_max=2,                     # λ cutoff of every panel
                    fit_validity=1.2,
                    real_axis_scale="linear",           # panel A y axis: "linear" | "log"
                    real_axis_ticks=None,
                    real_split_axis_ticks=[0.0, 0.5, 1.0],   # panel C (Re splitting / Ω_S)
                    splitting_fit="discriminant",       # "sqrt" layout overlay:
                                                        #   "discriminant" → √ of step 3's Delta line
                                                        #     (√(−Delta)/2 below λ_EP, √Delta/2 above,
                                                        #     over step 3's window); no step 4
                                                        #   "sqrt" → step 4's sqrt fit (needs step 4)
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
            lambda_range=(0, 1.65),
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
                n_boot=100,
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
                    half_width=0.2,
                    lambda_max=None,
                    plot_discriminant=True,
                )

            # Step 4 — sqrt fit → λ_c.   out: third_sqrt_fit (also on third_result)
            if (THIRD_CRITICAL_POINT := True):
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
                )

            # Step 6 — paper figure (Delta line + λ_EP = step 3's).   out: third_fig
            if (THIRD_FIGURE := True):
                third_fig = run_figure(
                    third_result, third_ep,
                    layout="sqrt",
                    lambda_max=1.35,
                    fit_validity=1.3,
                    real_axis_scale="linear",
                    real_axis_ticks=[0.0, 1.0, 2.0],
                    real_split_axis_ticks=[0.0, 0.4, 0.8],
                    splitting_fit="discriminant",   # "sqrt" layout overlay: √ of step 3's Delta line
                                                    #   | "sqrt" (step 4's fit)
                )

            # Step 7 — snapshot of the figure inputs, written into a NEW
            # <parent>/fitting_results_<UTC stamp>/  folder (old folders are never
            # touched).  run_files/make_third_transition_figure.py does not need it:
            # it draws the newest third_json and reloads that run's data itself.
            #   mpm_analysis_results_*.json      panel B — third_json as it stands now
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
                    parent=ENS_AVG_DROPBOX,     # where the figure script reads from
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
                    half_width=0.3,
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
                    lambda_c_guess=1.2,
                    lambda_c_bounds=(0.85, 1.5),
                    lambda_max=1.5,
                    per_realization=True,
                )

            # Step 5 — trace browser.   out: mc_slider
            if (MC_SLIDER := False):
                mc_slider = run_slider_plot(
                    mc_records, mc_result,
                    scales={"S": "log"},
                    take_abs=False,
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

# Key under which load_records stamps the provenance into every record's
# metadata; run_mpm copies it into the result metadata (→ the saved JSON).
_PROVENANCE = "run_provenance"


def _show_and_save(fig, figure_tag: str, *, data_mode: str = "unknown",
                   scan_number: int | None = None) -> None:
    """Save the figure as png into OUTPUT_DIR (if enabled) and show it."""
    if SAVE_FIGURES and not DEBUG:
        filename = build_figure_filename(
            figure_tag=figure_tag, data_mode=data_mode,
            scan_number=scan_number, extension="png",
        )
        path = OUTPUT_DIR / filename
        fig.savefig(path, bbox_inches="tight")
        print(f"  Figure saved: {path.name}")
    if SHOW_PLOTS:
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


def _order(result: ZenoAnalysisResult) -> int:
    """MPM model order of the result."""
    return int(result.parameters["pole_order"])


def _central_poles(result: ZenoAnalysisResult) -> np.ndarray:
    """Central poles  -decay + i·frequency, shape (n_lambda, order)."""
    return -result.poles.decay_rates + 1j * result.poles.frequencies


def _check_pole_ordering(result: ZenoAnalysisResult) -> None:
    """Every bootstrap draw must label its poles the way the central fit does.

    Per λ: (1) no decay rate <= 0 (a growing pole), central or draw; (2) the
    EP pair the discriminant follows along λ is slots 0/1 in every draw —
    the splittings, the sqrt fit and ksi read slots 0/1/2 directly; (3) the
    stored central value (decay rate per slot, real and imaginary splitting)
    lies inside its draws' 16-84 band.  The aggregated half-widths are kept
    as |c - p16|, |p84 - c|, so a central value outside the band (mislabelled
    draws, or a bias larger than the error bar) would otherwise go unseen.
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


# ===========================================================================
# STEP FUNCTIONS  (one per step of main(); settings are keyword-only)
# ===========================================================================

def load_records(
    loader, folder: str | Path, *,
    data_mode: str,
    observables: list[str],
    lambda_range: tuple[float, float] | None,
    t_range: tuple[float | None, float | None] | None,
    drop_first_point: bool,
    file_glob: str = "*.json",
) -> list[ObservableRecord]:
    """Step 1 — load the traces.  Returns the records.

    One ``ObservableRecord`` per (λ file × observable), sorted by λ then
    observable; ``record.metadata`` carries the bootstrap statistics the
    loader found, plus the provenance of this load (data mode, folder, file
    glob, λ source, observables, t_range, drop_first_point) that run_mpm
    writes into the result JSON.  ``t_range = (t_min, t_max)`` in µs keeps
    only ``t_min <= t <= t_max`` (either end ``None`` → open); ``None`` keeps
    the full record.  ``file_glob`` keeps only the files whose name matches
    (one source run of a merged folder, POST_SELECTED_SUBSETS).
    ``MAX_FILES`` (global) caps the number of files.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"[{data_mode}] folder not found: {folder}")
    names = [p.name for p in folder.glob("*.json")]
    if not any(fnmatch.fnmatch(n, file_glob) for n in names):
        stamps = sorted({m.group(1) for n in names
                         if (m := re.search(r"_kHz_(.+)\.json$", n))})
        raise FileNotFoundError(
            f"[{data_mode}] no file in {folder} matches {file_glob!r}; the file stamps "
            f"there are {stamps}.  POST_SELECTED_DATASETS pairs each subset with its folder.")
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
        max_files=MAX_FILES,
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
        "file_glob": file_glob,
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
          + ("" if file_glob == "*.json" else f"  (files {file_glob})"))
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


def run_mpm(
    records: list[ObservableRecord], *,
    order: int,
    uncertainty_source: str,
    n_boot: int,
    refine_poles: bool,
    conjugate_pairs: bool,
    plot_svd: bool,
    bootstrap_center: str = "measured",
) -> tuple[ZenoAnalysisResult, Path | None]:
    """Step 2 — matrix pencil + bootstrap on the records.

    Returns ``(result, json_path)``: the ``ZenoAnalysisResult`` (central
    poles, 16/84 bands, raw bootstrap draws in ``result.poles.raw_samples``)
    and the path of the saved result JSON (``None`` in DEBUG).  Also draws
    the spectrum panels.
    """
    prov = _provenance(records)
    print(f"\n=== MPM ({prov['data_mode']}, observables={prov['observables']}, order={order}) ===")
    pipeline, result = _mpm_pipeline(
        records, order=order, uncertainty_source=uncertainty_source, n_boot=n_boot,
        refine_poles=refine_poles, conjugate_pairs=conjugate_pairs, plot_svd=plot_svd,
        bootstrap_center=bootstrap_center,
    )
    _check_pole_ordering(result)

    fig, _ = plot_spectrum_panels(result, title=prov["data_mode"])
    fig.tight_layout()
    tag_mode, scan_number = _result_tags(result)
    _show_and_save(fig, "spectrum", data_mode=tag_mode, scan_number=scan_number)

    json_path = None
    if not DEBUG:
        json_path = pipeline.save(OUTPUT_DIR, full=SAVE_FULL_ANALYSIS)
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
) -> tuple[MPMPipeline, ZenoAnalysisResult]:
    """MPM → (Refine) → Bootstrap on the records, run (step 2 without its
    figure and JSON; the window sweep reruns it per window).  Returns
    ``(pipeline, result)``."""
    prov = _provenance(records)
    data_mode, observables = prov["data_mode"], prov["observables"]
    src = _resolve_uncertainty_source(uncertainty_source, data_mode)
    n_boot = min(n_boot, 20) if DEBUG else n_boot
    steps = [
        MPMStep(order=order, observable_key=observables,
                plot_svd=plot_svd, n_sv=10, save_dir=OUTPUT_DIR),
    ]
    if refine_poles or src == "fit_covariance":
        # fit_covariance needs the refinement covariance → RefineStep forced on
        steps.append(RefineStep(observable_key=observables,
                                conjugate_pairs=conjugate_pairs))
    mode = {"data_bootstrap": "data", "errorbar_bootstrap": "errorbar",
            "fit_covariance": "parametric", "residual_bootstrap": "residual"}[src]
    steps.append(BootstrapStep(n_boot=n_boot, seed=SEED, mode=mode, n_jobs=N_JOBS,
                               center=bootstrap_center))

    # Same metadata MPMPipeline.from_loader would have written.
    metadata = dict(
        source_folder=prov["source_folder"],
        data_mode=data_mode,
        source_tag=data_mode,          # legacy alias kept for old readers
        observables=observables,
        lambda_source=prov["lambda_source"],
        uncertainty_source=src,
        bootstrap_center=bootstrap_center,
    )
    if prov["drop_first_point"]:
        metadata["drop_first_point"] = True
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
    half_width: float,
    lambda_max: float | None,
    plot_discriminant: bool,
    points_below: int | None = None,
    points_above: int | None = None,
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
    ``json_path`` (unless DEBUG / None).
    """
    data_mode, scan_number = _result_tags(result)
    obs_tag = "+".join(_observables(result))
    print(f"\n=== Exceptional point ({data_mode}, {obs_tag}) ===")

    c = _ep_crossing(result, method=method, half_width=half_width, lambda_max=lambda_max,
                     points_below=points_below, points_above=points_above)
    lam_d, d_central, d_sigma = c["lam_d"], c["d_central"], c["d_sigma"]
    lam_ep, err, fit, first = c["lam_ep"], c["lam_ep_err"], c["fit"], c["first"]
    ep: dict = {"delta_crossing_method": method, "delta_half_width": half_width,
                "delta_points_below": points_below, "delta_points_above": points_above,
                "delta_lambda_max": lambda_max, "lambda_ep": float(lam_ep),
                "lambda_ep_error": err, "chi2": c["chi2"], "dof": c["dof"]}
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
    if fit["window"] is not None:
        wl = lam_d[fit["window"]]
        n_lo = int(np.sum(wl < lam_ep))
        rule = (f"fixed: {points_below} before / {points_above} after Delta turns positive"
                if points_below is not None or points_above is not None
                else f"+-{half_width}" + (" held fewer than 5" if fit.get("widened") else ""))
        print(f"                  fit points: lam {', '.join(f'{x:.3f}' for x in wl)}  "
              f"({n_lo} below / {wl.size - n_lo} above lam_EP; {rule}"
              + "); raw first sign change at " + (f"{first[0]:.4f}" if first else "none"))

    if plot_discriminant:
        ax_d = plot_discriminant_curve(
            lam_d, d_central, fit=fit, lambda_ep=lam_ep, lambda_ep_sigma=err,
            first_crossing=first[0] if first else None, delta_sigma=d_sigma,
            title=f"Exceptional point: Delta = 0  ({data_mode}, {obs_tag})",
        )
        _show_and_save(ax_d.get_figure(), "exceptional_point_discriminant",
                       data_mode=data_mode, scan_number=scan_number)

    if json_path is not None and not DEBUG and np.isfinite(lam_ep):
        saved = {"delta_crossing_method": method, "delta_linfit_central": float(lam_ep)}
        if err is not None:
            saved.update(delta_linfit_sigma=float(err), delta_linfit_chi2=c["chi2"],
                         delta_linfit_dof=c["dof"])
        augment_json_with_transitions(json_path, saved)
    return ep


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
    )
    crossing = dict(method=ep["delta_crossing_method"], half_width=ep["delta_half_width"],
                    lambda_max=ep.get("delta_lambda_max"),
                    points_below=ep.get("delta_points_below"),
                    points_above=ep.get("delta_points_above"))
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
        print(f"  [{r['t_window'][0]:5.2f}, {r['t_window'][1]:6.2f}] {r['n_points']:6d} "
              f"{r['lam_ep']:8.4f} {err:>7} {chi:>10}  {pts}")

    for r in rows:
        a, b = r["t_window"]
        r["label"] = rf"$t \in [{a:.2f}, {b:.2f}]$ µs"
    data_mode, scan_number = _result_tags(result)
    obs_tag = "+".join(_observables(result))
    sub_tag = "" if subset == "*.json" else f", files {subset}"
    fig = plot_discriminant_window_sweep(
        rows, title=f"Exceptional point vs fit window ({data_mode}, {obs_tag}{sub_tag})")
    _show_and_save(fig, f"discriminant_window_sweep_{vary}",
                   data_mode=data_mode, scan_number=scan_number)
    return rows


def _ep_crossing(
    result: ZenoAnalysisResult, *,
    method: str,
    half_width: float,
    lambda_max: float | None,
    points_below: int | None = None,
    points_above: int | None = None,
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
       ±``half_width`` around the sign change (≥ 5 nearest).

    ``method="first"``: λ_EP = the first sign change, no error.  Returns a
    dict: ``lam_d``, ``d_central``, ``d_sigma`` (None without draws),
    ``lam_ep``, ``lam_ep_err`` (None without draws), ``chi2``, ``dof``,
    ``fit`` (window, coef, …), ``first`` (raw sign changes), ``has_draws``.
    """
    import warnings

    lam_d = np.asarray(result.lambda_values, dtype=float)
    d_central = discriminant_from_poles(_central_poles(result))
    keep = np.ones(lam_d.shape, dtype=bool) if lambda_max is None else lam_d <= lambda_max
    lam_d, d_central = lam_d[keep], d_central[keep]

    d_sigma = None
    raw = result.poles.raw_samples
    if raw is not None:
        d_draws = discriminant_from_poles(np.asarray(raw.samples))[keep]   # (n_lam, n_boot)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # failed draws: NaN
            lo, hi = np.nanpercentile(d_draws, [16, 84], axis=1)
        d_sigma = (hi - lo) / 2

    lam_ep, fit = linfit_zero_crossing(lam_d, d_central, sigma=d_sigma,
                                       half_width=half_width, return_fit=True,
                                       n_below=points_below, n_above=points_above)
    first = find_zero_crossings(lam_d, d_central)
    lam_ep_err = fit.get("sigma_zero")
    if method == "first":
        lam_ep, lam_ep_err = (first[0] if first else np.nan), None
    return dict(lam_d=lam_d, d_central=d_central, d_sigma=d_sigma,
                lam_ep=float(lam_ep), lam_ep_err=lam_ep_err,
                chi2=fit.get("chi2"), dof=fit.get("dof"), fit=fit, first=first,
                has_draws=raw is not None)


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
):
    """Step 4 — sqrt fit of the pole splitting → lambda_c (aggregate + per realization).

    Returns the ``SqrtFitResult``; it is also attached to the result as
    ``result.critical_point_sqrt_fit`` (the "sqrt" figure layout reads it)
    and written into ``json_path`` with the per-realization quantiles.
    """
    data_mode, scan_number = _result_tags(result)
    print(f"\n=== Critical-point fit ({data_mode}, {'+'.join(_observables(result))}) ===")
    cp = CriticalPointPipeline(result, lambda_max=lambda_max)
    fit = cp.fit(
        lambda_c_guess=lambda_c_guess, soft_bounds=True, lambda_c_bounds=lambda_c_bounds,
        window_left=50, window_right=15,
    )
    print(f"  lam_c (aggregate fit) = {fit.lambda_c:.4f} +/- {fit.lambda_c_err:.4f}   "
          f"[guess {lambda_c_guess}, bounds {lambda_c_bounds}]")
    print(f"  chi2_red = {fit.chi2_reduced:.3f}  R2 = {fit.r_squared:.4f}")

    if per_realization and result.poles.raw_samples is not None:
        raw = result.poles.raw_samples
        if lambda_max is not None:
            m = raw.lambda_values <= lambda_max
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
    fig = plot_appendix_bootstrap(
        cp.result, lambda_max=lambda_max, fit_validity_limit=lambda_max,
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
):
    """Step 5 — interactive per-λ trace browser; the fit overlay uses the result's poles.

    Returns the slider (also kept in ``_LIVE_WIDGETS`` so it stays responsive).
    """
    central = _central_poles(result)
    fixed_poles = {float(lam): central[i] for i, lam in enumerate(result.lambda_values)}
    slider = plot_time_traces_slider(
        records,
        observable_key=_observables(result),
        scales=scales,
        take_abs=take_abs,
        mpm_order=_order(result),
        fixed_poles=fixed_poles,
    )
    _LIVE_WIDGETS.append(slider)
    return slider


def run_figure(
    result: ZenoAnalysisResult, ep: dict, *,
    layout: str,
    lambda_max: float,
    fit_validity: float | None,
    real_axis_scale: str,
    real_axis_ticks: list | None,
    real_split_axis_ticks: list | None,
    splitting_fit: str,
):
    """Step 6 — paper transition figure from this run's result.

    The Δ line fit and λ_EP drawn are step 3's (``ep["delta_fit"]``), not a
    refit, so the figure shows exactly what step 3 reported.
    ``layout="sqrt"`` with ``splitting_fit="sqrt"`` needs
    ``result.critical_point_sqrt_fit`` (step 4); with ``"discriminant"`` the
    overlay is the square root of step 3's line and step 4 is not needed.
    Returns the figure, or ``None`` when the sqrt fit is missing.  With
    SAVE_FIGURES: PDF + SVG + high-res PNG into PAPER_FIGURES_DIR.
    """
    from mpm_analysis.plotting.paper_figures.first_transition_main import (
        plot_first_transition_figure,
    )
    data_mode, scan_number = _result_tags(result)
    observables, order = _observables(result), _order(result)
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
        discriminant_fit=ep["delta_fit"],
    )
    # Paper figures: own folder, PDF + SVG + high-res PNG, stem naming the
    # layout, data mode, scan, observables and order + UTC stamp.
    if SAVE_FIGURES and not DEBUG:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        layout_tag = "sqrt_from_disc" if (layout, splitting_fit) == ("sqrt", "discriminant") else layout
        stem = build_figure_filename(
            figure_tag=f"transition_{layout_tag}_{'+'.join(observables)}_order{order}",
            data_mode=data_mode, scan_number=scan_number, extension="",
        ).rstrip(".")
        save_figure_pdf_svg(fig, stem, PAPER_FIGURES_DIR, dpi=PAPER_FIGURES_DPI, png=True)
    if SHOW_PLOTS:
        plt.show(block=False)
    return fig


def run_export_figure_data(
    records: list[ObservableRecord], result: ZenoAnalysisResult, *,
    json_path: Path | None,
    parent: str | Path,
    observable: str,
    n_smooth: int,
) -> Path:
    """Step 7 — write the files run_files/make_third_transition_figure.py reads.

    A fresh ``<parent>/fitting_results_<UTC stamp>/`` folder receives this
    run's result JSON (panel B; ``json_path`` copied as it stands, or a fresh
    save when there is none) and one ``<scan file>_mpm_fit.json`` per λ
    (panel A overlays): ``times_smooth`` / ``fit_curve`` of the modal model
    with the result's poles at that λ and amplitudes re-solved on the
    ``observable`` trace of ``records`` — the same construction as the slider
    overlay.  Returns the folder.
    """
    prov = _provenance(records)
    traces = [r for r in records if r.observable_key == observable]
    if not traces:
        raise ValueError(f"export observable '{observable}' is not among the loaded "
                         f"observables {prov['observables']}.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(parent) / f"fitting_results_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    print(f"\n=== Export for make_third_transition_figure.py ({prov['data_mode']}) ===")

    # Panel B: the result JSON (already augmented with the sqrt fit if the
    # critical point ran; DEBUG never saved one, so save a fresh copy here).
    if json_path is not None:
        result_name = Path(shutil.copy2(json_path, out / Path(json_path).name)).name
    else:
        result_name = save_result(result, out, full=False).name
    print(f"  {result_name}")

    # Panel A: per-λ fit curves of `observable` with the result's poles.
    lam_res = np.asarray(result.lambda_values, dtype=float)
    poles_res = _central_poles(result)
    source_folder = Path(prov["source_folder"])
    n_written = n_skipped = 0
    for rec in traces:
        i = int(np.argmin(np.abs(lam_res - rec.lambda_val)))
        poles = poles_res[i]
        if abs(lam_res[i] - rec.lambda_val) > 1e-6 or not np.all(np.isfinite(poles)):
            n_skipped += 1
            continue
        amps = solve_linear_amplitudes(rec.t, rec.signal, poles)
        t_smooth = np.linspace(float(rec.t[0]), float(rec.t[-1]), int(n_smooth))
        payload = {
            "times_smooth": t_smooth.tolist(),
            "fit_curve": reconstruct_signal(t_smooth, poles, amps).tolist(),
            "lambda": float(rec.lambda_val),
            "observable": observable,
            "poles_re_1_per_us": np.real(poles).tolist(),
            "poles_im_rad_per_us": np.imag(poles).tolist(),
            "amplitudes_re": np.real(amps).tolist(),
            "amplitudes_im": np.imag(amps).tolist(),
            "mpm_order": _order(result),
            "lambda_source": prov["lambda_source"],
            "analysis_json": result_name,
            "original_file": str(source_folder / rec.source_file),
        }
        with open(out / f"{Path(rec.source_file).stem}_mpm_fit.json", "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        n_written += 1
    print(f"  {n_written} *_mpm_fit.json files ({observable}, {n_smooth} points)"
          + (f", {n_skipped} records without a matching λ / finite poles" if n_skipped else ""))
    print(f"  Folder: {out}")
    return out


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
