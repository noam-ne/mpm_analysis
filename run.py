from __future__ import annotations

from pathlib import Path
from zeno_analysis.utils.lab_paths import get_lab_path

# ===========================================================================
# GLOBAL SETTINGS
# ===========================================================================

DEBUG = False          # True → small n_boot, few files, no saving
N_BOOT = 20         # bootstrap resamples (use 50 for quick tests)
ORDER = 3              # MPM model order
N_LAMBDA_TEST = None   # set to e.g. 10 to load only first 10 files (debug)
LAMBDA_RANGE = (0,1.5)    # e.g. (0.1, 1.6) — restrict lambda window
SEED = 15              # RNG seed for bootstrap reproducibility

SHOW_PLOTS  = True     # plt.show() after each task
SAVE_FIGURES = False   # save figures as PDF into OUTPUT_DIR

if DEBUG:
    N_BOOT = 50
    N_LAMBDA_TEST = 10

# ===========================================================================
# PATHS
# ===========================================================================
# Dropbox root is auto-detected from the current hostname — see
# utils/lab_paths.py.  Add your machine there if you get a "hostname not
# recognised" ValueError.

_ZT = r"Experiments\Zeno\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons"

# ---- Ensemble-average scan ----
ENS_AVG_FOLDER = get_lab_path(
    _ZT,
    r"survival_probability\Omega_BG0_scan\scan_experiment_results_50",
)

# ---- Post-selected scan + CSV for HMM lambda scale ----
_PS = _ZT + r"\click_record_with_post_selected_tomography\Omega_BG0_scan\scan_experiment_results_10"

_PARITY = _ZT + r"\click_record_repeated\Omega_BG0_scan\scan_experiment_results_5"

POST_SELECTED_FOLDER = get_lab_path(_PS)
PARITY_FOLDER = get_lab_path(_PARITY)
ENSEMBLE_AVERAGE_CSV = get_lab_path(_PS, "results.csv")
# HMM results CSV; set to None to use lambda_zeno_estimation from JSON

# ---- Output directory ----
OUTPUT_DIR = Path(__file__).parent / "results"

# ===========================================================================
# TASKS
# ===========================================================================

RUN_ENS_AVG_MPM     = False   # MPM + bootstrap on ensemble-average data
RUN_POST_SEL_MPM    = False   # MPM + bootstrap on post-selected data
RUN_PARITY_MPM      = True   # MPM + bootstrap on parity data
RUN_CRITICAL_POINT  = False # Fit sqrt model from an existing analysis JSON
RUN_SIMULATION_TEST = False   # Small synthetic simulation (no real data needed)
RUN_BENCHMARK       = False  # Robustness benchmark with noise injection

# For RUN_CRITICAL_POINT: path to a previously saved analysis JSON
# CRITICAL_POINT_JSON = Path(
#     r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code"
#     r"\mpm_analysis_results_ens_avg_scan_experiment_results_50_20260129_225403.json"
# )

CRITICAL_POINT_JSON = None  # set to None to require a fresh MPM result for critical point fit
LAMBDA_C_GUESS = 1     # initial guess for critical point
CP_LAMBDA_MAX  = 1.55    # exclude lambda above this for the fit

if RUN_ENS_AVG_MPM:
    OBSERVABLE_KEY = "survival" 
elif RUN_POST_SEL_MPM:
    OBSERVABLE_KEY = "survival"
elif RUN_PARITY_MPM:
    OBSERVABLE_KEY = "parity_real"


# ===========================================================================
# IMPORTS
# ===========================================================================

global MPMPipeline, CriticalPointPipeline, BenchmarkPipeline
global MPMStep, BootstrapStep
global PostSelectedDynamicsSimulator, EnsembleAverageLoader, PostSelectedLoader
global plot_raw_survival, plot_eigenvalue_spectrum
import matplotlib.pyplot as plt
from zeno_analysis.pipeline.mpm_pipeline import MPMPipeline
from zeno_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline
from zeno_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline
from zeno_analysis.analysis.steps import MPMStep, BootstrapStep, MPMRRHAStep
from zeno_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator
from zeno_analysis.simulation.ensemble_average import EnsembleAverageDynamics
from zeno_analysis.io.experimental import EnsembleAverageLoader, PostSelectedLoader, ParityLoader
from zeno_analysis.simulation.noise import ShotNoise
from zeno_analysis.plotting.exploratory import (
    plot_raw_survival, plot_eigenvalue_spectrum, plot_time_traces_slider
)
from zeno_analysis.plotting.paper_figures.appendix_bootstrap import plot_appendix_bootstrap
from zeno_analysis.analysis.critical_point_guesses import get_guess, list_presets
from zeno_analysis.io.file_naming import build_figure_filename


# ===========================================================================
# PLOT HELPER
# ===========================================================================

def _show_and_save(
    fig,
    figure_tag: str,
    *,
    source_tag: str = "unknown",
    scan_number: int | None = None,
) -> None:
    """Optionally save figure to OUTPUT_DIR and/or show it."""
    import matplotlib.pyplot as plt
    if SAVE_FIGURES and not DEBUG:
        filename = build_figure_filename(
            figure_tag=figure_tag,
            source_tag=source_tag,
            scan_number=scan_number,
            extension="pdf",
        )
        path = OUTPUT_DIR / filename
        fig.savefig(path, bbox_inches="tight")
        print(f"  Figure saved: {path.name}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _plot_mpm_result(records, result, tag: str) -> None:
    """Standard 3-panel diagnostic: raw data, decay rates, frequencies."""
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(tag, fontsize=14)

    plot_raw_survival(records, observable_key=OBSERVABLE_KEY, ax=axes[0])
    raw_sigs = np.concatenate([r.signal for r in records if r.observable_key == OBSERVABLE_KEY])
    finite_pos = raw_sigs[np.isfinite(raw_sigs) & (raw_sigs > 0)]
    if finite_pos.size == len(raw_sigs[np.isfinite(raw_sigs)]):
        axes[0].set_yscale("log")
    axes[0].set_title("Raw curves", fontsize=12)

    plot_eigenvalue_spectrum(result, ax=axes[1], mode="decay")
    axes[1].set_title("Decay rates", fontsize=12)
    axes[1].tick_params(labelsize=10)

    plot_eigenvalue_spectrum(result, ax=axes[2], mode="frequency")
    axes[2].set_title("Frequencies", fontsize=12)
    axes[2].tick_params(labelsize=10)

    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    source_tag = str(result.metadata.get("source_tag", tag))
    scan_number = result.metadata.get("scan_number")
    if scan_number is not None:
        scan_number = int(scan_number)
    _show_and_save(
        fig,
        "spectrum",
        source_tag=source_tag,
        scan_number=scan_number,
    )


# ===========================================================================
# TASKS
# ===========================================================================

def run_ens_avg_mpm():
    """Run MPM + bootstrap on ensemble-average experimental data."""
    if not ENS_AVG_FOLDER.exists():
        print(f"[ens_avg] Folder not found: {ENS_AVG_FOLDER}")
        return None

    print(f"\n=== Ensemble-average MPM ===")
    steps = [MPMStep(order=ORDER, plot_svd=True, n_sv=10), BootstrapStep(n_boot=N_BOOT, seed=SEED)]
    csv = ENSEMBLE_AVERAGE_CSV if (ENSEMBLE_AVERAGE_CSV and ENSEMBLE_AVERAGE_CSV.exists()) else None
    if csv is None:
        print("[post_sel] CSV not found, using lambda_zeno_estimation from JSON")
    pipeline = MPMPipeline.from_ensemble_average(
        ENS_AVG_FOLDER,
        observables=OBSERVABLE_KEY,
        steps=steps,
        lambda_range=LAMBDA_RANGE,
        max_files=N_LAMBDA_TEST,
        csv_path=csv,
    )
    result = pipeline.run()
    _plot_mpm_result(pipeline.records, result, "ens_avg")
    saved_path = None
    if not DEBUG:
        saved_path = pipeline.save(OUTPUT_DIR)
        print(f"Saved: {saved_path.name}")
    return result, saved_path


def run_post_sel_mpm():
    """Run MPM + bootstrap on post-selected experimental data."""
    if not POST_SELECTED_FOLDER.exists():
        print(f"[post_sel] Folder not found: {POST_SELECTED_FOLDER}")
        return None

    

    print(f"\n=== Post-selected MPM ===")
    steps = [MPMStep(order=ORDER), BootstrapStep(n_boot=N_BOOT, seed=SEED)]
    pipeline = MPMPipeline.from_post_selected(
        POST_SELECTED_FOLDER,
        observables=OBSERVABLE_KEY,
        steps=steps,
        lambda_range=LAMBDA_RANGE,
        max_files=N_LAMBDA_TEST,
    )
    result = pipeline.run()
    _plot_mpm_result(pipeline.records, result, "post_sel")
    saved_path = None
    if not DEBUG:
        saved_path = pipeline.save(OUTPUT_DIR)
        print(f"Saved: {saved_path.name}")
    return result, saved_path

def run_parity_mpm():
    folder = Path(PARITY_FOLDER)
    records = ParityLoader().load(folder, observables=OBSERVABLE_KEY)
    print(f"\n===plotting raw curves===")
    fig, ax = plt.subplots()
    plot_raw_survival(records, observable_key=OBSERVABLE_KEY, ax=ax)
    ax.set_title("Parity curves")
    print(f"\n=== Parity MPM ===")
    steps = [MPMStep(order=ORDER + 1, plot_svd=True), BootstrapStep(n_boot=N_BOOT, seed=SEED)]
    pipeline = MPMPipeline.from_parity(
        folder,
        observables=OBSERVABLE_KEY,
        steps=steps,
        lambda_range=LAMBDA_RANGE,
        max_files=N_LAMBDA_TEST,
    )
    result = pipeline.run()
    _plot_mpm_result(pipeline.records, result, "parity")
    saved_path = None
    if not DEBUG:
        saved_path = pipeline.save(OUTPUT_DIR)
        print(f"Saved: {saved_path.name}")
    return result, saved_path

def run_parity_curves_mpm_fit():
    folder = Path(PARITY_FOLDER)
    records = ParityLoader().load(folder, observables=OBSERVABLE_KEY,
                                  lambda_range=LAMBDA_RANGE, max_files=N_LAMBDA_TEST)
    plot_time_traces_slider(records, observable_key=OBSERVABLE_KEY, log_scale=False, mpm_order=ORDER + 1)


def run_critical_point(result=None, guess=None, json_path=None):
    """Fit critical-point sqrt model."""
    if result is not None:
        cp = CriticalPointPipeline(result, lambda_max=CP_LAMBDA_MAX)
    elif CRITICAL_POINT_JSON is not None and CRITICAL_POINT_JSON.exists():
        cp = CriticalPointPipeline.from_json(CRITICAL_POINT_JSON, lambda_max=CP_LAMBDA_MAX)
        json_path = CRITICAL_POINT_JSON
    else:
        print(f"[critical_point] No result and no valid JSON — skipping.")
        return None

    print(f"\n=== Critical-point fit ===")
    lambda_c_guess, a_guess, b_guess = guess["lambda_c"], guess["a"], guess["b"]
    fit = cp.fit(lambda_c_guess=lambda_c_guess, a_guess=a_guess, b_guess=b_guess, window_left=50, window_right=15)
    print(f"  lam_c = {fit.lambda_c:.4f} +/- {fit.lambda_c_err:.4f}")
    print(f"  chi2_red = {fit.chi2_reduced:.3f}  R2 = {fit.r_squared:.4f}")

    # Plot: 3-panel appendix style
    cp.result.critical_point_sqrt_fit = fit
    fig = plot_appendix_bootstrap(
        cp.result,
        lambda_max=CP_LAMBDA_MAX,
        fit_validity_limit=CP_LAMBDA_MAX,
        real_axis_ticks=[0.0, 1, 2],
        real_split_axis_ticks=[0.0, 0.6, 1.2],
    )
    cp_source_tag = str(cp.result.metadata.get("source_tag", "unknown"))
    cp_scan_number = cp.result.metadata.get("scan_number")
    if cp_scan_number is not None:
        cp_scan_number = int(cp_scan_number)
    _show_and_save(
        fig,
        "critical_point_appendix",
        source_tag=cp_source_tag,
        scan_number=cp_scan_number,
    )

    if not DEBUG:
        path = cp.save(json_path)
        print(f"Saved: {path.name}")
    return fit


def run_simulation_test():
    """Quick end-to-end test on synthetic data (no real data needed)."""
    print(f"\n=== Simulation test ===")
    sim = PostSelectedDynamicsSimulator(
        num_lambdas=30,
        simulation_duration=25.0,
        min_alpha=0.1,
        max_alpha=2.0,
    )
    records = sim.to_observable_records()
    print(f"  Simulated {len(records)} records")

    steps = [MPMStep(order=ORDER), BootstrapStep(n_boot=N_BOOT, seed=SEED)]
    result = MPMPipeline.from_records(records, steps=steps).run()
    print(f"  Poles shape: {result.poles.decay_rates.shape}")
    _plot_mpm_result(records, result, "simulation")
    return result


def run_benchmark():
    """Test analysis robustness with injected Gaussian noise."""
    print(f"\n=== Benchmark ===")
    sim = PostSelectedDynamicsSimulator(
        num_lambdas=20, simulation_duration=40.0,
        min_alpha=0.1, max_alpha=2.0,
    )
    clean_records = sim.to_observable_records()

    bench = BenchmarkPipeline(
        clean_records=clean_records,
        noise_models=[ShotNoise(n_shots=6*3e4, seed=0)],
        analysis_steps=[MPMStep(order=ORDER), BootstrapStep(n_boot=N_BOOT, seed=SEED)],
        n_trials=5 if DEBUG else 20,
    )
    result = bench.run(lambda_c_guess=1.25)
    print(f"  lam_c bias = {result.lambda_c_bias:.4f}  std = {result.lambda_c_std:.4f}")
    return result


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
  
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mpm_result = None
    mpm_json_path = None
    guess = None

    if RUN_ENS_AVG_MPM:
        mpm_result, mpm_json_path = run_ens_avg_mpm()
        guess = get_guess("ens_avg_experimental")

    if RUN_POST_SEL_MPM:
        mpm_result, mpm_json_path = run_post_sel_mpm()
        guess = get_guess("post_selected_experimental")
    
    if RUN_PARITY_MPM:
        mpm_result, mpm_json_path = run_parity_mpm()
        run_parity_curves_mpm_fit()
        

    if RUN_CRITICAL_POINT:
        run_critical_point(result=mpm_result, guess=guess, json_path=mpm_json_path)

    if RUN_SIMULATION_TEST:
        run_simulation_test()

    if RUN_BENCHMARK:
        run_benchmark()

    print("\nDone.")
