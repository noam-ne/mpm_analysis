"""
make_third_transition_figure.py
--------------------------------
Entry point: the third-transition main-text figure, drawn from a result that
``run_files/run.py`` saved for the third transition (step 2, ``THIRD_MPM``;
steps 3/4 append to the same JSON).

Everything comes from that one result JSON:

    panel B   its poles; the dashed frequency curve and the transition line
              from the step-3 exceptional-point line (TRANSITION_FIT =
              "discriminant": √(−Δ_line)/2 and λ_EP, the appendix figure's
              fit — the line is recomputed from the JSON's poles and draws
              with run.py's own step-3 function and checked against the λ_EP
              the JSON stores) or from the step-4 sqrt fit ("sqrt": λ_c)
    panel A   the traces of run.py's third-transition data folder
              (ENS_AVG_FOLDER), on the observable and λ axis the analysis ran
              on (read from the JSON's provenance; every analysed λ must be
              found again in the data, or the script stops), with the modal
              fit of the result's poles at each λ — amplitudes re-solved on the
              trace, the construction of run.py's slider and THIRD_EXPORT

The result JSON is read from the paper data folder, where run.py's step 7
(``THIRD_EXPORT``) writes one ``fitting_results_<stamp>`` folder per run:
``combined_results_0_and_1_merged/pole_analysis_data_for_main_text_figure``.
``RESULT_JSON = None`` draws the newest of them; pin it to redraw a given run.

The figure is saved into the paper's plots folder, ``Plots/transitions``, as
``FIGURE_NAME`` .pdf / .svg / .png — a rerun overwrites the paper's files.

Edit the configuration section, then run:
    python make_third_transition_figure.py
"""
from pathlib import Path
import numpy as np

import run                                # run.py: the paper folders + step 3's fit
from mpm_analysis.io.analysis_json import load_analysis_result
from mpm_analysis.io.experimental import EnsembleAverageLoader
from mpm_analysis.plotting.paper_figures import make_third_transition_figure
from mpm_analysis.utils.windows_paths import win_long_path
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

# Where run.py's step 7 (THIRD_EXPORT) writes its fitting_results_<stamp> folders:
# zeno_paper_data/combined_results_0_and_1_merged/pole_analysis_data_for_main_text_figure
POLE_DATA_DIR = run.THIRD_POLE_DATA_FOLDER

# None → the mpm_ens_avg_*.json of the newest fitting_results_* folder; or an explicit Path.
# Pinned to the 2 Nov run the paper figure is drawn from (λ_EP 1.0845 ± 0.0075).
RESULT_JSON: Path | None = (POLE_DATA_DIR / "fitting_results_20261118_140622"
                            / "mpm_ens_avg_N70_lam0.18-1.66_20261102_172702.json")

# The paper's plots folder; the figure is saved there under FIGURE_NAME (overwriting it).
SAVE_DIR = run.PAPER_PLOTS_DROPBOX / "transitions"
FIGURE_NAME = "third_transition_main_text_figure_with_QZ"   # with the "Quantum Zeno" label

# "discriminant" → panel B's dashed curve and transition line from run.py step 3's
#   exceptional-point line, refitted here over EP_LAMBDA_WINDOW exactly as step 3
#   does (λ_EP checked against the value stored in the JSON); "sqrt" → the step-4
#   sqrt fit stored in the JSON (λ_c)
TRANSITION_FIT = "discriminant"
EP_LAMBDA_WINDOW = (0.9, 1.3)   # run.py: lambda_window of the third transition's step 3
TRANSITION_LINE_X = 1.09        # draw the transition line + label at this λ (the value the
                                #   text quotes); None → the fit's own λ_EP / λ_c

TARGET_LAMBDAS = np.array([0.4, 1.08, 1.5])   # nearest analysed λ: 0.4032, 1.0825 (at λ_EP), 1.4999
LEGEND_FMT = r"$\lambda={:.2f}$"               # panel A legend, filled with the plotted λ
                                               #   (all math: digits in the tick-label font)
LAMBDA_MAX_B = 1.6
TRANSITION_LABEL = "Quantum\nZeno"
DOWNSAMPLE_A = 3        # panel A: plot every n-th time point (1 → all; the paper figure used 3)
DOWNSAMPLE_B = 3        # panel B: plot every n-th λ point (1 → all; the paper figure used 3)
KEEP_ALL_NEAR_B = 0.15  # panel B: every λ point within ± this of the transition line (λ_c) is
                        #   plotted; DOWNSAMPLE_B thins only the points farther away (None → off)
KEEP_FIRST_B = 2        # panel B: the first n λ points are always plotted (the low-λ end is sparse)
YTICKS_A = [0.2, 0.5, 0.8]   # panel A y ticks (None → matplotlib's)
YTICKS_FREQ = [0.0, 0.4, 0.8]
YLIM_DECAY_B = (0.25, 0.9)  # panel B decay-rate axis, the paper figure's range; error bars
                             #   beyond it are clipped at the frame (None → autoscale)

# ============================================================
# RUN
# ============================================================


def _lambda_table(folder: Path, lambda_source: str):
    """``EnsembleAverageLoader`` csv_path reproducing the result's λ axis.

    The result records only the table's name (``"hmm_csv:<name>"``): the
    folder's own table of that name, else left to the loader's discovery.
    The λ check below catches a wrong pick.
    """
    kind, _, name = lambda_source.partition(":")
    if kind == "json":
        return False
    table = folder / name
    return table if Path(win_long_path(table)).exists() else "auto"


if RESULT_JSON is None:
    # newest export folder (the stamp in its name sorts by time)
    jsons = sorted(Path(win_long_path(POLE_DATA_DIR)).glob("fitting_results_*/mpm_ens_avg_*.json"),
                   key=lambda p: p.parent.name)
    if not jsons:
        raise FileNotFoundError(f"No fitting_results_*/mpm_ens_avg_*.json in {POLE_DATA_DIR} "
                                "- run run.py's third transition with THIRD_EXPORT.")
    RESULT_JSON = POLE_DATA_DIR / jsons[-1].parent.name / jsons[-1].name
result = load_analysis_result(RESULT_JSON)
md = result.metadata
folder = run.ENS_AVG_FOLDER
observable = md["observables"][0]
print(f"Result : {RESULT_JSON.name}  ({result.n_lambda} lambda points, "
      f"{'+'.join(md['observables'])}, lambda = {md['lambda_source']})")
print(f"Data   : {folder}")

# Panel A: the analysed traces, on the analysed λ axis
records = EnsembleAverageLoader(csv_path=_lambda_table(folder, md["lambda_source"])).load(
    folder,
    observables=[observable],
    drop_first_point=bool(md.get("drop_first_point", False)),
)
lam_data = np.array([r.lambda_val for r in records])
missing = [lam for lam in result.lambda_values
           if not np.any(np.isclose(lam_data, lam, rtol=0.0, atol=1e-9))]
if missing:
    raise ValueError(
        f"{len(missing)} analysed lambda values are not in {folder} on this "
        f"lambda axis (first: {missing[:3]}) — the data or the lambda table "
        f"differ from the analysis in {RESULT_JSON.name}."
    )
records = [r for r in records
           if np.any(np.isclose(result.lambda_values, r.lambda_val, rtol=0.0, atol=1e-9))]
print(f"Panel A: {len(records)} '{observable}' traces, lambda axis matches the result")
SAVE = Path(win_long_path(SAVE_DIR)).is_dir()
if not SAVE:
    print(f"Not saved: {SAVE_DIR} does not exist (figure shown only)")

# Panel B transition: the step-3 line (λ_EP) or the stored step-4 sqrt fit (λ_c)
discriminant_fit = None
if TRANSITION_FIT == "discriminant":
    c = run._ep_crossing(result, method="linfit", half_width=0.3, lambda_max=None,
                         lambda_window=EP_LAMBDA_WINDOW)     # run.py's step-3 fit, unchanged
    lam_ep, err, fit = c["lam_ep"], c["lam_ep_err"], c["fit"]
    if not np.isfinite(lam_ep) or fit["coef"] is None:
        raise ValueError("Discriminant line: no zero crossing in "
                         f"lambda {EP_LAMBDA_WINDOW[0]:g}-{EP_LAMBDA_WINDOW[1]:g}.")
    import json
    stored = (json.loads(Path(win_long_path(RESULT_JSON)).read_text()).get("transition_samples") or {}) \
        .get("delta_linfit_central")
    if stored is not None and not np.isclose(lam_ep, stored, atol=1e-6):
        raise ValueError(f"Refitted lambda_EP {lam_ep:.4f} != {stored:.4f} stored in "
                         f"{RESULT_JSON.name}: EP_LAMBDA_WINDOW differs from the run's step 3.")
    wl = c["lam_d"][fit["window"]]
    discriminant_fit = dict(lambda_ep=float(lam_ep), coef=fit["coef"],
                            lambda_window=(float(wl.min()), float(wl.max())))
    print(f"Panel B: discriminant line, lambda_EP = {lam_ep:.4f} +/- {err:.4f} "
          f"(window lambda {wl.min():.3f}-{wl.max():.3f}, {wl.size} points"
          + (", matches the JSON)" if stored is not None else ", JSON stores no value)"))
elif TRANSITION_FIT == "sqrt":
    sf = result.critical_point_sqrt_fit
    print("Panel B: sqrt fit, lambda_c = " + (f"{sf.lambda_c:.4f}" if sf is not None
          else "none stored in the JSON (no curve, no line)"))
else:
    raise ValueError(f"TRANSITION_FIT must be 'discriminant' or 'sqrt', got {TRANSITION_FIT!r}")

fig = make_third_transition_figure(
    panel_b_result=result,
    panel_a_records=records,
    panel_a_fit_result=result,
    panel_a_observable=observable,
    target_lambdas=TARGET_LAMBDAS,
    label_fmt=LEGEND_FMT,
    lambda_max_b=LAMBDA_MAX_B,
    downsample_a=DOWNSAMPLE_A,
    downsample_b=DOWNSAMPLE_B,
    keep_all_within_b=KEEP_ALL_NEAR_B,
    keep_first_b=KEEP_FIRST_B,
    save_dir=SAVE_DIR if SAVE else None,
    figure_stem=FIGURE_NAME,
    yticks_a=YTICKS_A,
    yticks_freq=YTICKS_FREQ,
    ylim_decay_b=YLIM_DECAY_B,
    discriminant_fit=discriminant_fit,
    crit_line_x=TRANSITION_LINE_X,
)
plt.show()
