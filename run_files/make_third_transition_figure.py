"""
make_third_transition_figure.py
--------------------------------
Entry point: the third-transition main-text figure, drawn from a result that
``run_files/run.py`` saved for the third transition (step 2, ``THIRD_MPM``;
steps 3/4 append to the same JSON).

Everything comes from that one result JSON:

    panel B   its poles, and the step-4 sqrt fit (λ_c line + frequency overlay)
    panel A   the traces of the SAME folder, observable and λ axis the analysis
              ran on (read from the JSON's provenance; every analysed λ must
              be found again in the data, or the script stops), with the modal
              fit of the result's poles at each λ — amplitudes re-solved on the
              trace, the construction of run.py's slider and THIRD_EXPORT

By default the newest ens_avg result in run.py's OUTPUT_DIR is used, i.e. the
run you just made; pin ``RESULT_JSON`` to redraw an older one (the copy inside
an exported ``fitting_results_*`` folder works too).

Edit the configuration section, then run:
    python make_third_transition_figure.py
"""
from pathlib import Path
import numpy as np

from mpm_analysis.io.analysis_json import load_analysis_result
from mpm_analysis.io.experimental import EnsembleAverageLoader
from mpm_analysis.plotting.paper_figures import make_third_transition_figure
from mpm_analysis.utils.lab_paths import get_lab_path
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

# run.py's OUTPUT_DIR, where every run saves mpm_<data mode>_*.json
OUTPUT_DIR = Path(__file__).parent.parent / "mpm_analysis_output"

# None → the newest mpm_ens_avg_*.json in OUTPUT_DIR; or an explicit Path.
RESULT_JSON: Path | None = None

SAVE_DIR = Path(r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code\mpm_analysis\paper_figures")

TARGET_LAMBDAS = np.array([0.4, 1.1, 1.5])
LAMBDA_MAX_B = 1.6
TRANSITION_LABEL = "Quantum\nZeno"
DOWNSAMPLE_A = 3        # panel A: plot every n-th time point (1 → all; the paper figure used 3)
DOWNSAMPLE_B = 3        # panel B: plot every n-th λ point (1 → all; the paper figure used 3)
KEEP_ALL_NEAR_B = 0.15  # panel B: every λ point within ± this of the transition line (λ_c) is
                        #   plotted; DOWNSAMPLE_B thins only the points farther away (None → off)
KEEP_FIRST_B = 2        # panel B: the first n λ points are always plotted (the low-λ end is sparse)
YTICKS_FREQ = [0.0, 0.4, 0.8]
YLIM_DECAY_B = (0.25, 0.9)  # panel B decay-rate axis, the paper figure's range; error bars
                             #   beyond it are clipped at the frame (None → autoscale)

# ============================================================
# RUN
# ============================================================


def _lambda_table(folder: Path, lambda_source: str):
    """``EnsembleAverageLoader`` csv_path reproducing the result's λ axis.

    The result records only the table's name (``"hmm_csv:<name>"``); it is
    looked up where run.py looks (the folder's own table, then the newest
    ``alpha_measurement/hmm_results_*``), else left to the loader's discovery.
    The λ check below catches a wrong pick.
    """
    kind, _, name = lambda_source.partition(":")
    if kind == "json":
        return False
    candidates = [folder / name] + sorted(
        (folder / "alpha_measurement").glob(f"hmm_results_*/{name}"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((p for p in candidates if p.exists()), "auto")


if RESULT_JSON is None:
    jsons = sorted(OUTPUT_DIR.glob("mpm_ens_avg_*.json"), key=lambda p: p.stat().st_mtime)
    if not jsons:
        raise FileNotFoundError(f"No mpm_ens_avg_*.json result in {OUTPUT_DIR}")
    RESULT_JSON = jsons[-1]
result = load_analysis_result(RESULT_JSON)
md = result.metadata
folder = Path(md["source_folder"])
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
if not SAVE_DIR.exists():
    print(f"Not saved: {SAVE_DIR} does not exist (figure shown only)")

fig = make_third_transition_figure(
    panel_b_result=result,
    panel_a_records=records,
    panel_a_fit_result=result,
    panel_a_observable=observable,
    target_lambdas=TARGET_LAMBDAS,
    lambda_max_b=LAMBDA_MAX_B,
    downsample_a=DOWNSAMPLE_A,
    downsample_b=DOWNSAMPLE_B,
    keep_all_within_b=KEEP_ALL_NEAR_B,
    keep_first_b=KEEP_FIRST_B,
    save_dir=SAVE_DIR if SAVE_DIR.exists() else None,
    yticks_freq=YTICKS_FREQ,
    ylim_decay_b=YLIM_DECAY_B,
)
plt.show()
