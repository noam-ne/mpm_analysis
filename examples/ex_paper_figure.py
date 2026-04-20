"""
ex_paper_figure.py
------------------
Example: reproduce the third-transition main-text figure using the new package.

This is the thin-caller replacement for
``third_transition_main_text_figure_plotting.py``.
"""
from pathlib import Path

from zeno_analysis.io.analysis_json import load_analysis_result
from zeno_analysis.io.experimental import EnsembleAverageLoader
from zeno_analysis.plotting.paper_figures import make_third_transition_figure

# ---- PATHS — change to your actual locations ----
PANEL_B_JSON = Path(
r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code\mpm_analysis_results_ens_avg_scan_experiment_results_50_20260129_204832.json"
)

PANEL_A_DATA_FOLDER = Path(
    r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab"
    r"\Experiments\Zeno\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons"
    r"\survival_probability\Omega_BG0_scan\scan_experiment_results_50"
)

PANEL_A_FIT_DIR = Path(
    r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab\Experiments\Zeno\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons\survival_probability\Omega_BG0_scan\scan_experiment_results_50\fitting_results_20260102_155334"
)

SAVE_DIR = Path(
    r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab"
    r"\Common\QuantumZeno\Paper\Plots\transitions"
)

# ---- Load data ----
if not PANEL_B_JSON.exists():
    print(f"Panel B JSON not found: {PANEL_B_JSON}")
    raise SystemExit(1)

result = load_analysis_result(PANEL_B_JSON)
print(f"Loaded panel B result: {result.n_lambda} lambda points")

panel_a_records = []
if PANEL_A_DATA_FOLDER.exists():
    loader = EnsembleAverageLoader()
    panel_a_records = loader.load(PANEL_A_DATA_FOLDER)
    print(f"Loaded {len(panel_a_records)} panel A records")
else:
    print(f"Panel A folder not found — panel A will be empty")

# ---- Make figure ----
import numpy as np

fig = make_third_transition_figure(
    panel_b_result=result,
    panel_a_records=panel_a_records,
    fit_dir=PANEL_A_FIT_DIR if PANEL_A_FIT_DIR.exists() else None,
    target_lambdas=np.array([0.4, 1.1, 1.5]),
    lambda_max_b=1.55,
    save_dir=SAVE_DIR if SAVE_DIR.exists() else None,
    yticks_freq=[0.0, 0.4, 0.8],
)

import matplotlib.pyplot as plt
plt.show()
