"""
ex_load_experimental.py
-----------------------
Example: load 3 lambda points from an experimental folder and inspect shapes.

Run this script to verify the loader works on your data.
"""
from pathlib import Path

from zeno_analysis.io.experimental import EnsembleAverageLoader, PostSelectedLoader

# ---- CONFIGURE PATHS ----
# Change these to point to your actual data folder
ENS_AVG_FOLDER = Path(
    r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab"
    r"\Experiments\Zeno\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons"
    r"\survival_probability\Omega_BG0_scan\scan_experiment_results_50"
)

loader = EnsembleAverageLoader()
records = loader.load(ENS_AVG_FOLDER, observables=["survival"], max_files=5)

print(f"Loaded {len(records)} records from {ENS_AVG_FOLDER.name}")
for r in records[:3]:
    print(f"  lam={r.lambda_val:.3f}  obs='{r.observable_key}'  "
            f"t.shape={r.t.shape}  signal range=[{r.signal.min():.3f}, {r.signal.max():.3f}]")
