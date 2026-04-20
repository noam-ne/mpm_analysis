"""
ex_critical_point_fit.py
------------------------
Load an existing analysis JSON, fit the sqrt model, and produce the
3-panel appendix figure (decay rates + imaginary splitting + real splitting)
with the fit overlaid.

The fit is recomputed from the raw splitting data in the JSON — it does NOT
use any previously stored ``critical_point_sqrt_fit`` block.

Usage:
    python ex_critical_point_fit.py                  # third transition (default)
    python ex_critical_point_fit.py first            # first transition (postselected)
    python ex_critical_point_fit.py path/to/file.json
"""
import sys
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Per-transition configurations  (matching critical_points_analysis.py)
# ---------------------------------------------------------------------------

_BASE = Path(r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code")

CONFIGS = {
    "third": dict(
        json_path=_BASE / "mpm_analysis_results_ens_avg_scan_experiment_results_50_20260129_225403.json",
        lambda_c_guess=1.1,
        a_guess=1.0,
        b_guess=0.1,
        lambda_max=1.55,
        window_left=50,
        window_right=15,
        fit_validity_limit=1.3,
        real_axis_scale="linear",
        real_axis_ticks=[0.0, 1, 2],
        real_split_axis_ticks=[0.0, 0.6, 1.2],
        label="Ens. avg. (third transition)",
    ),
    "first": dict(
        json_path=_BASE / "mpm_analysis_results_post_selected_scan_experiment_results_10_20260130_141124.json",
        lambda_c_guess=1.023,   # from JSON critical_point.lambda_c_median
        a_guess=1.12,
        b_guess=0.5,
        lambda_max=1.6,         # covers full lambda range of this dataset
        window_left=50,
        window_right=15,
        fit_validity_limit=1.2,
        real_axis_scale="log",
        real_axis_ticks=None,   # handled by log formatter
        real_split_axis_ticks=[0.0, 1.0, 2.0, 3.0],
        label="Post-selected (first transition)",
    ),
}

# ---------------------------------------------------------------------------
# Resolve which config to use
# ---------------------------------------------------------------------------

arg = sys.argv[1] if len(sys.argv) > 1 else "third"

if arg in CONFIGS:
    cfg = CONFIGS[arg]
    json_path = cfg["json_path"]
else:
    # Treat as an explicit file path; default to third-transition plot params
    json_path = Path(arg)
    cfg = CONFIGS["third"]
    cfg["json_path"] = json_path

if not json_path.exists():
    print(f"JSON not found: {json_path}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

from zeno_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline
from zeno_analysis.plotting.paper_figures.appendix_bootstrap import plot_appendix_bootstrap
import matplotlib.pyplot as plt

cp  = CriticalPointPipeline.from_json(json_path, lambda_max=cfg["lambda_max"])
fit = cp.fit(
    lambda_c_guess=cfg["lambda_c_guess"],
    window_left=cfg["window_left"],
    window_right=cfg["window_right"],
    a_guess=cfg["a_guess"],
    b_guess=cfg["b_guess"],
)

print(f"\nFit result ({cfg['label']}):")
print(f"  lam_c    = {fit.lambda_c:.4f} +/- {fit.lambda_c_err:.4f}")
print(f"  a        = {fit.a:.4f} +/- {fit.a_err:.4f}")
print(f"  b        = {fit.b:.4f} +/- {fit.b_err:.4f}")
print(f"  chi2_red = {fit.chi2_reduced:.3f}")

# ---------------------------------------------------------------------------
# Window calibration
# ---------------------------------------------------------------------------

print("\nRunning window calibration ...")
cal = cp.calibrate(
    lambda_c_guess=cfg["lambda_c_guess"],
    a_guess=cfg["a_guess"],
    b_guess=cfg["b_guess"],
    left_range=(20, 60),
    right_range=(5, 20),
)
best_l, best_r = cal.best_window
finite_chi2 = cal.chi2_values[np.isfinite(cal.chi2_values)]
print(f"Best window: left={best_l}, right={best_r}  (lowest chi2_red = {finite_chi2.min():.3f})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

result = cp.result
result.critical_point_sqrt_fit = fit   # attach so the plotter picks up the curves

fig = plot_appendix_bootstrap(
    result,
    lambda_max=cfg["lambda_max"],
    fit_validity_limit=cfg["fit_validity_limit"],
    real_axis_scale=cfg["real_axis_scale"],
    real_axis_ticks=cfg["real_axis_ticks"],
    real_split_axis_ticks=cfg["real_split_axis_ticks"],
)
plt.show()
