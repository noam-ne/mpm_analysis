"""
ex_run_bootstrap_small.py
-------------------------
Example: run the MPM pipeline with a small n_boot on synthetic data for
fast local testing (takes ~10-30 s instead of minutes).

What you see:
  Figure 1 — Raw survival curves (all lambda overlaid, colour = lambda)
  Figure 2 — Grid: one panel per lambda slice
  Figure 3 — Decay rates + frequencies vs lambda (with error bars)
"""
import matplotlib.pyplot as plt

from zeno_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator
from zeno_analysis.simulation.ensemble_average import EnsembleAverageDynamics
from zeno_analysis.pipeline.mpm_pipeline import MPMPipeline
from zeno_analysis.analysis.steps import MPMStep, BootstrapStep
from zeno_analysis.plotting.exploratory import (
    plot_raw_survival,

    plot_eigenvalue_spectrum,
)

# ---- Small simulation (50 lambda points, short duration) ----
sim = PostSelectedDynamicsSimulator(
    num_lambdas=80,
    simulation_duration=25.0,
    min_alpha=0.4,
    max_alpha=2.0,
)
# sim = EnsembleAverageDynamics(
#     num_lambdas=100,
#     simulation_duration=25.0,
#     min_alpha=0.4,
#     max_alpha=2.0,
# )
print("Running simulation...")
observable_key = "Z"
records = sim.to_observable_records(observable_key=observable_key)
print(f"Generated {len(records)} records, lam range: [{records[0].lambda_val:.2f}, {records[-1].lambda_val:.2f}]")

# ---- Figure 1: raw survival curves ----
fig1, ax = plt.subplots(figsize=(6, 4))
plot_raw_survival(records, observable_key=observable_key, ax=ax, title="Raw survival curves (colour = lambda)")
fig1.tight_layout()

# ---- Run analysis ----
#order for ensemble average: 4 
steps = [MPMStep(order=3, observable_key=observable_key), BootstrapStep(n_boot=2, seed=42)]
pipeline = MPMPipeline.from_records(records, steps=steps)

print("\nRunning MPM + Bootstrap (n_boot=50)...")
result = pipeline.run()

print(f"\nResult: {result.n_lambda} lambda points, poles shape {result.poles.decay_rates.shape}")
print(f"Lambda range: [{result.lambda_values.min():.2f}, {result.lambda_values.max():.2f}]")
print(f"Decay rate (median) at first lambda: {result.poles.decay_rates.median[0]}")

# ---- Figure 3: decay rates + frequencies ----
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_eigenvalue_spectrum(result, ax=ax1, mode="decay",     title="Decay rates / omega_S vs lambda")
plot_eigenvalue_spectrum(result, ax=ax2, mode="frequency", title="Frequencies / omega_S vs lambda")
fig3.tight_layout()


plt.show()
print("\nDone.")
