"""
ex_benchmark_noise.py
---------------------
Example: test analysis robustness by injecting Shot noise on
analytical simulation data.

Runs BenchmarkPipeline with:
  - PostSelectedDynamicsSimulator (small: 20 lambdas, short duration)
  - ShotNoise
  - n_trials=5 (increase for real use)

Expected output:
  λ_c bias and std across trials.
"""
import numpy as np
import matplotlib.pyplot as plt

from mpm_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator
from mpm_analysis.simulation.noise import ShotNoise,CompositeNoise
from mpm_analysis.analysis.steps import MPMStep, BootstrapStep
from mpm_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline

# ---- Small simulation ----
sim = PostSelectedDynamicsSimulator(
    num_lambdas=20,
    simulation_duration=25.0,
    min_alpha=1,
    max_alpha=2.0,
)
print("Generating clean simulation data...")
clean_records = sim.to_observable_records()
print(f"  {len(clean_records)} records")

# ---- Benchmark with Shot noise ----
bench = BenchmarkPipeline(
    clean_records=clean_records,
    noise_models=[ShotNoise(n_shots=6e6, seed=0)],
    analysis_steps=[MPMStep(order=3), BootstrapStep(n_boot=150, seed=42)],
    n_trials=20,  # increase for real use
)

print("\nRunning benchmark (5 trials)...")
result = bench.run(lambda_c_guess=1.25, plot=True)

print(f"\nResults:")
print(f"  Ground-truth lam_c: {result.ground_truth_lambda_c:.4f}")
print(f"  Extracted samples: {result.extracted_lambda_c_samples}")
print(f"  Bias             : {result.lambda_c_bias:.4f}")
print(f"  Std              : {result.lambda_c_std:.4f}")

# ---- Noise sweep example ----
print("\nRunning noise sweep (n_shots = 10000, 30000, 50000)...")
sweep_results = bench.sweep_noise(
    noise_factory=lambda n_shots: [ShotNoise(n_shots=n_shots, seed=0)],
    param_values=[1.8e5],
    lambda_c_guess=1.25, plot=True)

n_shots = [1.8e5]
biases = [r.lambda_c_bias for r in sweep_results]
stds   = [r.lambda_c_std  for r in sweep_results]

print("\nNoise sweep summary:")
for s, b, v in zip(n_shots, biases, stds):
    print(f"  n_shots={s}  bias={b:+.4f}  std={v:.4f}")

