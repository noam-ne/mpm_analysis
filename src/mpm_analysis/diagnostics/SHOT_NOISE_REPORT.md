# Shot-Noise Parity Report

**Date:** 2026-04-15  
**Script:** `mpm_analysis/diagnostics/shot_noise_parity.py`

---

## Summary

The new `PostSelectedDynamicsSimulator` pipeline shows worse pole recovery
under shot noise compared to the old `object_oriented_poles.py` results.
This report identifies the root causes.

---

## Setup

| Parameter | Value |
|-----------|-------|
| N_shots | 120 000 (= 4 × 30 000, old-code default) |
| N_lambda | 20 |
| N_realizations | 30 per lambda per method |
| Model order | 3 |
| Simulation duration | 25 µs |

---

## Findings

### Root cause 1 — The old code ran WITHOUT shot noise (primary cause)

The most important finding: in `object_oriented_poles.py`, the shot-noise
injection block at **lines 391–393** is **commented out**:

```python
# N_trials = 3e5
# shot_noise = np.random.normal(0, np.sqrt(sig*(1-sig)/N_trials), size=sig.shape)
# sig = sig + shot_noise
```

This means **all previously published results from `object_oriented_poles.py`
were computed on the clean analytical signal without any injected noise**.
The improved-looking poles were a consequence of running on noiseless data,
not superior algorithm design.

When the new benchmark pipeline injects shot noise with `N_shots=120 000`,
it is doing something the old code never did in practice.

### Root cause 2 — Old bootstrap used RRHA on every bootstrap draw (secondary cause)

In `object_oriented_poles.py`, the bootstrap loop (lines 443–445):

```python
rrha_matrix, singular_values = self.get_approximated_reduced_hankel(H_boot, order=order)
s_poles, singular_values = self.get_mpm_poles_from_hankel(rrha_matrix, dt, order=order)
```

Every bootstrap resample goes through a full RRHA iteration before the pencil
solve.  The new `BootstrapStep` uses plain `matrix_pencil_method` (no RRHA)
on each resample.

The diagnostic measured the effect on pole recovery accuracy on a single noisy
signal (no bootstrap, just single-shot extraction):

| Method | Mean |Re(pole) − Re(truth)| (poles 0,1, all λ) |
|--------|----------------------------------------------|
| Standard MPM | **0.174** |
| RRHA MPM | **0.112** |
| Ratio | 1.56× (RRHA is 56% more accurate) |

The improvement concentrates near the exceptional point (λ ≈ 1.1–1.6) where
poles nearly coalesce and small noise causes large real-part jumps.  RRHA's
rank reduction suppresses the noise-floor singular values before the pencil
step, giving more stable eigenvalues.

### Root cause 3 — Different signal length in simulation vs real data

The `PostSelectedDynamicsSimulator` uses `T_int = 0.32 µs` and
`simulation_duration = 25 µs`, giving N ≈ 78 time points.  The experimental
data typically has longer traces.  A shorter signal means a smaller Hankel
matrix and weaker denoising power in RRHA.

---

## Recommendations

1. **For benchmark results to be comparable with old-code "clean" results:**
   Run the benchmark with `fit_critical_point=False` on the clean analytical
   data (`ShotNoise(n_shots=0)` or no noise model) to reproduce the old
   noiseless baseline.

2. **For noisy benchmarks:** Use `MPMRRHAStep` instead of `MPMStep` in
   `analysis_steps`.  The diagnostic shows a 1.56× reduction in pole-recovery
   error near the EP:

   ```python
   from mpm_analysis.analysis.steps import MPMRRHAStep, BootstrapStep
   bench = BenchmarkPipeline(
       clean_records=records,
       noise_models=[ShotNoise(n_shots=120_000)],
       analysis_steps=[MPMRRHAStep(order=3), BootstrapStep(n_boot=500)],
   )
   ```

3. **For bootstrap inner loop (RRHA per resample):** The most accurate
   approach — matching the old code exactly — would also apply RRHA inside
   `run_bootstrap()` on each bootstrap Hankel resample.  This is not yet
   implemented in `BootstrapStep` but can be added as a flag if needed.

4. **Old results were produced without noise injection.**  If you want to
   reproduce old figures exactly, use the clean analytical signal directly
   (no `ShotNoise`).

---

## Quantitative results (from diagnostic run, 2026-04-15)

```
lam     MPM err (p0)   RRHA err (p0)    MPM err (p1)   RRHA err (p1)
0.796        0.01158        0.01117         0.01158        0.01117
0.838        0.01552        0.01558         0.01552        0.01558
0.880        0.02296        0.02096         0.02296        0.02096
0.922        0.01915        0.02609         0.01915        0.02609
0.964        0.02326        0.02720         0.02326        0.02720
1.006        0.03403        0.02692         0.03403        0.02692
1.048        0.04702        0.02991         0.03994        0.02832
1.090        0.04111        0.06206         0.04111        0.04216
1.131        0.16380        0.09152         0.05608        0.04351
1.173        0.14990        0.08551         0.05783        0.05293
1.215        0.21907        0.11294         0.08712        0.06307
1.257        0.30437        0.13849         0.13488        0.08036
1.299        0.34086        0.16921         0.16424        0.10179
1.341        0.28758        0.20286         0.10713        0.06641
1.383        0.31644        0.28362         0.08689        0.08478
1.425        0.49934        0.29214         0.10460        0.09019
1.467        0.39860        0.32019         0.09883        0.09224
1.509        0.85779        0.33693         0.12663        0.07661
1.550        0.64588        0.48927         0.14048        0.09355
1.592        1.07305        0.54049         0.11994        0.13596

Aggregate (poles 0,1):
  MPM  : 0.174
  RRHA : 0.112
  Ratio MPM/RRHA: 1.56x
```

Figure saved as `shot_noise_parity.pdf`.
