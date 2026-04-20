# zeno_analysis

Analysis package: loading experimental data, extracting poles via the Matrix
Pencil Method (MPM), bootstrap uncertainty quantification, critical-point
fitting, and simulation benchmarks.

---

## Quick start

```bash
# Install
pip install -e .

# Run everything (toggle tasks in run.py first)
python run.py

# Or run a self-contained test on synthetic data (no real data needed)
python examples/ex_run_bootstrap_small.py
```

---

## Multi-machine path setup

Dropbox paths differ per machine.  `run.py` auto-detects the correct root
using the hostname.  The mapping lives in
`src/zeno_analysis/utils/lab_paths.py` and mirrors `ZenoPlotting/base.py`.

**Currently registered machines:**

| Hostname | User | Drive |
|----------|------|-------|
| `DESKTOP-BAEVOT2` | Noam Nezer | C: |
| `DESKTOP-FQN6OUI` | Danielle Gov | C: |
| `CMD-SRStudent1` | Barkay Guttel | E: |
| `cmd-SRLenovoPc2` | Serge Rosenblum | E: |

**Adding a new computer:**

1. Find your hostname: `python -c "import socket; print(socket.gethostname())"`
2. Add an entry to `_DROPBOX_ROOTS` in `src/zeno_analysis/utils/lab_paths.py`.
.

If your hostname is not registered you will get a clear `ValueError` when
`run.py` starts — not a silent wrong-path failure.

---

## No silent defaults — fail-loud policy

This package never guesses unknown inputs or substitutes placeholders.
If a required value is missing (e.g. `lambda_c_guess`, `observable_key`),
the code raises a `ValueError` or `KeyError` immediately.

---

## Package structure

```
zeno_analysis/
├── run.py                   ← Main orchestrator (toggleable tasks, edit paths here)
├── examples/                ← Small runnable demos per module
│   ├── ex_run_bootstrap_small.py
│   ├── ex_load_experimental.py
│   ├── ex_critical_point_fit.py
│   ├── ex_benchmark_noise.py
│   └── ex_paper_figure.py
├── docs/
│   └── adding_a_new_json_format.md  ← How to plug in a new data source
└── src/zeno_analysis/
    ├── data_types/          ← Typed containers (no I/O, no heavy deps)
    ├── io/                  ← Load/save: JSON, NPZ, experimental folders
    ├── analysis/            ← Numerical core: MPM, RRHA, bootstrap, ksi, guesses
    ├── simulation/          ← Forward simulators + noise models
    ├── pipeline/            ← High-level orchestrators
    ├── plotting/            ← Exploratory + paper figures
    ├── diagnostics/         ← Parity tests and written reports
    └── utils/               ← Path helpers, physics constants, lab paths
```

---

## Data flow

```
Experimental data           Simulation
(JSON files / NPZ)          (PostSelectedDynamicsSimulator /
       |                     EnsembleAverageDynamics)
       |                           |
       v                           v
  io/experimental/          simulation/*.py
  EnsembleAverageLoader  ─────────────────────>  list[ObservableRecord]
  PostSelectedLoader                                      |
  (or your custom loader)                                 |
                           ┌──────────────────────────────┘
                           v
                    MPMPipeline.run()
                           |
                    ┌──────┴──────────────┐
                    v                     v
          MPMStep / MPMRRHAStep    BootstrapStep
          (raw poles per λ)        (residual resampling)
                    └──────┬──────────────┘
                           v
                   ZenoAnalysisResult
                   ├── lambda_values
                   └── poles (PolesResult)
                           ├── decay_rates / _lower / _upper  (n_lambda, order)
                           ├── frequencies / _lower / _upper
                           ├── imag_splitting / real_splitting (n_lambda,)
                           └── raw_samples (PoleSamples | None)
                                   |
                    ┌──────────────┴──────────────┐
                    v                             v
        CriticalPointPipeline.fit()     KsiCrossingPipeline.run()
                    |                             |
                    v                             v
             SqrtFitResult                KsiCrossingResult
             └── lambda_c ± err           └── zero_crossings
```

---

## Data types

### `ObservableRecord` — one lambda, one observable

```
ObservableRecord
├── lambda_val: float          dimensionless Zeno parameter λ
├── t: np.ndarray              time axis in µs
├── signal: np.ndarray         measured signal (same length as t)
├── observable_key: str        "survival", "Z", "G_probability", ...
├── source_file: str           original filename (for tracing)
└── metadata: dict             omega_DG, navg_repeat_experiment, scan_number, ...
```

### `PolesResult` — bootstrap-aggregated pole spectrum

All uncertainty fields are **half-widths** (distance from median to the 16th /
84th percentile), not absolute percentile values.

```
PolesResult
├── decay_rates:          np.ndarray   (n_lambda, order) — median
├── decay_rates_lower:    np.ndarray   (n_lambda, order) — half-width to 16th pct
├── decay_rates_upper:    np.ndarray   (n_lambda, order) — half-width to 84th pct
├── frequencies:          np.ndarray   (n_lambda, order)
├── frequencies_lower:    np.ndarray
├── frequencies_upper:    np.ndarray
├── imag_splitting:       np.ndarray | None   (n_lambda,) — |Im(p1-p0)|/2
├── imag_splitting_lower: np.ndarray | None
├── imag_splitting_upper: np.ndarray | None
├── real_splitting:       np.ndarray | None   (n_lambda,) — |Re(p1-p0)|/2
├── real_splitting_lower: np.ndarray | None
├── real_splitting_upper: np.ndarray | None
└── raw_samples:          PoleSamples | None  — (n_lambda, n_boot, order) complex

# Slicing propagates to all arrays:
subset = poles[mask]        # PolesResult with all fields filtered
```

### `ZenoAnalysisResult` — complete pipeline output

```
ZenoAnalysisResult
├── lambda_values: np.ndarray           (n_lambda,)
├── poles: PolesResult
├── critical_point_sqrt_fit: SqrtFitResult | None
├── metadata: dict
└── parameters: dict
```

### `PoleSamples` — raw bootstrap draws

```python
from zeno_analysis.data_types import PoleSamples

ps = result.poles.raw_samples   # populated by BootstrapStep
ps.samples                      # complex, shape (n_lambda, n_boot, order)
ps.lambda_values                # float, shape (n_lambda,)
ps.get_lambda(i)                # (n_boot, order) slice for lambda index i
```

---

## Pole sorting — canonical rule

All producers (MPMStep, MPMRRHAStep, bootstrap inner loop) sort poles by
**real part ascending** (ties broken by imaginary part ascending).

```python
from zeno_analysis.analysis.pole_sorting import sort_poles_canonical

sorted_poles = sort_poles_canonical(poles)  # shape (order,)
# poles[0]: most negative real (fastest decay / EP pole)
# poles[1]: second-most negative  (other EP pole)
# poles[2]: least negative    (slow / non-EP pole)
```

Splittings use poles 0 and 1 (the EP pair):
- `imag_splitting = |Im(p1 - p0)| / 2`  — non-zero in Zeno regime (λ < λ_c)
- `real_splitting = |Re(p1 - p0)| / 2`  — non-zero in classical regime (λ > λ_c)

---

## Loading experimental data

### Ensemble-average (`scan_experiment_results_50`)

```python
from zeno_analysis.io.experimental import EnsembleAverageLoader

loader = EnsembleAverageLoader()
records = loader.load(folder, observables=["survival", "Z"])
# "survival" <- data.D_probability
# "Z"        <- data.D_probability - data.G_probability
```

### Post-selected (`click_record_with_post_selected_tomography`)

```python
from zeno_analysis.io.experimental import PostSelectedLoader

loader = PostSelectedLoader()
records = loader.load(folder, observables=["survival"])
```

### Adding a new observable to an existing loader

```python
loader.add_extractor(
    "my_observable",
    lambda jd, t_us: np.asarray(jd["data"]["my_key"], dtype=float),
)
records = loader.load(folder, observables=["survival", "my_observable"])
```

### Adding a completely new JSON format

See **[docs/adding_a_new_json_format.md](docs/adding_a_new_json_format.md)**
for a step-by-step guide with a worked example.

### Scan number

The loaders automatically parse a trailing integer from the folder name (e.g.
`scan_experiment_results_50` → `scan_number=50`) and attach it to each
record's metadata.

```python
from zeno_analysis.io.file_naming import parse_result_filename
info = parse_result_filename("mpm_ens_avg_scan50_N50_lam0.40-1.60_20260410_143022.json")
# {'analysis_type': 'mpm', 'source_tag': 'ens_avg', 'scan_number': 50, ...}
```

---

## Running the analysis pipeline

### Typical usage (one call)

```python
from zeno_analysis.pipeline.mpm_pipeline import MPMPipeline

result = MPMPipeline.from_ensemble_average(folder, n_boot=2000).run()
```

### With explicit steps — standard MPM

```python
from zeno_analysis.analysis.steps import MPMStep, BootstrapStep

steps = [
    MPMStep(order=3),
    BootstrapStep(n_boot=2000, seed=42),
]
result = MPMPipeline.from_ensemble_average(folder, steps=steps).run()
```

### With RRHA-enhanced MPM (better noise robustness)

```python
from zeno_analysis.analysis.steps import MPMRRHAStep, BootstrapStep

steps = [
    MPMRRHAStep(order=3, max_iter=5),
    BootstrapStep(n_boot=2000, seed=42),
]
result = MPMPipeline.from_ensemble_average(folder, steps=steps).run()
```

`MPMRRHAStep` is a drop-in replacement for `MPMStep`.  It uses iterative
SVD-reduction + Hankel projection to suppress noise before the pencil solve.
Benchmark shows ~1.56× lower real-part error near the EP under shot noise
(see `diagnostics/SHOT_NOISE_REPORT.md`).

### From a pre-built record list (custom loaders)

```python
result = MPMPipeline.from_records(records, steps=steps).run()
```

---

## Critical-point fitting

### With a preset (recommended)

```python
from zeno_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline

cp = CriticalPointPipeline.from_json("result.json", lambda_max=1.55)
fit = cp.fit(preset="ens_avg_experimental", window_left=50, window_right=15)
print(f"lambda_c = {fit.lambda_c:.4f} +/- {fit.lambda_c_err:.4f}")
```

### Available presets

```python
from zeno_analysis.analysis.critical_point_guesses import get_guess, list_presets

list_presets()
# ['post_selected_experimental', 'ens_avg_experimental', 'post_selected_simulation']
```

| Preset | λ_c | a | b | Use for |
|--------|-----|---|---|---------|
| `"post_selected_experimental"` | 1.0 | 1.12 | 0.5 | First transition, post-selected lab data |
| `"ens_avg_experimental"` | 1.1 | 0.62 | −0.15 | Third transition, ensemble-average lab data |
| `"post_selected_simulation"` | 1.3 | 1.0 | 0.0 | Analytical simulator |

### With explicit guesses

```python
fit = cp.fit(lambda_c_guess=1.1, a_guess=0.62, b_guess=-0.15)
```

The sqrt model is fitted jointly to both sides of the EP:

```
Im(Δe₁₂) = a·√|λ−λ_c| + b·(λ−λ_c)^1.5   (λ < λ_c, imaginary splitting)
Re(Δe₁₂) = a·√|λ−λ_c| + b·(λ−λ_c)^1.5   (λ > λ_c, real splitting)
```

`a` is constrained ≥ 0 (physically required); `λ_c` and `b` are unconstrained.

---

## Ksi crossing-point detection

```python
from zeno_analysis.analysis.ksi_crossing import KsiCrossingPipeline
from zeno_analysis.plotting.ksi_plot import plot_ksi_curve

ksi_pipe = KsiCrossingPipeline.from_result(result, lambda_max=1.55)
ksi_result = ksi_pipe.run()
print(f"EP from ksi: lambda = {ksi_result.lambda_ep:.4f}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plot_ksi_curve(ksi_result, ax=ax)
plt.show()
```

---

## Simulation benchmark

```python
from zeno_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline
from zeno_analysis.simulation.noise import ShotNoise

bench = BenchmarkPipeline(
    clean_records=clean_records,
    noise_models=[ShotNoise(n_shots=1.8e5)],
    analysis_steps=[MPMStep(order=3), BootstrapStep(n_boot=500)],
    n_trials=20,
)
result = bench.run(preset="post_selected_simulation")
print(f"bias = {result.lambda_c_bias:.4f},  std = {result.lambda_c_std:.4f}")
```

### Available noise models

| Class | Parameters | Effect |
|-------|-----------|--------|
| `ShotNoise` | `n_shots`, `seed` | Binomial shot noise: σ = √(s(1−s)/n) |
| `ReadoutError` | `false_positive`, `false_negative` | P_meas = P·(1−fn) + (1−P)·fp |
| `FiniteSampling` | `n_shots`, `seed` | Same formula as ShotNoise |
| `CompositeNoise` | `[model1, model2, ...]` | Apply models in sequence |

---

## Lambda scale: which one to use?

| Data type | Lambda source | How |
|-----------|-------------|-----|
| `ens_avg` | `parameters.lambda_zeno_estimation` in JSON | Automatically read |
| `post_selected` (no CSV) | `parameters.lambda_zeno_estimation` in JSON | Fallback |
| `post_selected` (with CSV) | `Gamma_up_3_states_hmm / (2·Ω_DG·1e-6)` | Pass `csv_path=` to loader |

---

## Debugging tips

### 1. Use `run.py` with `DEBUG = True`

```python
DEBUG = True    # forces n_boot=50, max_files=10
```

### 2. Inspect intermediate state

```python
from zeno_analysis.analysis.steps import MPMStep
state = MPMStep(order=3).process(records, {})
print(state["poles_raw"][0])   # poles for first lambda
```

### 3. Inspect raw bootstrap samples

```python
ps = result.poles.raw_samples  # PoleSamples
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(ps.samples[10].real, ps.samples[10].imag, s=5)
plt.show()
```

### 4. Run examples

```bash
python examples/ex_run_bootstrap_small.py  # synthetic, ~15 s, no data needed
python examples/ex_load_experimental.py    # just loads + prints, no analysis
python examples/ex_critical_point_fit.py   # needs a saved JSON
python examples/ex_benchmark_noise.py      # noise sweep, ~1 min
```

### 5. Run shot-noise parity diagnostic

```bash
python -m zeno_analysis.diagnostics.shot_noise_parity
```

---

## File naming convention

Saved analysis files are auto-named:

```
mpm_ens_avg_scan50_N50_lam0.40-1.60_20260410_143022.json
     ^          ^   ^               ^
   type    scan no  n_lambda      timestamp
```

The `_scanNN_` segment is present only when a scan number was parsed from the
folder name.  Synthetic results omit it.

---

## Shot-noise parity

See `diagnostics/SHOT_NOISE_REPORT.md`.  Short version:

1. The old `object_oriented_poles.py` had shot-noise injection **commented out**.
   All previously published results were on clean noiseless signals.
2. The new pipeline correctly injects noise in benchmarks.
3. **RRHA suppresses noise significantly** near the EP (~1.56× lower error).
