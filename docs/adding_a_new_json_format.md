# Working with data: loading, analysing, and plotting

This guide covers:
1. [Adding a new JSON format](#1-adding-a-new-json-format)
2. [Plotting raw time traces — including the slider](#2-plotting-raw-time-traces)
3. [Plotting poles with uncertainties](#3-plotting-poles-with-uncertainties)
4. [Custom analysis on top of the poles](#4-custom-analysis-on-top-of-the-poles)
5. [Controlling the number of poles and the EP-pair selection](#5-controlling-poles-and-ep-pair-selection)

---

## 1. Adding a new JSON format

### What the pipeline needs

The only contract between your data and the rest of the pipeline is
`ObservableRecord`:

| Field | Type | Description |
|-------|------|-------------|
| `lambda_val` | float | Dimensionless Zeno parameter λ |
| `t` | 1-D array (µs) | Time axis |
| `signal` | 1-D array | Observable value at each time point |
| `observable_key` | str | Name string, e.g. `"survival"` |

Everything else (bootstrap, critical-point fitting, plotting) works unchanged.

### Example: minimal custom JSON

Suppose each file looks like this (one file per λ):

```json
{
  "lambda": 0.72,
  "time_ns": [0, 100, 200, 300, 400],
  "survival_probability": [1.0, 0.91, 0.75, 0.58, 0.43],
  "omega_DG_hz": 1570796.0
}
```

### Write a loader

Subclass `ExperimentalLoader` and implement three methods:

```python
# my_loader.py
from pathlib import Path
import numpy as np
from zeno_analysis.io.experimental.base import ExperimentalLoader

class MyFormatLoader(ExperimentalLoader):

    def _register_default_extractors(self) -> None:
        self.add_extractor(
            "survival",
            lambda jd, t: np.asarray(jd["survival_probability"], dtype=float),
        )

    def _discover_files(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.json"))   # sort however fits your naming

    def _extract_lambda_and_time(self, jd: dict, fp: Path) -> tuple[float, np.ndarray]:
        lam = float(jd["lambda"])
        t_us = np.asarray(jd["time_ns"], dtype=float) / 1e3   # ns → µs
        return lam, t_us
```

The base class handles: lambda filtering, `max_files`, per-file error skipping,
and metadata extraction.

### Load and run the full pipeline

```python
from pathlib import Path
from my_loader import MyFormatLoader
from zeno_analysis.pipeline.mpm_pipeline import MPMPipeline
from zeno_analysis.analysis.steps import MPMStep, BootstrapStep

folder = Path("/path/to/my/data")
records = MyFormatLoader().load(folder, observables=["survival"])

steps  = [MPMStep(order=3), BootstrapStep(n_boot=1000, seed=42)]
result = MPMPipeline.from_records(records, steps=steps).run()
```

### Add extra observables (without a new subclass)

```python
loader = MyFormatLoader()
loader.add_extractor(
    "excited_population",
    lambda jd, t: np.asarray(jd["excited_state"], dtype=float),
)
records = loader.load(folder, observables=["survival", "excited_population"])
```

Each observable becomes its own `ObservableRecord` with `observable_key` set
to the given string.  The pipeline analyses each key independently.

### How the base class processes files

```
_discover_files(folder)         → list[Path], sorted
  for each file:
    _load_json(fp)              → raw dict
    _extract_lambda_and_time()  → (float, np.ndarray µs)
    apply lambda_range / max_files filter
    for each observable key:
      _extractors[key](jd, t)   → 1-D np.ndarray (signal)
      → ObservableRecord(lambda_val, t, signal, key, ...)
```

If any file raises an exception during load or extraction a warning is printed
and that file is skipped — the rest of the scan still loads.

---

## 2. Plotting raw time traces

### All curves at once (colour-coded by λ)

```python
from zeno_analysis.plotting.exploratory import plot_raw_survival
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
plot_raw_survival(records, observable_key="survival", ax=ax)
plt.show()
```

### Selected linecuts (paper-quality)

```python
from zeno_analysis.plotting.survival_curves import plot_survival_curves_panel
import numpy as np

fig, ax = plt.subplots()
ax, plotted = plot_survival_curves_panel(
    records,
    target_lambdas=np.array([0.4, 0.9, 1.3]),  # closest available λ used
    observable_key="survival",
    ax=ax,
)
plt.show()
```

Pass `fit_dir=Path("path/to/fit_jsons")` to overlay pre-computed MPM fit curves
(the JSON must have keys `"time smooth"` and `"fit_curve"`).

### Interactive slider

Browse all lambda slices with a scrollable slider.  The **MPM-reconstructed
signal** (clean model evaluated from the fitted poles) is overlaid in red when
you pass `boot_results`.

```python
from zeno_analysis.plotting.exploratory import plot_time_traces_slider

# Raw traces only
plot_time_traces_slider(records)

# With reconstructed fit overlay
from zeno_analysis.analysis.bootstrap import run_bootstrap
import numpy as np

boot_results = run_bootstrap(
    records, order=3, n_boot=200, observable_key="survival",
    rng=np.random.default_rng(42)
)
plot_time_traces_slider(records, boot_results=boot_results)
```

Use `log_scale=False` for linear y-axis.

---

## 3. Plotting poles with uncertainties

### Quick spectrum (decay rates or frequencies)

```python
from zeno_analysis.plotting.exploratory import plot_eigenvalue_spectrum
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_eigenvalue_spectrum(result, mode="decay",     ax=axes[0])
plot_eigenvalue_spectrum(result, mode="frequency", ax=axes[1])
plt.tight_layout()
plt.show()
```

Error bars show the 16th–84th percentile half-widths from the bootstrap.

### Pole splitting (imaginary and real)

Splitting is stored directly in `PolesResult`:

```python
import matplotlib.pyplot as plt
import numpy as np

lams = result.lambda_values
poles = result.poles

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Imaginary splitting (non-zero for λ < λ_c, Zeno regime)
ax = axes[0]
ax.errorbar(
    lams, poles.imag_splitting,
    yerr=[poles.imag_splitting_lower, poles.imag_splitting_upper],
    fmt="o", capsize=3, label="Im splitting",
)
ax.set_xlabel(r"$\lambda$");  ax.set_ylabel(r"$|\Delta\,\mathrm{Im}| / 2$")
ax.legend()

# Real splitting (non-zero for λ > λ_c, classical regime)
ax = axes[1]
ax.errorbar(
    lams, poles.real_splitting,
    yerr=[poles.real_splitting_lower, poles.real_splitting_upper],
    fmt="o", capsize=3, color="C1", label="Re splitting",
)
ax.set_xlabel(r"$\lambda$");  ax.set_ylabel(r"$|\Delta\,\mathrm{Re}| / 2$")
ax.legend()

plt.tight_layout()
plt.show()
```

### Bootstrap distribution at a specific λ

```python
from zeno_analysis.plotting.exploratory import plot_bootstrap_distribution
from zeno_analysis.analysis.bootstrap import run_bootstrap
import numpy as np

boot_results = run_bootstrap(records, order=3, n_boot=500, observable_key="survival")
plot_bootstrap_distribution(boot_results, lambda_idx=10, pole_idx=0, component="real")
```

### Paper-quality 3-panel figure (decay + splitting + fit)

```python
from zeno_analysis.plotting.paper_figures.appendix_bootstrap import plot_appendix_bootstrap

fig = plot_appendix_bootstrap(
    result,
    lambda_max=1.55,
    fit_validity_limit=1.55,
    real_axis_ticks=[0.0, 1, 2],
    real_split_axis_ticks=[0.0, 0.6, 1.2],
)
fig.savefig("appendix.pdf", bbox_inches="tight")
```

---

## 4. Custom analysis on top of the poles

### Access raw bootstrap samples

The full `(n_lambda, n_boot, order)` complex pole array is in `raw_samples`:

```python
ps = result.poles.raw_samples   # PoleSamples
poles_lam10 = ps.samples[10]    # (n_boot, order) complex — lambda index 10
print(f"median real pole 0 at λ={ps.lambda_values[10]:.3f}: "
      f"{np.median(poles_lam10[:, 0].real):.4f}")
```

### Compute a derived quantity per bootstrap draw

```python
import numpy as np

# Example: ratio of decay rates of poles 0 and 1
lams = result.lambda_values
ratio_median = np.zeros(len(lams))
ratio_lo     = np.zeros(len(lams))
ratio_hi     = np.zeros(len(lams))

for i, lam_samples in enumerate(result.poles.raw_samples.samples):
    # lam_samples shape: (n_boot, order)
    r = -lam_samples[:, 0].real / (-lam_samples[:, 1].real + 1e-15)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        continue
    ratio_median[i] = np.median(r)
    ratio_lo[i]     = ratio_median[i] - np.percentile(r, 16)
    ratio_hi[i]     = np.percentile(r, 84) - ratio_median[i]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(lams, ratio_median, yerr=[ratio_lo, ratio_hi], fmt="o", capsize=3)
ax.set_xlabel(r"$\lambda$");  ax.set_ylabel("Γ₀ / Γ₁")
plt.show()
```

### Fit a custom model to the poles

You can extract the `PolesResult` arrays and fit any scipy model to them:

```python
from scipy.optimize import curve_fit
import numpy as np

lams   = result.lambda_values
dr     = result.poles.decay_rates[:, 0]   # median decay rate, pole 0
dr_err = np.sqrt(
    0.5 * (result.poles.decay_rates_lower[:, 0]**2 +
           result.poles.decay_rates_upper[:, 0]**2)
)

def linear_model(lam, a, b):
    return a * lam + b

popt, pcov = curve_fit(linear_model, lams, dr, sigma=dr_err, absolute_sigma=False)
print(f"slope = {popt[0]:.4f},  intercept = {popt[1]:.4f}")
```

---

## 5. Controlling poles and EP-pair selection

### Number of poles (model order)

Set `order` in `MPMStep` (and optionally in `BootstrapStep`):

```python
from zeno_analysis.analysis.steps import MPMStep, BootstrapStep

steps = [
    MPMStep(order=4),                      # extract 4 poles per λ
    BootstrapStep(n_boot=1000, order=4),   # must match (or omit to auto-detect)
]
```

The result will have `result.poles.decay_rates.shape == (n_lambda, 4)`.

**How to choose `order`:**
- Start with `order=3` (the physical minimum for a 3-state Zeno system).
- If pole spectra look noisy or inconsistent, try `order=4` or `order=5`.
- Higher order is slower and can introduce spurious poles — inspect the
  eigenvalue spectrum to decide.

### Pencil parameter L

The pencil parameter `L` controls the Hankel matrix shape.  Default is
`N // 3`.  Pass explicitly to override:

```python
MPMStep(order=3, L=20)
```

Smaller `L` → faster, less stable.  Larger `L` → slower, more robust.

### Which pole pair is used for the critical-point fit

The splitting is always computed between **poles 0 and 1** in the canonical
sorted order (real part ascending, ties broken by imaginary part ascending).

```
poles[0]  — most negative real part (fastest decay / one EP pole)
poles[1]  — second-most negative    (other EP pole)
poles[2+] — remaining slow poles
```

`imag_splitting = |Im(p₁ − p₀)| / 2`
`real_splitting = |Re(p₁ − p₀)| / 2`

**To use a different pair** (e.g. poles 1 and 2):

1. Open `src/zeno_analysis/analysis/bootstrap.py`, function `_compute_splitting`.
2. Change the index `[:, 1]` / `[:, 0]` at lines ~236 and ~242 to the pair
   you want.
3. Re-run the bootstrap — the `PolesResult` splitting fields will reflect the
   new pair.

There is currently no runtime parameter for this; modifying the source is the
intended extension point.

### Restricting the lambda window fed to the critical-point fitter

```python
from zeno_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline

# Only use λ ≤ 1.55 for the fit
cp = CriticalPointPipeline(result, lambda_max=1.55)

# Or load from a previously saved JSON
cp = CriticalPointPipeline.from_json("result.json", lambda_max=1.55)

fit = cp.fit(
    preset="ens_avg_experimental",
    window_left=50,    # number of λ points left of λ_c_guess in fit window
    window_right=15,   # number of λ points right of λ_c_guess in fit window
)
```

The `window_left / window_right` parameters control how many data points on
each side of the initial `lambda_c_guess` are included in the optimisation
window.  Wider windows are more robust but can include λ far from the EP where
the sqrt model is no longer accurate.
