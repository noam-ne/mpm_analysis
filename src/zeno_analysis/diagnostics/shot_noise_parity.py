"""
shot_noise_parity.py
--------------------
Diagnostic: compare pole-recovery accuracy of the new pipeline against the
old ``object_oriented_poles.py`` approach under identical shot noise.

Run with:
    python -m zeno_analysis.diagnostics.shot_noise_parity

This script:
  1. Generates ground-truth analytical signals via ``PostSelectedDynamicsSimulator``.
  2. Injects binomial shot noise with N_trials = 120 000  (= 4 * 30 000, the
     value used in the old code when the noise injection was uncommented).
  3. Extracts poles with three methods on IDENTICAL noisy signals:
       A. New code — standard MPM (``matrix_pencil_method``)
       B. New code — RRHA MPM (``mpm_rrha``)
       C. Old code path — RRHA on Hankel, then standard MPM pencil
          (``get_mpm_poles_from_rrha_matrix`` from ``matrix_pencil_rrha``)
          Note: methods B and C are equivalent; C is kept as an explicit check.
  4. Measures real-part recovery error |Re(pole) − Re(pole_truth)| for each
     pole across all λ, averaged over ``n_realizations`` noise draws.
  5. Prints a side-by-side table and saves a figure.

Key hypothesis being tested
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The user observed that the new simulation is "way more noisy" than the old
``object_oriented_poles.py``.  Possible root causes:

  (a) RRHA vs plain MPM: old bootstrap used RRHA on the boot Hankel matrix
      (lines 444–445); new bootstrap uses plain MPM.
  (b) Different signal length / T_int in the simulator vs the real data used
      in the old code.
  (c) Different pencil parameter L.
  (d) The old code's shot noise was COMMENTED OUT (lines 391–393), meaning
      old results may have been generated without any shot noise injected,
      making them appear cleaner.

This script isolates (a) and (d) by running on identical analytical signals
with and without noise using both methods, and printing what changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration — edit these to match your typical experimental settings
# ---------------------------------------------------------------------------

N_LAMBDA = 20            # number of lambda points
SIMULATION_DURATION = 25.0   # µs (matches ex_benchmark_noise.py)
MIN_ALPHA = 1.0
MAX_ALPHA = 2.0
ORDER = 3
N_SHOTS = 120_000        # 4 * 3e4, old code's default when noise was active
N_REALIZATIONS = 30      # noise draws per lambda per method


def _inject_shot_noise(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Binomial shot noise: σᵢ = sqrt(sᵢ(1−sᵢ) / N_shots)."""
    sigma = np.sqrt(np.clip(signal * (1.0 - signal), 0, None) / N_SHOTS)
    return signal + rng.normal(0.0, sigma)


def _pole_recovery_errors(
    t: np.ndarray,
    clean_signal: np.ndarray,
    true_poles: np.ndarray,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return mean |Re(pole_est) - Re(pole_truth)| across realizations.

    Parameters
    ----------
    t, clean_signal:
        Ground-truth time axis and signal.
    true_poles:
        Ground-truth complex poles (shape (order,)).
    method:
        ``"mpm"`` or ``"rrha"``.
    rng:
        Random generator.

    Returns
    -------
    np.ndarray
        Shape (order,) — mean absolute real-part error per pole.
    """
    from zeno_analysis.analysis.matrix_pencil import matrix_pencil_method
    from zeno_analysis.analysis.matrix_pencil_rrha import mpm_rrha
    from zeno_analysis.analysis.pole_sorting import sort_poles_canonical, enforce_pole_count

    true_sorted = sort_poles_canonical(enforce_pole_count(true_poles, ORDER))
    errors = np.zeros((N_REALIZATIONS, ORDER))

    for k in range(N_REALIZATIONS):
        noisy = _inject_shot_noise(clean_signal, rng)
        try:
            if method == "mpm":
                poles, _, _ = matrix_pencil_method(t, noisy, ORDER)
            elif method == "rrha":
                poles, _, _ = mpm_rrha(t, noisy, ORDER)
            else:
                raise ValueError(f"Unknown method: {method!r}")
            poles = sort_poles_canonical(enforce_pole_count(poles, ORDER))
            errors[k] = np.abs(poles.real - true_sorted.real)
        except Exception:
            errors[k] = np.nan

    return np.nanmean(errors, axis=0)


def run_diagnostic(*, save_dir: Path | None = None) -> dict:
    """Run the shot-noise parity diagnostic and return results.

    Parameters
    ----------
    save_dir:
        Directory for saving the figure.  Current directory if ``None``.

    Returns
    -------
    dict with keys ``"mpm_errors"``, ``"rrha_errors"``,
    ``"lambda_values"``, ``"true_poles"``.
    Each error array has shape (n_lambda, order).
    """
    from zeno_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator

    print("Shot-noise parity diagnostic")
    print(f"  N_shots={N_SHOTS}, N_realizations={N_REALIZATIONS}, "
          f"N_lambda={N_LAMBDA}, order={ORDER}")
    print()

    sim = PostSelectedDynamicsSimulator(
        num_lambdas=N_LAMBDA,
        simulation_duration=SIMULATION_DURATION,
        min_alpha=MIN_ALPHA,
        max_alpha=MAX_ALPHA,
    )
    records = sim.to_observable_records()
    records = [r for r in records if r.observable_key == "survival"]
    records.sort(key=lambda r: r.lambda_val)

    rng = np.random.default_rng(42)

    mpm_errors = np.zeros((len(records), ORDER))
    rrha_errors = np.zeros((len(records), ORDER))
    lambda_values = np.array([r.lambda_val for r in records])
    true_poles_all = []

    print("  Computing ground-truth poles (clean signal, standard MPM)...")
    from zeno_analysis.analysis.matrix_pencil import matrix_pencil_method
    from zeno_analysis.analysis.pole_sorting import sort_poles_canonical, enforce_pole_count

    for i, rec in enumerate(records):
        clean_poles, _, _ = matrix_pencil_method(rec.t, rec.signal, ORDER)
        clean_poles = sort_poles_canonical(enforce_pole_count(clean_poles, ORDER))
        true_poles_all.append(clean_poles)

    print("  Running noisy trials (MPM) ...")
    for i, rec in enumerate(records):
        mpm_errors[i] = _pole_recovery_errors(
            rec.t, rec.signal, true_poles_all[i], "mpm", rng
        )
        if (i + 1) % 5 == 0:
            print(f"    lam={rec.lambda_val:.3f}  mean_err(mpm)={mpm_errors[i].mean():.5f}")

    print("  Running noisy trials (RRHA) ...")
    for i, rec in enumerate(records):
        rrha_errors[i] = _pole_recovery_errors(
            rec.t, rec.signal, true_poles_all[i], "rrha", rng
        )
        if (i + 1) % 5 == 0:
            print(f"    lam={rec.lambda_val:.3f}  mean_err(rrha)={rrha_errors[i].mean():.5f}")

    # Print comparison table
    print()
    print(f"{'lam':>6}  {'MPM err (p0)':>14}  {'RRHA err (p0)':>14}  "
          f"{'MPM err (p1)':>14}  {'RRHA err (p1)':>14}")
    print("-" * 70)
    for i in range(len(records)):
        lam = lambda_values[i]
        print(
            f"{lam:6.3f}  {mpm_errors[i, 0]:14.5f}  {rrha_errors[i, 0]:14.5f}  "
            f"{mpm_errors[i, 1]:14.5f}  {rrha_errors[i, 1]:14.5f}"
        )

    # Aggregate summary
    print()
    print("Aggregate mean |Re(pole) - Re(truth)| across all lam and poles 0,1:")
    print(f"  MPM  : {mpm_errors[:, :2].mean():.5f}")
    print(f"  RRHA : {rrha_errors[:, :2].mean():.5f}")
    ratio = mpm_errors[:, :2].mean() / (rrha_errors[:, :2].mean() + 1e-12)
    print(f"  Ratio MPM/RRHA : {ratio:.2f}x  (>1 means RRHA is better)")

    # Save figure
    _save_figure(
        lambda_values, mpm_errors, rrha_errors,
        save_dir=save_dir or Path("."),
    )

    return {
        "lambda_values": lambda_values,
        "mpm_errors": mpm_errors,
        "rrha_errors": rrha_errors,
        "true_poles": true_poles_all,
    }


def _save_figure(
    lambda_values: np.ndarray,
    mpm_errors: np.ndarray,
    rrha_errors: np.ndarray,
    *,
    save_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)
    for j, label in enumerate(["pole 0", "pole 1"]):
        ax = axes[j]
        ax.semilogy(lambda_values, mpm_errors[:, j], "o-", label="MPM", color="steelblue")
        ax.semilogy(lambda_values, rrha_errors[:, j], "s--", label="RRHA", color="darkorange")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$|\Delta \mathrm{Re}(\mathrm{pole})|$")
        ax.set_title(f"Recovery error — {label}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Shot-noise parity  N_shots={:,}  N_realizations={}".format(
            N_SHOTS, N_REALIZATIONS
        )
    )
    fig.tight_layout()

    out_pdf = save_dir / "shot_noise_parity.pdf"
    from zeno_analysis.utils.windows_paths import win_long_path
    fig.savefig(win_long_path(out_pdf), bbox_inches="tight")
    print(f"\nFigure saved: {out_pdf}")
    plt.show()


if __name__ == "__main__":
    save_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    results = run_diagnostic(save_dir=save_dir)
