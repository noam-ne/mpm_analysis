"""discriminant_fit.py — the EP discriminant from a DIRECT weighted least-squares fit of the
survival, no SVD, plus the sensitivity the analytic solution demands.

    python run_files/discriminant_fit.py

The fit
-------
Three exponentials, but parametrised so that the exceptional point is a regular point of
the parameter space.  The slow pole is  -g ; the pair has mean rate  a  and discriminant
Delta  (Delta < 0: oscillating pair  -a ± i·sqrt(-Delta)/2;  Delta > 0: two real poles
-a ± sqrt(Delta)/2;  Delta = 0: the EP):

    S(t) = A e^{-g t} + e^{-a t} [ B f(Delta, t) + C g(Delta, t) ]
    f = cos(w t)        g = sin(w t)/w        w = sqrt(-Delta)/2       (Delta < 0)
    f = cosh(k t)       g = sinh(k t)/k       k = sqrt(Delta)/2        (Delta > 0)
    f = 1               g = t                                          (Delta = 0)

Delta here is exactly the production discriminant c1^2 - 4 c0 of the pair's monic quadratic,
so it is comparable one-to-one with the pencil's.  The amplitudes A, B, C are linear and
are solved exactly for every trial (g, a, Delta) (variable projection); scipy's
least_squares only works on the three nonlinear parameters, seeded from the production
matrix pencil of the same trace.  Weights are the MEASURED per-point error bars
(data.p_0_t_errorbar_meta, the block standard errors) and the quoted sigma of Delta is
the covariance from the full 6-parameter Jacobian at the solution (times sqrt(chi2_red)
when chi2_red > 1).  The t = 0 anchor is not fitted (exact 1, no error bar).

The sensitivity question (model section)
----------------------------------------
For the 3x3 generator with the sheet numbers, at each measured lambda:
  * the Cramér-Rao bound on Delta with YOUR per-point error bars (sigma_Delta), and with
    pure shot noise of the pooled attempts (what the error bars would be without drift);
  * d Delta / d lambda of the model, hence the lambda_EP resolution those bounds give;
  * the noise reduction factor needed for a given lambda_EP resolution;
  * where the oscillation's signature sits: the pair's envelope amplitude at its first
    zero crossing compared with the error bar and with S there.

Steps (toggle block below; products on the left):
    1  recs  = load_points(...)          S, S_pooled, sigmas, lambda axis
    2  seeds = pencil_seeds(recs, ...)   production pencil poles -> (g, a, Delta) per lambda
    3  fits  = fit_discriminant(...)     the direct fit per lambda and observable
    4  fig   = plot_delta(...)           Delta(lambda) ± sigma for S / S_pooled vs pencil and model
    5  sensitivity(...)                  the table of required sensitivities
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "mpm_benchmarks" / "scan3_p0"))
import common as K          # noqa: E402
import config as C          # noqa: E402
from mpm_analysis.analysis.matrix_pencil import matrix_pencil_method        # noqa: E402
from mpm_analysis.analysis.ksi_crossing import discriminant_from_poles, linfit_zero_crossing  # noqa: E402

# ===========================================================================
# SETTINGS
# ===========================================================================
_SCAN_ROOT = Path(r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab\Experiments\Zeno"
                  r"\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons\click_record_with_post_selected_tomography"
                  r"\Omega_BG0_scan")
FOLDER = _SCAN_ROOT / "scan_experiment_results_3" / "combined_iterations"
HMM_TABLE: Path | None = _SCAN_ROOT / "scan_experiment_results_3" / "hmm_rates_per_drive.csv"
LAMBDA_RANGE = (0.0, 2.0)           # beyond ~1.9 the third exponential is below the noise (Hankel s3)
OBSERVABLES = ("S_pooled", "S")
T_MIN = 0.2                         # µs; the t = 0 anchor is never fitted
T_MAX: float | None = None
EP_HALF_WIDTH = 0.3                 # λ half-window of the Delta(λ) line fit
T1_US, T2_US = C.T1_US, C.T2_US     # model for the sensitivity section and the reference curve
TARGET_DLAMBDA = 0.03               # λ_EP resolution to aim for
SHOW_PLOTS = True
SAVE_FIGURES = True
OUTPUT_DIR = CODE / "mpm_analysis" / "mpm_analysis_output" / "discriminant_fit" / FOLDER.name


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    K.style()
    recs = load_points(FOLDER, hmm_table=HMM_TABLE, lambda_range=LAMBDA_RANGE)
    seeds = pencil_seeds(recs, observables=OBSERVABLES, t_min=T_MIN, t_max=T_MAX)
    fits = fit_discriminant(recs, seeds, observables=OBSERVABLES, t_min=T_MIN, t_max=T_MAX)
    if (DISC_FIGURE := True):
        fig = plot_delta(recs, seeds, fits, half_width=EP_HALF_WIDTH)
    if (DISC_SENSITIVITY := True):
        sensitivity(recs, T1_us=T1_US, T2_us=T2_US, t_min=T_MIN, target_dlambda=TARGET_DLAMBDA)
    if SHOW_PLOTS:
        plt.show()


# ===========================================================================
# STEP FUNCTIONS
# ===========================================================================

def load_points(folder, *, hmm_table, lambda_range) -> dict:
    """Step 1 — stacked curves of the folder (same layout as survival_explorer.load_points)."""
    pts = K.load_combined(lambda_range=lambda_range, with_counts=True, folder=folder,
                          lambda_source="hmm" if hmm_table is not None else "json", hmm_table=hmm_table)
    t = pts[0]["t"]
    n, m = len(pts), t.size
    out = dict(lam=np.array([p["lam"] for p in pts]), t=t, omega_s=float(pts[0]["omega_s"]),
               folder=Path(folder), n_att_pooled=[p["n_att_pooled"] for p in pts])
    for key in ("S", "S_pooled"):
        out[key] = np.vstack([p[key] for p in pts])
        sig = np.full((n, m), np.nan)
        for i, p in enumerate(pts):
            if p.get(f"sigma_{key}") is not None:
                sig[i] = p[f"sigma_{key}"]
        out[f"sigma_{key}"] = sig
    print(f"=== Load ({out['folder'].name}) ===\n  {n} points, lambda {out['lam'][0]:.3f} .. {out['lam'][-1]:.3f}, "
          f"error bars: {'yes' if np.isfinite(out['sigma_S_pooled']).any() else 'NO (unweighted fits)'}")
    return out


def _window(t, t_min, t_max):
    m = t >= t_min
    if t_max is not None:
        m &= t <= t_max
    return m


def pencil_seeds(recs, *, observables, t_min, t_max) -> dict:
    """Step 2 — production matrix pencil (order 3) on the same window → (g, a, Delta) per λ."""
    t = recs["t"]
    m = _window(t, t_min, t_max)
    out = {}
    for key in observables:
        rows = []
        for y in recs[key]:
            poles, _, _ = matrix_pencil_method(t[m], y[m], 3)
            rows.append(_seed_from_poles(poles))
        out[key] = dict(g=np.array([r[0] for r in rows]), a=np.array([r[1] for r in rows]),
                        delta=np.array([r[2] for r in rows]))
    print("=== Pencil seeds ===\n  order 3 on t in "
          f"[{t[m][0]:.1f}, {t[m][-1]:.1f}] us; Delta(lambda) of S_pooled: "
          + " ".join(f"{l:.2f}:{d:+.3f}" for l, d in zip(recs["lam"], out[observables[0]]["delta"])))
    return out


def _seed_from_poles(poles) -> tuple[float, float, float]:
    p = np.asarray(poles, complex)
    delta = float(discriminant_from_poles(p[None, :])[0])
    # slow pole = the one with the smallest |Re| among the real-ish ones; the other two are the pair
    i_slow = int(np.argmin(np.abs(p.real) + 10 * np.abs(p.imag)))
    pair = np.delete(p, i_slow)
    g = float(-p[i_slow].real)
    a = float(-pair.real.mean())
    if not np.isfinite(delta):
        delta = float(((pair[0] - pair[1]) ** 2).real)
    return g, a, delta


def _basis(t, g, a, delta):
    """Columns [e^{-g t}, e^{-a t} f, e^{-a t} g] of the Delta-parametrised model."""
    if delta < 0:
        w = np.sqrt(-delta) / 2
        f, gg = np.cos(w * t), np.sin(w * t) / w
    elif delta > 0:
        k = np.sqrt(delta) / 2
        f, gg = np.cosh(k * t), np.sinh(k * t) / k
    else:
        f, gg = np.ones_like(t), t
    e = np.exp(-a * t)
    return np.column_stack([np.exp(-g * t), e * f, e * gg])


def _amplitudes(t, y, w, theta):
    X = _basis(t, *theta) * w[:, None]
    amps, *_ = np.linalg.lstsq(X, y * w, rcond=None)
    return amps


def _resid(theta, t, y, w):
    X = _basis(t, *theta)
    amps = _amplitudes(t, y, w, theta)
    return (X @ amps - y) * w


def fit_one(t, y, sigma, seed, *, t0_shift=True):
    """Weighted variable-projection fit; returns dict(g, a, delta, amps, cov, chi2_red, ...)."""
    tt = t - t[0] if t0_shift else t          # origin at the first fitted sample (conditioning)
    w = 1.0 / np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nanmedian(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0)
    g0, a0, d0 = seed
    theta0 = np.array([max(g0, 1e-3), max(a0, 1e-3), d0])
    lb = [1e-4, 1e-4, -np.inf]
    ub = [np.inf, np.inf, np.inf]
    best = None
    for d_start in (d0, -abs(d0) if d0 != 0 else -0.05, abs(d0) if d0 != 0 else 0.05):
        th = theta0.copy(); th[2] = d_start
        try:
            sol = least_squares(_resid, th, args=(tt, y, w), bounds=(lb, ub), x_scale="jac",
                                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    sol = best
    theta = sol.x
    amps = _amplitudes(tt, y, w, theta)
    # full 6-parameter Jacobian of the weighted residual at the solution (numerical)
    def full_resid(params):
        th, am = params[:3], params[3:]
        return (_basis(tt, *th) @ am - y) * w
    p0 = np.concatenate([theta, amps])
    r0 = full_resid(p0)
    J = np.empty((r0.size, p0.size))
    for j in range(p0.size):
        h = 1e-6 * max(abs(p0[j]), 1e-3)
        pp = p0.copy(); pp[j] += h
        J[:, j] = (full_resid(pp) - r0) / h
    dof = max(r0.size - p0.size, 1)
    chi2_red = float(r0 @ r0 / dof)
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((6, 6), np.nan)
    scale = max(1.0, np.sqrt(chi2_red))
    sig = np.sqrt(np.clip(np.diag(cov), 0, None)) * scale
    return dict(g=theta[0], a=theta[1], delta=theta[2], amps=amps, sigma_delta=sig[2], sigma_a=sig[1],
                sigma_g=sig[0], chi2_red=chi2_red, cov=cov, n=r0.size, success=sol.success)


def fit_discriminant(recs, seeds, *, observables, t_min, t_max) -> dict:
    """Step 3 — the direct fit per λ and observable.  Returns {key: dict of arrays}."""
    t = recs["t"]
    m = _window(t, t_min, t_max)
    out = {}
    for key in observables:
        rows = []
        for i, lam in enumerate(recs["lam"]):
            seed = (seeds[key]["g"][i], seeds[key]["a"][i], seeds[key]["delta"][i])
            rows.append(fit_one(t[m], recs[key][i][m], recs[f"sigma_{key}"][i][m], seed))
        out[key] = {k: np.array([r[k] for r in rows]) for k in ("g", "a", "delta", "sigma_delta", "sigma_a", "sigma_g", "chi2_red")}
    print("=== Direct fit ===")
    print(f"{'lambda':>7} | {'pooled Delta':>12} {'+- sigma':>8} {'chi2r':>6} {'a':>6} {'g':>6} | {'S Delta':>9} {'+- sigma':>8} {'chi2r':>6} | {'pencil pooled':>13} {'pencil S':>9}")
    for i, lam in enumerate(recs["lam"]):
        fp, fs = out["S_pooled"], out.get("S")
        line = (f"{lam:7.3f} | {fp['delta'][i]:+12.4f} {fp['sigma_delta'][i]:8.4f} {fp['chi2_red'][i]:6.1f} "
                f"{fp['a'][i]:6.3f} {fp['g'][i]:6.3f} | ")
        line += (f"{fs['delta'][i]:+9.4f} {fs['sigma_delta'][i]:8.4f} {fs['chi2_red'][i]:6.1f} | " if fs is not None else f"{'':>27}| ")
        line += f"{seeds['S_pooled']['delta'][i]:+13.4f} {seeds['S']['delta'][i] if 'S' in seeds else np.nan:+9.4f}"
        print(line)
    return out


def plot_delta(recs, seeds, fits, *, half_width):
    """Step 4 — Delta(λ) with error bars: direct fit vs pencil vs model; line-fit EP of each."""
    lam = recs["lam"]
    w = recs["omega_s"]
    lam_fine = np.linspace(lam[0], lam[-1], 400)
    model_delta = np.array([float(discriminant_from_poles(
        np.linalg.eigvals(K.generator(l, omega_s=w, g1=1 / T1_US, g2=1 / T2_US))[None, :])[0]) for l in lam_fine])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, key in zip(axes, ("S_pooled", "S")):
        f = fits[key]
        ax.errorbar(lam, f["delta"], yerr=f["sigma_delta"], fmt="o", ms=4, color=K.COL[key],
                    label=f"direct weighted fit of {key}  (± σ from the covariance)")
        ax.plot(lam, seeds[key]["delta"], "x", ms=6, color="0.3", label="matrix pencil, same window")
        ax.plot(lam_fine, model_delta, "-", color=K.COL["model"], lw=1, label=f"3x3 model T1={T1_US:g} T2={T2_US:g}")
        ax.axhline(0, color="0.6", lw=0.7)
        ep_fit, _ = linfit_zero_crossing(lam, f["delta"], sigma=f["sigma_delta"], half_width=half_width, return_fit=True)
        ep_pen, _ = linfit_zero_crossing(lam, seeds[key]["delta"], half_width=half_width, return_fit=True)
        for x, c, lbl in ((ep_fit, K.COL[key], "fit"), (ep_pen, "0.3", "pencil")):
            if np.isfinite(x):
                ax.axvline(x, color=c, lw=0.8, ls="--", label=f"λ_EP {lbl} = {x:.3f}")
        ax.set_xlabel("λ"); ax.set_title(key); ax.legend(fontsize=7, loc="upper left")
        lim = np.nanpercentile(np.abs(np.concatenate([f["delta"], seeds[key]["delta"]])), 90)
        ax.set_ylim(-1.5 * lim, 1.5 * lim)
    axes[0].set_ylabel("Δ = c₁² − 4c₀  (1/µs²)")
    fig.suptitle(f"pair discriminant vs λ — {recs['folder'].name}", fontsize=10)
    fig.tight_layout()
    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "delta_vs_lambda.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"  figure saved: {path}")
    return fig


def sensitivity(recs, *, T1_us, T2_us, t_min, target_dlambda) -> None:
    """Step 5 — Cramér-Rao bound on Delta from the analytic solution with the measured error
    bars, the λ_EP resolution it implies, the noise reduction needed, and where the
    oscillation signature sits relative to the noise."""
    t, lam, w = recs["t"], recs["lam"], recs["omega_s"]
    m = t >= t_min
    tt = t[m] - t[m][0]
    g1, g2 = 1 / T1_us, 1 / T2_us
    lam_fine = np.linspace(0.3, 2.0, 1701)
    d_fine = np.array([float(discriminant_from_poles(np.linalg.eigvals(K.generator(l, omega_s=w, g1=g1, g2=g2))[None, :])[0])
                       for l in lam_fine])
    ep = K.model_lambda_ep(omega_s=w, g1=g1, g2=g2)
    print(f"\n=== Sensitivity (3x3 model T1={T1_us:g} T2={T2_us:g}; EP at lambda = {ep:.3f}) ===")
    print(f"{'lambda':>6} {'Delta':>8} {'dDelta/dlam':>11} | {'sig_Delta meas':>14} {'-> sig_lamEP':>12} {'need x':>7} | "
          f"{'sig_Delta shot':>14} {'-> sig_lamEP':>12} | {'osc: t_dip':>10} {'S(t_dip)':>9} {'env/sigma':>9} {'env/S':>7}")
    for i, l in enumerate(lam):
        if not (0.6 <= l <= 1.7):
            continue
        poles = np.linalg.eigvals(K.generator(l, omega_s=w, g1=g1, g2=g2))
        g, a, delta = _seed_from_poles(poles)
        y = K.model_survival(l, t, omega_s=w, g1=g1, g2=g2)[m]
        amps = _amplitudes(tt, y, np.ones_like(tt), (g, a, delta))
        slope = np.interp(l, lam_fine, np.gradient(d_fine, lam_fine))
        res = []
        for sig in (recs["sigma_S_pooled"][i][m], _shot_sigma(recs["S_pooled"][i], recs["n_att_pooled"][i])[m]):
            sd = _crlb_delta(tt, y, sig, (g, a, delta), amps)
            res.append((sd, sd / abs(slope) if slope else np.nan))
        need = res[0][1] / target_dlambda if np.isfinite(res[0][1]) else np.nan
        # oscillation signature: pair envelope at its first zero crossing
        if delta < 0:
            om = np.sqrt(-delta) / 2
            B, Cc = amps[1], amps[2]
            R, phi = np.hypot(B, Cc / om), np.arctan2(Cc / om, B)
            t_dip = ((np.pi / 2 + phi) % np.pi) / om  # first zero of R cos(om t - phi) at t > 0
            k = int(np.argmin(np.abs(tt - t_dip)))
            env = R * np.exp(-a * t_dip)
            osc = (f"{t_dip + t[m][0]:10.2f} {y[k]:9.4f} {env / recs['sigma_S_pooled'][i][m][k]:9.0f} {env / y[k]:7.3f}")
        else:
            osc = f"{'(real pair)':>10} {'':>9} {'':>9} {'':>7}"
        print(f"{l:6.3f} {delta:+8.3f} {slope:+11.3f} | {res[0][0]:14.4f} {res[0][1]:12.3f} {need:7.1f} | "
              f"{res[1][0]:14.4f} {res[1][1]:12.3f} | {osc}")
    print("  sig_Delta: Cramer-Rao bound of the 6-parameter fit with the given per-point sigmas (no model error);\n"
          "  'need x' = factor by which the measured sigma must shrink for sig_lamEP = "
          f"{target_dlambda:g} at that lambda (shots scale with its square);\n"
          "  osc: time of the first zero of the pair's oscillation, S there, the pair's envelope amplitude in units of the\n"
          "  measured sigma and of S.")


def _shot_sigma(S, n_att):
    n_att = np.asarray(n_att, float)
    if n_att.size != S.size:
        return np.full(S.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.clip(S * (1 - S), 0, None) / np.maximum(n_att, 1))


def _crlb_delta(tt, y, sigma, theta, amps):
    sigma = np.asarray(sigma, float)
    ok = np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < 8:
        return np.nan
    w = 1.0 / sigma[ok]
    p0 = np.concatenate([theta, amps])

    def model(params):
        return _basis(tt[ok], *params[:3]) @ params[3:]
    f0 = model(p0)
    J = np.empty((f0.size, 6))
    for j in range(6):
        h = 1e-6 * max(abs(p0[j]), 1e-3)
        pp = p0.copy(); pp[j] += h
        J[:, j] = (model(pp) - f0) / h * w
    try:
        cov = np.linalg.inv(J.T @ J)
        return float(np.sqrt(cov[2, 2]))
    except np.linalg.LinAlgError:
        return np.nan


if __name__ == "__main__":
    main()
