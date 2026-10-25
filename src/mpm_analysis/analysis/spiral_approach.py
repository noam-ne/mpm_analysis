"""
spiral_approach.py
------------------
State freezing from the **direct fit of the conditional Bloch spiral** — a
pencil-free route to the second transition.

Background (paper, Methods "Extracting the observed transition locations" and
SI 6 E / 7 B)
~~~~~~~~~~~~~~
Conditioned on no clicks, the Bloch vector ``R(t) = (X(t), Z(t))`` spirals (or
relaxes monotonically) into an attracting fixed point ``R_inf``.  At late
times the deviation from the fixed point is governed by the pair of
post-selected eigenvalues ``e1, e2`` measured relative to the slow one ``e3``::

    R(t) = R_inf + e^{-gamma t} [ a cosh(s t) + b sinh(s t) / s ]        (eq. 4)

with ``s`` real for a monotonic approach and imaginary for a spiral.  The
approach rate is the slower of the two decay rates ``gamma -/+ Re s``::

    gamma_a = gamma - Re s   ( = e3 - Re e2 )

The click rate at the fixed point ``gamma_0`` is the asymptotic decay rate of
the no-click survival ``P(0)(t)`` ( = -e3 ).  The critical exponent of the
dwell-time density, ``xi = gamma_0 / gamma_a - 1``, crosses zero at the
state-freezing transition; the paper locates the zero with a local quadratic
fit through the ten measured values nearest the sign change.

This module fits eq. (4) **directly** to the tomography traces (variable
projection: the linear amplitudes ``R_inf, a, b`` per component are solved
exactly for every trial ``(gamma, Delta)``), takes ``gamma_0`` either from the
slow pole of the "regular" matrix-pencil result on the survival
(``gamma0_source="mpm_slow_pole"``) or from a linear fit to ``ln S`` over the
last points (``"lnS_tail"``), and bootstraps the whole chain on the
measurement statistics.  The result object shares the ξ-curve attributes of
``ksi_crossing.KsiCrossingResult`` so the ξ plot and JSON writer accept it.

Parametrisation
~~~~~~~~~~~~~~~
Like ``run_files/discriminant_fit.py`` the nonlinear parameters are
``theta = (gamma, Delta)`` with ``Delta = (2 s)^2``: ``Delta > 0`` →
``cosh / sinh`` with ``k = sqrt(Delta)/2 = Re s``; ``Delta < 0`` →
``cos / sin`` with ``omega = sqrt(-Delta)/2 = Im s``; ``Delta = 0`` → the
critically damped ``(1, t)``.  ``Delta`` is continuous through zero, so the
optimiser can cross from spiral to monotonic without a branch switch.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from mpm_analysis.analysis.ksi_crossing import (
    zero_crossing, crossing_samples_from_curves,
)

# Observable keys of the trajectory and of the survival in the tomography
# records (PostSelectedLoader): raw conditional <X>, <Z> on the survivors.
SPIRAL_KEYS: tuple[str, str] = ("X_cond", "Z_cond")
SURVIVAL_KEY: str = "S"
# Slot of the slow (non-pair) pole in the pencil result.  None → picked per λ
# (and per bootstrap draw) as the real pole with the smallest decay rate; the
# canonical order puts the EP pair in slots 0/1 and the real poles after it
# by decay (order 3: slot 2; order 4: the slow one is slot 3), so a fixed slot
# is only right for one model order.
SLOW_POLE_SLOT: int | None = None


def slow_pole_index(decay: np.ndarray, freq: np.ndarray, *, tol: float = 1e-6):
    """Index of the slow pole of one pole set: the finite pole with the smallest
    decay rate among the real ones (``|Im| < tol``), or among all finite poles
    when none is real.  ``None`` when every pole is NaN."""
    decay = np.asarray(decay, dtype=float)
    freq = np.asarray(freq, dtype=float)
    finite = np.isfinite(decay)
    real = finite & (np.abs(freq) < tol)
    cand = real if real.any() else finite
    if not cand.any():
        return None
    return int(np.flatnonzero(cand)[np.argmin(decay[cand])])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def spiral_basis(t: np.ndarray, gamma: float, delta: float) -> np.ndarray:
    """Design matrix ``[1, e^{-gamma t} f(t), e^{-gamma t} g(t)]`` of eq. (4).

    ``(f, g) = (cosh kt, sinh kt / k)`` for ``delta > 0`` (``k = sqrt(delta)/2``),
    ``(cos wt, sin wt / w)`` for ``delta < 0`` (``w = sqrt(-delta)/2``) and
    ``(1, t)`` at ``delta = 0``.  Shape ``(N, 3)``; the constant column carries
    the fixed point ``R_inf``.
    """
    t = np.asarray(t, dtype=float)
    if delta > 0:
        # e^{-γt}cosh(kt), e^{-γt}sinh(kt)/k as the two pure exponentials
        # e^{-(γ∓k)t} — no cosh overflow when the optimiser tries a large Δ.
        k = np.sqrt(delta) / 2
        slow = np.exp(np.clip(-(gamma - k) * t, -700, 700))
        fast = np.exp(np.clip(-(gamma + k) * t, -700, 700))
        return np.column_stack([np.ones_like(t), (slow + fast) / 2, (slow - fast) / (2 * k)])
    e = np.exp(np.clip(-gamma * t, -700, 700))
    if delta < 0:
        w = np.sqrt(-delta) / 2
        f, g = np.cos(w * t), np.sin(w * t) / w
    else:
        f, g = np.ones_like(t), t
    return np.column_stack([np.ones_like(t), e * f, e * g])


def _amplitudes(B: np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted lstsq amplitudes ``(k, 3)`` of every component of ``Y`` (k, N)."""
    amps = np.empty((Y.shape[0], B.shape[1]))
    for c in range(Y.shape[0]):
        w = W[c]
        a, *_ = np.linalg.lstsq(B * w[:, None], Y[c] * w, rcond=None)
        amps[c] = a
    return amps


def rates_from_theta(gamma: float, delta: float) -> tuple[float, float]:
    """``(gamma_a, omega)`` of eq. (4): ``gamma - Re s`` and ``Im s``."""
    if delta > 0:
        return float(gamma - np.sqrt(delta) / 2), 0.0
    return float(gamma), float(np.sqrt(-delta) / 2)


def gamma_from_rates(gamma_a: float, delta: float) -> float:
    """Inverse of :func:`rates_from_theta`: ``gamma = gamma_a + Re s``."""
    return float(gamma_a + (np.sqrt(delta) / 2 if delta > 0 else 0.0))


# The optimiser works in theta = (gamma_a, Delta): the approach rate is then a
# bounded parameter (>= GAMMA_A_FLOOR), so the slow decay can never turn into
# growth, and its error comes straight out of the covariance.
GAMMA_A_FLOOR = 1e-3


def _resid(theta: np.ndarray, t: np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    ga, delta = theta
    B = spiral_basis(t, gamma_from_rates(ga, delta), delta)
    amps = _amplitudes(B, Y, W)
    return ((amps @ B.T - Y) * W).ravel()


@dataclass
class SpiralFit:
    """One fit of eq. (4) to a late-time trajectory.

    Attributes
    ----------
    gamma, delta:
        Fitted nonlinear parameters (``Delta = (2s)^2``).
    gamma_a, omega:
        Approach rate ``gamma - Re s`` and spiral frequency ``Im s`` (1/µs).
    sigma_gamma_a:
        1-σ error of ``gamma_a`` from the fit covariance (√χ²_red-scaled when
        χ²_red > 1).
    r_inf:
        Fixed point, shape ``(k,)`` — the constant amplitude of each component.
    amps:
        Linear amplitudes ``(k, 3)`` = ``[R_inf, a, b]`` per component.
    cov:
        Covariance of ``(gamma, delta, amps.ravel())`` or ``None``.
    chi2_red, n_points, success:
        Fit quality.
    t0:
        Time origin of the fit (the first fitted sample when ``t0_shift``).
    """
    gamma: float
    delta: float
    gamma_a: float
    omega: float
    sigma_gamma_a: float
    r_inf: np.ndarray
    amps: np.ndarray
    cov: np.ndarray | None
    chi2_red: float
    n_points: int
    success: bool
    t0: float = 0.0

    def curve(self, t: np.ndarray) -> np.ndarray:
        """Model trajectory ``(k, N)`` on ``t`` (absolute times)."""
        B = spiral_basis(np.asarray(t, dtype=float) - self.t0, self.gamma, self.delta)
        return self.amps @ B.T


def _default_seed(t: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """Crude ``(gamma, Delta)`` start: decay over a third of the window, spiral
    frequency from the strongest Fourier component of the most varying trace."""
    span = max(float(t[-1] - t[0]), 1e-6)
    gamma0 = 3.0 / span
    c = int(np.argmax(np.nanvar(Y, axis=1)))
    y = Y[c] - np.nanmean(Y[c])
    y = np.where(np.isfinite(y), y, 0.0)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), dt)
    if len(spec) > 1:
        i = 1 + int(np.argmax(spec[1:]))
        omega = 2 * np.pi * freqs[i]
    else:
        omega = 2 * np.pi / span
    return gamma0, -4.0 * omega ** 2


def fit_spiral_approach(
    t: np.ndarray,
    R: np.ndarray,
    sigma: np.ndarray | None = None,
    *,
    seed: tuple[float, float] | None = None,
    t0_shift: bool = True,
    max_nfev: int = 4000,
) -> SpiralFit:
    """Fit eq. (4) to a late-time trajectory by variable projection.

    Parameters
    ----------
    t:
        Times (µs) of the fitted window, shape ``(N,)``.
    R:
        Trajectory ``(k, N)`` — e.g. ``[X_cond, Z_cond]`` — or ``(N,)`` for a
        single component.  Every component shares ``(gamma, Delta)`` and has
        its own ``(R_inf, a, b)``.
    sigma:
        Optional 1-σ error bars of the same shape (or ``(N,)`` shared by all
        components); points with missing / zero σ get the median σ.
    seed:
        Start ``(gamma_a, Delta)``; ``None`` → :func:`_default_seed`.  Several
        ``Delta`` starts (seed, ± |seed|/4, 4 |seed|) are tried and the lowest
        cost wins, so the fit can land on either side of ``Delta = 0``.
    t0_shift:
        Measure time from the first fitted sample (conditioning); the
        amplitudes then refer to that origin, ``R_inf`` is unaffected.

    Returns
    -------
    SpiralFit
        ``success`` is ``False`` when the optimiser did not converge or
        ``gamma_a`` sits on its floor (``GAMMA_A_FLOOR``) — the window holds
        no resolvable approach; the pipeline treats such ``gamma_a`` as NaN.
    """
    t = np.asarray(t, dtype=float)
    Y = np.atleast_2d(np.asarray(R, dtype=float))
    if Y.shape[1] != t.size:
        raise ValueError(f"fit_spiral_approach: R {Y.shape} does not match t ({t.size},).")
    k, N = Y.shape
    if N < 6:
        raise ValueError(f"fit_spiral_approach: {N} points — need at least 6 "
                         "to fit (gamma_a, Delta) + 3 amplitudes per component.")
    if sigma is None:
        W = np.ones_like(Y)
    else:
        s = np.broadcast_to(np.asarray(sigma, dtype=float), Y.shape).copy()
        ok = np.isfinite(s) & (s > 0)
        fill = np.median(s[ok]) if ok.any() else 1.0
        W = 1.0 / np.where(ok, s, fill)
    finite = np.all(np.isfinite(Y), axis=0)
    t, Y, W = t[finite], Y[:, finite], W[:, finite]
    t0 = float(t[0]) if t0_shift else 0.0
    tt = t - t0

    ga0, d0 = seed if seed is not None else _default_seed(tt, Y)
    ga0 = max(float(ga0), 2 * GAMMA_A_FLOOR)
    d0 = float(d0)
    starts = [d0, -abs(d0) / 4 if d0 else -0.05, abs(d0) / 4 if d0 else 0.05,
              4 * abs(d0) if d0 else 0.5]
    best = None
    for d_start in starts:
        try:
            sol = least_squares(
                _resid, np.array([ga0, d_start]), args=(tt, Y, W),
                bounds=([GAMMA_A_FLOOR, -np.inf], [np.inf, np.inf]), x_scale="jac",
                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=max_nfev,
            )
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise RuntimeError("fit_spiral_approach: every start of the optimiser failed.")
    gamma_a, delta = (float(v) for v in best.x)
    gamma = gamma_from_rates(gamma_a, delta)
    B = spiral_basis(tt, gamma, delta)
    amps = _amplitudes(B, Y, W)

    # Covariance of the full parameter vector (gamma_a, delta, amps) from a
    # numerical Jacobian of the weighted residual at the solution.
    def full_resid(p):
        (ga, dl), am = p[:2], p[2:].reshape(k, 3)
        return ((am @ spiral_basis(tt, gamma_from_rates(ga, dl), dl).T - Y) * W).ravel()

    p0 = np.concatenate([[gamma_a, delta], amps.ravel()])
    r0 = full_resid(p0)
    J = np.empty((r0.size, p0.size))
    for j in range(p0.size):
        h = 1e-6 * max(abs(p0[j]), 1e-3)
        pp = p0.copy()
        pp[j] += h
        J[:, j] = (full_resid(pp) - r0) / h
    dof = max(r0.size - p0.size, 1)
    chi2_red = float(r0 @ r0 / dof)
    try:
        cov = np.linalg.inv(J.T @ J) * max(1.0, chi2_red)
    except np.linalg.LinAlgError:
        cov = None
    _, omega = rates_from_theta(gamma, delta)
    sigma_ga = float(np.sqrt(cov[0, 0])) if cov is not None and cov[0, 0] >= 0 else np.nan
    resolved = gamma_a > 2 * GAMMA_A_FLOOR
    return SpiralFit(
        gamma=gamma, delta=delta, gamma_a=gamma_a, omega=omega, sigma_gamma_a=sigma_ga,
        r_inf=amps[:, 0].copy(), amps=amps, cov=cov, chi2_red=chi2_red,
        n_points=int(N), success=bool(best.success and resolved), t0=t0,
    )


# ---------------------------------------------------------------------------
# Rates and the late-time window
# ---------------------------------------------------------------------------

def tail_mask(
    t: np.ndarray,
    S: np.ndarray | None,
    *,
    t_fit_min: float | None,
    min_survival: float = 0.05,
) -> np.ndarray:
    """Late-time window of the spiral fit: ``t >= t_fit_min`` and
    ``S >= min_survival`` (finite); ``S=None`` / ``t_fit_min=None`` drop the
    respective condition."""
    t = np.asarray(t, dtype=float)
    m = np.isfinite(t)
    if t_fit_min is not None:
        m &= t >= float(t_fit_min)
    if S is not None:
        S = np.asarray(S, dtype=float)
        m &= np.isfinite(S) & (S >= float(min_survival))
    return m


def gamma0_from_survival_tail(
    t: np.ndarray,
    S: np.ndarray,
    *,
    n_points: int = 10,
    min_survival: float = 0.05,
) -> tuple[float, float]:
    """Click rate at the fixed point: ``-slope`` of ``ln S`` over the last
    ``n_points`` time points with ``S >= min_survival`` (paper Methods).
    Returns ``(gamma0, sigma)``; NaN when fewer than 3 points qualify."""
    t = np.asarray(t, dtype=float)
    S = np.asarray(S, dtype=float)
    ok = np.isfinite(S) & (S >= float(min_survival)) & (S > 0) & np.isfinite(t)
    idx = np.flatnonzero(ok)[-int(n_points):]
    if idx.size < 3:
        return np.nan, np.nan
    coef, cov = np.polyfit(t[idx], np.log(S[idx]), 1, cov=True)
    return float(-coef[0]), float(np.sqrt(max(cov[0, 0], 0.0)))


def ksi_from_rates(gamma0, gamma_a):
    """``xi = gamma0 / gamma_a - 1`` (array-safe; NaN where ``gamma_a`` ≤ 0)."""
    g0 = np.asarray(gamma0, dtype=float)
    ga = np.asarray(gamma_a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(ga > 0, g0 / ga - 1.0, np.nan)
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SpiralKsiResult:
    """ξ(λ) from the spiral approach rate; shares the ξ-curve attributes of
    ``KsiCrossingResult`` (``plot_ksi_curve`` / ``augment_json_with_ksi``
    accept both).

    Per λ: ``gamma_a`` (+ ``gamma_a_err``, fit 1-σ), ``gamma0`` (+ ``gamma0_err``:
    pencil 16/84 half-width or ln-S fit 1-σ), ``omega`` (spiral frequency),
    ``r_inf`` ``(n_lambda, 2)`` fixed points, ``chi2_red`` and the ``fits``.
    Bootstrap: ``ksi_lower/upper`` 16/84 band, ``crossing_samples`` per draw,
    ``crossing_quantiles`` (16/50/84).  ``crossing_fit`` is the central
    quadratic / line fit for plotting.
    """
    lambda_values: np.ndarray
    ksi_values: np.ndarray
    zero_crossings: list[float]
    gamma_a: np.ndarray
    gamma_a_err: np.ndarray
    gamma0: np.ndarray
    gamma0_err: np.ndarray
    omega: np.ndarray
    r_inf: np.ndarray
    chi2_red: np.ndarray
    fits: list = field(default_factory=list, repr=False)
    ksi_lower: np.ndarray | None = None
    ksi_upper: np.ndarray | None = None
    crossing_samples: np.ndarray | None = None
    crossing_quantiles: tuple[float, float, float] | None = None
    crossing_fit: dict | None = None
    gamma0_source: str = "mpm_slow_pole"
    crossing_method: str = "quadfit"
    t_fit_min: float | None = None
    min_survival: float = 0.05
    n_points: int = 10
    half_width: float = 0.2
    keys: tuple[str, ...] = SPIRAL_KEYS
    ksi_label: str = r"$\xi = \gamma_0/\gamma_a - 1$  (spiral fit)"

    @property
    def lambda_ksi_zero(self) -> float | None:
        """Central state-freezing λ (the fitted crossing); ``None`` without one."""
        return self.zero_crossings[0] if self.zero_crossings else None


# ---------------------------------------------------------------------------
# Bootstrap worker (top level: process pools pickle it)
# ---------------------------------------------------------------------------

def _spiral_bootstrap_worker(payload: dict) -> tuple[np.ndarray, np.ndarray, int, str | None]:
    """All draws of one λ: resample the measurement statistics, refit eq. (4)
    on the fixed central window, take γ0 from the paired pencil draw or the
    resampled ln-S tail.  Returns ``(gamma_a (n_boot,), gamma0 (n_boot,),
    n_failed, first_error)``."""
    from mpm_analysis.analysis.bootstrap import _MEASUREMENT_RESAMPLERS

    stats = payload["stats"]
    resample = _MEASUREMENT_RESAMPLERS[stats["data_mode"]]
    rng = np.random.default_rng(payload["seed_seq"])
    t, keys, mask = payload["t"], payload["keys"], payload["mask"]
    n_boot = int(payload["n_boot"])
    seed = payload["seed"]
    g0_draws = payload["gamma0_draws"]          # (n_mpm,) or None → lnS tail
    ga = np.full(n_boot, np.nan)
    g0 = np.full(n_boot, np.nan)
    n_failed, first_err = 0, None
    sigma = payload.get("sigma")
    for b in range(n_boot):
        try:
            Y = resample(stats, keys, rng)          # rows: S, X_cond, Z_cond
            S_b, R_b = Y[0], Y[1:]
            fit = fit_spiral_approach(t[mask], R_b[:, mask], sigma, seed=seed)
            ga[b] = fit.gamma_a if fit.gamma_a > 2 * GAMMA_A_FLOOR else np.nan
            if g0_draws is None:
                g0[b], _ = gamma0_from_survival_tail(
                    t, S_b, n_points=payload["n_tail_points"],
                    min_survival=payload["min_survival"])
            else:
                g0[b] = g0_draws[b % len(g0_draws)]
        except Exception as exc:          # one bad draw must not kill the λ
            n_failed += 1
            if first_err is None:
                first_err = f"{type(exc).__name__}: {exc}"
    return ga, g0, n_failed, first_err


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SpiralKsiPipeline:
    """ξ = γ0/γa − 1 from the direct spiral fit, per λ, with bootstrap.

    Parameters
    ----------
    tomo_records:
        ``ObservableRecord`` list holding, per λ, the survival ``survival_key``
        and the raw conditional trajectory ``keys`` (PostSelectedLoader
        ``["S", "X_cond", "Z_cond"]``), full time range — the late-time window
        is applied here.
    result:
        The first-transition ``ZenoAnalysisResult`` (pencil on S).  Required
        for ``gamma0_source="mpm_slow_pole"``: γ0 = decay rate of the slow pole
        at the same λ (exact match) — ``slow_pole_slot`` fixes its slot, the
        default ``None`` picks per λ and per draw the real pole with the
        smallest decay (:func:`slow_pole_index`; right for any model order).
        Its bootstrap draws pair with the spiral draws (draw ``b`` ↔ draw
        ``b``; the two experiments are independent, so that is one joint
        realisation).  When given, the pair pole also seeds the spiral fit.
    gamma0_source:
        ``"mpm_slow_pole"`` | ``"lnS_tail"`` (linear fit to ln S over the last
        ``n_tail_points`` with S ≥ ``min_survival``, paper Methods).
    t_fit_min, min_survival:
        Late-time window of the spiral fit (:func:`tail_mask`).
    lambda_min, lambda_max:
        Restrict to lambda_min ≤ λ ≤ lambda_max.  Below λ ≈ 0.65 the window
        is not a decaying approach (near-undamped Rabi, χ²_red ≫ 1, ξ ≫ 0)
        and would offer the crossing search a spurious first sign change.
    n_boot, seed, n_jobs:
        Bootstrap draws per λ (0 → none), base seed, process-pool workers
        (``-1`` all cores, ``1`` serial) — the conventions of
        ``bootstrap.run_data_bootstrap``.
    crossing_method, n_points, half_width:
        How ξ = 0 is located (:func:`ksi_crossing.zero_crossing`):
        ``"quadfit"`` (paper, ``n_points`` nearest the sign change),
        ``"linfit"`` (±``half_width``) or ``"first"``.
    """

    def __init__(
        self,
        tomo_records,
        result=None,
        *,
        gamma0_source: str = "mpm_slow_pole",
        t_fit_min: float | None = None,
        min_survival: float = 0.05,
        n_tail_points: int = 10,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        n_boot: int = 0,
        seed: int | None = None,
        n_jobs: int = -1,
        crossing_method: str = "quadfit",
        n_points: int = 10,
        half_width: float = 0.2,
        keys: tuple[str, str] = SPIRAL_KEYS,
        survival_key: str = SURVIVAL_KEY,
        slow_pole_slot: int | None = SLOW_POLE_SLOT,
    ) -> None:
        if gamma0_source not in ("mpm_slow_pole", "lnS_tail"):
            raise ValueError(f"gamma0_source '{gamma0_source}' — expected "
                             "'mpm_slow_pole' or 'lnS_tail'.")
        if gamma0_source == "mpm_slow_pole" and result is None:
            raise ValueError("gamma0_source='mpm_slow_pole' needs the pencil result "
                             "(run_mpm on the survival) — pass it, or use 'lnS_tail'.")
        self._records = tomo_records
        self._result = result
        self._gamma0_source = gamma0_source
        self._t_fit_min = t_fit_min
        self._min_survival = float(min_survival)
        self._n_tail_points = int(n_tail_points)
        self._lambda_min = lambda_min
        self._lambda_max = lambda_max
        self._n_boot = int(n_boot)
        self._seed = seed
        self._n_jobs = n_jobs
        self._method = crossing_method
        self._n_points = int(n_points)
        self._half_width = float(half_width)
        self._keys = tuple(keys)
        self._survival_key = survival_key
        self._slot = None if slow_pole_slot is None else int(slow_pole_slot)

    # -- helpers -----------------------------------------------------------

    def _result_index(self, lam: float) -> int | None:
        """Row of ``result`` at exactly this λ (``None`` when absent)."""
        if self._result is None:
            return None
        lam_res = np.asarray(self._result.lambda_values, dtype=float)
        i = int(np.argmin(np.abs(lam_res - lam)))
        return i if abs(lam_res[i] - lam) <= 1e-6 else None

    def _seed_from_result(self, i: int | None) -> tuple[float, float] | None:
        """(gamma_a, Delta) start from the pencil pair relative to the slow pole:
        gamma_a = (slower pair decay) − (slow decay), Delta = −4ω² (spiral) or
        the squared real split (monotonic)."""
        if i is None:
            return None
        dec = np.asarray(self._result.poles.decay_rates[i], dtype=float)
        frq = np.asarray(self._result.poles.frequencies[i], dtype=float)
        i_slow = self._slow_slot(dec, frq)
        pair = [k for k in (0, 1) if k != i_slow]      # canonical: EP pair in slots 0/1
        if i_slow is None or len(pair) < 2 or not np.all(np.isfinite(dec[pair])):
            return None
        slow = dec[i_slow]
        om = float(np.max(np.abs(frq[pair])))
        delta = -4.0 * om ** 2 if om > 1e-9 else float((dec[pair[0]] - dec[pair[1]]) ** 2)
        gamma_a = float(np.min(dec[pair]) - slow)
        return max(gamma_a, 2 * GAMMA_A_FLOOR), delta

    def _slow_slot(self, dec: np.ndarray, frq: np.ndarray):
        """Slow-pole slot of one pole set: the fixed ``slow_pole_slot`` or
        :func:`slow_pole_index`."""
        return self._slot if self._slot is not None else slow_pole_index(dec, frq)

    @staticmethod
    def _sigma_from_stats(stats: dict | None, keys: tuple[str, ...], mask: np.ndarray):
        """Binomial 1-σ of the conditional coordinates, sqrt((1−V²)/N_V), from
        the loader's bootstrap statistics (``None`` without them) — the
        weights of the spiral fit.  Shape (k, N_window)."""
        if not stats:
            return None
        from mpm_analysis.analysis.bootstrap import _POST_SELECTED_TOMOGRAPHY
        rows = []
        for key in keys:
            spec = _POST_SELECTED_TOMOGRAPHY.get(key)
            if spec is None or stats.get(spec[0]) is None or stats.get(spec[1]) is None:
                return None
            v = np.clip(np.asarray(stats[spec[0]], dtype=float), -1, 1)
            n = np.asarray(stats[spec[1]], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rows.append(np.where(n > 0, np.sqrt(np.clip(1 - v ** 2, 1e-4, 1) / n), np.nan))
        return np.vstack(rows)[:, mask]

    def _gamma0_mpm(self, i: int | None) -> tuple[float, float, np.ndarray | None]:
        """γ0 = decay rate of the slow pole at row ``i``: (value, err, draws)."""
        if i is None:
            return np.nan, np.nan, None
        p = self._result.poles
        slot = self._slow_slot(p.decay_rates[i], p.frequencies[i])
        if slot is None:
            return np.nan, np.nan, None
        val = float(p.decay_rates[i, slot])
        lo = getattr(p, "decay_rates_lower", None)
        hi = getattr(p, "decay_rates_upper", None)
        err = (float(lo[i, slot] + hi[i, slot]) / 2
               if lo is not None and hi is not None else np.nan)
        raw = p.raw_samples
        draws = None
        if raw is not None:
            # the slow pole of every draw, picked the same way (draws are
            # slot-ordered canonically but a fixed slot can still be a NaN or
            # an aliased pole in an individual draw)
            arr = np.asarray(raw.samples)[i]                 # (n_boot, order)
            draws = np.full(arr.shape[0], np.nan)
            for b in range(arr.shape[0]):
                sb = self._slow_slot(-arr[b].real, arr[b].imag)
                if sb is not None:
                    draws[b] = -arr[b, sb].real
        return val, err, draws

    # -- run ---------------------------------------------------------------

    def run(self) -> SpiralKsiResult:
        from mpm_analysis.data_types.observables import group_records_by_lambda

        keys = [self._survival_key, *self._keys]
        groups = group_records_by_lambda(self._records, keys)
        if self._lambda_min is not None:
            groups = [g for g in groups if g[0] >= self._lambda_min]
        if self._lambda_max is not None:
            groups = [g for g in groups if g[0] <= self._lambda_max]
        if not groups:
            raise ValueError(f"SpiralKsiPipeline: no λ with all of {keys} in "
                             f"[{self._lambda_min}, {self._lambda_max}].")

        n = len(groups)
        lam = np.array([g[0] for g in groups], dtype=float)
        ga = np.full(n, np.nan); ga_err = np.full(n, np.nan)
        g0 = np.full(n, np.nan); g0_err = np.full(n, np.nan)
        om = np.full(n, np.nan); chi2 = np.full(n, np.nan)
        r_inf = np.full((n, len(self._keys)), np.nan)
        fits: list = []
        payloads: list = []
        n_unresolved = 0
        for j, (lam_j, t, Y, recs) in enumerate(groups):
            S, R = Y[0], Y[1:]
            mask = tail_mask(t, S, t_fit_min=self._t_fit_min, min_survival=self._min_survival)
            stats = recs[0].metadata.get("bootstrap_stats")
            sigma = self._sigma_from_stats(stats, self._keys, mask)
            i_res = self._result_index(lam_j)
            seed = self._seed_from_result(i_res)
            fit = None
            if mask.sum() >= 6:
                try:
                    fit = fit_spiral_approach(t[mask], R[:, mask], sigma, seed=seed)
                except Exception as exc:
                    print(f"  spiral fit failed at lam={lam_j:.3f}: {exc}")
            else:
                print(f"  spiral fit skipped at lam={lam_j:.3f}: only {int(mask.sum())} "
                      f"points in the window (t >= {self._t_fit_min}, S >= {self._min_survival}).")
            fits.append(fit)
            if fit is not None:
                om[j], r_inf[j], chi2[j] = fit.omega, fit.r_inf, fit.chi2_red
                if fit.gamma_a > 2 * GAMMA_A_FLOOR:
                    ga[j], ga_err[j] = fit.gamma_a, fit.sigma_gamma_a
                else:               # approach rate on its floor: nothing resolved
                    n_unresolved += 1
            g0_draws = None
            if self._gamma0_source == "mpm_slow_pole":
                g0[j], g0_err[j], g0_draws = self._gamma0_mpm(i_res)
                if i_res is None:
                    print(f"  lam={lam_j:.3f} not in the pencil result — gamma0 is NaN.")
            else:
                g0[j], g0_err[j] = gamma0_from_survival_tail(
                    t, S, n_points=self._n_tail_points, min_survival=self._min_survival)
            payloads.append(dict(
                lambda_val=lam_j, t=t, keys=keys, mask=mask, stats=stats, sigma=sigma,
                seed=(fit.gamma_a, fit.delta) if fit is not None else seed,
                gamma0_draws=g0_draws, n_tail_points=self._n_tail_points,
                min_survival=self._min_survival,
            ))
        if n_unresolved:
            print(f"  {n_unresolved}/{n} lambdas: approach rate on its floor "
                  f"(no resolvable approach in the window) — gamma_a set to NaN; "
                  f"try an earlier t_fit_min.")

        ksi = ksi_from_rates(g0, ga)
        x0, cfit = zero_crossing(lam, ksi, method=self._method, half_width=self._half_width,
                                 n_points=self._n_points, return_fit=True)
        crossings = [float(x0)] if np.isfinite(x0) else []
        if crossings:
            print(f"Ksi (state freezing, spiral fit, {self._method}): lam(ksi=0) ~ {x0:.4f}")
        else:
            print("Ksi (state freezing, spiral fit): no zero crossing found.")

        ksi_lo = ksi_hi = cross_samples = quantiles = None
        if self._n_boot > 0:
            ksi_lo, ksi_hi, cross_samples, quantiles = self._bootstrap(lam, payloads)

        return SpiralKsiResult(
            lambda_values=lam, ksi_values=ksi, zero_crossings=crossings,
            gamma_a=ga, gamma_a_err=ga_err, gamma0=g0, gamma0_err=g0_err, omega=om,
            r_inf=r_inf, chi2_red=chi2, fits=fits,
            ksi_lower=ksi_lo, ksi_upper=ksi_hi, crossing_samples=cross_samples,
            crossing_quantiles=quantiles, crossing_fit=cfit,
            gamma0_source=self._gamma0_source, crossing_method=self._method,
            t_fit_min=self._t_fit_min, min_survival=self._min_survival,
            n_points=self._n_points, half_width=self._half_width, keys=self._keys,
        )

    def _bootstrap(self, lam, payloads):
        n_boot = self._n_boot
        todo = [p for p in payloads if p["stats"] is not None and p["seed"] is not None]
        for p in payloads:
            if p["stats"] is None:
                print(f"Warning: spiral bootstrap skipped for lam={p['lambda_val']:.3f} — "
                      "no bootstrap_stats in record metadata.")
        if not todo:
            raise ValueError("SpiralKsiPipeline: no record carries "
                             "metadata['bootstrap_stats'] — the data bootstrap needs "
                             "the loader's measurement statistics.")
        if self._gamma0_source == "mpm_slow_pole":
            n_mpm = {len(p["gamma0_draws"]) for p in todo if p["gamma0_draws"] is not None}
            if not n_mpm:
                raise ValueError("SpiralKsiPipeline: gamma0_source='mpm_slow_pole' with "
                                 "n_boot > 0 needs the pencil's raw pole draws "
                                 "(run_mpm with n_boot >= 1 / SAVE of raw samples).")
            if any(m != n_boot for m in n_mpm):
                print(f"  Note: spiral n_boot={n_boot} differs from the pencil draws "
                      f"{sorted(n_mpm)} — pencil draws are reused cyclically.")
        children = np.random.SeedSequence(self._seed).spawn(len(todo))
        for p, c in zip(todo, children):
            p["seed_seq"] = c
            p["n_boot"] = n_boot

        if self._n_jobs == 1 or len(todo) <= 1:
            outputs = [_spiral_bootstrap_worker(p) for p in todo]
        else:
            from concurrent.futures import ProcessPoolExecutor
            max_workers = None if self._n_jobs in (-1, 0, None) else int(self._n_jobs)
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                outputs = list(pool.map(_spiral_bootstrap_worker, todo))

        ksi_curves = np.full((n_boot, len(lam)), np.nan)
        done = {id(p): o for p, o in zip(todo, outputs)}
        n_failed, first_err = 0, None
        for j, p in enumerate(payloads):
            o = done.get(id(p))
            if o is None:
                continue
            ga_b, g0_b, nf, err = o
            ksi_curves[:, j] = ksi_from_rates(g0_b, ga_b)
            n_failed += nf
            first_err = first_err or err
        if n_failed:
            print(f"Warning: {n_failed}/{n_boot * len(todo)} spiral bootstrap draws failed; "
                  f"first error: {first_err}")
        cross, lo, hi = crossing_samples_from_curves(
            lam, ksi_curves, method=self._method, half_width=self._half_width,
            n_points=self._n_points,
        )
        quantiles = None
        n_valid = int(np.sum(np.isfinite(cross)))
        if n_valid:
            q16, q50, q84 = np.nanquantile(cross, [0.16, 0.5, 0.84])
            quantiles = (float(q16), float(q50), float(q84))
            print(f"Ksi (state freezing, spiral fit, bootstrap n_boot={n_boot}): "
                  f"lam(ksi=0) = {q50:.4f} +{q84 - q50:.4f} / -{q50 - q16:.4f}  "
                  f"(16/50/84 = {q16:.4f}/{q50:.4f}/{q84:.4f}; "
                  f"{1 - n_valid / n_boot:.1%} of draws had no crossing)")
        else:
            print("Ksi (state freezing, spiral fit): no bootstrap draw had a zero crossing.")
        return lo, hi, cross, quantiles
