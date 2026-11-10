"""Spiral-approach-rate state freezing (analysis/spiral_approach.py).

Synthetic only: eq. (4) curves with known rates, the 3x3 post-selected
Liouvillian of the paper (eq. S30) for the physics check, and hand-built
post_selected records for the pipeline + bootstrap plumbing.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from mpm_analysis.analysis.spiral_approach import (
    SpiralKsiPipeline, fit_spiral_approach, gamma0_from_survival_tail,
    ksi_from_rates, rates_from_theta, spiral_basis, tail_mask,
)
from mpm_analysis.analysis.ksi_crossing import quadfit_zero_crossing, zero_crossing
from mpm_analysis.analysis.bootstrap import bootstrap_post_selected_measurements
from mpm_analysis.data_types.observables import ObservableRecord


# ---------------------------------------------------------------------------
# eq. (4) with known parameters
# ---------------------------------------------------------------------------

def _eq4(t, r_inf, gamma, delta, a, b):
    B = spiral_basis(t, gamma, delta)
    amps = np.column_stack([np.atleast_1d(r_inf), np.atleast_1d(a), np.atleast_1d(b)])
    return amps @ B.T


@pytest.mark.parametrize("delta", [-4 * 0.6 ** 2, 4 * 0.15 ** 2])
def test_fit_recovers_rates_2d(delta):
    t = np.linspace(0, 20, 120)
    R = _eq4(t, r_inf=[-0.3, 0.2], gamma=0.4, delta=delta, a=[0.5, -0.6], b=[0.2, 0.3])
    fit = fit_spiral_approach(t, R)
    ga, om = rates_from_theta(0.4, delta)
    assert fit.gamma == pytest.approx(0.4, abs=1e-5)
    assert fit.delta == pytest.approx(delta, abs=1e-5)
    assert fit.gamma_a == pytest.approx(ga, abs=1e-5)
    assert fit.omega == pytest.approx(om, abs=1e-5)
    assert np.allclose(fit.r_inf, [-0.3, 0.2], atol=1e-5)
    assert np.allclose(fit.curve(t), R, atol=1e-6)


def test_fit_1d_and_shifted_origin():
    t = np.linspace(5, 25, 80)                       # window not starting at 0
    R = _eq4(t - t[0], r_inf=0.1, gamma=0.3, delta=-4 * 0.5 ** 2, a=0.7, b=0.1)[0]
    fit = fit_spiral_approach(t, R, seed=(0.5, -0.5))
    assert fit.t0 == t[0]
    assert fit.gamma_a == pytest.approx(0.3, abs=1e-5)
    assert fit.r_inf[0] == pytest.approx(0.1, abs=1e-5)


def test_fit_noisy_within_errors():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 20, 120)
    R = _eq4(t, r_inf=[-0.3, 0.2], gamma=0.4, delta=-4 * 0.6 ** 2, a=[0.5, -0.6], b=[0.2, 0.3])
    fit = fit_spiral_approach(t, R + 0.01 * rng.standard_normal(R.shape), sigma=0.01)
    assert abs(fit.gamma_a - 0.4) < 4 * fit.sigma_gamma_a
    assert 0.3 < fit.chi2_red < 3


# ---------------------------------------------------------------------------
# physics: post-selected Liouvillian (paper eq. S30), s = [pS, x, z]
# ---------------------------------------------------------------------------

def _liouvillian(lam, omega=1.0, gamma1=0.0, gamma2=0.0):
    a = 2 * lam * omega
    return np.array([[-a / 2, 0.0, a / 2],
                     [0.0, -gamma2 - a / 2, -omega],
                     [a / 2 - gamma1, omega, -a / 2 - gamma1]])


def _trajectory(L, t, s0=(1.0, 0.0, 1.0)):
    s = np.array([expm(L * ti) @ np.asarray(s0, float) for ti in t]).T
    pS, x, z = s
    return pS, x / pS, z / pS


def _rates(L):
    ev = np.linalg.eigvals(L)
    ev = ev[np.argsort(ev.real)]            # e1, e2, e3 by real part ascending
    e3 = ev[-1].real
    return -e3, e3 - ev[1].real             # gamma0, gamma_a


def _asymptotic_window(t, S, gamma_a):
    """Late-time window of a noiseless trajectory: eq. (4) is the ASYMPTOTIC
    form (X = x/pS carries second-order terms ~ e^{2(Re e2 - e3)t}), so start
    at 4/gamma_a; stop at 12/gamma_a, where the deviation is ~1e-5 and before
    the expm round-off floor (S ~ 1e-7) takes over."""
    return tail_mask(t, S, t_fit_min=4 / gamma_a, min_survival=1e-7) & (t <= 12 / gamma_a)


# spiral on both sides of xi = 0 (0.8/0.15: xi ≈ +0.45; 0.8/0.25: xi ≈ −0.14),
# ideal monotonic approach (1.3/0), spiral near the ideal EP (0.95/0.05).
# A strongly positive xi (e.g. lam 0.7, gamma2 0.08: xi ≈ 3) is NOT testable
# this way: the survival dies before the asymptotic regime of eq. (4) starts.
@pytest.mark.parametrize("lam, gamma2", [(0.8, 0.15), (0.8, 0.25), (1.3, 0.0), (0.95, 0.05)])
def test_gamma_a_matches_liouvillian(lam, gamma2):
    L = _liouvillian(lam, gamma2=gamma2)
    t = np.linspace(0, 80, 1601)
    S, X, Z = _trajectory(L, t)
    g0_true, ga_true = _rates(L)
    m = _asymptotic_window(t, S, ga_true)
    fit = fit_spiral_approach(t[m], np.vstack([X, Z])[:, m])
    assert fit.gamma_a == pytest.approx(ga_true, rel=0.02)
    g0, _ = gamma0_from_survival_tail(t, S, n_points=10, min_survival=1e-7)
    assert g0 == pytest.approx(g0_true, rel=0.02)


def test_ksi_zero_ideal_is_2_over_sqrt3():
    lams = np.arange(1.05, 1.40, 0.02)
    t = np.linspace(0, 60, 1201)
    ksi = []
    for lam in lams:
        L = _liouvillian(lam)
        S, X, Z = _trajectory(L, t)
        m = _asymptotic_window(t, S, np.sqrt(lam ** 2 - 1))     # ideal e3 - e2
        fit = fit_spiral_approach(t[m], np.vstack([X, Z])[:, m])
        g0, _ = gamma0_from_survival_tail(t, S, n_points=10, min_survival=1e-7)
        ksi.append(ksi_from_rates(g0, fit.gamma_a))
    x0 = quadfit_zero_crossing(lams, np.array(ksi), n_points=10)
    assert x0 == pytest.approx(2 / np.sqrt(3), abs=0.01)


# ---------------------------------------------------------------------------
# crossing helpers
# ---------------------------------------------------------------------------

def test_slow_pole_index_any_order():
    from mpm_analysis.analysis.spiral_approach import slow_pole_index
    # order 3, canonical: pair in 0/1, slow real in 2
    assert slow_pole_index([0.8, 0.8, 0.3], [0.5, -0.5, 0.0]) == 2
    # order 4: pair, then two real poles by decay ascending in Re → fast real
    # in slot 2, SLOW real in slot 3 (a fixed slot 2 would pick the fast one)
    assert slow_pole_index([0.8, 0.8, 6.0, 0.3], [0.5, -0.5, 0.0, 0.0]) == 3
    # NaN slot skipped; all-NaN → None; no real pole → least damped overall
    assert slow_pole_index([np.nan, 0.8, 0.3], [0.0, 0.0, 0.0]) == 2
    assert slow_pole_index([np.nan, np.nan], [0.0, 0.0]) is None
    assert slow_pole_index([0.8, 0.4], [0.5, -0.5]) == 1


def test_quadfit_zero_crossing_parabola():
    lam = np.linspace(0.7, 1.3, 25)
    val = (lam - 0.93) * (lam + 2.0)
    x0, fit = quadfit_zero_crossing(lam, val, n_points=10, return_fit=True)
    assert x0 == pytest.approx(0.93, abs=1e-9)
    assert fit["window"].sum() == 10
    assert np.isnan(quadfit_zero_crossing(lam, val + 10))      # no sign change
    assert zero_crossing(lam, val, method="quadfit") == pytest.approx(0.93, abs=1e-9)
    with pytest.raises(ValueError):
        zero_crossing(lam, val, method="cubic")


# ---------------------------------------------------------------------------
# bootstrap rule for the raw conditional coordinates
# ---------------------------------------------------------------------------

def test_bootstrap_x_cond_shares_draw_with_product():
    n = 30
    stats = {
        "data_mode": "post_selected",
        "survival": np.linspace(1, 0.2, n), "N_survival": 6000,
        "SX": np.linspace(1, 0.25, n), "N_attempts_SX": 4000,
        "SZ": np.linspace(1, 0.22, n), "N_attempts_SZ": 2000,
        "X": np.cos(np.linspace(0, 3, n)), "Z": np.sin(np.linspace(0, 3, n)),
        "N_X": np.full(n, 500), "N_Z": np.full(n, 500),
    }
    Y = bootstrap_post_selected_measurements(stats, ["X_cond", "Z_cond", "X", "SX"],
                                             np.random.default_rng(0))
    x_cond, z_cond, x_prod, sx = Y
    assert np.all(np.abs(x_cond) <= 1) and np.all(np.abs(z_cond) <= 1)
    assert np.allclose(x_prod, sx * x_cond)         # same <X>* draw, same SX*
    with pytest.raises(KeyError):
        bootstrap_post_selected_measurements(stats, ["W_cond"], np.random.default_rng(0))


# ---------------------------------------------------------------------------
# pipeline plumbing on synthetic records (lnS tail and pencil gamma0)
# ---------------------------------------------------------------------------

def _records(lams, t, rng, n_shots=4000):
    recs = []
    for lam in lams:
        S, X, Z = _trajectory(_liouvillian(lam, gamma2=0.05), t)
        stats = {
            "data_mode": "post_selected",
            "survival": S, "N_survival": 6 * n_shots,
            "X": X, "Z": Z,
            "N_X": np.maximum((4 * n_shots * S).astype(int), 1),
            "N_Z": np.maximum((2 * n_shots * S).astype(int), 1),
        }
        meta = {"bootstrap_stats": stats, "scan_number": 99}
        for key, sig in (("S", S), ("X_cond", X), ("Z_cond", Z)):
            recs.append(ObservableRecord(lambda_val=float(lam), t=t, signal=sig,
                                         observable_key=key, metadata=meta))
    return recs


def test_pipeline_lns_tail_with_bootstrap():
    lams = np.round(np.arange(0.9, 1.26, 0.05), 3)
    t = np.linspace(0, 40, 161)
    recs = _records(lams, t, np.random.default_rng(0))
    res = SpiralKsiPipeline(
        recs, None, gamma0_source="lnS_tail", t_fit_min=10.0, min_survival=1e-9,
        n_boot=8, seed=3, n_jobs=1, crossing_method="quadfit", n_points=6,
    ).run()
    assert res.lambda_values.shape == lams.shape
    assert np.all(np.isfinite(res.gamma_a)) and np.all(np.isfinite(res.gamma0))
    assert res.ksi_lower.shape == lams.shape and res.crossing_samples.shape == (8,)
    # dephasing lowers lambda_SF below the ideal 2/sqrt(3)
    assert res.lambda_ksi_zero is not None and 0.95 < res.lambda_ksi_zero < 1.16
    assert res.ksi_label.startswith("$")


def test_pipeline_mpm_gamma0_pairs_draws():
    lams = np.round(np.arange(0.9, 1.21, 0.1), 3)
    t = np.linspace(0, 40, 161)
    recs = _records(lams, t, np.random.default_rng(1))
    # fake pencil result: slot 2 = slow pole with decay gamma0, pair at slots 0/1
    n_boot = 5
    dec = np.zeros((len(lams), 3)); frq = np.zeros((len(lams), 3))
    samples = np.zeros((len(lams), n_boot, 3), complex)
    for i, lam in enumerate(lams):
        ev = np.linalg.eigvals(_liouvillian(lam, gamma2=0.05))
        ev = ev[np.argsort(ev.real)]
        order = [0, 1, 2]
        dec[i] = -ev[order].real; frq[i] = ev[order].imag
        samples[i] = ev[order][None, :] * (1 + 0.01 * np.arange(n_boot))[:, None]
    poles = SimpleNamespace(
        decay_rates=dec, decay_rates_lower=0.01 * dec, decay_rates_upper=0.01 * dec,
        frequencies=frq, raw_samples=SimpleNamespace(samples=samples, lambda_values=lams),
    )
    result = SimpleNamespace(lambda_values=lams, poles=poles)
    res = SpiralKsiPipeline(
        recs, result, gamma0_source="mpm_slow_pole", t_fit_min=10.0, min_survival=1e-9,
        n_boot=n_boot, seed=0, n_jobs=1, crossing_method="first",
    ).run()
    assert np.allclose(res.gamma0, dec[:, 2])
    assert res.gamma0_source == "mpm_slow_pole"
    assert res.ksi_lower.shape == lams.shape
    with pytest.raises(ValueError):
        SpiralKsiPipeline(recs, None, gamma0_source="mpm_slow_pole")
