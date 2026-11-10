"""Error bars of the paper transition figure (appendix_bootstrap, first_transition_main):
- Im / Re splitting bars = the attempts' 16-84 % range of the discriminant (not symmetrized), converted
- the Delta points of the discriminant layout keep the symmetrized bar the line fit uses
- error bars faded, markers solid
- a decay rate outside its own 16-84 % band: the bar ends at the point, no mirroring
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpm_analysis.analysis.bootstrap import _LambdaBootstrapResult, aggregate_bootstrap
from mpm_analysis.analysis.ksi_crossing import discriminant_from_poles
from mpm_analysis.analysis.steps import BootstrapStep, MPMStep
from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline
from mpm_analysis.plotting.paper_figures.appendix_bootstrap import (
    fade_error_bars,
    plot_appendix_bootstrap,
)
from mpm_analysis.plotting.paper_figures.first_transition_main import plot_first_transition_figure


def _im(d):
    return np.sqrt(np.clip(-np.asarray(d, float), 0.0, None)) / 2


def _re(d):
    return np.sqrt(np.clip(np.asarray(d, float), 0.0, None)) / 2


# ---------------------------------------------------------------------------
# Fading
# ---------------------------------------------------------------------------

def test_fade_error_bars_leaves_the_markers_solid():
    fig, ax = plt.subplots()
    faded = ax.errorbar([1, 2], [1, 2], yerr=[0.1, 0.1], capsize=2, marker="o")
    fade_error_bars(faded, 0.4)
    line, caps, bars = faded.lines
    assert line.get_alpha() is None
    assert all(c.get_alpha() == 0.4 for c in caps) and all(b.get_alpha() == 0.4 for b in bars)

    solid = ax.errorbar([1, 2], [1, 2], yerr=[0.1, 0.1], capsize=2, marker="o")
    fade_error_bars(solid, 1.0)
    assert all(a.get_alpha() is None for a in (*solid.lines[1], *solid.lines[2]))
    plt.close(fig)


# ---------------------------------------------------------------------------
# A decay rate outside its own band
# ---------------------------------------------------------------------------

def test_decay_bar_ends_at_a_point_outside_its_band():
    n = 400
    fast, slow = np.linspace(1.0, 2.0, n), np.linspace(0.5, 0.6, n)
    real = -np.column_stack([fast, slow])
    results = [_LambdaBootstrapResult(1.0, real, np.zeros_like(real))]

    above = aggregate_bootstrap(results, observable_key="S", center_poles=[np.array([-1.5 + 0j, -0.9 + 0j])])
    p16, p84 = np.percentile(slow, [16, 84])
    assert above.decay_rates_lower[0, 1] == pytest.approx(0.9 - p16)
    assert above.decay_rates_upper[0, 1] == 0.0

    below = aggregate_bootstrap(results, observable_key="S", center_poles=[np.array([-1.5 + 0j, -0.3 + 0j])])
    assert below.decay_rates_lower[0, 1] == 0.0
    assert below.decay_rates_upper[0, 1] == pytest.approx(p84 - 0.3)

    # A point inside its band keeps both arms.
    p16, p84 = np.percentile(fast, [16, 84])
    assert above.decay_rates_lower[0, 0] == pytest.approx(1.5 - p16)
    assert above.decay_rates_upper[0, 0] == pytest.approx(p84 - 1.5)


# ---------------------------------------------------------------------------
# The whole figure
# ---------------------------------------------------------------------------

def _curve(t, delta):
    """A survival curve with pair discriminant ``delta`` (around -1.2 /us) and a slow pole at -0.25 /us."""
    a3, s3, m = 1.56, -0.25, -1.2
    beta = -a3 * s3 / (1 - a3) - m                       # S'(0) = 0
    q = np.sqrt(abs(delta)) / 2
    if delta < 0:
        pair = np.cos(q * t) + beta * np.sin(q * t) / q
    else:
        pair = np.cosh(q * t) + beta * np.sinh(q * t) / q
    return a3 * np.exp(s3 * t) + (1 - a3) * np.exp(m * t) * pair


@pytest.fixture(scope="module")
def result():
    t = np.arange(63) * 0.2
    rng = np.random.default_rng(11)
    n_shots = 300_000
    records = []
    for lam, delta in zip((0.9, 1.1, 1.3, 1.5, 1.7), (-0.8, -0.4, -0.05, 0.3, 0.9)):
        y = rng.binomial(n_shots, np.clip(_curve(t, delta), 1e-6, 1.0)) / n_shots
        y[0] = 1.0
        stats = {"data_mode": "post_selected", "survival": y, "N_survival": n_shots}
        records.append(ObservableRecord(lambda_val=lam, t=t, signal=y, observable_key="survival",
                                        metadata={"bootstrap_stats": stats}))
    steps = [MPMStep(order=3, observable_key=["survival"], plot_svd=False),
             BootstrapStep(n_boot=400, mode="data", n_jobs=1, seed=3)]
    res = MPMPipeline.from_records(records, steps=steps, data_mode="post_selected",
                                   observables=["survival"]).run()
    res.parameters["omega_DG"] = 2 * np.pi * 0.1         # so the figure's normalization is tested too
    return res


def _bars(ax):
    """Per errorbar container of ``ax``: (lower ends, upper ends, data line, cap artists, bar artists)."""
    out = []
    for c in ax.containers:
        line, caps, bars = c.lines
        seg = bars[0].get_segments()
        out.append((np.array([s[:, 1].min() for s in seg]), np.array([s[:, 1].max() for s in seg]),
                    line, caps, bars))
    return out


def test_splitting_bars_are_the_unsymmetrized_discriminant_range(result):
    """Panels B/C: both bars are the attempts' own 16-84 % of Delta through the square root, measured points."""
    om = result.omega_s
    central = -result.poles.decay_rates + 1j * result.poles.frequencies
    delta = discriminant_from_poles(central) / om**2
    draws = discriminant_from_poles(np.asarray(result.poles.raw_samples.samples)) / om**2
    p16, p84 = np.nanpercentile(draws, [16, 84], axis=1)
    half = (p84 - p16) / 2
    assert np.any(np.abs((p84 + p16) / 2 - delta) > 0.02 * half)     # the range is not centred on the point

    with matplotlib.rc_context():
        fig = plot_appendix_bootstrap(result, lambda_max=2.0, splitting_fit="discriminant", error_alpha=0.4)
        _, ax_im, ax_re = fig.axes[:3]
        (im_lo, im_hi, im_line, *_), = _bars(ax_im)
        (re_lo, re_hi, re_line, *_), = _bars(ax_re)
        assert np.allclose(im_line.get_ydata(), _im(delta)) and np.allclose(re_line.get_ydata(), _re(delta))
        # (equal up to the percentile's interpolation between neighbouring attempts)
        tol = dict(rtol=0.01, atol=1e-3)
        assert np.allclose(im_lo, _im(p84), **tol) and np.allclose(im_hi, _im(p16), **tol)
        assert np.allclose(re_lo, _re(p16), **tol) and np.allclose(re_hi, _re(p84), **tol)
        # not the symmetrized bar of the line fit
        sym_top = _im(delta - half)
        assert not np.allclose(im_hi, sym_top, atol=1e-3)
        plt.close(fig)


def test_discriminant_layout_points_keep_the_symmetrized_bar(result):
    om = result.omega_s
    central = -result.poles.decay_rates + 1j * result.poles.frequencies
    delta = discriminant_from_poles(central) / om**2
    draws = discriminant_from_poles(np.asarray(result.poles.raw_samples.samples)) / om**2
    p16, p84 = np.nanpercentile(draws, [16, 84], axis=1)
    with matplotlib.rc_context():
        fig = plot_first_transition_figure(result, layout="discriminant", lambda_max=2.0, error_alpha=0.4)
        (lo, hi, *_), = _bars(fig.axes[1])
        assert np.allclose(lo, delta - (p84 - p16) / 2) and np.allclose(hi, delta + (p84 - p16) / 2)
        plt.close(fig)


def test_figure_bars_faded_markers_solid_in_every_panel(result):
    with matplotlib.rc_context():
        fig = plot_first_transition_figure(result, layout="sqrt", lambda_max=2.0, splitting_fit="discriminant",
                                           error_alpha=0.4)
        for ax in fig.axes[:3]:
            for _, _, line, caps, bars in _bars(ax):
                assert line.get_alpha() is None
                assert all(a.get_alpha() == 0.4 for a in (*caps, *bars))
        plt.close(fig)

        fig = plot_first_transition_figure(result, layout="discriminant", lambda_max=2.0, error_alpha=0.4)
        for ax in fig.axes[:2]:
            for _, _, line, caps, bars in _bars(ax):
                assert line.get_alpha() is None
                assert all(a.get_alpha() == 0.4 for a in (*caps, *bars))
        plt.close(fig)


def test_figure_default_is_solid(result):
    with matplotlib.rc_context():
        fig = plot_appendix_bootstrap(result, lambda_max=2.0, splitting_fit="discriminant")
        for ax in fig.axes[:3]:
            for _, _, _, caps, bars in _bars(ax):
                assert all(a.get_alpha() is None for a in (*caps, *bars))
        plt.close(fig)


def test_decay_panel_keeps_each_poles_own_band(result):
    om = result.omega_s
    p = result.poles
    with matplotlib.rc_context():
        fig = plot_appendix_bootstrap(result, lambda_max=2.0, splitting_fit="discriminant", error_alpha=0.4)
        for k, (lo, hi, *_rest) in enumerate(_bars(fig.axes[0])):
            assert np.allclose(lo, (p.decay_rates - p.decay_rates_lower)[:, k] / om)
            assert np.allclose(hi, (p.decay_rates + p.decay_rates_upper)[:, k] / om)
        plt.close(fig)
