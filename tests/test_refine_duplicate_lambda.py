"""RefineStep with two records at the SAME λ (two scans side by side at one drive): each slice
must be refined on its own record's signal, paired by position with MPMStep's slices — a
λ-keyed lookup hands both slices the last record's data."""
import numpy as np

from mpm_analysis.analysis.steps.mpm_step import MPMStep
from mpm_analysis.analysis.steps.refine_step import RefineStep
from mpm_analysis.data_types.observables import ObservableRecord


def _record(rates, name):
    t = np.linspace(0.0, 12.0, 61)
    a = np.array([0.5, 0.3, 0.2])
    y = (a[:, None] * np.exp(-np.asarray(rates)[:, None] * t)).sum(axis=0)
    return ObservableRecord(lambda_val=1.474, t=t, signal=y, observable_key="S",
                            source_file=name, sigma=np.full_like(t, 1e-4))


def test_refine_pairs_duplicate_lambda_by_position():
    slow, fast = (0.25, 1.1, 2.0), (0.32, 1.4, 2.6)
    recs = [_record(slow, "scan3.json"), _record(fast, "scan45.json")]
    state = MPMStep(order=3, observable_key="S").process(recs, {})
    state = RefineStep(observable_key="S", conjugate_pairs=True).process(recs, state)
    refined = [np.sort(-np.real(p)) for p in state["poles_raw"]]
    assert np.allclose(refined[0], np.sort(slow), rtol=1e-3)
    assert np.allclose(refined[1], np.sort(fast), rtol=1e-3)
