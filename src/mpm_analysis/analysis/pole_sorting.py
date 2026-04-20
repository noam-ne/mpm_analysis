"""
pole_sorting.py
---------------
Single canonical pole-sorting routine shared by every component that
produces or consumes complex MPM poles.

Rule (matches ``object_oriented_poles.py`` reference implementation):
    Sort by real part ascending.  Ties (e.g. degenerate EP poles) are
    broken by imaginary part ascending.

Why one canonical function?
    Without a shared routine, each module (MPMStep, bootstrap, simulation,
    RRHA) may sort differently, causing index-swap artifacts when comparing
    poles across lambda or between clean/noisy runs.  Everything goes through
    this function so the ordering is guaranteed to be consistent.
"""
from __future__ import annotations

import numpy as np


def sort_poles_canonical(poles: np.ndarray) -> np.ndarray:
    """Sort complex poles by real part ascending (ties broken by imag ascending).

    Parameters
    ----------
    poles:
        1-D complex array of length ``order``.

    Returns
    -------
    np.ndarray
        Same values, re-ordered so ``poles.real`` is non-decreasing.

    Examples
    --------
    >>> sort_poles_canonical(np.array([-1+2j, -3+0j, -1-2j]))
    array([-3.+0.j, -1.-2.j, -1.+2.j])
    """
    poles = np.asarray(poles, dtype=complex)
    # Primary key: real part ascending; secondary: imaginary part ascending
    tol = 1e-2
    binned_real = np.round(poles.real / tol) * tol
    idx = np.lexsort((poles.imag, binned_real))
    return poles[idx]


def enforce_pole_count(poles: np.ndarray, order: int) -> np.ndarray:
    """Guarantee that ``poles`` has exactly ``order`` elements.

    If ``len(poles) == order``, returns as-is.
    If shorter, pads with ``NaN + NaN*j`` on the right.
    If longer, truncates to the first ``order`` elements.

    This should only be needed as a defensive guard; ``matrix_pencil_method``
    should always return exactly ``order`` poles.

    Parameters
    ----------
    poles:
        1-D complex array.
    order:
        Required number of poles.

    Returns
    -------
    np.ndarray
        Complex array of length ``order``.
    """
    poles = np.asarray(poles, dtype=complex)
    n = len(poles)
    if n == order:
        return poles
    if n > order:
        return poles[:order]
    # Pad with NaN
    out = np.full(order, complex(float("nan"), float("nan")), dtype=complex)
    out[:n] = poles
    return out
