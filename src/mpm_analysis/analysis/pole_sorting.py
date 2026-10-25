"""
pole_sorting.py
---------------
Single canonical pole-sorting routine shared by every component that
produces or consumes complex MPM poles.

Rule:
    Complex-conjugate pairs first (adjacent, −Im before +Im, pairs ordered
    by their common real part), then the real poles by real part ascending,
    NaN last.  The slot meaning is therefore the same on both sides of an
    exceptional point: for order 3, slots 0 and 1 hold the EP pair — the
    conjugate pair while it is closed, the two most strongly decaying real
    poles once it has split — and slot 2 the slow (least decaying) pole.

Why one canonical function, applied per pole set?
    A per-λ rule cannot swap labels between neighbouring λ or between a
    draw and the measured fit, which is what happened when the slots were
    chained across λ by nearest-neighbour matching: at the EP the matching
    is ambiguous, and once two slots had swapped every downstream quantity
    built from slot indices (splittings, the sqrt fit, ξ) was wrong there.
    Everything goes through this function so the ordering is consistent.
"""
from __future__ import annotations

import numpy as np


def sort_poles_canonical(poles: np.ndarray, *, pair_tol: float = 2e-2) -> np.ndarray:
    """Sort complex poles by real part ascending, conjugate pairs adjacent.

    Parameters
    ----------
    poles:
        1-D complex array of length ``order``.
    pair_tol:
        Two poles are a conjugate pair when ``|p_i - conj(p_j)|`` is below
        ``pair_tol * (1 + |p_i|)``; both then sort on their mean real part,
        so a pair is never split by small differences in ``Re`` (exact for
        the pencil on real data; a refinement without the conjugate
        constraint, or noise added to a pair, leaves them a few % apart).

    Returns
    -------
    np.ndarray
        Same values, re-ordered (NaN last).

    Examples
    --------
    >>> sort_poles_canonical(np.array([-1+2j, -3+0j, -1-2j]))
    array([-3.+0.j, -1.-2.j, -1.+2.j])
    """
    poles = np.asarray(poles, dtype=complex)
    return poles[canonical_order(poles, pair_tol=pair_tol)]


def canonical_order(poles: np.ndarray, *, pair_tol: float = 2e-2) -> np.ndarray:
    """Permutation ``idx`` with ``poles[idx]`` in canonical order (see above)."""
    poles = np.asarray(poles, dtype=complex)
    n = len(poles)
    finite = np.isfinite(poles)
    key_re = np.where(finite, poles.real, np.inf)
    key_im = np.where(finite, poles.imag, np.inf)
    paired = np.zeros(n, dtype=bool)
    for i in range(n):
        if paired[i] or not finite[i] or abs(poles[i].imag) < 1e-12:
            continue
        d = np.abs(poles - np.conj(poles[i]))
        d[i] = np.inf
        d[paired | ~finite] = np.inf
        j = int(np.argmin(d))
        if d[j] <= pair_tol * (1.0 + abs(poles[i])):
            key_re[i] = key_re[j] = 0.5 * (poles[i].real + poles[j].real)
            paired[i] = paired[j] = True
    # Conjugate pairs first (by Re), then real poles (by Re), NaN last: the
    # EP pair is slots 0, 1 whenever it exists, and once it has split the
    # two most strongly decaying real poles take its place.
    group = np.where(paired, 0, np.where(finite, 1, 2))
    return np.lexsort((key_im, key_re, group))


def aliased_pole_mask(poles: np.ndarray, dt: float, *, frac: float = 0.9) -> np.ndarray:
    """Which poles sit at the sampling Nyquist frequency ``|Im| ≈ π/dt``.

    On a noisy trace the pencil sometimes returns a pole whose frequency is
    the Nyquist limit of the time grid (no conjugate partner, ``Im = π/dt``
    exactly).  That is a numerical artefact of the sampling, not a mode of
    the system, and such a pole set must not enter the error bars.  Any
    ``|Im| >= frac·π/dt`` counts (the physical frequencies are far below).

    ``poles`` may have any leading shape; the mask has the same shape.
    """
    nyq = np.pi / float(dt)
    im = np.abs(np.asarray(poles, dtype=complex).imag)
    return np.isfinite(im) & (im >= frac * nyq)


def match_poles_hungarian(
    poles: np.ndarray,
    reference: np.ndarray,
    *,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Permute ``poles`` to best match ``reference`` (Hungarian assignment).

    Minimises the total complex distance ``sum_i |poles[perm[i]] - reference[i]|``
    via ``scipy.optimize.linear_sum_assignment``.  Used to keep pole indices
    meaning the *same branch* across lambda and across bootstrap draws, where
    an independent canonical re-sort would swap labels near the EP.

    NaN poles (padding from ``enforce_pole_count``) are assigned last via a
    large finite cost, so they end up matched to whatever reference slots are
    left over.

    Parameters
    ----------
    poles:
        1-D complex array to be re-ordered.
    reference:
        1-D complex array of the same length defining the target order.
    return_indices:
        If ``True``, also return the permutation ``idx`` such that the result
        equals ``poles[idx]`` (useful to permute amplitudes alongside).

    Returns
    -------
    np.ndarray
        ``poles`` permuted so element ``i`` is the best match of ``reference[i]``
        (with ``(poles_matched, idx)`` when ``return_indices=True``).
    """
    from scipy.optimize import linear_sum_assignment

    poles = np.asarray(poles, dtype=complex)
    reference = np.asarray(reference, dtype=complex)
    if len(poles) != len(reference):
        poles = enforce_pole_count(poles, len(reference))

    cost = np.abs(poles[None, :] - reference[:, None])  # (ref, pole)
    # NaN in either side → large finite cost so the assignment stays solvable
    finite = cost[np.isfinite(cost)]
    big = (finite.max() * 1e3 + 1.0) if finite.size else 1.0
    cost = np.where(np.isfinite(cost), cost, big)
    _, col = linear_sum_assignment(cost)
    if return_indices:
        return poles[col], col
    return poles[col]


def track_branches_across_lambda(
    poles_list: list[np.ndarray],
    *,
    return_indices: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
    """Order poles consistently across a lambda scan (branch tracking).

    The first slice with finite poles is put in canonical order
    (``sort_poles_canonical``); every subsequent slice is Hungarian-matched to
    the previous finite slice, so index ``i`` follows one continuous branch
    through the scan (all-NaN slices are passed through and skipped as
    reference).

    Parameters
    ----------
    poles_list:
        One 1-D complex pole array per lambda, sorted by lambda.
    return_indices:
        If ``True``, also return the per-slice permutations (result ==
        ``poles_list[i][idx[i]]``), for permuting amplitudes alongside.

    Returns
    -------
    list[np.ndarray]
        Same arrays, re-ordered per slice (with the permutation list when
        ``return_indices=True``).
    """
    tracked: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    reference: np.ndarray | None = None
    for poles in poles_list:
        poles = np.asarray(poles, dtype=complex)
        if not np.any(np.isfinite(poles)):
            tracked.append(poles)
            indices.append(np.arange(len(poles)))
            continue
        if reference is None:
            idx = canonical_order(poles)
            aligned = poles[idx]
        else:
            aligned, idx = match_poles_hungarian(poles, reference, return_indices=True)
        tracked.append(aligned)
        indices.append(idx)
        reference = aligned
    if return_indices:
        return tracked, indices
    return tracked


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
