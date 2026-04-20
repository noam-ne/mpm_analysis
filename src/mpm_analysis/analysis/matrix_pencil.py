"""
matrix_pencil.py
----------------
Matrix Pencil Method (MPM) for extracting complex exponential poles from a
time-domain signal.

"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd, eig, pinv


# ---------------------------------------------------------------------------
# Pencil parameter selection — spectral gap criterion (matches reference)
# ---------------------------------------------------------------------------

def _optimal_pencil_parameter(y: np.ndarray, order: int) -> int:
    """Return L that maximises S[order-1]/S[order] over L in [N//3, N//2].

    This is the spectral-gap criterion from ``object_oriented_poles.py``
    ``optimize_pencil_parameter``: the gap between the last "signal" singular
    value and the first "noise" singular value is largest at the optimal L.
    """
    N = len(y)
    L_min = max(order + 1, N // 3)
    L_max = min(N - order - 1, N // 2)

    best_L = L_min
    best_ratio = -1.0

    for L in range(L_min, L_max + 1):
        H = build_hankel_matrix(y, L)
        try:
            s = np.linalg.svd(H, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        if len(s) <= order:
            continue
        denom = s[order]
        if denom > 1e-12:
            ratio = s[order - 1] / denom
            if ratio > best_ratio:
                best_ratio = ratio
                best_L = L

    return best_L


# ---------------------------------------------------------------------------
# Core MPM
# ---------------------------------------------------------------------------

def build_hankel_matrix(y: np.ndarray, L: int) -> np.ndarray:
    """Build the Hankel data matrix Y from signal y with pencil parameter L.

    Parameters
    ----------
    y:
        1-D signal array of length N.
    L:
        Pencil parameter.  Rule of thumb: L ≈ N/3.

    Returns
    -------
    np.ndarray
        Hankel matrix of shape (N - L, L + 1).
    """
    N = len(y)
    rows = N - L
    cols = L + 1
    H = np.zeros((rows, cols), dtype=complex)
    for c in range(cols):
        H[:, c] = y[c: c + rows]
    return H


def matrix_pencil_method(
    t: np.ndarray,
    y: np.ndarray,
    order: int,
    *,
    L: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract ``order`` complex poles from signal ``y`` sampled at times ``t``.

    Parameters
    ----------
    t:
        Time axis (uniformly spaced, units: µs).
    y:
        Signal values.
    order:
        Number of poles to extract.
    L:
        Pencil parameter.  Default: ``len(y) // 3``.

    Returns
    -------
    poles:
        Complex poles z_k (1/µs).
        Shape (order,).
    singular_values:
        All singular values of the Hankel matrix (useful for model-order
        selection).  Shape (min(N-L, L+1),).
    L_used:
        The pencil parameter actually used.
    """

    N = len(y)
    if L is None:
        L = _optimal_pencil_parameter(y, order)
    L = max(order, min(L, N - order - 1))

    H = build_hankel_matrix(y, L)

    # Truncated SVD
    U, s, Vh = svd(H, full_matrices=False)
   
    V1 = Vh[:order, :].conj().T 

    # Split V1 into V1- and V1+ (remove last / first row)
    V1_minus = V1[:-1, :]
    V1_plus  = V1[1:, :]

    # Solve the pencil  V1+ = V1- Z  via explicit pseudoinverse
    Z = pinv(V1_minus) @ V1_plus
    z_discrete, _ = eig(Z)  # discrete-time poles

    # Convert from discrete to continuous time
    dt = float(t[1] - t[0])  # µs
    poles_continuous = np.log(z_discrete.astype(complex)) / dt

    return poles_continuous, s, L


def solve_linear_amplitudes(
    t: np.ndarray,
    y: np.ndarray,
    poles: np.ndarray,
) -> np.ndarray:
    """Solve for the complex amplitudes A_k such that y ≈ Σ_k A_k exp(poles_k t).

    Parameters
    ----------
    t:
        Time axis in µs.
    y:
        Signal values.
    poles:
        Complex poles (rad/µs).  Shape (M,).

    Returns
    -------
    np.ndarray
        Complex amplitudes A_k.  Shape (M,).
    """
    Phi = np.exp(np.outer(t, poles))  # (N, M)
    amplitudes, *_ = np.linalg.lstsq(Phi, y.astype(complex), rcond=None)
    return amplitudes


def reconstruct_signal(
    t: np.ndarray,
    poles: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Reconstruct y(t) ≈ Σ_k A_k exp(poles_k t) (real part taken)."""
    Phi = np.exp(np.outer(t, poles))
    return np.real(Phi @ amplitudes)


def refine_poles(
    t: np.ndarray,
    y: np.ndarray,
    poles_init: np.ndarray,
    *,
    bounds_real: tuple[float, float] | None = None,
    bounds_imag: tuple[float, float] | None = None,
    method: str = "trf",
) -> np.ndarray:
    """Nonlinear refinement of MPM poles via residual minimisation.

    Parameters
    ----------
    t, y:
        Time axis and signal.
    poles_init:
        Initial poles from ``matrix_pencil_method``.
    bounds_real:
        Optional ``(lower, upper)`` bounds on Re(pole).  Negative → decaying.
    bounds_imag:
        Optional ``(lower, upper)`` bounds on Im(pole).
    method:
        Scipy minimisation method passed to ``scipy.optimize.least_squares``.

    Returns
    -------
    np.ndarray
        Refined complex poles.  Shape same as ``poles_init``.
    """
    from scipy.optimize import least_squares  

    M = len(poles_init)
    # Pack real and imag parts into a flat parameter vector
    p0 = np.concatenate([poles_init.real, poles_init.imag])

    def residuals(p: np.ndarray) -> np.ndarray:
        poles_c = p[:M] + 1j * p[M:]
        y_hat = reconstruct_signal(t, poles_c, solve_linear_amplitudes(t, y, poles_c))
        return y - y_hat

    # Build bounds arrays
    lb = np.full(2 * M, -np.inf)
    ub = np.full(2 * M,  np.inf)
    if bounds_real is not None:
        lb[:M] = bounds_real[0]
        ub[:M] = bounds_real[1]
    if bounds_imag is not None:
        lb[M:] = bounds_imag[0]
        ub[M:] = bounds_imag[1]

    result = least_squares(residuals, p0, bounds=(lb, ub), method=method)
    p_opt = result.x
    return p_opt[:M] + 1j * p_opt[M:]


# ---------------------------------------------------------------------------
# Optimal pencil parameter
# ---------------------------------------------------------------------------

def select_pencil_parameter(
    y: np.ndarray,
    order: int,
    *,
    L_range: tuple[int, int] | None = None,
    criterion: str = "condition_number",
) -> int:
    """Select the pencil parameter L that minimises a criterion.

    Parameters
    ----------
    y:
        Signal array of length N.
    order:
        Model order.
    L_range:
        Search range ``(L_min, L_max)``.  Default: ``(order+1, N//2)``.
    criterion:
        ``"condition_number"`` — minimise cond(H), or
        ``"residual"``         — minimise reconstruction residual (slower).

    Returns
    -------
    int
        Optimal L value.
    """
    N = len(y)
    t_dummy = np.arange(N, dtype=float)  # dummy for residual criterion
    L_min, L_max = L_range or (order + 1, N // 2)

    best_L = L_min
    best_score = np.inf

    for L in range(L_min, L_max + 1):
        H = build_hankel_matrix(y, L)
        if criterion == "condition_number":
            s = np.linalg.svd(H, compute_uv=False)
            score = s[0] / (s[order - 1] + 1e-15)
        elif criterion == "residual":
            poles, _, _ = matrix_pencil_method(t_dummy, y, order, L=L)
            amps = solve_linear_amplitudes(t_dummy, y, poles)
            y_hat = reconstruct_signal(t_dummy, poles, amps)
            score = float(np.mean((y - y_hat) ** 2))
        else:
            raise ValueError(f"Unknown criterion '{criterion}'.")
        if score < best_score:
            best_score = score
            best_L = L

    return best_L

def svd_reduction(Y, order=3):
        U,S,Vh = svd(Y, full_matrices=False)
        U_red = U[:, :order]
        S_red = np.diag(S[:order])
        Vh_red = Vh[:order, :]
        Y_reduced = U_red @ S_red @ Vh_red
        return Y_reduced, S[:2*order]
    
def hankel_approximation(reduced_matrix):
        rows, cols = reduced_matrix.shape
        H = np.zeros_like(reduced_matrix, dtype=reduced_matrix.dtype)

        dmax = rows + cols - 1
        sums = np.zeros(dmax, dtype=np.complex128 if np.iscomplexobj(reduced_matrix) else np.float64)
        counts = np.zeros(dmax, dtype=np.int64)

        # accumulate sums/counts per anti-diagonal (i+j constant)
        for i in range(rows):
            for j in range(cols):
                k = i + j
                sums[k] += reduced_matrix[i, j]
                counts[k] += 1

        means = sums / counts

        # fill Hankel matrix with those means - this is the approximation of the original matrix by a Hankel matrix
        for i in range(rows):
            for j in range(cols):
                H[i, j] = means[i + j]

        return H

def rrha_stop_criterion(A,order, rtol=1e-2):
        s = svd(A, compute_uv=False)
        
        tail_med = np.median(s[order:2*order + 1]) 
        tail_med = np.log(tail_med) # 5x safety factor
        rtol_drop = tail_med
        return np.log((s[order] / s[order - 1])) > rtol_drop


def get_approximated_reduced_hankel(matrix, order=3, iters=3,rtol=1e-3):

        rrha_matrix = matrix.copy()
        
        for i in range(iters):
            svd_reduction,S = svd_reduction(rrha_matrix, order=order)
            rrha_matrix = hankel_approximation(svd_reduction)
            reached_rank = rrha_stop_criterion(rrha_matrix,order=order, rtol=rtol)
            if reached_rank:
                break

        return rrha_matrix, S