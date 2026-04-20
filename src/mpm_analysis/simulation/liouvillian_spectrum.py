"""
liouvillian_spectrum.py
-----------------------
Compute eigenvalues of the 5×5 Liouvillian matrix

    L(λ, k) =
    ⎛ -2λΩ      iΩ/2    -iΩ/2    0       2λΩe^{ik} ⎞
    ⎜  iΩ/2  -λΩ-2γz      0    -iΩ/2       0        ⎟
    ⎜ -iΩ/2      0    -λΩ-2γz   iΩ/2       0        ⎟
    ⎜   0      -iΩ/2    iΩ/2     0          0        ⎟
    ⎝  2λΩ       0        0       0         -2λ      ⎠

as a function of λ at fixed k (default k=π).
"""
from __future__ import annotations

import numpy as np

from mpm_analysis.analysis.pole_sorting import sort_poles_canonical


def build_liouvillian(lam: float, Omega: float, gamma_z: float, k: float = np.pi) -> np.ndarray:
    """Return the 5×5 Liouvillian matrix as a complex numpy array."""
    eik = np.exp(1j * k)
    h = 1j * Omega / 2
    return np.array(
        [
            [-2 * lam * Omega,  h,              -h,             0,   2 * lam * Omega * eik],
            [ h,  -lam * Omega - 2 * gamma_z,   0,             -h,   0                    ],
            [-h,               0,  -lam * Omega - 2 * gamma_z,  h,   0                    ],
            [ 0,              -h,               h,              0,   0                    ],
            [ 2 * lam * Omega,  0,               0,             0,  -2 * lam             ],
        ],
        dtype=complex,
    )


class LiouvillianSpectrum:
    """Scan eigenvalues of the 5×5 Liouvillian over a range of λ values.

    Parameters
    ----------
    Omega:
        Rabi frequency Ω (rad/µs).  Default matches PostSelectedDynamicsSimulator.
    gamma_z:
        Dephasing rate γ_z = 1/T_D (µs⁻¹).  Default: 1/21.0.
    k:
        Quasi-momentum (radians).  Default: π.
    lambda_min, lambda_max, n_lambda:
        Scan range and resolution.
    """

    def __init__(
        self,
        Omega: float = 0.628,
        gamma_z: float = 1.0 / 66,
        k: float = np.pi,
        lambda_min: float = 0.6,
        lambda_max: float = 1.8,
        n_lambda: int = 200,
    ) -> None:
        self.Omega = Omega
        self.gamma_z = gamma_z
        self.k = k
        self.lambda_vals = np.linspace(lambda_min, lambda_max, n_lambda)

    def compute(self) -> np.ndarray:
        """Compute eigenvalues for all λ values.

        Returns
        -------
        np.ndarray, shape (n_lambda, 5), dtype complex
            Eigenvalues sorted by real part (most negative first) at each λ.
        """
        eigs = np.empty((len(self.lambda_vals), 5), dtype=complex)
        for i, lam in enumerate(self.lambda_vals):
            L = build_liouvillian(lam, self.Omega, self.gamma_z, self.k)
            ev = np.linalg.eigvals(L)
            eigs[i] = sort_poles_canonical(ev)
        return eigs

    def plot(self, ax_real=None, ax_imag=None):
        """Plot real and imaginary parts of eigenvalues vs λ.

        Parameters
        ----------
        ax_real, ax_imag:
            Matplotlib axes.  If None, a new figure with two subplots is created.

        Returns
        -------
        fig, (ax_real, ax_imag)
        """
        import matplotlib.pyplot as plt

        eigs = self.compute()
        lams = self.lambda_vals

        if ax_real is None or ax_imag is None:
            fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig = ax_real.get_figure()

        for j in range(5):
            ax_real.plot(lams, eigs[:, j].real, lw=1.2)
            ax_imag.plot(lams, eigs[:, j].imag, lw=1.2)

        ax_real.set_xlabel("λ")
        ax_real.set_ylabel("Re(eigenvalue)")
        ax_real.set_title(f"Decay rates  (k = {self.k/np.pi:.2g}π)")
        ax_real.axhline(0, color="k", lw=0.5, ls="--")
        ax_real.grid(True, alpha=0.3)

        ax_imag.set_xlabel("λ")
        ax_imag.set_ylabel("Im(eigenvalue)")
        ax_imag.set_title(f"Frequencies  (k = {self.k/np.pi:.2g}π)")
        ax_imag.axhline(0, color="k", lw=0.5, ls="--")
        ax_imag.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig, (ax_real, ax_imag)