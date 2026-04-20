"""
simulation
----------
Forward simulators for the quantum Zeno effect.

``sympy`` is imported lazily inside simulator ``__init__`` methods so that
``import mpm_analysis`` stays fast even when sympy is not needed.

Sub-modules
~~~~~~~~~~~
postselected_dynamics  — 3×3 SymPy EOM (post-selected tomography model)
ensemble_average       — 4×4 SymPy EOM (ensemble-average model)
liouvillian_spectrum   — 5×5 Liouvillian eigenvalue scan vs λ at fixed k
noise/                 — Modular noise injection for benchmark testing
"""
from mpm_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator
from mpm_analysis.simulation.ensemble_average import EnsembleAverageDynamics
from mpm_analysis.simulation.liouvillian_spectrum import LiouvillianSpectrum

__all__ = ["PostSelectedDynamicsSimulator", "EnsembleAverageDynamics", "LiouvillianSpectrum"]
