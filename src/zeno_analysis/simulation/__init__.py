"""
simulation
----------
Forward simulators for the quantum Zeno effect.

``sympy`` is imported lazily inside simulator ``__init__`` methods so that
``import zeno_analysis`` stays fast even when sympy is not needed.

Sub-modules
~~~~~~~~~~~
postselected_dynamics  — 3×3 SymPy EOM (post-selected tomography model)
ensemble_average       — 4×4 SymPy EOM (ensemble-average model)
noise/                 — Modular noise injection for benchmark testing
"""
from zeno_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator
from zeno_analysis.simulation.ensemble_average import EnsembleAverageDynamics

__all__ = ["PostSelectedDynamicsSimulator", "EnsembleAverageDynamics"]
