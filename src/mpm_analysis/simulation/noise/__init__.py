"""
simulation/noise
----------------
Modular noise injection for benchmark testing.

All noise models are stateless: they take a list of ``ObservableRecord``
objects and return a new list with noise applied.  Combine them with
``CompositeNoise``.

Usage
~~~~~
    from mpm_analysis.simulation.noise import (
        ShotNoise, GaussianNoise, CompositeNoise
    )

    noisy = ShotNoise(n_shots=1000).apply(clean_records)
    noisy = CompositeNoise([ShotNoise(n_shots=500), GaussianNoise(sigma=0.01)]).apply(clean_records)
"""
from mpm_analysis.simulation.noise.base import NoiseModel
from mpm_analysis.simulation.noise.shot_noise import ShotNoise
from mpm_analysis.simulation.noise.composite import CompositeNoise

__all__ = [
    "NoiseModel",
    "ShotNoise",
    "CompositeNoise",
]
