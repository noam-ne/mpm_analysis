"""
mpm_analysis
=============
Unified analysis and plotting package.

Submodules
----------
data_types  — typed data containers (no I/O, no heavy deps)
io          — file loading/saving (JSON, NPZ, experimental folders)
analysis    — stateless numerical routines (MPM, bootstrap, critical point)
simulation  — forward simulators + modular noise models
pipeline    — high-level orchestration classes
plotting    — exploratory and paper-quality figures
utils       — path helpers, physics constants
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mpm-analysis")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
