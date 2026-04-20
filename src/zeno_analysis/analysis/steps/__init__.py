"""
analysis/steps
--------------
Pluggable pipeline steps implementing the Strategy pattern.

Usage
~~~~~
    from zeno_analysis.analysis.steps import MPMStep, BootstrapStep, RefineStep

    steps = [MPMStep(order=3), BootstrapStep(n_boot=2000)]

Pass a custom ``steps`` list to ``MPMPipeline`` to change the analysis
sequence without subclassing.
"""
from zeno_analysis.analysis.steps.base import AnalysisStep
from zeno_analysis.analysis.steps.mpm_step import MPMStep
from zeno_analysis.analysis.steps.bootstrap_step import BootstrapStep
from zeno_analysis.analysis.steps.refine_step import RefineStep
from zeno_analysis.analysis.steps.mpm_rrha_step import MPMRRHAStep

__all__ = ["AnalysisStep", "MPMStep", "BootstrapStep", "RefineStep", "MPMRRHAStep"]
