"""
pipeline
--------
High-level orchestration classes that wire together io + analysis + saving.

Classes
~~~~~~~
MPMPipeline            — Load data → run steps → save ZenoAnalysisResult
CriticalPointPipeline  — Load result JSON → fit sqrt model → save back
BenchmarkPipeline      — Simulator → noise → analysis → bias/variance report
"""
from zeno_analysis.pipeline.mpm_pipeline import MPMPipeline
from zeno_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline
from zeno_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline

__all__ = ["MPMPipeline", "CriticalPointPipeline", "BenchmarkPipeline"]
