"""
analysis
--------
Stateless numerical routines for pole extraction and critical-point analysis.

Heavy imports (scipy, sympy) are deferred to function bodies so that
``import zeno_analysis`` stays fast.

Sub-modules
~~~~~~~~~~~
matrix_pencil   — Matrix Pencil Method (MPM) for pole extraction
bootstrap       — Bootstrap uncertainty quantification
critical_point  — Sqrt-fit for exceptional-point characterisation
steps/          — Pluggable AnalysisStep classes (Strategy pattern)
"""
