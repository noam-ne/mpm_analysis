"""
paper_figures
-------------
Publication-quality paper figure functions.

Functions
~~~~~~~~~
make_third_transition_figure  — Panel A + Panel B main-text figure
plot_appendix_bootstrap       — Appendix bootstrap analysis figure
"""
from zeno_analysis.plotting.paper_figures.third_transition_main import (
    make_third_transition_figure,
)
from zeno_analysis.plotting.paper_figures.appendix_bootstrap import (
    plot_appendix_bootstrap,
)

__all__ = ["make_third_transition_figure", "plot_appendix_bootstrap"]
