"""
io
--
All file I/O for zeno_analysis.

Sub-modules
~~~~~~~~~~~
analysis_json    — load / save ``ZenoAnalysisResult`` ↔ JSON
experimental/    — load experimental data folders → ``list[ObservableRecord]``
simulation       — load analytical / MC NPZ files → ``list[ObservableRecord]``
file_naming      — auto-generate and parse result filenames
"""
from zeno_analysis.io.analysis_json import (
    load_analysis_result,
    save_analysis_result,
    augment_json_with_sqrt_fit,
)
from zeno_analysis.io.file_naming import build_result_filename, save_result

__all__ = [
    "load_analysis_result",
    "save_analysis_result",
    "augment_json_with_sqrt_fit",
    "build_result_filename",
    "save_result",
]
