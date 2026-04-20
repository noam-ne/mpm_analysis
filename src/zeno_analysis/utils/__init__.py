"""
utils
-----
Lightweight helpers with no heavy deps.

    from zeno_analysis.utils import win_long_path, save_figure_pdf_svg
    from zeno_analysis.utils.physics import estimate_lambda_zeno, gamma_m
"""
from zeno_analysis.utils.windows_paths import win_long_path, save_figure_pdf_svg
from zeno_analysis.utils.lab_paths import get_lab_path, get_dropbox_root

__all__ = ["win_long_path", "save_figure_pdf_svg", "get_lab_path", "get_dropbox_root"]
