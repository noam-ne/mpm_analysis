"""
config
------
The package's two settings files, kept as JSON so they are easy to read, change
and track:

    paths.json            every lab path: the Dropbox folder of each machine and
                          the named folders and files under it.
                          Read through ``mpm_analysis.utils.lab_paths``
                          (``get_named_path("first_transition_data")``).
    plotting_params.json  the plot style: DPI, font size, line and marker
                          widths, column widths, colours.
                          Read through ``mpm_analysis.plotting._style``
                          (``from mpm_analysis.plotting._style import DPI``).

Code outside those two modules does not open the files itself.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_config(name: str) -> dict:
    """The contents of ``config/<name>.json`` (``"paths"`` or ``"plotting_params"``)."""
    with open(Path(__file__).with_name(f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)
