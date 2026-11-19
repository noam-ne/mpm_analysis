"""
lab_paths.py
------------
Lab paths, read from ``mpm_analysis/config/paths.json``.

The JSON holds two tables:

    dropbox_roots   hostname -> this machine's ``Quantum Circuits Lab`` Dropbox folder
    paths           name -> path under that folder

Usage::

    from mpm_analysis.utils.lab_paths import get_named_path

    POST_SELECTED_FOLDER = get_named_path("first_transition_data")
    SCAN_FOLDER = get_named_path("click_record_scans") / "scan_experiment_results_22"

A path is typed once, in the JSON, and used by name everywhere else.
``get_lab_path`` builds a path under the Dropbox root from typed parts; use it
only for a one-off path that has no name.

Adding a new computer
---------------------
1. Find your hostname:  python -c "import socket; print(socket.gethostname())"
2. Add a line to ``dropbox_roots`` in ``config/paths.json`` with the path to
   your local ``Quantum Circuits Lab`` Dropbox folder.
"""
from __future__ import annotations

import socket
from pathlib import Path

from mpm_analysis.config import load_config

_CONFIG = load_config("paths")

# Map hostname → local path to "Quantum Circuits Lab" Dropbox folder
_DROPBOX_ROOTS: dict[str, Path] = {
    host: Path(root) for host, root in _CONFIG["dropbox_roots"].items()
}
# Map name → path under the Dropbox root (the same on every machine)
_NAMED_PATHS: dict[str, str] = dict(_CONFIG["paths"])

# Sub-paths that are the same on every machine
CAVITY_STR = "CavityRecAl6"
CHIP_STR   = "Chip_TransSap46_5"


def get_dropbox_root() -> Path:
    """Return the Dropbox root for this machine.

    Raises
    ------
    ValueError
        If the current hostname is not in ``dropbox_roots`` of
        ``config/paths.json``.  Add it there to fix the error.
    """
    hostname = socket.gethostname()
    root = _DROPBOX_ROOTS.get(hostname)
    if root is None:
        raise ValueError(
            f"Hostname '{hostname}' is not in dropbox_roots of "
            f"mpm_analysis/src/mpm_analysis/config/paths.json. Add it there. "
            f"Known hosts: {list(_DROPBOX_ROOTS)}. "
            f"To find your hostname: python -c \"import socket; print(socket.gethostname())\""
        )
    return root


def get_named_path(name: str) -> Path:
    """The path called ``name`` in ``config/paths.json``, on this machine.

    Raises
    ------
    KeyError
        If ``name`` is not in the JSON's ``paths`` table.
    """
    if name not in _NAMED_PATHS:
        raise KeyError(
            f"'{name}' is not a path name in mpm_analysis/src/mpm_analysis/config/paths.json. "
            f"Known names: {sorted(_NAMED_PATHS)}."
        )
    return get_dropbox_root() / _NAMED_PATHS[name]


def get_lab_path(*parts: str) -> Path:
    """Build a path under the Dropbox root for this machine.

    Parameters
    ----------
    *parts:
        Path components relative to the ``Quantum Circuits Lab`` folder.
        Can be passed as one joined string or as separate segments.

    Returns
    -------
    Path
        Absolute path on this machine.  The path may not yet exist if the
        Dropbox hasn't synced it yet.

    Examples
    --------
    >>> from mpm_analysis.utils.lab_paths import get_lab_path
    >>> get_lab_path(r"Experiments\\Zeno", "CavityRecAl6", "scan50")
    WindowsPath('C:/Users/admin/Weizmann .../scan50')
    """
    return get_dropbox_root().joinpath(*parts)
