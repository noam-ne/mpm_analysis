"""Write shots_per_lambda.csv into a post-selected scan folder.

    python run_files/write_shots_per_lambda.py [folder]

One row per λ file with the shot counts the data bootstrap uses for that
file — read back through ``PostSelectedLoader`` (``bootstrap_stats``), so the
table IS what the error bars are computed from — plus the runs/iterations the
file was merged from (``provenance.json`` of the folder, and of scan 3's
``combined_iterations`` for files copied from there).  Default folder:
``POST_SELECTED_FOLDER`` of run.py.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from mpm_analysis.io.experimental import PostSelectedLoader
from mpm_analysis.utils.windows_paths import win_long_path

_SCAN3_COMBINED = (r"scan_experiment_results_3\combined_iterations")


def _provenance(folder: Path) -> dict:
    """points of folder/provenance.json keyed by combined file name."""
    fp = folder / "provenance.json"
    try:
        prov = json.load(open(win_long_path(fp), encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {pt["combined_file"]: pt for pt in prov.get("points", {}).values()}


def main(folder: Path) -> Path:
    points = _provenance(folder)
    # Files copied in from scan 3's merged iterations keep that provenance.
    points = {**_provenance(folder.parent / _SCAN3_COMBINED), **points}

    records = PostSelectedLoader(csv_path=False).load(folder, observables=["S"])
    rows = []
    for rec in records:
        st = rec.metadata["bootstrap_stats"]
        navg = int(st["N_survival"]) // 6
        pooled = st.get("N_attempts_S_pooled")
        pt = points.get(rec.source_file, {})
        sources = pt.get("sources", [])
        diff = (pt.get("parameters_differing_between_runs")
                or pt.get("parameters_differing_between_iterations") or {})
        per_run = diff.get("actual_navg_repeat_experiment")
        if per_run is None and sources:          # equal split when not listed
            per_run = [navg // len(sources)] * len(sources)
        rows.append({
            "Omega_BG0_kHz": rec.source_file.split("OBG0_")[1].split("_kHz")[0],
            "lambda": f"{rec.lambda_val:.6f}",
            "navg_per_setting": navg,
            "shots_per_point_S": int(st["N_survival"]),          # navg x 6 settings
            "pooled_attempts_t1": int(pooled[1]) if pooled is not None else "",
            "pooled_attempts_last": int(pooled[-1]) if pooled is not None else "",
            "n_source_runs": len(sources) if sources else 1,
            "navg_per_source_run": "+".join(str(int(n)) for n in per_run) if per_run else navg,
            "source_runs": ";".join(Path(s).parent.parent.name + "/" + Path(s).parent.name
                                    for s in sources) if sources else "single run",
            "file": rec.source_file,
        })
    rows.sort(key=lambda r: float(r["lambda"]))
    out = folder / "shots_per_lambda.csv"
    with open(win_long_path(out), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("runcfg", Path(__file__).with_name("run.py"))
        cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
        target = Path(cfg.POST_SELECTED_FOLDER)
    main(target)
