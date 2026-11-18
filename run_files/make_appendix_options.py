"""
make_appendix_options.py
------------------------
Layout options for the first-transition appendix figure (run.py step 6,
layout "sqrt": a decay rates, b Im / c Re splitting): step 3's Δ line fit
added as an inset, as a fourth panel or — the final choice, "disc_c" — in
place of panel c's Re splitting, over all of step 6's λ.  The options drawn
come from ONE run, so they compare like for like.

run.py's step 6 now draws the final choice itself (panel_c="discriminant",
with its ticks, y labels on one baseline and panel letters at one height),
so "current" IS that figure.  The inset and four-panel options were laid out
on the Re panel c (step 6 with panel_c="real", real_split_axis_ticks
[0, 0.5, 1]).

The discriminant view
    y      (Δe12)² / Ω_S² = Δ12 / (4 Ω_S²).  Panels b and c plot HALF the pole
           difference, Δe12 = (e1 − e2)/2, while step 3's discriminant is
           Δ12 = c1² − 4·c0 = (e1 − e2)²: the view is the square of what b and
           c plot, −Im(Δe12)² before the EP and +Re(Δe12)² after it.
    data   step 3's measured Δ12 ± the σ its line was weighted with, its line
           over the fitted λ, λ_EP = the line's zero (red dashed) and
           Δ = 0 (grey), over DISC_LAMBDA_RANGE (inset, panel d) or all of
           step 6's λ (panel c).
    style  panels b/c's: steelblue points, error bars at step 6's opacity,
           black dashed fit; font, line widths and ticks of the other panels.

Data: run.py's CURRENT first-transition configuration.  The settings of its
first_* clauses (1 load, 1c SVD cut when on, 2 pencil + bootstrap, 3 Δ line,
6 figure) are read from run.py's main() — nothing is copied here — and steps
1-3 are rerun (~10 s; nothing is written to run.py's OUTPUT_DIR).  Panels a-c
ARE run.py's step-6 figure (run_figure), ticks included; an option only
re-proportions or re-stacks them and adds the discriminant view.

Output: a new folder per run in run.py's PAPER_FIGURES_DIR,
    appendix_options_<observables>_order<n>_<data mode>_<UTC stamp>/
        opt<i>_<layout>.{pdf,svg,png}, one per entry of OPTIONS
Before saving, every option is checked: an inset inside its host's frame and
clear of the host's points, bars and lines; no text on another panel's text
or frame; the legend clear of the data.  A problem is printed (the figure is
still saved).

Edit the configuration section, then run:
    python make_appendix_options.py
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # one BLAS thread per bootstrap worker

import ast
import inspect
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from mpm_analysis.io.file_naming import build_figure_filename
from mpm_analysis.plotting._style import (
    LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH, LAMBDA_OBS_COLOR,
    add_panel_letter, make_ax_lines_wider,
)
from mpm_analysis.plotting.paper_figures.appendix_bootstrap import (
    PANEL_LETTER_DY_MM, PANEL_LETTER_X, fade_error_bars, place_ylabels,
)
from mpm_analysis.utils.windows_paths import save_figure_pdf_svg

# ============================================================
# CONFIGURATION
# ============================================================

# name → how step 6's figure is re-arranged.  "heights": gridspec height
# ratios of a, b, c (+ d with four panels; step 6 draws 2:1:1); "inset":
# (host panel, left, bottom, width, height), the inset's axes box in inches
# from the host's lower-left corner — its tick labels and y label lie outside
# that box, inside the host; "c": "disc" — panel c shows the discriminant view
# over all of step 6's λ in place of Re(Δe12), on c's own λ axis and ticks.
# The figure size is step 6's in every option.
OPTION_LAYOUTS = {
    "current":            dict(),
    "inset_c":            dict(heights=(2, 1, 1), inset=("c", 0.70, 0.35, 1.30, 0.66)),
    "inset_c_tall_c":     dict(heights=(1.8, 1, 1.2), inset=("c", 0.66, 0.40, 1.34, 0.78)),
    "inset_c_tall_bc":    dict(heights=(1.6, 1.15, 1.15), inset=("c", 0.66, 0.40, 1.34, 0.80)),
    "four_panels":        dict(heights=(1.5, 1, 1, 0.9)),
    "disc_c":             dict(c="disc"),
}
OPTIONS = ("current",)                  # the options to draw, in file order (opt0, opt1, ...):
                                        #   step 6 as run.py draws it (the final choice);
                                        #   tuple(OPTION_LAYOUTS) draws them all

DISC_LAMBDA_RANGE = (1.0, 1.65)         # λ shown by the discriminant view (to the first point past 1.6)
DISC_XTICKS = {"inset": (1.0, 1.2, 1.4, 1.6),          # the window's ends labelled
               "panel": (1.1, 1.2, 1.3, 1.4, 1.5, 1.6)}  # panel d: no label on its left/right edge
                                                       #   (it would stick out past the panels above)
DISC_YTICKS: tuple | None = (0.0, 0.5, 1.0)   # None → matplotlib's choice
DISC_C_YTICKS: tuple | None = (-1.0, 0.0, 1.0)  # panel c as the discriminant view (data −1 … +0.6)
DISC_LABEL = r"$\Delta e_{12}^2\,/\,\Omega_\mathrm{S}^2$"
INSET_FONT_SIZE = LABELS_FONT_SIZE * 0.7    # inset labels and tick labels

SAVE_DIR: Path | None = None            # None → run.py's PAPER_FIGURES_DIR
SHOW = True                             # plt.show() at the end

# error bars as panels b and c draw them (appendix_bootstrap)
ERR_KWS = dict(
    elinewidth=LINE_WIDTH * 0.6,
    capsize=MARKER_SIZE * 1.2,
    capthick=LINE_WIDTH * 0.6,
    zorder=10,
)

# ============================================================
# run.py's settings and data
# ============================================================

# the first-transition clauses read from run.py's main(): step → product it assigns
_CLAUSES = {"load_records": "first_records", "run_svd_gap_cut": "first_records",
            "run_mpm": "first_result", "run_transitions": "first_ep", "run_figure": "first_fig"}


def run_py_settings(run) -> dict:
    """The settings of run.py's first-transition steps, read from its main().

    For each clause of ``_CLAUSES`` (``first_<product> = <step>(...)``): its
    keyword arguments and the positional ones that are not earlier products
    (step 1's loader and folder), evaluated in run.py's namespace, and ``on``
    — whether every toggle above the clause is on.
    """
    tree = ast.parse(Path(run.__file__).read_text(encoding="utf-8"))
    main_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    namespace = vars(run)

    def value(node):
        return eval(compile(ast.Expression(node), run.__file__, "eval"), namespace)

    def is_product(node):
        return isinstance(node, ast.Name) and node.id.startswith("first_")

    steps: dict = {}

    def visit(body, on):
        for node in body:
            if isinstance(node, ast.If):
                test = node.test
                visit(node.body, on and (bool(value(test.value)) if isinstance(test, ast.NamedExpr) else True))
                visit(node.orelse, on)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                name = getattr(node.value.func, "id", None)
                target = next((t.id for t in ast.walk(node.targets[0]) if isinstance(t, ast.Name)), None)
                if name in _CLAUSES and _CLAUSES[name] == target and name not in steps:
                    steps[name] = dict(
                        args=[value(a) for a in node.value.args if not is_product(a)],
                        kwargs={k.arg: value(k.value) for k in node.value.keywords if not is_product(k.value)},
                        on=on,
                    )

    visit(main_def.body, True)
    missing = sorted(set(_CLAUSES) - {"run_svd_gap_cut"} - set(steps))
    if missing:
        raise RuntimeError(f"run.py's main() has no first-transition clause for {missing}.")
    return steps


def rerun_steps(run, steps: dict):
    """Steps 1-3 with run.py's settings: ``(result, ep, crossing)``.

    Step 2 runs without its JSON and figures (``_mpm_pipeline``, as the window
    sweep does); ``crossing`` holds step 3's Δ points and σ (``_ep_crossing``
    with run_transitions' own arguments).
    """
    load = steps["load_records"]
    records = run.load_records(*load["args"], **load["kwargs"])
    cut = steps.get("run_svd_gap_cut")
    if cut is not None and cut["on"]:
        records = run.run_svd_gap_cut(records, **{**cut["kwargs"], "plot": False})
    _, result = run._mpm_pipeline(records, **{**steps["run_mpm"]["kwargs"], "plot_svd": False})
    kw = {k: v for k, v in steps["run_transitions"]["kwargs"].items() if k != "json_path"}
    kw["plot_discriminant"] = False
    ep = run.run_transitions(result, json_path=None, **kw)
    a = inspect.signature(run.run_transitions).bind(result, json_path=None, **kw)
    a.apply_defaults()
    a = a.arguments
    crossing = run._ep_crossing(
        result, method=a["method"], half_width=a["half_width"], lambda_max=a["lambda_max"],
        points_below=a["points_below"], points_above=a["points_above"],
        lambda_window=a["lambda_window"],
    )
    return result, ep, crossing


def disc_view(result, ep: dict, crossing: dict, lambda_range: tuple = DISC_LAMBDA_RANGE) -> dict:
    """What the discriminant view draws over ``lambda_range``, in
    (Δe12)²/Ω_S² = Δ12/(4 Ω_S²)."""
    scale = 4.0 * result.omega_s ** 2
    lam = np.asarray(crossing["lam_d"], dtype=float)
    keep = (lam >= lambda_range[0]) & (lam <= lambda_range[1])
    sigma = crossing["d_sigma"]
    line = None
    fit = ep["delta_fit"]
    if fit["coef"] is not None:
        lo, hi = fit["lambda_window"]
        x = np.linspace(max(lo, lambda_range[0]), min(hi, lambda_range[1]), 50)
        line = (x, np.polyval(np.asarray(fit["coef"], dtype=float) / scale, x))
    return dict(
        lam=lam[keep],
        y=np.asarray(crossing["d_central"], dtype=float)[keep] / scale,
        sigma=None if sigma is None else np.asarray(sigma, dtype=float)[keep] / scale,
        line=line,
        lambda_ep=float(fit["lambda_ep"]),
    )


# ============================================================
# drawing
# ============================================================

def draw_disc(ax, view: dict, *, error_alpha: float, kind: str) -> None:
    """The discriminant view on ``ax``, in the style of panels b and c.

    ``kind``: "inset" (small type, no λ label — its host's λ axis names it),
    "panel" (panel d, on its own λ range) or "c" (in place of panel c's Re
    splitting: c keeps its λ axis, ticks and label).
    """
    fade_error_bars(ax.errorbar(
        view["lam"], view["y"], yerr=view["sigma"],
        color="steelblue", marker="o",
        linestyle="none", markersize=MARKER_SIZE,
        **ERR_KWS,
    ), error_alpha)
    if view["line"] is not None:
        ax.plot(*view["line"], color="k", linestyle="--", linewidth=LINE_WIDTH)
    ax.axhline(0.0, color="0.5", linewidth=LINE_WIDTH * 0.5, zorder=0)
    if np.isfinite(view["lambda_ep"]):
        ax.axvline(view["lambda_ep"], color=LAMBDA_OBS_COLOR, linestyle="--",
                   linewidth=LINE_WIDTH, alpha=0.8, zorder=0)
    fs = INSET_FONT_SIZE if kind == "inset" else LABELS_FONT_SIZE
    make_ax_lines_wider(ax, labelsize=fs)
    if kind != "c":
        ax.set_xlim(*DISC_LAMBDA_RANGE)
        ax.set_xticks(DISC_XTICKS[kind])
    yticks = DISC_C_YTICKS if kind == "c" else DISC_YTICKS
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_ylabel(DISC_LABEL, fontsize=fs)
    if kind == "panel":
        ax.set_xlabel(r"$\lambda$", fontsize=LABELS_FONT_SIZE)


def build_option(fig, layout: dict, view: dict, *, error_alpha: float, full_view: dict | None = None):
    """Re-arrange step 6's figure (axes a, b, c) as ``layout`` says;
    ``full_view`` is the discriminant view over all of step 6's λ (for
    ``c="disc"``).

    Returns the ``(inset, host)`` pairs drawn, for the layout check.
    """
    a, b, c = fig.axes[:3]
    panels = [a, b, c]
    if layout.get("c") == "disc":
        # c's Re points, fit and λ_EP line out (its letter, λ axis and ticks
        # stay); its y range then follows the new points alone
        for container in list(c.containers):
            container.remove()
        for art in [*c.lines, *c.collections]:
            art.remove()
        c.ignore_existing_data_limits = True
        draw_disc(c, full_view, error_alpha=error_alpha, kind="c")
    heights = layout.get("heights")
    if heights is not None and len(heights) == 4:
        # a, b, c restacked over a fourth row; c keeps its λ ticks, d has its own λ range
        gs = fig.add_gridspec(4, 1, height_ratios=heights)
        for i, ax in enumerate((a, b, c)):
            ax.set_subplotspec(gs[i])
        d = fig.add_subplot(gs[3])
        draw_disc(d, view, error_alpha=error_alpha, kind="panel")
        add_panel_letter(d, "d", x=PANEL_LETTER_X, y=1.0, dx_mm=0.0, dy_mm=PANEL_LETTER_DY_MM,
                         fontsize=LABELS_FONT_SIZE * 1.4)
        panels.append(d)
    elif heights is not None:
        a.get_subplotspec().get_gridspec().set_height_ratios(heights)
    # y labels on one line as step 6 sets them, panel d included
    place_ylabels(panels)
    insets = []
    if layout.get("inset") is not None:
        host_letter, left, bottom, width, height = layout["inset"]
        host = {"a": a, "b": b, "c": c}[host_letter]
        fig.canvas.draw()                   # settle the layout: the host's size is final
        w = host.get_window_extent().width / fig.dpi
        h = host.get_window_extent().height / fig.dpi
        ins = host.inset_axes([left / w, bottom / h, width / w, height / h], zorder=20)
        draw_disc(ins, view, error_alpha=error_alpha, kind="inset")
        insets.append((ins, host))
    return insets


# ============================================================
# layout check
# ============================================================

def _ink(ax, pad_pt: float) -> list:
    """Display boxes of ``ax``'s data artists: markers / caps (one box each),
    error bars (one box per bar) and lines (boxes along the path)."""
    px = ax.figure.dpi / 72.0
    boxes = []
    for art in ax.get_children():
        if not art.get_visible():
            continue
        if isinstance(art, LineCollection):
            half = (max(art.get_linewidths(), default=1.0) / 2 + pad_pt) * px
            for seg in art.get_segments():
                p = art.get_transform().transform(np.asarray(seg, dtype=float))
                boxes.append((p[:, 0].min() - half, p[:, 1].min() - half,
                              p[:, 0].max() + half, p[:, 1].max() + half))
        elif isinstance(art, Line2D) and len(art.get_xydata()):
            p = art.get_transform().transform(np.asarray(art.get_xydata(), dtype=float))
            marker = art.get_marker() not in (None, "None", "none", "", " ")
            half = (max(art.get_markersize() if marker else 0.0, art.get_linewidth()) / 2 + pad_pt) * px
            if art.get_linestyle() not in ("None", "none", "", " ") and len(p) > 1:
                p = np.concatenate([np.linspace(p0, p1, max(2, int(np.hypot(*(p1 - p0)) // 2)))
                                    for p0, p1 in zip(p[:-1], p[1:])])
            boxes += [(x - half, y - half, x + half, y + half) for x, y in p]
    return boxes


def _tick_labels(axis) -> list:
    """The tick labels ``axis`` drew: one per current tick inside the view
    (tick objects left over from an earlier draw keep stale positions)."""
    locs = axis.get_majorticklocs()
    lo, hi = sorted(axis.get_view_interval())
    return [t.label1 for t, loc in zip(axis.get_major_ticks(len(locs)), locs) if lo <= loc <= hi]


def _texts(ax, renderer) -> list:
    """(text, display box) of every label, tick label, text and legend of ``ax``."""
    items = [ax.xaxis.label, ax.yaxis.label, *_tick_labels(ax.xaxis), *_tick_labels(ax.yaxis), *ax.texts]
    out = [(t.get_text(), t.get_window_extent(renderer).extents) for t in items
           if t.get_visible() and t.get_text()]
    if ax.get_legend() is not None:
        out.append(("legend", ax.get_legend().get_window_extent(renderer).extents))
    return out


def _overlap(p, q) -> bool:
    return p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]


def check_layout(fig, insets: list, *, pad_pt: float = 2.0) -> list[str]:
    """Problems of the drawn figure (empty list → clean)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    problems = []
    inset_of = {id(i): h for i, h in insets}
    for ins, host in insets:
        box = ins.get_tightbbox(renderer).extents
        frame = host.get_window_extent(renderer).extents
        if (box[0] < frame[0] + pad_pt * px or box[1] < frame[1] + pad_pt * px
                or box[2] > frame[2] - pad_pt * px or box[3] > frame[3] - pad_pt * px):
            problems.append("the inset (with its labels) reaches the edge of its host panel")
        if any(_overlap(box, b) for b in _ink(host, pad_pt)):
            problems.append("the inset covers a point, error bar or line of its host panel")
    axes = fig.axes
    texts = {id(ax): _texts(ax, renderer) for ax in axes}
    for ax in axes:
        for name, box in texts[id(ax)]:
            for other in axes:
                if other is ax or inset_of.get(id(ax)) is other or inset_of.get(id(other)) is ax:
                    continue                # an inset sits inside its host by design
                if any(_overlap(box, b) for _, b in texts[id(other)]):
                    problems.append(f"text '{name}' overlaps text of another panel")
                if _overlap(box, other.get_window_extent(renderer).extents):
                    problems.append(f"text '{name}' overlaps the frame of another panel")
        legend = ax.get_legend()
        if legend is not None and any(_overlap(legend.get_window_extent(renderer).extents, b)
                                      for b in _ink(ax, 0.0)):
            problems.append("the legend covers data")
        own = texts[id(ax)]
        for i in range(len(own)):
            for j in range(i + 1, len(own)):
                if _overlap(own[i][1], own[j][1]):
                    problems.append(f"'{own[i][0]}' overlaps '{own[j][0]}'")
    canvas = fig.bbox.extents
    if any(not (canvas[0] - 1 <= e[0] and e[2] <= canvas[2] + 1 and canvas[1] - 1 <= e[1] and e[3] <= canvas[3] + 1)
           for e in (ax.get_tightbbox(renderer).extents for ax in axes)):
        problems.append("content outside the figure")
    return list(dict.fromkeys(problems))


# ============================================================
# RUN
# ============================================================

def main() -> None:
    import run

    run.SHOW_PLOTS, run.SAVE_FIGURES = False, False
    steps = run_py_settings(run)
    print("run.py first-transition settings:")
    for name, s in steps.items():
        shown = {k: v for k, v in s["kwargs"].items() if k != "json_path"}
        print(f"  {name:<16} {'on ' if s['on'] else 'off'}  {shown}")

    result, ep, crossing = rerun_steps(run, steps)
    fig_kwargs = steps["run_figure"]["kwargs"]
    view = disc_view(result, ep, crossing)
    full_view = disc_view(result, ep, crossing, lambda_range=(-np.inf, fig_kwargs["lambda_max"]))
    error_alpha = fig_kwargs.get("error_bar_alpha", 1.0)

    data_mode, scan_number = run._result_tags(result)
    observables = "+".join(run._observables(result))
    folder = Path(SAVE_DIR or run.PAPER_FIGURES_DIR) / build_figure_filename(
        figure_tag=f"appendix_options_{observables}_order{run._order(result)}",
        data_mode=data_mode, scan_number=scan_number, extension="",
    ).rstrip(".")
    print(f"\n=== Appendix options: (Delta e12)^2 / Omega_S^2 over lam "
          f"{DISC_LAMBDA_RANGE[0]:g}-{DISC_LAMBDA_RANGE[1]:g}, {view['lam'].size} points "
          f"(panel c: all {full_view['lam'].size}), "
          f"lam_EP = {ep['lambda_ep']:.4f}"
          + (f" +/- {ep['lambda_ep_error']:.4f}" if ep["lambda_ep_error"] is not None else "")
          + " ===")
    for i, name in enumerate(OPTIONS):
        fig = run.run_figure(result, ep, json_path=None, **fig_kwargs)
        insets = build_option(fig, OPTION_LAYOUTS[name], view, error_alpha=error_alpha, full_view=full_view)
        problems = check_layout(fig, insets)
        print(f"\n[opt{i}_{name}] layout check: "
              + ("OK" if not problems else "PROBLEMS\n    " + "\n    ".join(problems)))
        save_figure_pdf_svg(fig, f"opt{i}_{name}", folder, dpi=run.PAPER_FIGURES_DPI, png=True)
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
