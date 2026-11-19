"""What ZenoPlotting's paper-figure scripts use from this package.

ZenoPlotting/src/zeno_plotting/mpm_transition_figures/ holds three scripts
(first_transition_appendix.py, third_transition_appendix.py,
third_transition_main_text.py) that import ONLY mpm_analysis and call it with
the paper's settings.  This file lists everything they import, every keyword
they pass, and the path names, style constant and observables they ask for.

A failure here means: the change you made would break those scripts.  Either
keep the old name / keyword working, or change the ZenoPlotting scripts in the
same go (and then this list).

The list is a copy of the scripts' calls, kept by hand.  When a ZenoPlotting
script changes what it calls, update this file too.  No experimental data is
read: the calls are bound to the current signatures, and one test runs the
scripts' steps end to end on synthetic traces.
"""
import inspect

import pytest

from mpm_analysis.config import load_config


def _binds(fn, *args, **kwargs):
    """Fail unless ``fn(*args, **kwargs)`` is a valid call of ``fn`` (never calls it)."""
    try:
        inspect.signature(fn).bind(*args, **kwargs)
    except TypeError as e:
        pytest.fail(f"{fn.__module__}.{fn.__name__} no longer accepts ZenoPlotting's call: {e}")


def test_imports():
    """The import lines of the three scripts."""
    from mpm_analysis.io.experimental import EnsembleAverageLoader, PostSelectedLoader  # noqa: F401
    from mpm_analysis.pipeline.transition_steps import (  # noqa: F401
        load_records, run_figure, run_mpm, run_transitions,
    )
    from mpm_analysis.plotting._style import DPI  # noqa: F401
    from mpm_analysis.plotting.paper_figures import make_third_transition_figure  # noqa: F401
    from mpm_analysis.utils.lab_paths import get_named_path  # noqa: F401
    from mpm_analysis.utils.windows_paths import save_figure_pdf_svg  # noqa: F401


def test_loaders_take_the_lambda_table():
    from mpm_analysis.io.experimental import EnsembleAverageLoader, PostSelectedLoader

    _binds(PostSelectedLoader, csv_path="table.csv")
    _binds(EnsembleAverageLoader, csv_path="table.csv")


def test_observables_exist():
    """``OBSERVABLE = "p_0_t"`` (first transition) and ``"D"`` (third)."""
    from mpm_analysis.analysis.bootstrap import _POST_SELECTED_SURVIVAL
    from mpm_analysis.io.experimental import EnsembleAverageLoader, PostSelectedLoader

    assert "p_0_t" in PostSelectedLoader(csv_path=False)._extractors
    assert "D" in EnsembleAverageLoader(csv_path=False)._extractors
    # the shot-noise bootstrap knows how to redraw p_0_t
    assert _POST_SELECTED_SURVIVAL["p_0_t"] == ("p_0_t", "N_attempts_p_0_t")


def test_step_calls():
    """Steps 1-3 of each script's main(), with the keywords the scripts pass."""
    from mpm_analysis.pipeline.transition_steps import load_records, run_mpm, run_transitions

    loader, folder, records, result = object(), "folder", [], object()
    _binds(load_records, loader, folder,
           data_mode="post_selected", observables=["p_0_t"], lambda_range=None,
           t_range=None, drop_first_point=False)
    _binds(run_mpm, records,
           order=3, uncertainty_source="data_bootstrap", n_boot=2000,
           bootstrap_center="measured", redraw_around="svd_truncated",
           refine_poles=False, conjugate_pairs=True, plot_svd=False, plot_spectrum=False,
           seed=7, n_jobs=8, show_plots=False)
    _binds(run_transitions, result,
           json_path=None, method="linfit", lambda_window=(0.9, 1.3), lambda_max=None,
           plot_discriminant=False, show_plots=False)


def test_appendix_figure_call():
    """run_figure as first_transition_appendix.py and third_transition_appendix.py call it."""
    from mpm_analysis.pipeline.transition_steps import run_figure

    result, ep = object(), {}
    _binds(run_figure, result, ep,
           json_path=None, figure_name="first_transition_appendix", layout="sqrt",
           panel_c="discriminant", lambda_max=2, fit_validity=1.2, real_axis_scale="linear",
           real_axis_ticks=[0, 2, 4], real_split_axis_ticks=[-1, 0, 1],
           split_lambda_ticks=[0.5, 0.9, 1.3, 1.7], splitting_fit="discriminant",
           error_bar_alpha=0.4, crit_line_decimals=2, show_plots=False)


def test_main_text_figure_call():
    """make_third_transition_figure as third_transition_main_text.py calls it."""
    from mpm_analysis.plotting.paper_figures import make_third_transition_figure

    result, records = object(), []
    _binds(make_third_transition_figure,
           panel_b_result=result, panel_a_records=records, panel_a_fit_result=result,
           panel_a_observable="D", target_lambdas=[0.4, 1.08, 1.5], label_fmt="{:.2f}",
           lambda_max_b=1.6, downsample_a=3, downsample_b=3, keep_all_within_b=0.15,
           keep_first_b=2, transition_label="Quantum\nZeno", save_dir=None,
           figure_stem="third_transition_main_text_figure_with_QZ", yticks_a=[0.2, 0.5, 0.8],
           yticks_freq=[0.0, 0.4, 0.8], ylim_decay_b=(0.25, 0.9), discriminant_fit={},
           crit_line_decimals=2)


def _write_synthetic_scan(folder, *, n_seg=62, n_axes=2, n_reps=150_000):
    """Nine post-selected scan files (a half record each) in the scanner's JSON format.

    The no-click probability of file j is a three-pole curve whose pair
    discriminant crosses zero near lambda = 1.3; each file stores the
    first-click histogram a measurement of that curve gives, which is what the
    loader builds ``p_0_t`` from.
    """
    import json

    import numpy as np
    from mpm_analysis.io.experimental.click_record_npz import pooled_survival_from_counts

    def curve(t, delta):
        a3, s3, m = 1.56, -0.25, -1.2
        beta = -a3 * s3 / (1 - a3) - m                       # S'(0) = 0
        q = np.sqrt(abs(delta)) / 2
        pair = (np.cos(q * t) + beta * np.sin(q * t) / q if delta < 0
                else np.cosh(q * t) + beta * np.sinh(q * t) / q)
        return a3 * np.exp(s3 * t) + (1 - a3) * np.exp(m * t) * pair

    rng = np.random.default_rng(11)
    t = np.arange(n_seg + 1) * 0.2
    lengths = np.arange(1, n_seg + 1)
    for j, lam in enumerate(np.linspace(0.9, 1.7, 9)):
        surv = np.clip(curve(t, 2.0 * (lam - 1.3) + 1e-3), 1e-6, 1.0)
        counts = np.zeros((n_axes, n_seg, n_seg + 1), dtype=np.int64)
        for a in range(n_axes):
            for L in lengths:                                # slot 0: no click; slot k: first click in bin k
                first = np.clip(surv[:L] - surv[1:L + 1], 0.0, None)
                counts[a, L - 1, :L + 1] = rng.multinomial(n_reps, np.concatenate(([1.0 - first.sum()], first)))
        P, _, nP = pooled_survival_from_counts(counts)
        Q, _, nQ = pooled_survival_from_counts(counts, min_extra_bins=1)
        own = np.concatenate(([1.0], counts[:, :, 0].sum(0) / counts.sum(-1).sum(0)))
        flat = [0.0] * (n_seg + 1)
        data = {
            "time_array": (t * 1000.0).tolist(),
            "p_0_t_array": own.tolist(),
            "phase_axes_number_of_measurements": [100] * (n_seg + 1),
            "z_axis_number_of_measurements": [100] * (n_seg + 1),
            "main_phase_axis_array": flat, "orthogonal_phase_axis_array": flat, "z_axis_array": flat,
            "p_0_t_array_pooled": P.tolist(),
            "p_0_t_array_pooled_strict": [None if not np.isfinite(v) else float(v) for v in Q],
            "p_0_t_pooled_meta": {
                "n_attempts": nP.tolist(),
                "first_click_counts": [[counts[a, L - 1, :L + 1].tolist() for L in lengths]
                                       for a in range(n_axes)],
                "strict": {"min_extra_bins": 1, "n_attempts": nQ.tolist()},
            },
        }
        jd = {"parameters": {"Omega_DG": 628318.53, "lambda_zeno_estimation": float(lam),
                             "navg_repeat_experiment": n_reps},
              "data": data}
        name = (f"click_record_with_post_selected_tomography_data__OBG0_{1000 + 10 * j}"
                "_kHz_15_10_2026__16_57_51.json")
        (folder / name).write_text(json.dumps(jd))


def test_the_scripts_main_runs_end_to_end(tmp_path):
    """Steps 1-4 of the scripts' main() and the save, on synthetic scan files.

    Same calls and keywords as the scripts (a small bootstrap, one worker): the
    result of each step is what the next one is given, so a changed return
    value or result key fails here too.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpm_analysis.io.experimental import PostSelectedLoader
    from mpm_analysis.pipeline.transition_steps import (
        load_records, run_figure, run_mpm, run_transitions,
    )
    from mpm_analysis.plotting._style import DPI, round_half_up
    from mpm_analysis.plotting.paper_figures import make_third_transition_figure
    from mpm_analysis.utils.windows_paths import save_figure_pdf_svg

    data_folder = tmp_path / "scan"
    data_folder.mkdir()
    _write_synthetic_scan(data_folder)
    records = load_records(
        PostSelectedLoader(csv_path=False), data_folder, data_mode="post_selected",
        observables=["p_0_t"], lambda_range=None, t_range=None, drop_first_point=False)
    assert len(records) == 9 and records[0].observable_key == "p_0_t"
    result, _ = run_mpm(
        records, order=3, uncertainty_source="data_bootstrap", n_boot=60,
        bootstrap_center="measured", redraw_around="svd_truncated",
        refine_poles=False, conjugate_pairs=True, plot_svd=False, plot_spectrum=False,
        seed=7, n_jobs=1, show_plots=False)
    ep = run_transitions(
        result, json_path=None, method="linfit", lambda_window=(1.0, 1.6), lambda_max=None,
        plot_discriminant=False, show_plots=False)
    assert 1.0 < ep["lambda_ep"] < 1.6 and ep["lambda_ep_error"] > 0
    assert ep["delta_fit"]["lambda_ep"] == ep["lambda_ep"]

    fig = run_figure(
        result, ep, json_path=None, figure_name="appendix", layout="sqrt",
        panel_c="discriminant", lambda_max=2, fit_validity=1.2, real_axis_scale="linear",
        real_axis_ticks=[0, 2, 4], real_split_axis_ticks=[-1, 0, 1],
        split_lambda_ticks=[0.5, 0.9, 1.3, 1.7], splitting_fit="discriminant",
        error_bar_alpha=0.4, crit_line_decimals=2, show_plots=False)
    save_figure_pdf_svg(fig, "appendix", tmp_path, dpi=DPI, png=True)
    assert (tmp_path / "appendix.pdf").exists() and (tmp_path / "appendix.png").exists()

    fig = make_third_transition_figure(
        panel_b_result=result, panel_a_records=records, panel_a_fit_result=result,
        panel_a_observable="p_0_t", target_lambdas=np.array([0.9, 1.3, 1.7]),
        label_fmt=r"$\lambda={:.2f}$", lambda_max_b=1.8, downsample_a=3, downsample_b=3,
        keep_all_within_b=0.15, keep_first_b=2, transition_label="Quantum\nZeno", save_dir=None,
        figure_stem="main_text", yticks_a=[0.2, 0.5, 0.8], yticks_freq=[0.0, 0.4, 0.8],
        ylim_decay_b=None, discriminant_fit=ep["delta_fit"], crit_line_decimals=2)
    # the transition line of panel b sits at lambda_EP rounded to 2 digits
    line_x = round_half_up(ep["lambda_ep"], 2)
    drawn = [float(np.ravel(line.get_xdata())[0]) for ax in fig.axes for line in ax.get_lines()
             if len(np.ravel(line.get_xdata())) == 2
             and np.ravel(line.get_xdata())[0] == np.ravel(line.get_xdata())[1]]
    assert line_x in drawn, (line_x, drawn)
    plt.close("all")


def test_save_call():
    from mpm_analysis.utils.windows_paths import save_figure_pdf_svg

    _binds(save_figure_pdf_svg, object(), "figure_name", "save_dir", dpi=600, png=True)


def test_path_names():
    """The names the scripts pass to get_named_path are in config/paths.json."""
    from mpm_analysis.utils.lab_paths import get_named_path

    paths = load_config("paths")["paths"]
    for name in ("first_transition_data", "first_transition_lambda_table",
                 "third_transition_data", "third_transition_lambda_table", "paper_plots"):
        assert name in paths, f"'{name}' is missing from config/paths.json"
    with pytest.raises(KeyError):
        get_named_path("no_such_path_name")


def test_style_constants_are_the_json_values():
    """plotting/_style.py exposes config/plotting_params.json, value for value."""
    from mpm_analysis.plotting import _style

    params = load_config("plotting_params")
    assert _style.DPI == params["dpi"]
    assert _style.LABELS_FONT_SIZE == params["labels_font_size"]
    assert _style.MARKER_SIZE == params["marker_size"]
    assert _style.LINE_WIDTH == params["line_width"]
    assert _style.AXES_ARROW_WIDTH == params["axes_arrow_width"]
    assert _style.LAMBDA_OBS_COLOR == params["lambda_obs_color"]
    assert _style.SURVIVAL_PROBABILITY_LABEL == params["survival_probability_label"]
    assert _style.ONE_COLUMN_WIDTH == pytest.approx(params["one_column_width_mm"] / 25.4)
    assert _style.TWO_COLUMN_WIDTH == pytest.approx(params["two_column_width_mm"] / 25.4)
    for name, value in params["colors"].items():
        assert getattr(_style, name.upper()) == value


def test_line_rounding():
    """The transition lines sit at lambda_EP rounded half up (2 digits in ZenoPlotting)."""
    from mpm_analysis.plotting._style import round_half_up

    assert round_half_up(1.085478, 2) == 1.09      # third transition, seed 7
    assert round_half_up(1.295591, 2) == 1.30      # first transition, seed 0
    assert round_half_up(1.0845, 3) == 1.085       # half up, not Python's round()
