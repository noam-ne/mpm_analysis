"""survival_explorer.py — raw post-selected survival vs the forward model, with the hazard.

    python run_files/survival_explorer.py

What it shows, for one scan folder (default: scan 3 combined_iterations)
------------------------------------------------------------------------
Per λ point the three curves on the same time grid:

    S          stored survival  p_0_t_array        (segment i alone, its own last bin)
    S_pooled   pooled survival  p_0_t_array_pooled (every segment L >= i)
    model      the 3x3 post-selected generator of mpm_benchmarks/scan3_p0/common.py
               (D/G qubit, Omega_DG rotation, clicks at 2·λ·Omega_DG from G, T1, T2),
               initial state (1-EPS_INIT) D + EPS_INIT G, read out with a per-window
               false-bright probability Q (and Q_LAST extra on the stored own last bin)

and their HAZARDS — the per-window click probability given survival to the previous
window, h_i = 1 - S_i / S_{i-1}.  The hazard is where the difference between the curves
lives: S is the cumulative product of (1 - h), so a localised readout effect (a bin that
reads bright too often) is a spike in h but a smooth slope change in S.  From the
first-click histogram the hazard is also available PER SEGMENT, H[L, k] = clicks at bin k
of the L-bin segment / shots of that segment alive before bin k, without any pooling:
its diagonal H[L, L] is each segment's OWN last bin (what the stored S point L is made
of), the column k over L >= k is what the pooled point k averages.

Steps (toggles below, one clause each — the products are the names on the left):
    1  recs     = load_points(...)         the folder's points, HMM λ axis, counts, error bars
    2  model    = model_curves(recs, ...)  the generator on the data's time grid
    3  haz      = hazards(recs, model)     h of S / S_pooled / model, H[L, k] per λ
    4  slider   = plot_slider(...)         per-λ browser: curves, ratios, hazards, H[L, k]
    5  heatmaps = plot_heatmaps(...)       λ × t maps: pooled/stored, data/model, hazard
                                           differences, own-last-bin excess (λ × L)
    6  print_summary(...)

The folder, the λ table and the model numbers are the only things to change for
another scan (scan 4+5 combined: its own hmm_rates_per_drive.csv; scan 8: the
scan8_hmm table).  Figures go to mpm_analysis_output/survival_explorer/<folder>/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "mpm_benchmarks" / "scan3_p0"))
import common as K          # noqa: E402  loader + 3x3 generator + readout transform
import config as C          # noqa: E402  the suite's defaults (sheet T1/T2, scan-3 paths)

# ===========================================================================
# SETTINGS
# ===========================================================================
_SCAN_ROOT = Path(r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab\Experiments\Zeno"
                  r"\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons\click_record_with_post_selected_tomography"
                  r"\Omega_BG0_scan")
FOLDER = _SCAN_ROOT / "scan_experiment_results_3" / "combined_iterations"
HMM_TABLE: Path | None = _SCAN_ROOT / "scan_experiment_results_3" / "hmm_rates_per_drive.csv"
#   per-drive HMM λ table of THIS folder (exact drive lookup, no interpolation);
#   None → each file's stored lambda_zeno_estimation.
#   scan 4+5 combined → its own hmm_rates_per_drive.csv (merged over scans 3+4+5)
LAMBDA_RANGE = (0.0, 4.0)

# Forward model (sheet numbers; the generator needs 1/T1 and 1/T2)
T1_US = C.T1_US             # 75
T2_US = C.T2_US             # 19   (tomography says 13-16: try it)
EPS_INIT = 0.0              # incoherent G admixture of the initial state
Q_FALSE_BRIGHT = 0.0        # per-window false-bright probability (both curves)
Q_LAST = 0.0                # EXTRA false bright on the stored segment's own last bin
RELAX_TO = "G"              # "G" (T1 takes D -> G, feeds the click channel) | "D"

# Figures
SHOW_PLOTS = True
SAVE_FIGURES = True
OUTPUT_DIR = CODE / "mpm_analysis" / "mpm_analysis_output" / "survival_explorer" / FOLDER.name
HEATMAP_T_MAX: float | None = 15.0      # µs shown in the λ × t maps (None → all)
SLIDER_T_MAX: float | None = None       # µs shown in the browser (None → all)
FIG_DPI = 130

_LIVE: list = []    # strong references to the slider widgets


# ===========================================================================
# MAIN — step by step
# ===========================================================================

def main() -> None:
    K.style()

    # Step 1 — the folder's points.   out: recs (dict of stacked arrays, see load_points)
    recs = load_points(FOLDER, hmm_table=HMM_TABLE, lambda_range=LAMBDA_RANGE)

    # Step 2 — the forward model on the data's time grid.   out: model
    model = model_curves(recs, T1_us=T1_US, T2_us=T2_US, eps=EPS_INIT,
                         q=Q_FALSE_BRIGHT, q_last=Q_LAST, relax_to=RELAX_TO)

    # Step 3 — hazards of every curve + the per-segment hazard matrices.   out: haz
    haz = hazards(recs, model)

    # Step 4 — per-λ browser.   out: slider
    if (EXPLORER_SLIDER := True):
        slider = plot_slider(recs, model, haz, t_max=SLIDER_T_MAX)

    # Step 5 — λ × t heatmaps.   out: heatmaps (list of figures)
    if (EXPLORER_HEATMAPS := True):
        heatmaps = plot_heatmaps(recs, model, haz, t_max=HEATMAP_T_MAX)

    # Step 6 — numbers behind the pictures
    print_summary(recs, model, haz)

    if SHOW_PLOTS:
        plt.show()


# ===========================================================================
# STEP FUNCTIONS
# ===========================================================================

def load_points(folder: Path, *, hmm_table: Path | None, lambda_range) -> dict:
    """Step 1 — every drive point of ``folder`` as stacked arrays, sorted by λ.

    Returns a dict: ``lam`` (n,), ``khz`` (n,), ``t`` (m,) µs, ``S`` / ``S_pooled``
    (n, m), ``sigma_S`` / ``sigma_S_pooled`` (n, m; NaN when the file carries no
    block error bars), ``counts`` list of n first-click histograms
    (axes × segments × slots), ``omega_s`` (rad/µs), ``files``.
    """
    pts = K.load_combined(lambda_range=lambda_range, with_counts=True, folder=folder,
                          lambda_source="hmm" if hmm_table is not None else "json",
                          hmm_table=hmm_table)
    if not pts:
        raise RuntimeError(f"no points in {folder} (lambda_range={lambda_range})")
    t = pts[0]["t"]
    for p in pts:
        if p["t"].shape != t.shape or not np.allclose(p["t"], t):
            raise ValueError(f"{p['file']}: time grid differs from {pts[0]['file']}")
    n, m = len(pts), t.size
    sig = {k: np.full((n, m), np.nan) for k in ("S", "S_pooled")}
    for i, p in enumerate(pts):
        for k in ("S", "S_pooled"):
            if p.get(f"sigma_{k}") is not None:
                sig[k][i] = p[f"sigma_{k}"]
    out = dict(
        lam=np.array([p["lam"] for p in pts]), khz=np.array([p["khz"] for p in pts]),
        t=t, S=np.vstack([p["S"] for p in pts]), S_pooled=np.vstack([p["S_pooled"] for p in pts]),
        sigma_S=sig["S"], sigma_S_pooled=sig["S_pooled"],
        counts=[p.get("counts") for p in pts], omega_s=float(pts[0]["omega_s"]),
        files=[p["file"] for p in pts], folder=Path(folder),
        lambda_source=("hmm:" + Path(hmm_table).name) if hmm_table is not None else "json",
    )
    print(f"=== Load ({out['folder'].name}) ===\n  {n} points, lambda {out['lam'][0]:.3f} .. "
          f"{out['lam'][-1]:.3f} ({out['lambda_source']}), {m} time points, dt = {t[1] - t[0]:.3f} us, "
          f"error bars: {'yes' if np.isfinite(out['sigma_S']).any() else 'no'}, "
          f"histograms: {sum(c is not None for c in out['counts'])}/{n}")
    return out


def model_curves(recs: dict, *, T1_us: float, T2_us: float, eps: float, q: float,
                 q_last: float, relax_to: str) -> dict:
    """Step 2 — the 3x3 post-selected generator on the data's grid, per λ.

    ``S_phys`` is the generator's survival with the mixed initial state;
    ``S_stored`` adds the readout model of the stored curve (q every window,
    q + q_last on the own last bin), ``S_pooled`` the pooled one (q only).
    """
    g1 = 1.0 / T1_us
    g2 = 1.0 / T2_us
    t, w = recs["t"], recs["omega_s"]
    S_phys = np.vstack([K.mixed_init_survival(lam, t, omega_s=w, eps=eps, g1=g1, g2=g2,
                                              relax_to=relax_to) for lam in recs["lam"]])
    S_stored = np.vstack([K.readout_transform(s, q=q, q_last=q_last) for s in S_phys])
    S_pooled = np.vstack([K.readout_transform(s, q=q, q_last=0.0) for s in S_phys])
    label = (f"model: T1={T1_us:g} T2={T2_us:g} us, eps={eps:g}, q={q:g}, q_last={q_last:g}, "
             f"relax->{relax_to}")
    print(f"=== Model ===\n  {label}")
    return dict(S_phys=S_phys, S_stored=S_stored, S_pooled=S_pooled, label=label,
                T1=T1_us, T2=T2_us, eps=eps, q=q, q_last=q_last, relax_to=relax_to)


def hazard(S: np.ndarray) -> np.ndarray:
    """Per-window click probability given survival, h_i = 1 - S_i / S_{i-1}; h_0 = NaN."""
    S = np.asarray(S, float)
    h = np.full(S.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        h[..., 1:] = 1.0 - S[..., 1:] / S[..., :-1]
    return h


def hazard_matrix(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment hazard from a first-click histogram, no pooling involved.

    ``counts`` is (axes, n_seg, n_seg + 1) (slot 0 = never bright, slot k = first
    bright at bin k of the L = row + 1 bin segment).  Returns

    * ``H`` (n_seg + 1, n_seg + 1): H[L, k] = first bright at k in segment L /
      shots of segment L still dark before bin k, for 1 <= k <= L (NaN elsewhere);
    * ``h_own`` (n_seg + 1,): the diagonal H[L, L] — each segment's own last bin;
    * ``h_pool`` (n_seg + 1,): the pooled hazard, all segments L >= k together.
    """
    c = np.asarray(counts, float).sum(axis=0)            # (n_seg, n_seg+1)
    n_seg = c.shape[0]
    n_L = c.sum(axis=1)                                   # shots per segment
    clicks = c[:, 1:]                                     # clicks at bin k = 1..n_seg
    before = n_L[:, None] - np.concatenate([np.zeros((n_seg, 1)), np.cumsum(clicks, axis=1)[:, :-1]],
                                           axis=1)       # alive before bin k
    L = np.arange(1, n_seg + 1)[:, None]
    k = np.arange(1, n_seg + 1)[None, :]
    valid = k <= L
    H = np.full((n_seg + 1, n_seg + 1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        H[1:, 1:] = np.where(valid, clicks / before, np.nan)
        h_pool = np.full(n_seg + 1, np.nan)
        h_pool[1:] = np.where(valid, clicks, 0).sum(axis=0) / np.where(valid, before, 0).sum(axis=0)
    h_own = np.full(n_seg + 1, np.nan)
    h_own[1:] = np.diag(H[1:, 1:])
    return H, h_own, h_pool


def hazards(recs: dict, model: dict) -> dict:
    """Step 3 — hazards of S, S_pooled and the model curves (n, m), plus per λ the
    per-segment matrix H[L, k], its diagonal h_own and the pooled h_pool from the
    first-click histogram (None where the file has no histogram)."""
    out = dict(h_S=hazard(recs["S"]), h_pooled=hazard(recs["S_pooled"]),
               h_model_stored=hazard(model["S_stored"]), h_model_pooled=hazard(model["S_pooled"]),
               H=[], h_own=[], h_pool=[])
    for cnt in recs["counts"]:
        if cnt is None:
            out["H"].append(None); out["h_own"].append(None); out["h_pool"].append(None)
            continue
        H, h_own, h_pool = hazard_matrix(cnt)
        out["H"].append(H); out["h_own"].append(h_own); out["h_pool"].append(h_pool)
    n_hist = sum(h is not None for h in out["H"])
    print(f"=== Hazards ===\n  per-window h of S / S_pooled / model; per-segment H[L, k] for "
          f"{n_hist}/{len(recs['lam'])} points")
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _mask(t, t_max):
    return np.ones(t.shape, bool) if t_max is None else t <= t_max


def plot_slider(recs: dict, model: dict, haz: dict, *, t_max=None):
    """Step 4 — one λ at a time: curves, relative differences, hazards, H[L, k]."""
    t, lam = recs["t"], recs["lam"]
    m = _mask(t, t_max)
    tt = t[m]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    fig.subplots_adjust(bottom=0.12, hspace=0.32, wspace=0.25)
    (ax_s, ax_h), (ax_r, ax_H) = axes

    i0 = int(np.argmin(np.abs(lam - 1.0)))
    cS, cP, cM = K.COL["S"], K.COL["S_pooled"], K.COL["model"]

    # (a) survival curves
    eb_S = ax_s.errorbar(tt, recs["S"][i0][m], yerr=_err(recs["sigma_S"][i0][m]), fmt="o", ms=3,
                         color=cS, label="S (stored)", zorder=3)
    eb_P = ax_s.errorbar(tt, recs["S_pooled"][i0][m], yerr=_err(recs["sigma_S_pooled"][i0][m]), fmt="s",
                         ms=2.5, color=cP, label="S_pooled", zorder=3)
    (ln_M,) = ax_s.plot(tt, model["S_stored"][i0][m], "-", color=cM, lw=1.3, label="model (stored readout)")
    (ln_Mp,) = ax_s.plot(tt, model["S_pooled"][i0][m], "--", color=cM, lw=1.0, label="model (pooled readout)")
    ax_s.set_yscale("log"); ax_s.set_ylabel("survival"); ax_s.legend(loc="lower left")

    # (b) relative differences
    (ln_PS,) = ax_r.plot(tt, _rel(recs["S_pooled"][i0], recs["S"][i0])[m], "-", color=cP, label="S_pooled / S − 1")
    (ln_SM,) = ax_r.plot(tt, _rel(recs["S"][i0], model["S_stored"][i0])[m], "o-", ms=2.5, color=cS,
                         label="S / model − 1")
    (ln_PM,) = ax_r.plot(tt, _rel(recs["S_pooled"][i0], model["S_pooled"][i0])[m], "s-", ms=2.5, color="#6250d6",
                         label="S_pooled / model − 1")
    ax_r.axhline(0, color="0.5", lw=0.7); ax_r.set_ylabel("relative difference"); ax_r.set_xlabel("t (µs)")
    ax_r.legend(loc="upper left")

    # (c) hazards
    (ln_hS,) = ax_h.plot(tt, haz["h_S"][i0][m], "o-", ms=2.5, color=cS, label="h of S (stored: segment i vs i−1)")
    (ln_hP,) = ax_h.plot(tt, haz["h_pooled"][i0][m], "s-", ms=2.5, color=cP, label="h of S_pooled")
    h_own = haz["h_own"][i0]
    (ln_hO,) = ax_h.plot(tt, (h_own if h_own is not None else np.full(t.shape, np.nan))[m], "^", ms=3.5,
                         color=K.COL["warn"], label="own last bin H[L, L] (histogram)")
    (ln_hM,) = ax_h.plot(tt, haz["h_model_stored"][i0][m], "-", color=cM, lw=1.3, label="model h (stored readout)")
    (ln_hMp,) = ax_h.plot(tt, haz["h_model_pooled"][i0][m], "--", color=cM, lw=1.0, label="model h (pooled readout)")
    ax_h.set_ylabel("hazard  1 − S_i / S_{i−1}"); ax_h.legend(loc="upper right", fontsize=7)

    # (d) per-segment hazard matrix relative to the pooled hazard of that bin
    H = haz["H"][i0]
    img = _H_rel(H, haz["h_pool"][i0]) if H is not None else np.full((t.size, t.size), np.nan)
    n_show = int(m.sum())
    im = ax_H.imshow(img[1:n_show, 1:n_show], origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                     extent=[tt[1] - 0.5 * (t[1] - t[0]), tt[-1] + 0.5 * (t[1] - t[0])] * 2)
    ax_H.set_xlabel("bin time k (µs)"); ax_H.set_ylabel("segment length L (µs)")
    ax_H.set_title("H[L, k] / h_pooled[k] − 1   (diagonal = own last bin)", fontsize=9)
    fig.colorbar(im, ax=ax_H, fraction=0.04, pad=0.02)

    title = fig.suptitle("", fontsize=11)

    def _set(i: int) -> None:
        for eb, y, s in ((eb_S, recs["S"][i], recs["sigma_S"][i]), (eb_P, recs["S_pooled"][i], recs["sigma_S_pooled"][i])):
            _update_errorbar(eb, tt, y[m], _err(s[m]))
        ln_M.set_ydata(model["S_stored"][i][m]); ln_Mp.set_ydata(model["S_pooled"][i][m])
        ln_PS.set_ydata(_rel(recs["S_pooled"][i], recs["S"][i])[m])
        ln_SM.set_ydata(_rel(recs["S"][i], model["S_stored"][i])[m])
        ln_PM.set_ydata(_rel(recs["S_pooled"][i], model["S_pooled"][i])[m])
        ln_hS.set_ydata(haz["h_S"][i][m]); ln_hP.set_ydata(haz["h_pooled"][i][m])
        ln_hM.set_ydata(haz["h_model_stored"][i][m]); ln_hMp.set_ydata(haz["h_model_pooled"][i][m])
        ho = haz["h_own"][i]
        ln_hO.set_ydata((ho if ho is not None else np.full(t.shape, np.nan))[m])
        Hi = haz["H"][i]
        im.set_data((_H_rel(Hi, haz["h_pool"][i]) if Hi is not None else np.full((t.size, t.size), np.nan))[1:n_show, 1:n_show])
        for ax in (ax_r, ax_h):
            ax.relim(); ax.autoscale_view()
        ax_s.relim(); ax_s.autoscale_view()
        title.set_text(f"{recs['folder'].name}   λ = {lam[i]:.3f}  ({recs['khz'][i]} kHz, {recs['lambda_source']})"
                       f"\n{model['label']}")
        fig.canvas.draw_idle()

    _set(i0)
    ax_sl = fig.add_axes([0.15, 0.03, 0.7, 0.025])
    sl = Slider(ax_sl, "λ index", 0, len(lam) - 1, valinit=i0, valstep=1)
    sl.on_changed(lambda v: _set(int(v)))
    _LIVE.append(sl)
    if SAVE_FIGURES:
        _save(fig, f"slider_lambda{lam[i0]:.2f}")
    return sl


def plot_heatmaps(recs: dict, model: dict, haz: dict, *, t_max=None) -> list:
    """Step 5 — λ × t maps.  Figure 1: survival ratios; figure 2: hazard differences
    and the own-last-bin excess (λ × segment length)."""
    t, lam = recs["t"], recs["lam"]
    m = _mask(t, t_max)
    figs = []

    maps1 = [
        (_rel(recs["S_pooled"], recs["S"]), "S_pooled / S − 1", "RdBu_r"),
        (_rel(recs["S"], model["S_stored"]), "S / model − 1  (stored readout model)", "RdBu_r"),
        (_rel(recs["S_pooled"], model["S_pooled"]), "S_pooled / model − 1  (pooled readout model)", "RdBu_r"),
    ]
    figs.append(_maps(t[m], lam, [(M[:, m], ttl, cm) for M, ttl, cm in maps1],
                      f"survival ratios — {recs['folder'].name}\n{model['label']}", "survival_ratios"))

    n = len(lam)
    own_excess = np.full((n, t.size), np.nan)
    for i in range(n):
        if haz["h_own"][i] is not None:
            own_excess[i] = haz["h_own"][i] - haz["h_pool"][i]
    maps2 = [
        (haz["h_S"] - haz["h_model_stored"], "h(S) − h(model)", "RdBu_r"),
        (haz["h_pooled"] - haz["h_model_pooled"], "h(S_pooled) − h(model)", "RdBu_r"),
        (own_excess, "own last bin: H[L, L] − h_pooled[L]   (x = segment length L)", "RdBu_r"),
    ]
    figs.append(_maps(t[m], lam, [(M[:, m], ttl, cm) for M, ttl, cm in maps2],
                      f"hazard differences per window — {recs['folder'].name}", "hazard_differences"))

    if np.isfinite(recs["sigma_S"]).any():
        with np.errstate(invalid="ignore", divide="ignore"):
            z_PS = (recs["S_pooled"] - recs["S"]) / np.hypot(recs["sigma_S"], recs["sigma_S_pooled"])
            z_SM = (recs["S"] - model["S_stored"]) / recs["sigma_S"]
            z_PM = (recs["S_pooled"] - model["S_pooled"]) / recs["sigma_S_pooled"]
        maps3 = [(z_PS[:, m], "(S_pooled − S) / σ   (block error bars)", "RdBu_r"),
                 (z_SM[:, m], "(S − model) / σ_S", "RdBu_r"),
                 (z_PM[:, m], "(S_pooled − model) / σ_pooled", "RdBu_r")]
        figs.append(_maps(t[m], lam, maps3, f"deviations in measured-error-bar units — {recs['folder'].name}",
                          "z_scores", symmetric_limit=10.0))
    return figs


def print_summary(recs: dict, model: dict, haz: dict) -> None:
    """Step 6 — per λ: early pooled/stored excess, data/model deviation, own-last-bin excess."""
    t, lam = recs["t"], recs["lam"]
    early = (t > 0) & (t <= 2.0)
    mid = (t > 0) & (t <= 10.0)
    print(f"\n{'lambda':>7} {'kHz':>5} {'pooled/S-1 t<=2us':>18} {'S/model-1 t<=10':>16} "
          f"{'pooled/model-1':>15} {'own-bin h excess':>17} {'h_pool t<=10':>13}")
    for i in range(len(lam)):
        ps = np.nanmean(_rel(recs["S_pooled"][i], recs["S"][i])[early])
        sm = np.nanmean(_rel(recs["S"][i], model["S_stored"][i])[mid])
        pm = np.nanmean(_rel(recs["S_pooled"][i], model["S_pooled"][i])[mid])
        ho = haz["h_own"][i]
        ex = np.nanmean((ho - haz["h_pool"][i])[mid]) if ho is not None else np.nan
        hp = np.nanmean(haz["h_pooled"][i][mid])
        print(f"{lam[i]:7.3f} {recs['khz'][i]:5d} {ps:+18.4f} {sm:+16.4f} {pm:+15.4f} {ex:+17.5f} {hp:13.5f}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rel(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(a, float) / np.asarray(b, float) - 1.0


def _err(sigma):
    s = np.asarray(sigma, float)
    return None if not np.isfinite(s).any() else np.where(np.isfinite(s), s, 0.0)


def _H_rel(H, h_pool):
    with np.errstate(invalid="ignore", divide="ignore"):
        return H / h_pool[None, :] - 1.0


def _update_errorbar(eb, x, y, yerr):
    line, caps, bars = eb
    line.set_data(x, y)
    if bars:
        if yerr is None:
            yerr = np.zeros_like(y)
        segs = [np.array([[xi, yi - ei], [xi, yi + ei]]) for xi, yi, ei in zip(x, y, yerr)]
        bars[0].set_segments(segs)


def _maps(t, lam, maps, suptitle, name, *, symmetric_limit=None):
    fig, axes = plt.subplots(1, len(maps), figsize=(5.2 * len(maps), 5.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (M, title, cmap) in zip(axes, maps):
        M = np.asarray(M, float)
        finite = M[np.isfinite(M)]
        lim = (symmetric_limit if symmetric_limit is not None
               else (np.nanpercentile(np.abs(finite), 98) if finite.size else 1.0))
        im = ax.pcolormesh(t, lam, M, cmap=cmap, vmin=-lim, vmax=lim, shading="nearest")
        ax.set_title(title, fontsize=9); ax.set_xlabel("t (µs)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_ylabel("λ")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    if SAVE_FIGURES:
        _save(fig, name)
    return fig


def _save(fig, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  figure saved: {path}")
    return path


if __name__ == "__main__":
    main()
