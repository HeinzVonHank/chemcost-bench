#!/usr/bin/env python3
"""Reaction-type difficulty: 2x2 figure that decomposes ChemCost difficulty.

The earlier version showed three monotonic-decreasing curves and concluded
"harder reactions are harder." That collapses three correlated axes (route
depth, per-step component count, product MW) into one statement and does
not back the diagnostic claim. This version:

(A) Step depth as the headline axis -- aggregate CTA@25 with inter-model
    spread shown as shaded band. Marker size encodes per-bin n.
(B) Per-step component count, stratified by single/multi. Within
    single-step the curve is approximately flat; multi-step collapses
    above ~6 components.
(C) Product MW, stratified by single/multi. The MW effect collapses
    within single-step (disentanglement: MW was largely a route-depth
    proxy).
(D) Per-model difficulty fingerprint. Each model gets three sensitivity
    values (depth / per-step chemicals / MW within single-step). Different
    models live in different corners of this 3-axis space, supporting
    the "diagnostic benchmark" claim.

CTA@25 is computed per (model, bucket) and averaged across models.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "manuscript/neurips_2025/figures/reaction_type.pdf"

METRIC = os.environ.get("METRIC", "cta@25")
THRESHOLD = 0.10 if METRIC == "cta@10" else 0.25
if METRIC == "cta@10":
    OUT = ROOT / "manuscript/neurips_2025/figures/reaction_type_c10.pdf"

INK = "#2A2417"
MUTED = "#857A66"
GRID = "#E5DDD0"

COLOR_SINGLE = "#86cfcc"
COLOR_MULTI  = "#cc526a"
COLOR_AGG    = "#9aaf7e"
COLOR_STEP   = "#f18982"

MODELS = [
    ("Qwen3.5-Plus",   "dev_react_qwen35plus_clean.json",      "dev_multi_qwen35plus_clean.json"),
    ("DeepSeek V4 Pro","dev_react_deepseek_v4_pro_clean.json", "dev_multi40_deepseek_v4_pro_clean.json"),
    ("GPT-5",          "dev_react_gpt5_traj.json",             "dev_multi40_gpt5_clean.json"),
    ("Kimi K2.5",      "dev_react_kimi_k25_clean.json",        "dev_multi40_kimi_k25_clean.json"),
    ("Sonnet 4.6",     "dev_react_sonnet46_traj.json",         "dev_multi40_sonnet46_clean.json"),
]
N10_EXT_FN = {
    "Qwen3.5-Plus":    "test_multi_n10plus_qwen35plus.json",
    "DeepSeek V4 Pro": "test_multi_n10plus_deepseek_v4_pro.json",
    "Sonnet 4.6":      "test_multi_n10plus_sonnet46.json",
}

MODEL_COLOR = {
    "Qwen3.5-Plus":    "#86cfcc",
    "DeepSeek V4 Pro": "#FF7F0E",
    "GPT-5":           "#9aaf7e",
    "Kimi K2.5":       "#d9a87a",
    "Sonnet 4.6":      "#9467BD",
}


def load_truth():
    truth = {}
    for fp in (ROOT / "data/processed/splits/dev.jsonl",
               ROOT / "data/processed/splits/dev_multistep_v2.jsonl",
               ROOT / "data/processed/splits/test_multistep_n10plus.jsonl"):
        with fp.open() as f:
            for line in f:
                r = json.loads(line)
                truth[r["reaction_id"]] = r
    return truth


def load_preds(label, sfn, mfn):
    out = {}
    for fn in (sfn, mfn):
        p = R / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for pp in d.get("predictions", []):
            out[pp["reaction_id"]] = pp
    ext_fn = N10_EXT_FN.get(label)
    if ext_fn is not None:
        p_ext = R / ext_fn
        if p_ext.exists():
            d = json.loads(p_ext.read_text())
            for pp in d.get("predictions", []):
                out[pp["reaction_id"]] = pp
    return out


def n_nonsolv(t):
    return sum(1 for c in t["components"] if (c.get("role") or "").lower() != "solvent")


def is_single(t):
    return t.get("type") == "single_step"


def per_model_rate(models_preds, truth, axis_fn):
    """Returns {model_label: {bucket: (cta_pct, n)}}."""
    out = {}
    for label, preds in models_preds.items():
        per_b = defaultdict(lambda: {"n": 0, "hit": 0})
        for rid, p in preds.items():
            t = truth.get(rid)
            if t is None:
                continue
            bk = axis_fn(rid, t)
            if bk is None:
                continue
            per_b[bk]["n"] += 1
            tcre = p.get("tcre")
            if tcre is not None and tcre <= THRESHOLD:
                per_b[bk]["hit"] += 1
        out[label] = {bk: (b["hit"] / b["n"] * 100, b["n"]) for bk, b in per_b.items() if b["n"] > 0}
    return out


def aggregate_with_spread(per_model, min_models=3, min_n_per_model=5):
    """Aggregate per-model rates into mean ± std across models for each bucket.

    Per-model contributions with fewer than ``min_n_per_model`` reactions are
    dropped before averaging to prevent small-n survivorship inflation
    (e.g., GPT-5 / Kimi seeing only 2 of 9 high-MW multi-step reactions).
    """
    by_bucket = defaultdict(list)
    n_per_bucket = defaultdict(list)
    for label, rates in per_model.items():
        for bk, (rate, n) in rates.items():
            if n < min_n_per_model:
                continue
            by_bucket[bk].append(rate)
            n_per_bucket[bk].append(n)
    out = {}
    for bk, vals in by_bucket.items():
        if len(vals) < min_models:
            continue
        out[bk] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "n_models": len(vals),
            "n_react": int(np.mean(n_per_bucket[bk])),
        }
    return out


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "font.size": 6.6,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def halo():
    return [pe.withStroke(linewidth=1.4, foreground="white")]


def sized_marker(n):
    """Map per-bucket n to marker size (visual cue for sample count)."""
    return float(np.clip(8 + 1.0 * np.sqrt(max(n, 1)), 10, 60))


def panel_frame(ax, ylim=(-3, 55), yticks=(0, 10, 20, 30, 40, 50)):
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.grid(axis="y", color=GRID, lw=0.4, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    apply_style()
    truth = load_truth()
    models_preds = {label: load_preds(label, sfn, mfn) for label, sfn, mfn in MODELS}

    # ---- Axes definitions ----
    def step_axis(rid, t):
        return 1 if is_single(t) else t.get("n_steps")

    def comp_axis_strat(rid, t):
        n = n_nonsolv(t)
        typ = "single" if is_single(t) else "multi"
        return f"{typ}_{n}"

    # Last bin is open-ended (>=400) to stabilize the tail; the n=7 / n=9
    # MW>=450 buckets are too small on their own to be reliable.
    PMW_BINS = [(50, 150), (150, 250), (250, 350), (350, 1000)]
    PMW_X = [100, 200, 300, 450]

    def mw_bin(t):
        mw = t.get("product_mw")
        if mw is None:
            return None
        for i, (lo, hi) in enumerate(PMW_BINS):
            if lo <= mw < hi:
                return i
        return None

    def mw_axis_strat(rid, t):
        b = mw_bin(t)
        if b is None:
            return None
        return ("single" if is_single(t) else "multi", b)

    # ---- Per-model rates ----
    pm_step = per_model_rate(models_preds, truth, step_axis)
    pm_comp = per_model_rate(models_preds, truth, comp_axis_strat)
    pm_mw   = per_model_rate(models_preds, truth, mw_axis_strat)

    # Step depth: standard threshold drops step=6 (only 2--4 reactions per
    # model, sample too small to be reliable).
    agg_step = aggregate_with_spread(pm_step)
    agg_comp = aggregate_with_spread(pm_comp)
    # MW: relax per-model n threshold to 3 so the multi-step <150 g/mol bucket
    # is plotted (only 3 such reactions exist in the truth pool because
    # multi-step routes rarely converge to small products). The wide
    # shaded band on that point communicates the small-sample uncertainty.
    agg_mw   = aggregate_with_spread(pm_mw, min_n_per_model=3)

    # ============== Figure layout (2x2) ==============
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6),
                             gridspec_kw={"width_ratios": [1.0, 1.15],
                                          "height_ratios": [1.0, 1.0]})
    fig.patch.set_facecolor("white")
    axA, axB = axs[0]
    axC, axD = axs[1]

    # ============== Panel A: # steps ==============
    step_pts = sorted([(s, agg_step[s]) for s in agg_step
                       if isinstance(s, int) and 1 <= s <= 5])
    xs = [s for s, _ in step_pts]
    ys = [d["mean"] for _, d in step_pts]
    ss = [d["std"]  for _, d in step_pts]
    ns = [d["n_react"] for _, d in step_pts]

    xs_arr = np.array(xs); ys_arr = np.array(ys); ss_arr = np.array(ss)
    axA.fill_between(xs_arr, ys_arr - ss_arr, ys_arr + ss_arr,
                     color=COLOR_STEP, alpha=0.18, lw=0, zorder=2)
    axA.plot(xs_arr, ys_arr, color=COLOR_STEP, lw=1.4, zorder=3)
    for x, y, n in zip(xs, ys, ns):
        axA.scatter(x, y, s=sized_marker(n), c=COLOR_STEP, marker="o",
                    edgecolor=INK, linewidth=0.55, zorder=4)

    axA.set_xlim(0.5, 5.5)
    axA.set_xticks([1, 2, 3, 4, 5])
    axA.set_xlabel("Number of synthesis steps")
    axA.set_ylabel("CTA@25 (\\%)")
    panel_frame(axA)
    axA.text(0.0, 1.10, "(A)", transform=axA.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    # ============== Panel B: # chemicals (stratified by single/multi) ==============
    single_pts, multi_pts = [], []
    MIN_K, MIN_N = 3, 2
    SKIP_MULTI_N = {8}
    HIGH_THRESHOLD = 10
    for k, b in agg_comp.items():
        typ, n_str = k.split("_")
        n = int(n_str)
        if n < MIN_N:
            continue
        if typ == "single":
            if b["n_react"] >= MIN_K:
                single_pts.append((n, b["mean"], b["std"], b["n_react"]))
        else:
            if n >= HIGH_THRESHOLD or n in SKIP_MULTI_N:
                continue
            if b["n_react"] >= MIN_K:
                multi_pts.append((n, b["mean"], b["std"], b["n_react"]))

    # Tail buckets for multi >= 10
    def multi_tail(rid, t):
        if is_single(t):
            return None
        n = n_nonsolv(t)
        if n < 10: return None
        if n <= 12: return "10-12"
        if n <= 15: return "13-15"
        return "16+"
    pm_tail = per_model_rate(models_preds, truth, multi_tail)
    agg_tail = aggregate_with_spread(pm_tail)
    TAIL_X = {"10-12": 11, "13-15": 14, "16+": 17}
    for bk in ("10-12", "13-15", "16+"):
        if bk in agg_tail:
            b = agg_tail[bk]
            multi_pts.append((TAIL_X[bk], b["mean"], b["std"], b["n_react"]))
    single_pts.sort(); multi_pts.sort()

    OVERLAP_DX = 0.15
    s_xset = {p[0] for p in single_pts}
    m_xset = {p[0] for p in multi_pts}
    shared = s_xset & m_xset

    def draw_strat(ax, pts, color, marker, label, dx_sign):
        if not pts:
            return
        xs = np.array([p[0] + (dx_sign * OVERLAP_DX if p[0] in shared else 0) for p in pts])
        ys = np.array([p[1] for p in pts])
        ss = np.array([p[2] for p in pts])
        ax.fill_between(xs, ys - ss, ys + ss, color=color, alpha=0.15, lw=0, zorder=2)
        ax.plot(xs, ys, color=color, lw=1.4, zorder=3, label=label)
        for x, p in zip(xs, pts):
            ax.scatter(x, p[1], s=sized_marker(p[3]), c=color, marker=marker,
                       edgecolor=INK, linewidth=0.55, zorder=4)

    draw_strat(axB, single_pts, COLOR_SINGLE, "o", "Single-step", -1)
    draw_strat(axB, multi_pts,  COLOR_MULTI,  "s", "Multi-step", +1)
    axB.axvline(6, color=MUTED, lw=0.5, ls=(0, (1.5, 2.0)), zorder=1)
    axB.set_xlim(1, 19)
    axB.set_xticks([2, 3, 4, 5, 6, 7, 11, 14, 17])
    axB.set_xlabel("Number of non-solvent chemicals")
    panel_frame(axB)
    axB.tick_params(labelleft=False)
    axB.text(0.0, 1.10, "(B)", transform=axB.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    # ============== Panel C: MW stratified by single/multi (disentanglement) ==============
    single_mw_pts = []
    multi_mw_pts = []
    for (typ, idx), b in agg_mw.items():
        x = PMW_X[idx]
        pt = (x, b["mean"], b["std"], b["n_react"])
        if typ == "single":
            single_mw_pts.append(pt)
        else:
            multi_mw_pts.append(pt)
    single_mw_pts.sort(); multi_mw_pts.sort()

    def draw_mw(ax, pts, color, marker, label):
        if not pts:
            return
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        ss = np.array([p[2] for p in pts])
        ax.fill_between(xs, ys - ss, ys + ss, color=color, alpha=0.15, lw=0, zorder=2)
        ax.plot(xs, ys, color=color, lw=1.4, zorder=3, label=label)
        for x, p in zip(xs, pts):
            ax.scatter(x, p[1], s=sized_marker(p[3]), c=color, marker=marker,
                       edgecolor=INK, linewidth=0.55, zorder=4)

    draw_mw(axC, single_mw_pts, COLOR_SINGLE, "o", "Single-step")
    draw_mw(axC, multi_mw_pts,  COLOR_MULTI,  "s", "Multi-step")
    axC.set_xlim(60, 510)
    axC.set_xticks([100, 200, 300, 450])
    axC.set_xticklabels(["100", "200", "300", "$\\geq$400"])
    axC.set_xlabel("Product molecular weight (g/mol)")
    axC.set_ylabel("CTA@25 (\\%)")
    panel_frame(axC)
    axC.text(0.0, 1.10, "(C)", transform=axC.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    # ============== Panel D: per-model fingerprint ==============
    # depth_sens: CTA(steps=1) - CTA(steps>=3)
    # chem_sens : CTA(chemicals<=4, single) - CTA(chemicals>=7, single)
    # mw_sens   : CTA(MW<300, single) - CTA(MW>=300, single)
    def per_model_aggregate(preds_for_model, truth, predicate):
        n = hit = 0
        for rid, p in preds_for_model.items():
            t = truth.get(rid)
            if t is None or not predicate(t):
                continue
            n += 1
            tcre = p.get("tcre")
            if tcre is not None and tcre <= THRESHOLD:
                hit += 1
        return (hit / n * 100 if n else None, n)

    def cmp_axis(preds, truth, pred_lo, pred_hi):
        lo_rate, lo_n = per_model_aggregate(preds, truth, pred_lo)
        hi_rate, hi_n = per_model_aggregate(preds, truth, pred_hi)
        if lo_rate is None or hi_rate is None:
            return None
        return lo_rate - hi_rate, lo_n, hi_n

    # Single-step component counts span 1..6 only; use 3 / 5 as the split.
    AXES = [
        ("depth", lambda t: is_single(t),
                  lambda t: (not is_single(t)) and (t.get("n_steps") or 0) >= 3),
        ("# chem (single)",
                  lambda t: is_single(t) and n_nonsolv(t) <= 3,
                  lambda t: is_single(t) and n_nonsolv(t) >= 5),
        ("MW (single)",
                  lambda t: is_single(t) and (t.get("product_mw") or 0) < 300,
                  lambda t: is_single(t) and (t.get("product_mw") or 0) >= 300),
    ]
    sens = {label: {} for label, *_ in MODELS}
    for label, preds in models_preds.items():
        for ax_label, lo_p, hi_p in AXES:
            r = cmp_axis(preds, truth, lo_p, hi_p)
            sens[label][ax_label] = r

    n_models = len(MODELS)
    n_axes = len(AXES)
    bar_w = 0.22
    x_centers = np.arange(n_models)
    for j, (ax_label, _, _) in enumerate(AXES):
        offset = (j - (n_axes - 1) / 2) * (bar_w + 0.02)
        for i, (label, *_ ) in enumerate(MODELS):
            d = sens[label][ax_label]
            if d is None:
                continue
            delta, lo_n, hi_n = d
            color = ["#86cfcc", "#fbd7b3", "#cc526a"][j]
            axD.bar(x_centers[i] + offset, delta, bar_w,
                    color=color, edgecolor=INK, linewidth=0.45, zorder=3)
    axD.axhline(0, color=INK, lw=0.5, zorder=2)
    axD.set_xticks(x_centers)
    axD.set_xticklabels([label for label, *_ in MODELS], rotation=30, ha="right")
    axD.set_ylabel("$\\Delta$ CTA@25 (low $-$ high, pp)")
    axD.set_ylim(-15, 60)
    axD.set_yticks([-10, 0, 10, 20, 30, 40, 50])
    axD.grid(axis="y", color=GRID, lw=0.4, alpha=0.7, zorder=0)
    axD.spines["top"].set_visible(False)
    axD.spines["right"].set_visible(False)
    axD.text(0.0, 1.10, "(D)", transform=axD.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    # Combined external legend below the figure: Single/Multi line styles
    # (used by B and C) + the three D bar colours.
    sm_handles = [
        Line2D([0], [0], marker="o", color=COLOR_SINGLE,
               markerfacecolor=COLOR_SINGLE, markeredgecolor=INK,
               markeredgewidth=0.55, markersize=5, lw=1.4, label="Single-step"),
        Line2D([0], [0], marker="s", color=COLOR_MULTI,
               markerfacecolor=COLOR_MULTI, markeredgecolor=INK,
               markeredgewidth=0.55, markersize=4.5, lw=1.4, label="Multi-step"),
    ]
    bar_colors = ["#86cfcc", "#fbd7b3", "#cc526a"]
    d_handles = [
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=bar_colors[j],
               markeredgecolor=INK, markeredgewidth=0.45, markersize=6,
               label=lab)
        for j, (lab, *_ ) in enumerate(AXES)
    ]
    fig.legend(handles=sm_handles + d_handles,
               loc="lower center", ncol=5, frameon=False,
               handlelength=1.4, handletextpad=0.4, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.02), fontsize=6.0)

    plt.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.18,
                        wspace=0.18, hspace=0.45)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print()
    print("(A) steps: mean ± std across models | n_react")
    for x, _ in step_pts:
        d = agg_step[x]
        print(f"  steps={x}  mean={d['mean']:5.1f}  std={d['std']:4.1f}  "
              f"models={d['n_models']}  n_react/model={d['n_react']}")
    print()
    print("(C) MW × stratified:")
    for typ, pts in (("single", single_mw_pts), ("multi", multi_mw_pts)):
        for x, m, s, n in pts:
            print(f"  {typ:6s} MW~{x:.0f}  mean={m:5.1f}  std={s:4.1f}  n={n}")
    print()
    print("(D) sensitivity (Δpp): depth | # chem (single) | MW (single)")
    for label, *_ in MODELS:
        s = sens[label]
        cells = []
        for ax_label, *_ in AXES:
            v = s[ax_label]
            cells.append("  n/a   " if v is None else f"{v[0]:+6.1f}")
        print(f"  {label:<18s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
