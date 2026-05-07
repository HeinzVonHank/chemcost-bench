#!/usr/bin/env python3
"""Tool calling analysis (2 panels).

A. Mean tool calls per reaction, stacked by tool type, **split by trajectory
   outcome** (success: TCRE<=0.25 / fail: TCRE>0.25). Solid bar = success,
   hatched bar = fail. Same model, same tool color palette; the contrast
   exposes per-model differences in *how* tools are used between successful
   and failed runs (e.g., DeepSeek fail bars are taller — stuck retrieval
   loops; GPT-5 fail bars are shorter — early termination). Models with
   <10 successes are dropped from Panel A (kept in Panel B).

B. Mean tool calls per reaction vs. CTA@25 across all evaluated models.
   The cross-model correlation is high but, as Panel A shows, does not
   propagate to a within-model effect.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "manuscript/neurips_2025/figures/tool_usage.pdf"

INK = "#2A2417"
MUTED = "#857A66"
GRID = "#E5DDD0"

# Tool color palette (ColorHunt-style: teal / olive / peach / deep pink)
TOOL_COLORS = {
    "search_chemical":      "#86cfcc",
    "get_supplier_quotes":  "#fbd7b3",
    "compute_molar_mass":   "#9aaf7e",
    "calculate":            "#cc526a",
}
TOOL_LABEL = {
    "search_chemical":     "search_chemical",
    "get_supplier_quotes": "get_supplier_quotes",
    "compute_molar_mass":  "compute_molar_mass",
    "calculate":           "calculate",
}
TOOL_ORDER = ["search_chemical", "get_supplier_quotes", "compute_molar_mass", "calculate"]

# Per-model marker + color for panel B.
MODEL_STYLE = {
    "Qwen3.5-Plus":    ("o", "#86cfcc"),
    "DeepSeek V4 Pro": ("s", "#FF7F0E"),
    "GPT-5":           ("^", "#9aaf7e"),
    "Kimi K2.5":       ("D", "#d9a87a"),
    "Sonnet 4.6":      ("*", "#9467BD"),
    "Qwen3-235B":      ("P", "#E377C2"),
    "LlaSMol-7B":      ("X", "#7F7F7F"),
    "ChemDFM":         ("p", "#BCBD22"),
    "ChemLLM":         ("h", "#17BECF"),
}

MODELS = [
    ("Qwen3.5-Plus",   "dev_react_qwen35plus_clean.json",      "dev_multi_qwen35plus_clean.json"),
    ("DeepSeek V4 Pro","dev_react_deepseek_v4_pro_clean.json", "dev_multi40_deepseek_v4_pro_clean.json"),
    ("GPT-5",          "dev_react_gpt5_traj.json",             "dev_multi40_gpt5_clean.json"),
    ("Kimi K2.5",      "dev_react_kimi_k25_clean.json",        "dev_multi40_kimi_k25_clean.json"),
    ("Sonnet 4.6",     "dev_react_sonnet46_traj.json",         "dev_multi40_sonnet46_clean.json"),
    ("Qwen3-235B",     "dev_react_qwen3_235b_a22b_clean.json", "dev_multi40_qwen3_235b_a22b_clean.json"),
    ("LlaSMol-7B",     "dev_react_llasmol_7b_clean.json",      "dev_multi40_react_llasmol_7b_clean.json"),
    ("ChemDFM",        "dev_react_chemdfm_v2_clean.json",      "dev_multi40_react_chemdfm_v2_clean.json"),
    ("ChemLLM",        "dev_react_chemllm_20b_clean.json",     "dev_multi40_react_chemllm_20b_clean.json"),
]

MIN_PER_GROUP = 10  # Panel A inclusion threshold for both succ and fail


def classify(rec: dict) -> str:
    pc = rec.get("predicted_cost")
    t = rec.get("tcre")
    if pc is None or t is None:
        return "abstain"
    return "success" if t <= 0.25 else "fail"


def collect():
    """Return per-model aggregates including success/fail-stratified tool counts."""
    rows = []
    for label, sfn, mfn in MODELS:
        n_total = 0
        cta25_w = 0.0
        rec_w = 0.0
        tool_counter = Counter()
        # Per-class accumulators for Panel A
        n_by_class = defaultdict(int)
        tool_by_class = defaultdict(Counter)
        for fn in [sfn, mfn]:
            p = R / fn
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            n = d["metrics"]["n_total"]
            cta25_w += d["metrics"]["cta@25"] * n
            rec = d["metrics"].get("component_recall") or 0
            rec_w += rec * n
            n_total += n
            for pred in d.get("predictions", []):
                cls = classify(pred)
                n_by_class[cls] += 1
                for tc in pred.get("tool_calls") or []:
                    tn = tc.get("tool_name", "")
                    if tn in TOOL_COLORS:
                        tool_counter[tn] += 1
                        tool_by_class[cls][tn] += 1
        if n_total == 0:
            continue
        # Per-class mean tool calls per reaction (only computed when class non-empty)
        per_class = {}
        for cls in ("success", "fail"):
            if n_by_class[cls] > 0:
                per_class[cls] = {
                    "n": n_by_class[cls],
                    "by_tool": {t: tool_by_class[cls][t] / n_by_class[cls] for t in TOOL_ORDER},
                    "mean_total": sum(tool_by_class[cls].values()) / n_by_class[cls],
                }
        rows.append({
            "label": label,
            "n": n_total,
            "n_by_class": dict(n_by_class),
            "cta25": cta25_w / n_total * 100,
            "recall": rec_w / n_total * 100,
            "mean_total": sum(tool_counter.values()) / n_total,
            "by_tool": {t: tool_counter[t] / n_total for t in TOOL_ORDER},
            "per_class": per_class,
        })
    return rows


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
        "xtick.labelsize": 5.9,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 0,
        "ytick.major.size": 2.0,
        "hatch.linewidth": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def halo():
    return [pe.withStroke(linewidth=1.7, foreground="white")]


def main():
    apply_style()
    rows = collect()

    # Panel A = cross-model scatter (sets up the headline correlation);
    # Panel B = within-model success/fail bars (the contradiction).
    # Variables: axA holds bars (drawn on right), axB holds scatter (left) —
    # swap the unpacking so axB is the first/left axes.
    fig, (axB, axA) = plt.subplots(1, 2, figsize=(6.4, 2.5),
                                    gridspec_kw={"width_ratios": [1.0, 1.4]})
    fig.patch.set_facecolor("white")

    # ============== Panel B: Stacked bars, succ vs fail per model ==============
    eligible = [r for r in rows
                if r["per_class"].get("success", {}).get("n", 0) >= MIN_PER_GROUP
                and r["per_class"].get("fail", {}).get("n", 0) >= MIN_PER_GROUP]
    # Order by mean total over succ+fail (descending)
    eligible.sort(key=lambda r: -((r["per_class"]["success"]["mean_total"]
                                   + r["per_class"]["fail"]["mean_total"]) / 2))
    n = len(eligible)
    # Halve the inter-group stride (was 1.0) to tighten bar spacing.
    x = np.arange(n) * 0.5
    bar_w = 0.18
    off_s = -bar_w / 2 - 0.01
    off_f = bar_w / 2 + 0.01

    bottom_s = np.zeros(n)
    bottom_f = np.zeros(n)
    for tool in TOOL_ORDER:
        h_s = np.array([r["per_class"]["success"]["by_tool"][tool] for r in eligible])
        h_f = np.array([r["per_class"]["fail"]["by_tool"][tool] for r in eligible])
        axA.bar(x + off_s, h_s, bar_w, bottom=bottom_s,
                color=TOOL_COLORS[tool], linewidth=0, zorder=3)
        # For fail bars, edgecolor=INK + linewidth=0 keeps the hatch lines black
        # without drawing internal stack borders; outer outline is added below.
        axA.bar(x + off_f, h_f, bar_w, bottom=bottom_f,
                color=TOOL_COLORS[tool], linewidth=0, zorder=3,
                hatch="///", edgecolor=INK)
        bottom_s += h_s
        bottom_f += h_f

    # Outer outlines for each bar
    for i, r in enumerate(eligible):
        h_s = r["per_class"]["success"]["mean_total"]
        h_f = r["per_class"]["fail"]["mean_total"]
        if h_s > 0.05:
            axA.add_patch(Rectangle((x[i] + off_s - bar_w / 2, 0), bar_w, h_s,
                                    facecolor="none", edgecolor=INK,
                                    linewidth=0.55, zorder=4))
        if h_f > 0.05:
            axA.add_patch(Rectangle((x[i] + off_f - bar_w / 2, 0), bar_w, h_f,
                                    facecolor="none", edgecolor=INK,
                                    linewidth=0.55, zorder=4))
        # Tiny S / F annotations directly below bars
        axA.text(x[i] + off_s, -1.2, "S", ha="center", va="top",
                 fontsize=5.5, color=MUTED)
        axA.text(x[i] + off_f, -1.2, "F", ha="center", va="top",
                 fontsize=5.5, color=MUTED)

    axA.set_xticks(x)
    axA.set_xticklabels([r["label"] for r in eligible], rotation=18, ha="right")
    # Push model labels down a bit so the S/F markers don't collide
    axA.tick_params(axis="x", pad=8)
    axA.set_ylabel("Mean tool calls per reaction")
    ymax = max(r["per_class"]["fail"]["mean_total"] for r in eligible)
    ymax = max(ymax, max(r["per_class"]["success"]["mean_total"] for r in eligible))
    ytop = int(np.ceil((ymax + 2) / 5.0) * 5)
    axA.set_ylim(-2.5, ytop)
    axA.set_yticks(np.arange(0, ytop + 1, 5))
    axA.grid(axis="y", color=GRID, lw=0.4, alpha=0.7, zorder=0)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.spines["bottom"].set_position(("data", 0))

    axA.text(0.0, 1.08, "(B)", transform=axA.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    # ============== Panel A: Scatter ==============
    JITTER = {
        "ChemDFM":     (-1.0,  0.0),
        "ChemLLM":     ( 1.0,  0.0),
        "LlaSMol-7B":  ( 0.0,  1.5),
        "GPT-5":       (-0.6,  1.2),
        "Kimi K2.5":   ( 0.6, -1.0),
    }
    pos = {}
    for r in rows:
        jx, jy = JITTER.get(r["label"], (0.0, 0.0))
        pos[r["label"]] = (r["mean_total"] + jx, r["cta25"] + jy)

    for r in rows:
        marker, color = MODEL_STYLE[r["label"]]
        size = 38 if marker == "*" else 24
        xv, yv = pos[r["label"]]
        axB.scatter(xv, yv, s=size, c=color, marker=marker,
                    edgecolor=INK, linewidth=0.55,
                    alpha=0.95, zorder=3, label=r["label"])

    axB.set_xlabel("Mean tool calls per reaction")
    axB.set_ylabel("CTA@25 (\\%)")
    axB.set_xlim(-3.5, 46)
    axB.set_ylim(-3, 55)
    axB.set_xticks([0, 10, 20, 30, 40])
    axB.grid(color=GRID, lw=0.4, alpha=0.6, zorder=0)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    axB.text(0.0, 1.08, "(A)", transform=axB.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    xs = np.array([r["mean_total"] for r in rows])
    ys = np.array([r["cta25"] for r in rows])
    if len(xs) >= 3 and xs.std() > 0 and ys.std() > 0:
        from scipy import stats as _stats
        n_pts = len(xs)
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 100)
        y_line = slope * x_line + intercept
        residuals = ys - (slope * xs + intercept)
        s_err = np.sqrt(np.sum(residuals ** 2) / (n_pts - 2))
        x_mean = xs.mean()
        sxx = np.sum((xs - x_mean) ** 2)
        se_mean = s_err * np.sqrt(1.0 / n_pts + (x_line - x_mean) ** 2 / sxx)
        t_crit = _stats.t.ppf(0.975, n_pts - 2)
        band = t_crit * se_mean
        axB.fill_between(x_line, y_line - band, y_line + band,
                         color=MUTED, alpha=0.15, lw=0, zorder=1)
        axB.plot(x_line, y_line, color=MUTED, lw=0.7, ls=(0, (3, 2)),
                 alpha=0.85, zorder=2)

    # Tool legend below Panel A, plus succ/fail style legend
    tool_handles = [Patch(facecolor=TOOL_COLORS[t], edgecolor=INK, linewidth=0.55,
                          label=TOOL_LABEL[t]) for t in TOOL_ORDER]
    style_handles = [
        Patch(facecolor="#dddddd", edgecolor=INK, linewidth=0.55,
              label="success (TCRE $\\leq$ 0.25)"),
        Patch(facecolor="#dddddd", edgecolor=INK, linewidth=0.55,
              hatch="///", label="fail (TCRE $>$ 0.25)"),
    ]
    axA.legend(handles=tool_handles + style_handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.30), ncol=3,
               frameon=False, handlelength=1.2, handletextpad=0.4,
               columnspacing=1.0, labelspacing=0.3, fontsize=5.7)

    # Model legend below Panel B
    from matplotlib.lines import Line2D
    rows_for_legend = sorted(rows, key=lambda r: -r["mean_total"])
    model_handles = []
    for r in rows_for_legend:
        marker, color = MODEL_STYLE[r["label"]]
        ms = 6.5 if marker == "*" else 4.8
        model_handles.append(Line2D([0], [0], marker=marker, color="none",
                                    markerfacecolor=color, markeredgecolor=INK,
                                    markeredgewidth=0.55, markersize=ms,
                                    label=r["label"]))
    axB.legend(handles=model_handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.28), ncol=3,
               frameon=False, handlelength=1.0, handletextpad=0.5,
               columnspacing=1.2, labelspacing=0.35, fontsize=5.6)

    plt.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.36, wspace=0.28)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print()
    print("Panel A models (succ / fail mean total calls):")
    for r in eligible:
        s = r["per_class"]["success"]
        f = r["per_class"]["fail"]
        print(f"  {r['label']:<18s}  S(n={s['n']:3d})={s['mean_total']:5.1f}   "
              f"F(n={f['n']:3d})={f['mean_total']:5.1f}")
    skipped = [r['label'] for r in rows if r not in eligible]
    if skipped:
        print(f"\nSkipped from Panel A (insufficient succ or fail): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
