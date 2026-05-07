#!/usr/bin/env python3
"""Stage-level error decomposition figure.

Visualizes 4-stage pipeline hit rates per ReAct backbone:
  Stage 1 = Grounding (truth chemical present in predicted_components)
  Stage 2 = Retrieval (agent ran get_supplier_quotes on the right chemical)
  Stage 3 = Pack selection (chose the oracle pack within 5% on $/g)
  Stage 5 = Aggregation (truth-mass arithmetic reproduces agent's reported cost)

Each rate is conditional on the previous stage succeeding (S2 | S1, etc.).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/stage_analysis_summary.json"
OUT = ROOT / "manuscript/neurips_2025/figures/stage_pipeline.pdf"

INK = "#2A2417"
MUTED = "#857A66"
GRID = "#E5DDD0"

# Model order (frontier ReAct, ordered by Clean CTA@25 from tab:main)
MODEL_ORDER = ["Qwen3.5-Plus", "DS V4 Pro", "GPT-5", "Kimi K2.5", "Sonnet 4.6"]

# Stage colors (matching abstention.pdf palette: cool → warm)
STAGE_COLORS = {
    "S1: Grounding":   "#86cfcc",
    "S2: Retrieval":   "#9aaf7e",
    "S3: Pack":        "#fbd7b3",
    "S5: Aggregation": "#cc526a",
}
STAGE_KEYS = {
    "S1: Grounding":   "stage1_grounded_rate",
    "S2: Retrieval":   "stage2_retrieved_rate",
    "S3: Pack":        "stage3_pack_rate",
    "S5: Aggregation": "stage5_agg_rate",
}
STAGES = list(STAGE_COLORS.keys())


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
        "legend.fontsize": 5.6,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def halo():
    return [pe.withStroke(linewidth=1.4, foreground="white")]


def main():
    apply_style()
    summary = json.load(open(SUMMARY))
    by_model = {s["model"]: s for s in summary}

    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    fig.patch.set_facecolor("white")

    n_models = len(MODEL_ORDER)
    n_stages = len(STAGES)
    bar_w = 0.13
    intra_gap = 0.0
    group_w = n_stages * bar_w + (n_stages - 1) * intra_gap
    inter_gap = 0.22
    centers = []

    for gi, model in enumerate(MODEL_ORDER):
        if model not in by_model:
            centers.append(None)
            continue
        a = by_model[model]
        group_left = gi * (group_w + inter_gap)
        for si, stage in enumerate(STAGES):
            x_pos = group_left + si * (bar_w + intra_gap)
            v = a[STAGE_KEYS[stage]] * 100
            ax.bar(x_pos, v, bar_w, color=STAGE_COLORS[stage],
                   edgecolor=INK, linewidth=0.45, zorder=3)
            ax.text(x_pos + bar_w / 2, v + 1.5, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=5.0,
                    color=INK, weight="bold", path_effects=halo(), zorder=5)
        centers.append(group_left + group_w / 2 - bar_w / 2)

    ax.set_xticks([c for c in centers if c is not None])
    ax.set_xticklabels([m for m, c in zip(MODEL_ORDER, centers) if c is not None],
                       fontsize=5.9)
    ax.set_xlim(-0.10, max(c for c in centers if c is not None) + group_w / 2 + 0.08)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Stage hit rate (\\%)")
    ax.grid(axis="y", color=GRID, lw=0.4, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend below
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=STAGE_COLORS[s], edgecolor=INK,
                     linewidth=0.45, label=s) for s in STAGES]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=4,
               frameon=False, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.4, fontsize=5.8)

    plt.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.20)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print()
    for s in summary:
        print(f"{s['model']}: S1={100*s['stage1_grounded_rate']:.1f}% "
              f"S2|S1={100*s['stage2_retrieved_rate']:.1f}% "
              f"S3|S2={100*s['stage3_pack_rate']:.1f}% "
              f"S5={100*s['stage5_agg_rate']:.1f}%")


if __name__ == "__main__":
    main()
