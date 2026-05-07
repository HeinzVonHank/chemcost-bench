#!/usr/bin/env python3
"""Abstention + noise sub-type analysis figure (3 panels).

A. Abstention rate (%) under each noise stage, grouped bars per model.
B. Mechanism breakdown of abstentions under +Fmt: stacked bar per model
   showing Active / Budget / Refusal share of that model's abstentions.
C. Sub-type breakdown of CTA@1 / CTA@10 / CTA@25 / Precision / Recall
   under +Qty (mol% / vague / unit switch / approximate) and +Fmt
   (OCR only / NL only / NL+OCR), pooled across DeepSeek V4 Pro and
   Sonnet 4.6 at level=rich on the all set (n=121).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "manuscript/neurips_2025/figures/abstention.pdf"

INK = "#2A2417"
MUTED = "#857A66"
GRID = "#E5DDD0"

MODEL_ORDER = ["DeepSeek V4 Pro", "Qwen3.5-Plus", "Sonnet 4.6", "Kimi K2.5", "GPT-5"]

# 6-color noise-stage palette
STAGE_COLORS = {
    "Clean":     "#86cfcc",
    "+Name":     "#9aaf7e",
    "+Qty":      "#fbd7b3",
    "+Miss":     "#d9a87a",
    "+Fmt":      "#f18982",
    "All Noise": "#cc526a",
}

# Mechanism colors
MECH_COLORS = {
    "Active":  "#9aaf7e",
    "Budget":  "#d9a87a",
    "Refusal": "#cc526a",
}
MECH_ORDER = ["Active", "Budget", "Refusal"]

# Panel C metric colors (reuse the same palette as Panels A/B for visual cohesion)
METRIC_COLORS = {
    "CTA@1":     "#86cfcc",
    "CTA@10":    "#fbd7b3",
    "CTA@25":    "#cc526a",
    "Precision": "#9aaf7e",
    "Recall":    "#d9a87a",
}
METRIC_ORDER = ["CTA@1", "CTA@10", "CTA@25", "Precision", "Recall"]

STAGES = ["Clean", "+Name", "+Qty", "+Miss", "+Fmt", "All Noise"]

# ---- Panel A/B data ----
ABST_RATE = {
    "Qwen3.5-Plus":    [27.3, 39.7, 28.1, 42.1, 53.7, 68.6],
    "DeepSeek V4 Pro": [ 7.4,  5.0,  5.8,  5.0, 29.8, 24.0],
    "GPT-5":           [54.5, 57.9, 57.0, 59.5, 75.2, 81.8],
    "Kimi K2.5":       [35.5, 45.5, 38.8, 38.0, 69.4, 63.6],
    "Sonnet 4.6":      [28.9, 34.7, 28.9, 38.0, 57.0, 62.8],
}

MECH_PCT = {
    "Qwen3.5-Plus":    {"Active": 95, "Budget":   0, "Refusal":  5},
    "DeepSeek V4 Pro": {"Active":  0, "Budget": 100, "Refusal":  0},
    "GPT-5":           {"Active": 73, "Budget":   0, "Refusal": 27},
    "Kimi K2.5":       {"Active":  0, "Budget":  29, "Refusal": 71},
    "Sonnet 4.6":      {"Active": 94, "Budget":   0, "Refusal":  6},
}

# ---- Panel C data: (label, CTA@1, CTA@10, CTA@25, Precision, Recall) ----
QTY_DATA = [
    ("mol%",        20.0, 36.7, 56.7, 88.9, 63.3),
    ("vague",        0.0, 14.7, 29.4, 92.3, 58.3),
    ("unit switch",  8.3, 14.6, 29.2, 92.7, 58.1),
    ("approximate",  5.6, 16.2, 28.9, 91.4, 65.6),
]
FMT_DATA = [
    ("OCR only", 8.7, 21.9, 35.5, 76.5, 67.8),
    ("NL only",  6.6, 12.4, 21.1, 84.0, 48.7),
    ("NL+OCR",   7.4, 12.0, 20.7, 79.2, 48.5),
]


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
    # Width ratios: A is the widest (5 models x 6 stages); B is narrowest
    # (5 stacked bars); C has 7 subtypes x 5 metrics.
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(11.0, 2.6),
        gridspec_kw={"width_ratios": [1.5, 0.85, 1.5]},
    )
    fig.patch.set_facecolor("white")

    # ============== Panel A: model x stage abstention rate ==============
    n_models = len(MODEL_ORDER)
    n_stages = len(STAGES)
    bar_w = 0.13
    intra_gap = 0.0
    group_w = n_stages * bar_w + (n_stages - 1) * intra_gap
    inter_gap = 0.22
    centers = []
    for gi, model in enumerate(MODEL_ORDER):
        group_left = gi * (group_w + inter_gap)
        for si, stage in enumerate(STAGES):
            x_pos = group_left + si * (bar_w + intra_gap)
            y = ABST_RATE[model][si]
            axA.bar(x_pos, y, bar_w, color=STAGE_COLORS[stage],
                    edgecolor=INK, linewidth=0.45, zorder=3,
                    align="edge")
        centers.append(group_left + group_w / 2 - bar_w / 2)

    axA.set_xticks(centers)
    axA.set_xticklabels(MODEL_ORDER, fontsize=5.9)
    axA.set_ylabel("Abstention rate (\\%)")
    axA.set_ylim(0, 100)
    axA.set_yticks([0, 25, 50, 75, 100])
    axA.grid(axis="y", color=GRID, lw=0.4, alpha=0.7, zorder=0)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.set_title("Abstention rate by noise stage", fontsize=6.8,
                   color=INK, pad=3)
    axA.text(0.0, 1.10, "(A)", transform=axA.transAxes,
             fontsize=7.4, weight="bold", color=INK)
    axA.set_xlim(-0.10, centers[-1] + group_w / 2 + 0.08)

    handles_a = [Patch(facecolor=STAGE_COLORS[s], edgecolor=INK,
                       linewidth=0.45, label=s) for s in STAGES]
    axA.legend(handles=handles_a, loc="upper center",
               bbox_to_anchor=(0.5, -0.30), ncol=6,
               frameon=False, handlelength=1.0,
               handletextpad=0.35, columnspacing=0.9,
               labelspacing=0.25, borderpad=0.2)

    # ============== Panel B: mechanism stacked bar ==============
    n = len(MODEL_ORDER)
    xb = np.arange(n)
    bar_w_b = 0.66
    bottom = np.zeros(n)
    for mech in MECH_ORDER:
        heights = np.array([MECH_PCT[m][mech] for m in MODEL_ORDER])
        axB.bar(xb, heights, bar_w_b, bottom=bottom,
                color=MECH_COLORS[mech], linewidth=0, zorder=3,
                label=mech)
        bottom += heights

    for i in range(n):
        axB.add_patch(Rectangle((xb[i] - bar_w_b / 2, 0), bar_w_b, 100,
                                facecolor="none", edgecolor=INK,
                                linewidth=0.55, zorder=4))

    axB.set_xticks(xb)
    short_labels = ["DSV4 Pro", "Qwen3.5-Plus", "Sonnet 4.6", "Kimi K2.5", "GPT-5"]
    axB.set_xticklabels(short_labels, rotation=22, ha="right", fontsize=5.7)
    axB.set_ylabel("Share of abstentions (\\%)")
    axB.set_ylim(0, 102)
    axB.set_yticks([0, 25, 50, 75, 100])
    axB.grid(axis="y", color=GRID, lw=0.4, alpha=0.7, zorder=0)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.set_title("Mechanism under +Fmt", fontsize=6.8, color=INK, pad=3)
    axB.text(0.0, 1.10, "(B)", transform=axB.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    handles_b = [Patch(facecolor=MECH_COLORS[m], edgecolor=INK,
                       linewidth=0.55, label=m) for m in MECH_ORDER]
    axB.legend(handles=handles_b, loc="upper center",
               bbox_to_anchor=(0.5, -0.30), ncol=3,
               frameon=False, handlelength=1.0, handletextpad=0.4,
               columnspacing=0.9, borderpad=0.2)

    # ============== Panel C: noise sub-type metric breakdown ==============
    combined = QTY_DATA + FMT_DATA
    n_sub = len(combined)
    n_metrics = len(METRIC_ORDER)
    bar_w_c = 0.13
    intra_gap_c = 0.0
    group_w_c = n_metrics * bar_w_c + (n_metrics - 1) * intra_gap_c
    inter_gap_c = 0.18
    centers_c = []
    for i, row in enumerate(combined):
        group_left = i * (group_w_c + inter_gap_c)
        vals = list(row[1:])
        colors = [METRIC_COLORS[m] for m in METRIC_ORDER]
        for j, (v, c) in enumerate(zip(vals, colors)):
            x_pos = group_left + j * (bar_w_c + intra_gap_c)
            axC.bar(x_pos, v, bar_w_c, color=c,
                    edgecolor=INK, linewidth=0.45, zorder=3,
                    align="edge")
        centers_c.append(group_left + group_w_c / 2 - bar_w_c / 2)

    # Vertical separator between +Qty and +Fmt sub-groups
    n_qty = len(QTY_DATA)
    sep_x = n_qty * (group_w_c + inter_gap_c) - inter_gap_c / 2
    axC.axvline(sep_x, color=MUTED, lw=0.5, ls=(0, (1.5, 2.0)), zorder=1)

    axC.set_xticks(centers_c)
    axC.set_xticklabels([d[0] for d in combined], fontsize=5.7,
                         rotation=22, ha="right")
    axC.set_ylabel("Score (\\%)")
    axC.set_ylim(0, 102)
    axC.set_yticks([0, 25, 50, 75, 100])
    axC.grid(axis="y", color=GRID, lw=0.4, alpha=0.7, zorder=0)
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)
    axC.set_title("Sub-type metrics: +Qty | +Fmt", fontsize=6.8,
                   color=INK, pad=3)
    axC.text(0.0, 1.10, "(C)", transform=axC.transAxes,
             fontsize=7.4, weight="bold", color=INK)
    axC.set_xlim(-0.10, centers_c[-1] + group_w_c / 2 + 0.08)

    handles_c = [Patch(facecolor=METRIC_COLORS[m], edgecolor=INK,
                       linewidth=0.45, label=m) for m in METRIC_ORDER]
    axC.legend(handles=handles_c, loc="upper center",
               bbox_to_anchor=(0.5, -0.30), ncol=5,
               frameon=False, handlelength=1.0, handletextpad=0.4,
               columnspacing=1.0, labelspacing=0.25, borderpad=0.2)

    plt.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.26,
                        wspace=0.22)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
