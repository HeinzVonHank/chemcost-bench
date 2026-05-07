#!/usr/bin/env python3
"""Noise influence figure (single panel, line chart).

CTA@25 (%) on the all set ($n=121$) under each noise stage, one line per
frontier ReAct backbone.

Style matches reaction_type.pdf: minimalist, ColorHunt palette, no panel
titles or axis labels, legend below the figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "manuscript/neurips_2025/figures/noise.pdf"

INK = "#2A2417"
MUTED = "#857A66"
GRID = "#E5DDD0"

# Per-model color (cool -> warm by typical robustness)
MODEL_STYLE = {
    "DeepSeek V4 Pro": ("#86cfcc", "s"),
    "Qwen3.5-Plus":    ("#9aaf7e", "o"),
    "Sonnet 4.6":      ("#d9a87a", "^"),
    "Kimi K2.5":       ("#f18982", "D"),
    "GPT-5":           ("#cc526a", "*"),
}
MODEL_ORDER = ["DeepSeek V4 Pro", "Qwen3.5-Plus", "Sonnet 4.6", "Kimi K2.5", "GPT-5"]

STAGES = ["Clean", "+Name", "+Qty", "+Miss", "+Fmt", "All"]

# CTA@25 (%) per (model, stage), combined single+multi (n=121)
# Numbers from the dev/multi merged dataset used in tab:main.
CTA25 = {
    "Qwen3.5-Plus":    [40.5, 30.9, 34.6, 32.1, 18.5, 19.8],
    "DeepSeek V4 Pro": [46.3, 38.3, 37.0, 34.6, 22.2, 19.8],
    "GPT-5":           [30.6, 23.5, 23.5, 22.2, 13.6, 11.1],
    "Kimi K2.5":       [29.8, 21.0, 29.6, 25.9, 12.3, 12.3],
    "Sonnet 4.6":      [25.6, 23.5, 16.1, 22.2, 11.1, 12.3],
}


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
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 1.6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(STAGES))
    for model in MODEL_ORDER:
        color, marker = MODEL_STYLE[model]
        y = CTA25[model]
        ax.plot(x, y, color=color, lw=1.4, marker=marker, ms=4.4,
                markeredgecolor=INK, markeredgewidth=0.5,
                label=model, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=5.9)
    ax.set_ylim(0, 55)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.grid(axis="y", color=GRID, lw=0.4, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=MODEL_STYLE[m][0], marker=MODEL_STYLE[m][1],
               markersize=4.4, markeredgecolor=INK, markeredgewidth=0.5,
               lw=1.4, label=m)
        for m in MODEL_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, frameon=False, handlelength=1.4,
               handletextpad=0.35, columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.10), fontsize=5.8)

    plt.subplots_adjust(left=0.06, right=0.995, top=0.96, bottom=0.22)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
