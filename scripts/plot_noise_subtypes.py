#!/usr/bin/env python3
"""Noise sub-type analysis figure (3 panels).

Single backbone: DeepSeek V4 Pro at level=rich. Each panel decomposes a
noise stage into its sub-types and shows CTA@25.

A. +Qty (Requantification): mol% / unit_switch / approximate / vague
B. +Miss (Omission): mw_dropped / role_dropped / mw_and_role_dropped
C. +Fmt (Reformatting): nl_only / ocr_only / nl_plus_ocr (3 separate runs)

Style matches reaction_type.pdf and tool_usage.pdf (ColorHunt palette,
black bar outlines, compact ~6.6 in figure).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[1] / "manuscript/neurips_2025/figures/noise_subtypes.pdf"

INK = "#2A2417"
MUTED = "#857A66"
GRID = "#E5DDD0"

# ColorHunt palette (cool -> warm)
COL_TEAL  = "#86cfcc"
COL_SAGE  = "#9aaf7e"
COL_PEACH = "#fbd7b3"
COL_TAN   = "#d9a87a"
COL_CORAL = "#f18982"
COL_PINK  = "#cc526a"

# Sub-type metrics, raw counts pooled across 2 frontier backbones (DeepSeek
# V4 Pro and Sonnet 4.6) at level=rich, on the combined all set (n=121).
# Tuple: (label, CTA@1, CTA@10, CTA@25, Precision, Recall).
CLEAN_DATA = [
    ("overall", 9.9, 25.6, 36.0, 75.4, 66.2),
]
QTY_DATA = [
    ("mol%",        20.0, 36.7, 56.7, 88.9, 63.3),
    ("vague",        0.0, 14.7, 29.4, 92.3, 58.3),
    ("unit switch",  8.3, 14.6, 29.2, 92.7, 58.1),
    ("approximate",  5.6, 16.2, 28.9, 91.4, 65.6),
]
MISS_DATA = [
    ("mw+role",   4.8, 12.9, 29.8, 89.3, 63.1),
    ("mw only",   4.1, 11.0, 24.7, 89.1, 62.9),
    ("role only", 2.5,  5.8, 22.5, 88.4, 62.1),
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
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 5.6,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def halo():
    return [pe.withStroke(linewidth=1.3, foreground="white")]


METRIC_COLORS = {
    "CTA@1":     "#86cfcc",  # teal      (matches abstention Clean stage)
    "CTA@10":    "#fbd7b3",  # peach     (matches abstention +Qty stage)
    "CTA@25":    "#cc526a",  # deep pink (matches abstention All Noise / Refusal)
    "Precision": "#9aaf7e",  # sage      (matches abstention +Name / Active mechanism)
    "Recall":    "#d9a87a",  # tan       (matches abstention +Miss / Budget mechanism)
}
METRIC_ORDER = ["CTA@1", "CTA@10", "CTA@25", "Precision", "Recall"]

# Fixed x-axis extent (in data units). Keep this constant so that shrinking
# bar_w / inter_gap actually reduces rendered bar inches (matplotlib won't
# auto-zoom to the data range).
FIXED_XLIM = (-0.10, 4.5)


def draw_bar_panel(ax, data, ymax=110):
    """data: list of (label, c1, c10, c25, P, R). Layout matches abstention.pdf:
    bar_w = 0.13, intra_gap = 0.0 (bars touch within group), inter_gap = 0.044
    (gap between sub-type groups).
    """
    n = len(data)
    n_metrics = len(METRIC_ORDER)
    bar_w = 0.05
    intra_gap = 0.0
    group_w = n_metrics * bar_w + (n_metrics - 1) * intra_gap
    inter_gap = 0.044
    centers = []

    for i, row in enumerate(data):
        group_left = i * (group_w + inter_gap)
        vals = list(row[1:])
        colors = [METRIC_COLORS[m] for m in METRIC_ORDER]
        for j, (v, c) in enumerate(zip(vals, colors)):
            x_pos = group_left + j * (bar_w + intra_gap)
            ax.bar(x_pos, v, bar_w, color=c,
                   edgecolor=INK, linewidth=0.45, zorder=3)
        centers.append(group_left + group_w / 2 - bar_w / 2)

    ax.set_xticks(centers)
    ax.set_xticklabels([d[0] for d in data], fontsize=5.7)
    ax.set_xlim(*FIXED_XLIM)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, lw=0.4, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _shared_legend(fig, ncol=5, bottom_anchor=-0.10, fontsize=6.0):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=METRIC_COLORS[m], edgecolor=INK,
                     linewidth=0.45, label=m) for m in METRIC_ORDER]
    fig.legend(handles=handles, loc="lower center",
               ncol=ncol, frameon=False, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.5,
               bbox_to_anchor=(0.5, bottom_anchor), fontsize=fontsize)


def main():
    apply_style()

    # Single panel: +Qty + +Fmt sub-types on shared x/y axes
    fig, ax = plt.subplots(figsize=(6.6, 1.6))
    fig.patch.set_facecolor("white")

    combined = QTY_DATA + FMT_DATA
    draw_bar_panel(ax, combined)

    n_qty = len(QTY_DATA)
    n_metrics = len(METRIC_ORDER)
    bar_w = 0.05
    intra_gap = 0.0
    inter_gap = 0.044
    group_w = n_metrics * bar_w + (n_metrics - 1) * intra_gap

    _shared_legend(fig)

    plt.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.22)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
