#!/usr/bin/env python3
"""Multi-step Sankey: recall-loss decomposition for frontier ReAct agents.

Step 1: Truth components (3,335)
Step 2: Outcome — Kept (terminates) vs Missed
Step 3: Miss type — Never-mentioned / Unpriced->dropped / Priced->dropped
Step 4: Sub-causes within dropped buckets

Top-level counts are exact (from trajectory classifier on 5 frontier ReAct x
121 reactions); sub-cause shares are estimated from manual review of
trajectory samples.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "manuscript/neurips_2025/figures/recall_loss_sankey.pdf"

INK = "#172033"
MUTED = "#667085"
GRID = "#E2E8F0"

# Categorical palette: 6 distinct hues, each semantic category gets its own.
# Anchor / outcome colors follow Paul Tol "vibrant" + Wong (Nature 2011);
# the three miss-bucket colors are the user-specified palette.
COL_TRUTH      = "#4F5B6E"  # slate gray — anchor (Truth)
COL_TRUTH_TEXT = "#2F3848"
COL_KEPT       = "#A8C9A0"  # sage green — success, recedes
COL_KEPT_DARK  = "#5A7A5C"
COL_MISS       = "#7B8FA6"  # cool muted blue-gray — miss umbrella, distinct from PR
COL_MISS_TEXT  = "#4F5F77"
COL_NM         = "#81912F"  # olive — never-mentioned (smallest)
COL_NM_TEXT    = "#56611F"
COL_UP         = "#F8C463"  # gold — unpriced -> dropped (medium)
COL_UP_TEXT    = "#A8761D"
COL_PR         = "#FF8383"  # coral — priced -> dropped (largest)
COL_PR_TEXT    = "#B54A4A"

# Sub-cause palette: 6 distinct earthy/muted tones inspired by ColorHunt
# popular palettes. Warm -> cool gradient across (a)..(f).
SUB_COLORS = [
    "#B77466",   # (a) terracotta / rust
    "#F5BABB",   # (b) dusty pink
    "#C9A96E",   # (c) antique gold
    "#568F87",   # (d) sage teal
    "#3F6E64",   # (e) deep sage
    "#064232",   # (f) forest green
]

TOTAL    = 3335
KEPT     = 1589
MISSED   = TOTAL - KEPT     # 1746
NM_TOTAL = 169
UP_TOTAL = 552
PR_TOTAL = 1025

UP_SUB = [
    ("(a) Prose placeholder (MOMO, OTHP, abbreviations)",                0.10),
    ("(b) Drug R&D intermediates (no commercial supplier)",              0.50),
    ("(c) Pricing DB / query mismatch -> dropped",                       0.40),
]
PR_SUB = [
    ("(d) mol% catalyst: 'too small to count'",                          0.40),
    ("(e) Base / auxiliary: 'negligible cost' judgment",                 0.30),
    ("(f) Multi-step mid-stage reagents merged or dropped",              0.30),
]


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelcolor": INK,
        "text.color": INK,
        "font.size": 7.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def cubic(p0, p1, p2, p3, n=140):
    t = np.linspace(0, 1, n)[:, None]
    return ((1-t)**3)*p0 + 3*((1-t)**2)*t*p1 + 3*(1-t)*(t**2)*p2 + (t**3)*p3


def ribbon(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot, color, alpha=0.32):
    cx = (x1 - x0) * 0.55
    top = cubic(np.array([x0, y0_top]), np.array([x0+cx, y0_top]),
                np.array([x1-cx, y1_top]), np.array([x1, y1_top]))
    bot = cubic(np.array([x0, y0_bot]), np.array([x0+cx, y0_bot]),
                np.array([x1-cx, y1_bot]), np.array([x1, y1_bot]))
    poly = np.vstack([top, bot[::-1]])
    ax.add_patch(mpatches.Polygon(poly, closed=True, facecolor=color,
                                  edgecolor="none", alpha=alpha, zorder=1))


def push_apart(positions, min_dy):
    """Greedy 1D collision avoidance: ensure adjacent labels are >= min_dy apart.
    positions: list of (idx, y) tuples; modify y to satisfy constraint top-to-bottom.
    Returns dict idx -> adjusted y."""
    pos = sorted(positions, key=lambda p: -p[1])  # top to bottom
    out = {}
    last_y = None
    for idx, y in pos:
        if last_y is not None and last_y - y < min_dy:
            y = last_y - min_dy
        out[idx] = y
        last_y = y
    return out


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.55, 4.85))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axis_off()

    # Column x positions
    X = [0.00, 0.17, 0.36, 0.55]
    BAR_W = 0.022
    GAP = 0.018  # vertical gap between sibling blocks

    # ===== Column 1: Outcome (Kept | Missed) =====
    # Focus view: the kept branch is intentionally collapsed so visual area is
    # spent on the error decomposition.
    h_kept = 0.105
    h_miss = 0.835
    kept_top = 1.0
    kept_bot = kept_top - h_kept
    miss_top = kept_bot - GAP
    miss_bot = miss_top - h_miss

    # ===== Column 0: Truth — align with Step 2 vertical extent =====
    truth_top, truth_bot = 1.0, miss_bot
    ax.add_patch(mpatches.Rectangle((X[0], truth_bot), BAR_W, truth_top - truth_bot,
                                    facecolor=COL_TRUTH, edgecolor="none", zorder=3))

    ax.add_patch(mpatches.Rectangle((X[1], kept_bot), BAR_W, h_kept,
                                    facecolor=COL_KEPT, edgecolor="none", zorder=3))
    ax.add_patch(mpatches.Rectangle((X[1], miss_bot), BAR_W, h_miss,
                                    facecolor=COL_MISS, edgecolor="none", zorder=3))

    # Ribbons truth -> column 1: kept band on top, missed band below it,
    # both starting from inside the (now-shortened) truth bar
    truth_split = truth_top - h_kept
    ribbon(ax, X[0]+BAR_W, truth_top, truth_split, X[1], kept_top, kept_bot,
           color=COL_KEPT, alpha=0.38)
    ribbon(ax, X[0]+BAR_W, truth_split, truth_bot, X[1], miss_top, miss_bot,
           color=COL_MISS, alpha=0.18)

    # ===== Column 2: Miss type (NM | UP | PR) — only shown for Missed branch =====
    # Column 2 vertical extent matches Missed band in column 1
    avail2 = h_miss - 2 * GAP
    blocks2 = []
    cur = miss_top
    for cnt, color, label in [(NM_TOTAL, COL_NM, "Never-mentioned"),
                               (UP_TOTAL, COL_UP, "Unpriced $\\rightarrow$ dropped"),
                               (PR_TOTAL, COL_PR, "Priced $\\rightarrow$ dropped")]:
        h = (cnt / MISSED) * avail2
        y_top = cur
        y_bot = cur - h
        ax.add_patch(mpatches.Rectangle((X[2], y_bot), BAR_W, h,
                                        facecolor=color, edgecolor="none", zorder=3))
        blocks2.append((y_top, y_bot, color, label, cnt))
        cur = y_bot - GAP

    # Ribbons col1 Missed -> col2
    cum_in = miss_top
    for (y_top, y_bot, color, _, cnt) in blocks2:
        h_in = (cnt / MISSED) * h_miss
        y0_top = cum_in
        y0_bot = cum_in - h_in
        ribbon(ax, X[1]+BAR_W, y0_top, y0_bot, X[2], y_top, y_bot,
               color=color, alpha=0.36)
        cum_in = y0_bot

    # ===== Column 3: Sub-causes — each block its own color =====
    GAP3 = 0.0035
    sub_blocks = []  # (y_top, y_bot, color, label, cnt)
    sub_idx = 0
    for (y_top_p, y_bot_p, parent_color, parent_label, parent_cnt), subs in [
        (blocks2[1], UP_SUB), (blocks2[2], PR_SUB)
    ]:
        parent_h = y_top_p - y_bot_p
        n = len(subs)
        sub_avail = parent_h - GAP3 * (n - 1)
        cur = y_top_p
        for sub_label, share in subs:
            h = sub_avail * share
            y_top = cur
            y_bot = cur - h
            cnt = int(round(parent_cnt * share))
            this_color = SUB_COLORS[sub_idx]
            sub_blocks.append((y_top, y_bot, this_color, sub_label, cnt))
            ax.add_patch(mpatches.Rectangle((X[3], y_bot), BAR_W, h,
                                            facecolor=this_color,
                                            edgecolor="white", linewidth=0.4,
                                            zorder=3))
            sub_idx += 1
            cur = y_bot - GAP3

    # Ribbons col2 -> col3 — fade from parent color to each sub-block's color
    sub_idx = 0
    for blk2, subs in [(blocks2[1], UP_SUB), (blocks2[2], PR_SUB)]:
        y_top_p, y_bot_p, parent_color, _, parent_cnt = blk2
        parent_h = y_top_p - y_bot_p
        n = len(subs)
        sub_avail = parent_h - GAP3 * (n - 1)
        cur_src = y_top_p
        cur_dst = y_top_p
        for sub_label, share in subs:
            h_src = parent_h * share
            h_dst = sub_avail * share
            y0_top = cur_src
            y0_bot = cur_src - h_src
            y1_top = cur_dst
            y1_bot = cur_dst - h_dst
            ribbon(ax, X[2]+BAR_W, y0_top, y0_bot, X[3], y1_top, y1_bot,
                   color=SUB_COLORS[sub_idx], alpha=0.34)
            sub_idx += 1
            cur_src = y0_bot
            cur_dst = y1_bot - GAP3

    # ===== Sub-cause labels (with leader lines + global collision avoidance) =====
    # Natural y = block midpoint
    natural = [(i, (b[0] + b[1]) / 2, b) for i, b in enumerate(sub_blocks)]
    desired = [(i, y) for i, y, _ in natural]
    MIN_DY = 0.052
    adjusted_y = push_apart(desired, MIN_DY)


    ax.set_xlim(-0.02, 0.60)
    ax.set_ylim(miss_bot - 0.01, 1.01)

    plt.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.02)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
