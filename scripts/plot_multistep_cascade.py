#!/usr/bin/env python3
"""Multi-step procurement cost cascade analysis (appendix figure).

3 panels showing how procurement cost evolves through multi-step routes
(ground-truth only, no agent predictions):

A. Per-step cost distribution: log-scale boxplot of $/g at each step depth.
   Tests whether multi-step routes systematically reduce or amplify per-g
   cost as the synthesis proceeds.
B. Cost trajectory per reaction: each multi-step reaction is one line
   showing cost_per_g vs step_number. Color-coded by total step count.
C. Last-step contribution: ratio of (last step delta cost) / (final cost),
   showing how dominant the last step is in the final per-g cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/multistep_cascade_analysis.json"
OUT = ROOT / "manuscript/neurips_2025/figures/multistep_cascade.pdf"

INK = "#2A2417"
MUTED = "#857A66"
GRID = "#E5DDD0"

# Step-count colors (cool to warm: 2 steps cool, 6 steps warm)
STEP_COLORS = {
    2: "#86cfcc",
    3: "#9aaf7e",
    4: "#fbd7b3",
    5: "#f18982",
    6: "#cc526a",
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
        "legend.fontsize": 5.6,
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
    data = json.load(open(ANALYSIS))
    valid = [d for d in data if d["all_steps_priced"]]
    print(f"Total: {len(data)}, fully priced: {len(valid)}")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 2.4),
                                         gridspec_kw={"width_ratios":
                                                      [1.0, 1.0, 0.85]})
    fig.patch.set_facecolor("white")

    # ============== Panel A: Per-step cost distribution (log y) ==============
    # Group cost_per_g_at_step_N by step number across all valid reactions.
    by_step_pos: dict[int, list[float]] = {}
    for d in valid:
        for i, c in enumerate(d["step_costs"]):
            if c is not None and c > 0:
                by_step_pos.setdefault(i + 1, []).append(c)

    step_nums = sorted(by_step_pos.keys())
    box_data = [by_step_pos[s] for s in step_nums]
    n_per_step = [len(by_step_pos[s]) for s in step_nums]

    # Boxplot
    bp = axA.boxplot(box_data, positions=step_nums, widths=0.55,
                      patch_artist=True, showfliers=False,
                      medianprops={"color": INK, "linewidth": 0.8},
                      whiskerprops={"color": INK, "linewidth": 0.45},
                      capprops={"color": INK, "linewidth": 0.45},
                      boxprops={"linewidth": 0.55, "edgecolor": INK})
    for i, patch in enumerate(bp["boxes"]):
        s = step_nums[i]
        # Reuse the step-count color for the box at that position; box for
        # step 6 = warmest because deepest step
        c = STEP_COLORS.get(s, "#cc526a")
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    # Overlay individual points for transparency
    for s, vals in zip(step_nums, box_data):
        jitter = (np.random.RandomState(s).rand(len(vals)) - 0.5) * 0.25
        axA.scatter(np.full(len(vals), s) + jitter, vals,
                    s=4, c=INK, alpha=0.35, zorder=3, lw=0)

    axA.set_yscale("log")
    axA.set_xlabel("Step number")
    axA.set_ylabel("Cost \\$/g of step product (log)")
    axA.set_xlim(0.5, max(step_nums) + 0.5)
    axA.set_xticks(step_nums)
    axA.grid(axis="y", which="both", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.text(0.0, 1.05, "(A)", transform=axA.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    # n annotation under each box
    for s, n in zip(step_nums, n_per_step):
        axA.text(s, 0.92, f"$n${{=}}{n}", transform=axA.get_xaxis_transform(),
                 ha="center", va="top", fontsize=4.8, color=MUTED)

    # ============== Panel B: Cost trajectory per reaction ==============
    for d in valid:
        steps = list(range(1, d["n_steps"] + 1))
        costs = d["step_costs"]
        color = STEP_COLORS.get(d["n_steps"], MUTED)
        axB.plot(steps, costs, color=color, lw=0.7, alpha=0.65, zorder=2)
        axB.scatter(steps, costs, s=8, c=color, edgecolor=INK,
                    linewidth=0.3, alpha=0.75, zorder=3)

    axB.set_yscale("log")
    axB.set_xlabel("Step number")
    axB.set_ylabel("Cost \\$/g of step product (log)")
    axB.set_xlim(0.7, 6.3)
    axB.set_xticks([1, 2, 3, 4, 5, 6])
    axB.grid(axis="y", which="both", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.text(0.0, 1.05, "(B)", transform=axB.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    # Legend for step count colors
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color=STEP_COLORS[k],
                      markerfacecolor=STEP_COLORS[k], markeredgecolor=INK,
                      markeredgewidth=0.3, markersize=4, lw=0.7,
                      label=f"{k} steps")
               for k in sorted(STEP_COLORS)]
    axB.legend(handles=handles, loc="upper left", frameon=False,
               fontsize=5.0, handlelength=1.2, handletextpad=0.3,
               labelspacing=0.2, borderpad=0.1)

    # ============== Panel C: last-step delta / final cost ==============
    ratios = []
    n_steps_arr = []
    for d in valid:
        if d["final_cost"] is None or d["final_cost"] <= 0: continue
        last_delta = d["step_deltas"][-1]
        if last_delta is None: continue
        ratios.append(last_delta / d["final_cost"])
        n_steps_arr.append(d["n_steps"])

    ratios_pct = [r * 100 for r in ratios]

    # Color by step count
    colors = [STEP_COLORS.get(n, MUTED) for n in n_steps_arr]
    # Sort for visual: ascending
    idx = sorted(range(len(ratios_pct)), key=lambda i: ratios_pct[i])
    sorted_pct = [ratios_pct[i] for i in idx]
    sorted_colors = [colors[i] for i in idx]

    axC.bar(range(len(sorted_pct)), sorted_pct, width=1.0,
            color=sorted_colors, edgecolor="none", alpha=0.85, zorder=3)
    axC.axhline(0, color=INK, lw=0.5, zorder=2)
    axC.axhline(50, color=MUTED, lw=0.4, ls=(0, (3, 2)), alpha=0.5, zorder=2)
    axC.set_ylabel("Last-step $\\Delta$ / final cost (\\%)")
    axC.set_xlabel(f"Reaction (sorted, $n${{=}}{len(ratios_pct)})")
    axC.set_xticks([])
    axC.set_ylim(min(-200, min(sorted_pct) - 10), max(120, max(sorted_pct) + 10))
    axC.grid(axis="y", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)
    axC.text(0.0, 1.05, "(C)", transform=axC.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    plt.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.18,
                        wspace=0.30)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")

    # Print stats for caption
    import statistics
    pos = [r for r in ratios_pct if r > 0]
    neg = [r for r in ratios_pct if r < 0]
    print(f"\n=== last-step delta / final cost ratio ===")
    print(f"  positive (last step ADDS cost): {len(pos)}/{len(ratios_pct)}")
    print(f"  negative (last step REDUCES /g): {len(neg)}/{len(ratios_pct)}")
    print(f"  median: {statistics.median(ratios_pct):.1f}%")
    print(f"  IQR: [{statistics.quantiles(ratios_pct, n=4)[0]:.1f}%, "
          f"{statistics.quantiles(ratios_pct, n=4)[2]:.1f}%]")


if __name__ == "__main__":
    main()
