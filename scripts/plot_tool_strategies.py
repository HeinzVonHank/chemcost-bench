#!/usr/bin/env python3
"""Same API, different strategies: tool-call volume vs per-call success rate.

Scatter of frontier ReAct agents on the (calls/reaction, success-rate) plane.
Point size encodes CTA@25; color encodes abstention rate. Trendline shows the
exploration tax: agents that issue more calls per reaction tend to land a
lower per-call success rate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "manuscript/neurips_2025/figures/tool_strategies.pdf"

INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE9"
TREND = "#94A3B8"

AGENTS = [
    ("GPT-4.1",        RESULTS / "dev_react_gpt41_traj.json"),
    ("GPT-5.4",        RESULTS / "dev_react_gpt54_traj.json"),
    ("GPT-5",          RESULTS / "dev_react_gpt5_traj.json"),
    ("Qwen-plus",      RESULTS / "dev_react_qwenplus_traj.json"),
    ("Qwen3.5-plus",   RESULTS / "dev_react_qwen35plus_clean.json"),
    ("Kimi K2.5",      RESULTS / "dev_react_kimi_k25_clean.json"),
    ("DeepSeek V4 Pro", RESULTS / "dev_react_deepseek_v4_pro_clean.json"),
    ("Sonnet 4.6",     RESULTS / "dev_react_sonnet46_traj.json"),
]


def stats(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as f:
        d = json.load(f)
    preds = d["predictions"]
    n = len(preds)
    n_tc = sum(len(p.get("tool_calls", [])) for p in preds)
    n_succ = sum(1 for p in preds for tc in p.get("tool_calls", []) if tc.get("success"))
    n_abst = sum(1 for p in preds if p.get("predicted_cost") is None)
    return {
        "calls_per_rxn": n_tc / n,
        "success_rate": n_succ / n_tc * 100,
        "cta25": d["metrics"]["cta@25"] * 100,
        "abst": n_abst / n * 100,
        "n_tc": n_tc,
    }


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "font.size": 7.6,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def halo() -> list[pe.AbstractPathEffect]:
    return [pe.withStroke(linewidth=2.0, foreground="white")]


# Manual nudges for label placement; (dx, dy, ha, va) in axis-data units.
LABEL_OFFSETS: dict[str, tuple[float, float, str, str]] = {
    "GPT-4.1":         (0.55, -1.6, "left",   "top"),
    "GPT-5.4":         (0.55, 1.4,  "left",   "bottom"),
    "GPT-5":           (0.55, 1.4,  "left",   "bottom"),
    "Qwen-plus":       (-0.55, -1.6, "right",  "top"),
    "Qwen3.5-plus":    (0.55, 1.4,  "left",   "bottom"),
    "Kimi K2.5":       (0.55, -1.6, "left",   "top"),
    "DeepSeek V4 Pro": (-0.55, 1.4, "right",  "bottom"),
    "Sonnet 4.6":      (0.55, -1.6, "left",   "top"),
}


def main() -> None:
    apply_style()
    rows = [(name, stats(p)) for name, p in AGENTS]

    fig, ax = plt.subplots(figsize=(5.4, 3.55))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xs = np.array([r[1]["calls_per_rxn"] for r in rows])
    ys = np.array([r[1]["success_rate"] for r in rows])
    ctas = np.array([r[1]["cta25"] for r in rows])
    absts = np.array([r[1]["abst"] for r in rows])

    # Trendline: linear fit on the (calls, success_rate) cloud.
    slope, intercept = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min() - 1.0, xs.max() + 1.0, 100)
    yline = slope * xline + intercept
    r = np.corrcoef(xs, ys)[0, 1]
    ax.plot(xline, yline, color=TREND, lw=1.0, ls=(0, (3.5, 2.5)), zorder=1)
    # Place r-label along the trendline near its midpoint, just above the line.
    x_mid = xline[len(xline) * 3 // 5]
    y_mid = slope * x_mid + intercept
    ax.text(
        x_mid, y_mid + 1.6, f"$r = {r:+.2f}$",
        ha="left", va="bottom", color=TREND, fontsize=7.2, style="italic",
        path_effects=halo(), zorder=2,
    )

    # Size encodes CTA@25 (16–51% range -> 60–360 pt^2).
    s_min, s_max = ctas.min(), ctas.max()
    sizes = 60 + (ctas - s_min) / max(s_max - s_min, 1e-9) * 300

    sc = ax.scatter(
        xs, ys,
        s=sizes,
        c=absts,
        cmap="YlOrRd",
        vmin=0, vmax=45,
        edgecolors=INK,
        linewidths=0.8,
        alpha=0.92,
        zorder=4,
    )

    for (name, _), x, y in zip(rows, xs, ys):
        dx, dy, ha, va = LABEL_OFFSETS[name]
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            ha=ha, va=va,
            fontsize=7.0, color=INK,
            path_effects=halo(), zorder=5,
        )

    ax.set_xlabel("Average tool calls per reaction")
    ax.set_ylabel("Tool-call success rate (%)")
    ax.set_xlim(7.0, 30.5)
    ax.set_ylim(50, 96)
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.set_yticks([55, 65, 75, 85, 95])
    ax.grid(True, color=GRID, lw=0.55, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colorbar for abstention.
    cbar = fig.colorbar(sc, ax=ax, fraction=0.038, pad=0.025, aspect=22)
    cbar.set_label("Abstention rate (%)", fontsize=7.4)
    cbar.ax.tick_params(labelsize=6.6, width=0.5, length=2.2)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(INK)

    # Size legend (CTA@25).
    legend_ctas = [20, 35, 50]
    legend_handles = []
    for cv in legend_ctas:
        sz = 60 + (cv - s_min) / max(s_max - s_min, 1e-9) * 300
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#CBD5E1", markeredgecolor=INK,
                   markeredgewidth=0.7,
                   markersize=np.sqrt(sz), label=f"{cv}%")
        )
    leg = ax.legend(
        handles=legend_handles,
        title="CTA@25",
        loc="lower left",
        bbox_to_anchor=(0.012, 0.012),
        frameon=True,
        fancybox=False,
        edgecolor=GRID,
        facecolor="white",
        framealpha=0.9,
        borderpad=0.55,
        labelspacing=0.85,
        handletextpad=0.5,
        title_fontsize=7.0,
    )
    leg.get_title().set_color(INK)
    leg.get_frame().set_linewidth(0.5)

    plt.subplots_adjust(left=0.115, right=0.995, top=0.97, bottom=0.13)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print()
    print(f"{'agent':<18} {'calls/rxn':>10} {'succ%':>7} {'CTA25':>7} {'abst%':>7}")
    for name, m in rows:
        print(f"{name:<18} {m['calls_per_rxn']:>10.1f} {m['success_rate']:>7.1f} "
              f"{m['cta25']:>7.1f} {m['abst']:>7.1f}")
    print(f"\nlinear fit: success% = {slope:.2f} * calls + {intercept:.1f}  (r = {r:+.2f})")


if __name__ == "__main__":
    main()
