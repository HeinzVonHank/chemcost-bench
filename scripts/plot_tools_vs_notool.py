#!/usr/bin/env python3
"""Figure 3: same-backbone tool gap as a grouped bar chart."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "manuscript/neurips_2025/figures/tools_vs_notool.pdf"

INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE9"
REACT = "#155E9F"
COT = "#D99A00"
FEWSHOT = "#0E9F6E"
ZEROSHOT = "#8A8F98"
SATURATED = "#C2410C"
FLOOR = "#F7E7BE"

REGIME_COLOR = {
    "ReAct": REACT,
    "CoT": COT,
    "FewShot": FEWSHOT,
    "ZeroShot": ZEROSHOT,
    "High P/R": SATURATED,
}

GROUPS = [
    (
        "DeepSeek\nV4 Pro",
        [
            ("ReAct", RESULTS / "dev_react_deepseek_v4_pro_clean.json"),
            ("CoT", RESULTS / "dev_cot_deepseek_v4_pro.json"),
            ("FewShot", RESULTS / "dev_fewshot_deepseek_v4_pro.json"),
        ],
    ),
    (
        "Qwen3.5\nPlus",
        [
            ("ReAct", RESULTS / "dev_react_qwen35plus_clean.json"),
            ("CoT", RESULTS / "dev_cot_qwen-plus.json"),
            ("FewShot", RESULTS / "dev_fewshot_qwen-plus.json"),
            ("ZeroShot", RESULTS / "dev_zeroshot_qwen-plus.json"),
        ],
    ),
    (
        "Sonnet\n4.6",
        [
            ("ReAct", RESULTS / "dev_react_sonnet46_traj.json"),
            ("CoT", RESULTS / "dev_cot_sonnet46.json"),
            ("FewShot", RESULTS / "dev_few-shot_sonnet46.json"),
            ("ZeroShot", RESULTS / "dev_zero-shot_sonnet46.json"),
        ],
    ),
    (
        "GPT-5",
        [
            ("ReAct", RESULTS / "dev_react_gpt5_traj.json"),
            ("CoT", RESULTS / "dev_cot_gpt5.json"),
        ],
    ),
    (
        "near-saturated\nP/R",
        [
            ("High P/R", RESULTS / "dev_cot_deepseek-v3.json"),
            ("High P/R", RESULTS / "dev_cot_sonnet4.json"),
        ],
    ),
]

HIGH_PR_LABEL = {
    RESULTS / "dev_cot_deepseek-v3.json": "P/R\n98/97",
    RESULTS / "dev_cot_sonnet4.json": "P/R\n95/95",
}


def metric(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)["metrics"]
    n_total = data.get("n_total") or 0
    n_pred = data.get("n_predicted") or 0
    abstention = data.get("abstention_rate")
    if abstention is None and n_total:
        abstention = 1 - n_pred / n_total
    return {
        "cta25": data["cta@25"] * 100,
        "p": data.get("component_precision", 0) * 100,
        "r": data.get("component_recall", 0) * 100,
        "abs": (abstention or 0) * 100,
    }


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "font.size": 7.2,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 0,
            "ytick.major.size": 2.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def halo() -> list[pe.AbstractPathEffect]:
    return [pe.withStroke(linewidth=2.0, foreground="white")]


def main() -> None:
    apply_style()
    fig, (ax_hi, ax_lo) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(5.55, 2.32),
        gridspec_kw={"height_ratios": [1.18, 1.0], "hspace": 0.035},
    )
    fig.patch.set_facecolor("white")

    width = 0.46
    inner_gap = 0.055
    group_gap = 0.56
    x = 0.0
    centers: list[float] = []
    labels: list[str] = []
    standalone_start = None

    def draw_bar(xpos: float, value: float, color: str) -> None:
        for ax in (ax_hi, ax_lo):
            ax.bar(
                xpos,
                value,
                width=width,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )

    for group_label, runs in GROUPS:
        xs: list[float] = []
        if group_label.startswith("near"):
            standalone_start = x - 0.28
        for regime, path in runs:
            result = metric(path)
            if result is None:
                continue
            value = result["cta25"]
            color = REGIME_COLOR[regime]
            draw_bar(x, value, color)

            target = ax_hi if value >= 18 else ax_lo
            dy = 0.7 if value >= 18 else 0.22
            target.text(
                x,
                value + dy,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.0,
                color=color if value < 18 else INK,
                path_effects=halo(),
                zorder=5,
            )
            if result["abs"] >= 50:
                ax_lo.text(
                    x,
                    value + 1.0,
                    f"abs {result['abs']:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5.35,
                    color=SATURATED,
                    rotation=90,
                    path_effects=halo(),
                    zorder=5,
                )
            if path in HIGH_PR_LABEL:
                ax_lo.text(
                    x,
                    value + 1.15,
                    HIGH_PR_LABEL[path],
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    linespacing=0.85,
                    color=SATURATED,
                    path_effects=halo(),
                    zorder=5,
                )
            xs.append(x)
            x += width + inner_gap
        centers.append((xs[0] + xs[-1]) / 2)
        labels.append(group_label)
        x += group_gap

    if standalone_start is not None:
        for ax in (ax_hi, ax_lo):
            ax.axvline(standalone_start, color=MUTED, lw=0.65, ls=(0, (1.5, 2.2)), zorder=1)

    ax_lo.axhspan(0, 8.2, color=FLOOR, alpha=0.56, zorder=0)
    ax_lo.text(
        0.0,
        8.05,
        "no-tool floor",
        ha="left",
        va="top",
        fontsize=6.2,
        color="#8A5A00",
        style="italic",
    )

    ax_hi.set_ylim(31.5, 53.8)
    ax_lo.set_ylim(0, 9.0)
    ax_hi.set_yticks([35, 45])
    ax_lo.set_yticks([0, 4, 8])

    for ax in (ax_hi, ax_lo):
        ax.grid(axis="y", color=GRID, lw=0.55, alpha=0.72, zorder=0)
        ax.spines["right"].set_visible(False)
    ax_hi.spines["top"].set_visible(False)
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(labelbottom=False, bottom=False)

    break_kw = {
        "marker": [(-1, -0.45), (1, 0.45)],
        "markersize": 5.2,
        "linestyle": "none",
        "color": INK,
        "mec": INK,
        "mew": 0.8,
        "clip_on": False,
    }
    ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **break_kw)
    ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **break_kw)

    ax_lo.set_xticks(centers)
    ax_lo.set_xticklabels(labels)
    ax_lo.set_xlabel("Backbone / standalone control")
    fig.text(0.014, 0.52, "CTA@25 (%)", rotation=90, va="center", fontsize=7.8, color=INK)

    handles = [
        Patch(facecolor=REACT, edgecolor="none", label="ReAct (tools)"),
        Patch(facecolor=COT, edgecolor="none", label="CoT"),
        Patch(facecolor=FEWSHOT, edgecolor="none", label="FewShot"),
        Patch(facecolor=ZEROSHOT, edgecolor="none", label="ZeroShot"),
        Patch(facecolor=SATURATED, edgecolor="none", label="near-saturated P/R"),
    ]
    ax_hi.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.26),
        ncol=5,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.75,
        handletextpad=0.35,
    )

    ax_hi.set_xlim(-0.45, x - group_gap + 0.2)
    plt.subplots_adjust(left=0.085, right=0.995, top=0.82, bottom=0.23)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
