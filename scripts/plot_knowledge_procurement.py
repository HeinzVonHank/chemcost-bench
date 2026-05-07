#!/usr/bin/env python3
"""Figure 4: chemistry knowledge does not imply procurement capability."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "manuscript/neurips_2025/figures/knowledge_procurement.pdf"

INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE9"
BLUE = "#155E9F"
GREEN = "#168A5B"
VERMILLION = "#C2410C"
AMBER = "#D99A00"

POINTS = [
    ("Qwen3.5", "Qwen3.5-Plus", RESULTS / "dev_react_qwen35plus_clean.json", "frontier"),
    ("DeepSeek V4", "DeepSeek V4 Pro", RESULTS / "dev_react_deepseek_v4_pro_clean.json", "frontier"),
    ("GPT-5", "GPT-5", RESULTS / "dev_react_gpt5_traj.json", "frontier"),
    ("Kimi K2.5", "Kimi K2.5", RESULTS / "dev_react_kimi_k25_clean.json", "frontier"),
    ("Sonnet 4.6", "Sonnet 4.6", RESULTS / "dev_react_sonnet46_traj.json", "frontier"),
    ("DS-V3 CoT", "DeepSeek-V3 (CoT)", RESULTS / "dev_cot_deepseek-v3.json", "no-tool"),
    ("Sonnet 4 CoT", "Sonnet 4 (CoT)", RESULTS / "dev_cot_sonnet4.json", "no-tool"),
    ("S4.6 FewShot", "Sonnet 4.6 (FewShot)", RESULTS / "dev_few-shot_sonnet46.json", "no-tool"),
    ("S4.6 CoT", "Sonnet 4.6 (CoT)", RESULTS / "dev_cot_sonnet46.json", "no-tool"),
    ("S4.6 ZeroShot", "Sonnet 4.6 (ZeroShot)", RESULTS / "dev_zero-shot_sonnet46.json", "no-tool"),
    ("ChemDFM", "ChemDFM-v2.0-14B", RESULTS / "dev_react_chemdfm_v2_clean.json", "domain"),
    ("ChemLLM", "ChemLLM-20B", RESULTS / "dev_react_chemllm_20b_clean.json", "domain"),
    ("LlaSMol", "LlaSMol-7B", RESULTS / "dev_react_llasmol_7b_clean.json", "domain"),
    ("Qwen3-235B", "Qwen3-235B-A22B", RESULTS / "dev_react_qwen3_235b_a22b_clean.json", "domain"),
    ("Qwen3-14B", "Qwen3-14B (base)", RESULTS / "dev_react_qwen3_14b_clean.json", "domain"),
]

CATEGORY = {
    "frontier": {"color": BLUE, "marker": "o", "label": "Frontier ReAct (tools)", "size": 68},
    "no-tool": {"color": GREEN, "marker": "s", "label": "No-tool baselines", "size": 56},
    "domain": {"color": VERMILLION, "marker": "^", "label": "Chem-domain / open base", "size": 66},
}

LABEL_OFFSETS = {
    "Qwen3.5-Plus": (2.0, 1.8),
    "DeepSeek V4 Pro": (1.8, -2.2),
    "GPT-5": (-10.8, 1.5),
    "Kimi K2.5": (1.8, -2.0),
    "Sonnet 4.6": (-14.8, -2.0),
    "DeepSeek-V3 (CoT)": (-34.0, 2.5),
    "Sonnet 4 (CoT)": (-3.0, 4.6),
    "Sonnet 4.6 (FewShot)": (2.0, 3.2),
    "Sonnet 4.6 (CoT)": (2.0, 1.7),
    "Sonnet 4.6 (ZeroShot)": (-2.0, 3.0),
    "ChemDFM-v2.0-14B": (-8.0, 2.5),
    "ChemLLM-20B": (2.0, 2.0),
    "LlaSMol-7B": (2.0, 1.8),
    "Qwen3-235B-A22B": (2.0, 2.0),
    "Qwen3-14B (base)": (2.0, 2.4),
}


def metric(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)["metrics"]
    return {
        "p": data.get("component_precision", 0) * 100,
        "r": data.get("component_recall", 0) * 100,
        "cta25": data["cta@25"] * 100,
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
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def halo() -> list[pe.AbstractPathEffect]:
    return [pe.withStroke(linewidth=2.1, foreground="white")]


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.1, 3.28))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    zone = Rectangle(
        (45, -2.5),
        60,
        10.5,
        facecolor=VERMILLION,
        edgecolor=VERMILLION,
        lw=0.7,
        alpha=0.075,
        zorder=0,
    )
    ax.add_patch(zone)
    ax.text(
        74,
        2.35,
        "high recall,\nlow procurement",
        ha="center",
        va="center",
        fontsize=6.7,
        color="#8B1E0D",
        style="italic",
    )

    plotted: set[str] = set()
    label_targets = []
    for short, full, path, category in POINTS:
        result = metric(path)
        if result is None:
            continue
        props = CATEGORY[category]
        label = props["label"] if category not in plotted else None
        plotted.add(category)
        ax.scatter(
            result["r"],
            result["cta25"],
            s=props["size"],
            marker=props["marker"],
            color=props["color"],
            edgecolors="white",
            linewidths=0.75,
            alpha=0.94,
            label=label,
            zorder=4,
        )
        label_targets.append((short, full, result, category))

    for short, full, result, category in label_targets:
        dx, dy = LABEL_OFFSETS[full]
        color = CATEGORY[category]["color"] if category != "frontier" else INK
        arrow = None
        if abs(dx) > 10 or abs(dy) > 3.5:
            arrow = {
                "arrowstyle": "-",
                "color": MUTED,
                "lw": 0.45,
                "shrinkA": 0,
                "shrinkB": 4,
            }
        ax.annotate(
            short,
            xy=(result["r"], result["cta25"]),
            xytext=(result["r"] + dx, result["cta25"] + dy),
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=6.35,
            color=color,
            arrowprops=arrow,
            path_effects=halo(),
            zorder=5,
        )

    ax.set_xlim(-3, 105)
    ax.set_ylim(-2.5, 58)
    ax.set_xlabel("Component recall (%)")
    ax.set_ylabel("CTA@25 (%)")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.grid(color=GRID, lw=0.55, alpha=0.72, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker=props["marker"],
            lw=0,
            color=props["color"],
            markeredgecolor="white",
            markeredgewidth=0.75,
            label=props["label"],
            markersize=6,
        )
        for props in CATEGORY.values()
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=0.96,
        fancybox=False,
        edgecolor="#CBD5E1",
        borderpad=0.45,
        handletextpad=0.45,
    )

    plt.subplots_adjust(left=0.12, right=0.988, top=0.98, bottom=0.145)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
