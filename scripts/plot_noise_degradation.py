#!/usr/bin/env python3
"""Figure 5: Noise degradation across single-stage and composed noise."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "manuscript/neurips_2025/figures/noise_degradation.pdf"

INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE9"
SHADE = "#F7E7BE"

CONDS = ["Clean", "+Name", "+Qty", "+Miss", "+Fmt", "All"]

MODELS = [
    (
        "Qwen3.5-Plus",
        "#155E9F",
        [
            RESULTS / "dev_react_qwen35plus_clean.json",
            RESULTS / "dev_react_qwen35plus_noise_name.json",
            RESULTS / "dev_react_qwen35plus_noise_quantity.json",
            RESULTS / "dev_react_qwen35plus_noise_missing.json",
            RESULTS / "dev_react_qwen35plus_noise_format.json",
            RESULTS / "dev_react_qwen35plus_noise_name_quantity_missing_format.json",
        ],
    ),
    (
        "DeepSeek V4 Pro",
        "#D99A00",
        [
            RESULTS / "dev_react_deepseek_v4_pro_clean.json",
            RESULTS / "dev_react_deepseek_v4_pro_noise_name.json",
            RESULTS / "dev_react_deepseek_v4_pro_noise_quantity.json",
            RESULTS / "dev_react_deepseek_v4_pro_noise_missing.json",
            RESULTS / "dev_react_deepseek_v4_pro_noise_format.json",
            RESULTS / "dev_react_deepseek_v4_pro_noise_all.json",
        ],
    ),
    (
        "GPT-5",
        "#168A5B",
        [
            RESULTS / "dev_react_gpt5_traj.json",
            RESULTS / "dev_react_gpt5_noise_name.json",
            RESULTS / "dev_react_gpt5_noise_quantity.json",
            RESULTS / "dev_react_gpt5_noise_missing.json",
            RESULTS / "dev_react_gpt5_noise_format.json",
            RESULTS / "dev_react_gpt5_noise_name_quantity_missing_format.json",
        ],
    ),
    (
        "Kimi K2.5",
        "#B95784",
        [
            RESULTS / "dev_react_kimi_k25_clean.json",
            RESULTS / "dev_react_kimi_k25_noise_name.json",
            RESULTS / "dev_react_kimi_k25_noise_quantity.json",
            RESULTS / "dev_react_kimi_k25_noise_missing.json",
            RESULTS / "dev_react_kimi_k25_noise_format.json",
            RESULTS / "dev_react_kimi_k25_noise_name_quantity_missing_format.json",
        ],
    ),
    (
        "Sonnet 4.6",
        "#4E9BC7",
        [
            RESULTS / "dev_react_sonnet46_traj.json",
            RESULTS / "dev_react_sonnet46_noise_name.json",
            RESULTS / "dev_react_sonnet46_noise_quantity.json",
            RESULTS / "dev_react_sonnet46_noise_missing.json",
            RESULTS / "dev_react_sonnet46_noise_format.json",
            RESULTS / "dev_react_sonnet46_noise_name_quantity_missing_format.json",
        ],
    ),
]

PLACEHOLDERS = {
    "Qwen3.5-Plus": [34.6, 30.9, 34.6, 32.1, 18.5, 19.8],
    "DeepSeek V4 Pro": [37.0, 38.3, 37.0, 34.6, 22.2, 19.8],
    "GPT-5": [24.7, 23.5, 23.5, 22.2, 13.6, 11.1],
    "Kimi K2.5": [24.7, 21.0, 29.6, 25.9, 12.3, 12.3],
    "Sonnet 4.6": [24.7, 23.5, 16.1, 22.2, 11.1, 12.3],
}


def cta10(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)["metrics"]["cta@10"] * 100


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
            "xtick.labelsize": 7.0,
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
    fig, ax = plt.subplots(figsize=(5.45, 2.62))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(CONDS))
    series: dict[str, list[float]] = {}

    ax.axvspan(3.5, 4.5, color=SHADE, alpha=0.78, zorder=0)
    ax.axvline(4.5, color="#E0B15A", lw=0.65, ls=(0, (2, 2)), zorder=1)

    for label, color, paths in MODELS:
        ys = []
        for idx, path in enumerate(paths):
            value = cta10(path)
            if value is None:
                value = PLACEHOLDERS[label][idx]
                print(f"[placeholder] {label} {CONDS[idx]} -> {value}")
            ys.append(value)
        series[label] = ys
        ax.plot(
            x,
            ys,
            color=color,
            lw=1.65,
            marker="o",
            ms=4.2,
            mec="white",
            mew=0.65,
            label=label,
            zorder=3,
        )

    means = np.array(list(series.values())).mean(axis=0)
    ax.plot(
        x,
        means,
        color=INK,
        lw=2.0,
        marker="o",
        ms=4.4,
        mec="white",
        mew=0.7,
        zorder=5,
        label="Mean",
    )

    clean_mean = means[0]
    fmt_mean = means[4]
    ax.annotate(
        "",
        xy=(4.28, fmt_mean),
        xytext=(4.28, clean_mean),
        arrowprops={"arrowstyle": "<->", "color": "#8A5A00", "lw": 0.8},
        zorder=6,
    )
    ax.text(
        4.34,
        (clean_mean + fmt_mean) / 2,
        f"mean -{clean_mean - fmt_mean:.1f}",
        ha="left",
        va="center",
        fontsize=6.3,
        color="#8A5A00",
        path_effects=halo(),
    )
    ax.text(
        4.0,
        41.7,
        "format cliff",
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#8A5A00",
        style="italic",
    )

    right_labels = {
        "Qwen3.5-Plus": 1.25,
        "DeepSeek V4 Pro": -0.95,
        "Kimi K2.5": 1.55,
        "Sonnet 4.6": -0.70,
        "GPT-5": -1.70,
    }
    for label, color, _ in MODELS:
        y = series[label][-1]
        ax.text(
            5.08,
            y + right_labels[label],
            label.replace(" V4 Pro", " V4"),
            color=color,
            fontsize=6.15,
            va="center",
            ha="left",
            path_effects=halo(),
        )

    ax.set_xlim(-0.25, 6.08)
    ax.set_ylim(0, 44)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS)
    ax.set_xlabel("Noise condition")
    ax.set_ylabel("CTA@10 (%)")
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.grid(axis="y", color=GRID, lw=0.55, alpha=0.74, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.47, 1.18),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=0.8,
    )

    plt.subplots_adjust(left=0.09, right=0.965, top=0.80, bottom=0.22)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
