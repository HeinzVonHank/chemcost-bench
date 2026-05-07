#!/usr/bin/env python3
"""Compact failure decomposition for frontier ReAct (§4.1).

Grouped bar chart: 5 frontier models + 1 aggregate group on the right.
Each group has 5 side-by-side bars, one per procurement stage. Reader
compares stage profiles within a model (vertical scan) and stage shares
across models (horizontal scan within one color).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chemcost.evaluation.metrics import component_names_equivalent  # noqa: E402

R = ROOT / "results"
DEV_SINGLE = ROOT / "data/processed/splits/dev.jsonl"
DEV_MULTI = ROOT / "data/processed/splits/dev_multistep_v2.jsonl"
OUT = ROOT / "manuscript/neurips_2025/figures/failure_frontier.pdf"

INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE9"

CATEGORIES = [
    "Abstention",
    "Component omission",
    "Quote / pack selection",
    "Quantity normalization",
    "Arithmetic",
]
COLORS = {
    "Abstention":             "#999999",
    "Component omission":     "#009E73",
    "Quote / pack selection": "#0072B2",
    "Quantity normalization": "#CC79A7",
    "Arithmetic":             "#D55E00",
}

MODELS = [
    ("Qwen3.5-Plus",    "dev_react_qwen35plus_clean.json",   "dev_multi_qwen35plus_clean.json"),
    ("DeepSeek V4 Pro", "dev_react_deepseek_v4_pro_clean.json", "dev_multi40_deepseek_v4_pro_clean.json"),
    ("GPT-5",           "dev_react_gpt5_traj.json",          "dev_multi40_gpt5_clean.json"),
    ("Kimi K2.5",       "dev_react_kimi_k25_clean.json",     "dev_multi40_kimi_k25_clean.json"),
    ("Sonnet 4.6",      "dev_react_sonnet46_traj.json",      "dev_multi40_sonnet46_clean.json"),
]


def load_truth() -> dict:
    truth = {}
    for path in (DEV_SINGLE, DEV_MULTI):
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                truth[r["reaction_id"]] = r
    return truth


def classify(pred: dict, truth: dict) -> str | None:
    pcost = pred.get("predicted_cost")
    if pcost is None:
        return "Abstention"
    tcre = pred.get("tcre")
    if tcre is None:
        return "Abstention"
    if tcre <= 0.25:
        return None

    pred_components = pred.get("predicted_components") or []
    pred_names = [c.get("name", "") for c in pred_components if c.get("name")]

    truth_nonsolvent = [c for c in truth.get("components", [])
                        if (c.get("role") or "").lower() != "solvent"]
    truth_names = [c.get("name") or c.get("smiles") or "" for c in truth_nonsolvent]
    truth_names = [t for t in truth_names if t]

    if not truth_names:
        return "Arithmetic"
    if not pred_names or not pred_components:
        return "Component omission"

    matched = sum(
        1 for tname in truth_names
        if any(component_names_equivalent(tname, pname) for pname in pred_names)
    )
    recall = matched / len(truth_names)

    if recall < 0.4:
        return "Component omission"

    tcost = pred.get("true_cost")
    if not tcost or tcost <= 0:
        return "Arithmetic"

    log_err = abs(math.log10(pcost / tcost)) if pcost > 0 else 3.0
    if log_err > 1.5:
        return "Quantity normalization"
    if log_err > 0.5:
        return "Quote / pack selection"
    return "Arithmetic"


def per_model_breakdown(truth_map: dict) -> list[tuple[str, int, int, dict[str, int]]]:
    rows = []
    for label, single_fname, multi_fname in MODELS:
        counts = {c: 0 for c in CATEGORIES}
        n_fail = 0
        n_total = 0
        for fname in (single_fname, multi_fname):
            path = R / fname
            if not path.exists():
                print(f"[skip] {label}: {path} missing")
                continue
            d = json.loads(path.read_text())
            n_total += d["metrics"]["n_total"]
            for p in d.get("predictions", []):
                t = truth_map.get(p["reaction_id"])
                if t is None:
                    continue
                bucket = classify(p, t)
                if bucket is None:
                    continue
                counts[bucket] += 1
                n_fail += 1
        rows.append((label, n_fail, n_total, counts))
    return rows


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
        "font.size": 7.2,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 0,
        "ytick.major.size": 2.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    apply_style()
    truth_map = load_truth()
    rows = per_model_breakdown(truth_map)

    plot_labels: list[str] = []
    plot_pct: list[dict[str, float]] = []
    plot_n_fail: list[int] = []
    plot_n_total: list[int] = []

    for label, n_fail, n_total, counts in rows:
        plot_labels.append(label)
        plot_pct.append({c: counts[c] / n_fail * 100 if n_fail else 0 for c in CATEGORIES})
        plot_n_fail.append(n_fail)
        plot_n_total.append(n_total)

    agg_counts = {c: sum(r[3][c] for r in rows) for c in CATEGORIES}
    agg_n_fail = sum(r[1] for r in rows)
    agg_n_total = sum(r[2] for r in rows)
    plot_labels.append("Aggregate")
    plot_pct.append({c: agg_counts[c] / agg_n_fail * 100 for c in CATEGORIES})
    plot_n_fail.append(agg_n_fail)
    plot_n_total.append(agg_n_total)

    n_groups = len(plot_labels)
    n_cats = len(CATEGORIES)

    fig, ax = plt.subplots(figsize=(5.0, 2.3))
    fig.patch.set_facecolor("white")

    group_w = 1.0
    intra_gap = 0.04
    bar_w = (group_w - intra_gap * (n_cats - 1)) / n_cats
    inter_gap = 0.55  # extra space between groups

    centers = []
    for g in range(n_groups):
        # extra gap before the Aggregate group
        gap_extra = 0.25 if g == n_groups - 1 else 0.0
        group_left = g * (group_w + inter_gap) + gap_extra
        center = group_left + group_w / 2 - bar_w / 2
        centers.append(center)
        for ci, cat in enumerate(CATEGORIES):
            x = group_left + ci * (bar_w + intra_gap)
            h = plot_pct[g][cat]
            ax.bar(
                x, h, width=bar_w,
                color=COLORS[cat],
                linewidth=0,
                zorder=3,
                label=cat if g == 0 else None,
                align="edge",
            )
            if h >= 1.5:
                ax.text(
                    x + bar_w / 2, h + 1.0,
                    f"{h:.0f}",
                    ha="center", va="bottom",
                    fontsize=5.6, color=INK,
                    zorder=5,
                )

    # group labels: model name only
    ax.set_xticks(centers)
    ax.set_xticklabels(plot_labels)
    for tick, lbl in zip(ax.get_xticklabels(), plot_labels):
        if lbl == "Aggregate":
            tick.set_fontweight("bold")

    # dashed separator between models and Aggregate
    sep_x = (centers[-2] + centers[-1]) / 2
    ax.axvline(sep_x, color=MUTED, lw=0.55, ls=(0, (1.5, 2.0)), zorder=2)

    ax.set_ylim(0, 88)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("Failed cases (\\%)")
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", bottom=False)

    last_right = (n_groups - 1) * (group_w + inter_gap) + 0.25 + group_w
    ax.set_xlim(-0.35, last_right + 0.15)

    handles = [Patch(facecolor=COLORS[c], edgecolor="none", label=c) for c in CATEGORIES]
    ax.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.5, 1.16),
        ncol=5, frameon=False,
        handlelength=1.05, handletextpad=0.36, columnspacing=0.75,
    )

    plt.subplots_adjust(left=0.085, right=0.99, top=0.83, bottom=0.20)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print()
    print("Per-group %:")
    head = "  ".join(c[:6].rjust(6) for c in CATEGORIES)
    print(f"  {'group':18s}  {'n_fail/n':>10s}   {head}")
    for i in range(n_groups):
        cells = "  ".join(f"{plot_pct[i][c]:6.1f}" for c in CATEGORIES)
        print(f"  {plot_labels[i]:18s}  {plot_n_fail[i]:>4d}/{plot_n_total[i]:<4d}   {cells}")


if __name__ == "__main__":
    main()
