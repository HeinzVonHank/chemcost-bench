#!/usr/bin/env python3
"""Figure: stage-level failure decomposition per model.

For every reaction the agent fails (either by abstaining or by predicting a
cost outside the 25%% tolerance), we classify the failure into one of six
mutually exclusive stages:

    Abstention                — agent returned no cost.
    Chemical grounding        — listed components but most lack a resolvable
                                 supplier price (synonym/CAS form did not match
                                 the price index).
    Component omission        — listed too few non-solvent components vs. the
                                 ground truth set (recall < 0.4 with prices ok).
    Quote / pack selection    — components mostly correct, predicted cost off
                                 by 3-30x (wrong pack within an order of
                                 magnitude of the right one).
    Quantity normalization    — components mostly correct, predicted cost off
                                 by >30x (mol%/equiv confusion or wrong
                                 stoichiometric base).
    Arithmetic                — components correct, magnitude near correct
                                 (within 3x), but final number still > 25% off.

Bars are stacked to 100%% of FAILED cases per model. Below each model label
we annotate the absolute failure rate (n_fail / n_total).
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
DEV = ROOT / "data/processed/splits/dev.jsonl"
OUT = ROOT / "manuscript/neurips_2025/figures/failure_decomposition.pdf"

INK = "#172033"
GRID = "#D8DEE9"

CATEGORIES = [
    "Abstention",
    "Chemical grounding",
    "Component omission",
    "Quote / pack selection",
    "Quantity normalization",
    "Arithmetic",
]
COLORS = {
    "Abstention":             "#999999",  # gray
    "Chemical grounding":     "#E69F00",  # orange
    "Component omission":     "#009E73",  # bluish-green
    "Quote / pack selection": "#0072B2",  # blue
    "Quantity normalization": "#CC79A7",  # pink
    "Arithmetic":             "#D55E00",  # vermillion
}

MODELS = [
    ("Qwen3.5-Plus",    "dev_react_qwen35plus_clean.json"),
    ("DeepSeek V4 Pro", "dev_react_deepseek_v4_pro_clean.json"),
    ("GPT-5",           "dev_react_gpt5_traj.json"),
    ("Kimi K2.5",       "dev_react_kimi_k25_clean.json"),
    ("Sonnet 4.6",      "dev_react_sonnet46_traj.json"),
    ("Qwen3-14B",       "dev_react_qwen3_14b_clean.json"),
    ("Qwen3-235B",      "dev_react_qwen3_235b_a22b_clean.json"),
    ("LlaSMol-7B",      "dev_react_llasmol_7b_clean.json"),
    ("ChemDFM",         "dev_react_chemdfm_v2_clean.json"),
    ("ChemLLM",         "dev_react_chemllm_20b_clean.json"),
]


def load_truth() -> dict:
    truth = {}
    with DEV.open() as f:
        for line in f:
            r = json.loads(line)
            truth[r["reaction_id"]] = r
    return truth


def classify(pred: dict, truth: dict) -> str | None:
    """Return one of CATEGORIES for a failed reaction, or None if it succeeded."""
    pcost = pred.get("predicted_cost")
    if pcost is None:
        return "Abstention"

    tcre = pred.get("tcre")
    if tcre is None:
        return "Abstention"
    if tcre <= 0.25:
        return None  # success

    pred_components = pred.get("predicted_components") or []
    pred_names = [c.get("name", "") for c in pred_components if c.get("name")]
    pred_prices = [c.get("price_per_gram") or 0 for c in pred_components]

    truth_nonsolvent = [c for c in truth.get("components", [])
                        if (c.get("role") or "").lower() != "solvent"]
    truth_names = [c.get("name") or c.get("smiles") or "" for c in truth_nonsolvent]
    truth_names = [t for t in truth_names if t]

    if not truth_names:
        # default branch when truth has empty names — treat as arithmetic
        return "Arithmetic"

    # Component recall via the project's name equivalence
    matched = sum(
        1 for tname in truth_names
        if any(component_names_equivalent(tname, pname) for pname in pred_names)
    )
    recall = matched / len(truth_names)

    # Fraction of predicted components that have a non-zero price (proxy for
    # whether the agent successfully resolved them on the supplier index).
    if pred_components:
        valid_priced = sum(1 for p in pred_prices if p and p > 0)
        priced_frac = valid_priced / len(pred_components)
    else:
        priced_frac = 0.0

    if not pred_names or not pred_components:
        return "Component omission"

    if recall < 0.4:
        # Listed too few that match. Was it a grounding failure (listed
        # plenty but couldn't get prices) or omission (listed too few)?
        if len(pred_names) >= 0.6 * len(truth_names) and priced_frac < 0.5:
            return "Chemical grounding"
        return "Component omission"

    # Components mostly recovered: the failure is downstream.
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
    for label, fname in MODELS:
        path = R / fname
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        preds = d.get("predictions", [])
        n_total = d["metrics"]["n_total"]
        counts = {c: 0 for c in CATEGORIES}
        n_fail = 0
        for p in preds:
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
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 0,
        "ytick.major.size": 2.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def halo() -> list[pe.AbstractPathEffect]:
    return [pe.withStroke(linewidth=1.7, foreground="white")]


def main() -> None:
    apply_style()
    truth_map = load_truth()
    rows = per_model_breakdown(truth_map)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    fig.patch.set_facecolor("white")

    x = np.arange(n)
    bar_w = 0.72

    pct_matrix = np.zeros((len(CATEGORIES), n))
    for j, (_, n_fail, _, counts) in enumerate(rows):
        if n_fail == 0:
            continue
        for i, cat in enumerate(CATEGORIES):
            pct_matrix[i, j] = counts[cat] / n_fail * 100

    bottom = np.zeros(n)
    for i, cat in enumerate(CATEGORIES):
        ax.bar(
            x, pct_matrix[i], bar_w, bottom=bottom,
            color=COLORS[cat], linewidth=0, zorder=3, label=cat,
        )
        bottom += pct_matrix[i]

    # x labels: model name + (fail rate)
    xlabels = []
    for label, n_fail, n_total, _ in rows:
        rate = n_fail / n_total * 100 if n_total else 0
        xlabels.append(f"{label}\n({rate:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=22, ha="right")

    ax.set_ylabel("Failed cases (\\%)")
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3,
        frameon=False, handlelength=1.3, handletextpad=0.45, columnspacing=0.9,
    )

    plt.subplots_adjust(left=0.085, right=0.985, top=0.83, bottom=0.32)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print()
    print("Per-model breakdown:")
    print(f"  {'model':18s}  {'n_fail/n':>10s}   " +
          "  ".join(c[:6].rjust(6) for c in CATEGORIES))
    for label, n_fail, n_total, counts in rows:
        cells = "  ".join(f"{counts[c]:>6d}" for c in CATEGORIES)
        print(f"  {label:18s}  {n_fail:>4d}/{n_total:<4d}   {cells}")


if __name__ == "__main__":
    main()
