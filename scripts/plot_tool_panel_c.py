#!/usr/bin/env python3
"""Per-model failure fingerprint (Panel C of tool_usage figure).

Heatmap: rows = models (those with >= 10 succ AND >= 10 fail), columns =
trajectory features. Cell value = Cliff's delta(succ - fail). Positive =
successful trajectories have a higher value; negative = failed trajectories
have a higher value. Asterisk = Mann-Whitney U two-sided p < 0.05.

The point of the panel: each model's row is qualitatively different,
showing that ChemCost separates ordering, persistence, retrieval-grounding,
and stuck-loop behaviours into distinct signatures.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_within_model_traj import (  # noqa: E402
    classify,
    cliffs_delta,
    extract_features,
    mann_whitney,
)

R = ROOT / "results"
OUT = ROOT / "manuscript/neurips_2025/figures/tool_usage_panel_c.pdf"

INK = "#2A2417"
MUTED = "#857A66"
GRID = "#E5DDD0"

MODELS = [
    ("DeepSeek V4 Pro", "dev_react_deepseek_v4_pro_clean.json", "dev_multi40_deepseek_v4_pro_clean.json"),
    ("Qwen3.5-Plus",    "dev_react_qwen35plus_clean.json",      "dev_multi_qwen35plus_clean.json"),
    ("Sonnet 4.6",      "dev_react_sonnet46_traj.json",         "dev_multi40_sonnet46_clean.json"),
    ("Kimi K2.5",       "dev_react_kimi_k25_clean.json",        "dev_multi40_kimi_k25_clean.json"),
    ("GPT-5",           "dev_react_gpt5_traj.json",             "dev_multi40_gpt5_clean.json"),
]

# Feature columns: short label + the key in extract_features() output.
FEATURES = [
    ("total calls",     "total_calls"),
    ("# search",        "n_search"),
    ("# quote",         "n_quote"),
    ("# calc",          "n_calc"),
    ("empty quote",     "empty_quote"),
    ("quote grounded",  "quote_grounded_rate"),
    ("first quote step","first_quote_step"),
    ("retry rate",      "retry_rate"),
]


def collect_groups():
    """Return {model_label: {'success': [feat_dict,...], 'fail': [...]}}."""
    out = {}
    for label, sfn, mfn in MODELS:
        succ, fail = [], []
        for fn in [sfn, mfn]:
            p = R / fn
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for pred in d.get("predictions", []):
                cls = classify(pred)
                if cls == "abstain":
                    continue
                f = extract_features(pred.get("tool_calls") or [])
                if cls == "success":
                    succ.append(f)
                else:
                    fail.append(f)
        out[label] = {"success": succ, "fail": fail}
    return out


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
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main():
    apply_style()
    groups = collect_groups()

    n_rows = len(MODELS)
    n_cols = len(FEATURES)

    delta = np.full((n_rows, n_cols), np.nan)
    pvals = np.full((n_rows, n_cols), np.nan)
    sample_sizes = []
    for i, (label, _, _) in enumerate(MODELS):
        s = groups[label]["success"]
        f = groups[label]["fail"]
        sample_sizes.append((label, len(s), len(f)))
        if len(s) < 10 or len(f) < 10:
            continue
        for j, (_, key) in enumerate(FEATURES):
            xs = [r[key] for r in s]
            ys = [r[key] for r in f]
            d = cliffs_delta(xs, ys)
            _, p = mann_whitney(xs, ys)
            delta[i, j] = d
            pvals[i, j] = p

    # Diverging colormap matching the paper palette: teal (neg) -> ivory -> deep pink (pos)
    cmap = LinearSegmentedColormap.from_list(
        "tealpink", ["#5fa6a3", "#a8d3d1", "#fbf6ec", "#e6a3b1", "#cc526a"]
    )

    fig, ax = plt.subplots(figsize=(3.6, 1.85))
    fig.patch.set_facecolor("white")

    im = ax.imshow(delta, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")

    # Cell annotations: Cliff's delta value, asterisk if p<0.05
    for i in range(n_rows):
        for j in range(n_cols):
            d = delta[i, j]
            if np.isnan(d):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=5.4, color=MUTED)
                continue
            txt = f"{d:+.2f}"
            if pvals[i, j] < 0.05:
                txt = txt + "*"
            tcolor = "white" if abs(d) > 0.35 else INK
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=5.4, color=tcolor)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([lab for lab, _ in FEATURES], rotation=30, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([lab for lab, _, _ in MODELS])

    # Subtle gridlines between cells
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025,
                        ticks=[-0.5, 0.0, 0.5])
    cbar.ax.tick_params(labelsize=5.6, length=0)
    cbar.outline.set_visible(False)
    cbar.set_label("Cliff's $\\delta$\n(succ vs fail)",
                   fontsize=6.0, color=INK)

    ax.text(0.0, 1.10, "(C)", transform=ax.transAxes,
            fontsize=7.4, weight="bold", color=INK)

    plt.subplots_adjust(left=0.18, right=0.93, top=0.88, bottom=0.30)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print()
    print("Sample sizes (success / fail):")
    for label, ns, nf in sample_sizes:
        print(f"  {label:<18s}  S={ns:3d}  F={nf:3d}")
    print()
    print("Cliff's delta (Mann-Whitney p):")
    print(f"  {'model':<18s}  " + "  ".join(f"{lab:>10s}" for lab, _ in FEATURES))
    for i, (label, _, _) in enumerate(MODELS):
        cells = []
        for j in range(n_cols):
            d = delta[i, j]
            p = pvals[i, j]
            if np.isnan(d):
                cells.append(f"{'-':>10s}")
            else:
                star = "*" if p < 0.05 else " "
                cells.append(f"{d:+.2f}{star}    ".replace("    ", "    ")[:10].rjust(10))
        print(f"  {label:<18s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
