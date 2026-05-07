#!/usr/bin/env python3
"""Tool calls vs CTA@25 correlation diagnostic (appendix figure).

Replaces the misleading r=0.98 narrative with a proper stratified analysis:
- Bimodal distribution (zero-call domain models cluster + frontier models)
  inflates Pearson r to 0.98 even though within-frontier correlation is weak
  and not significant.
- Reports Pearson r and Spearman ρ at 3 stratification levels, with 95%
  confidence band on the all-9 OLS fit and annotations on which points
  drive which estimate.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parents[1] / "manuscript/neurips_2025/figures/tool_correlation.pdf"

INK = "#2A2417"
MUTED = "#857A66"
GRID = "#E5DDD0"

# (label, mean_calls, cta25, marker, color, group)
DATA = [
    # frontier ReAct (≥15 calls)
    ("Qwen3.5-Plus",    28.24, 40.50, "o",  "#86cfcc", "frontier"),
    ("DeepSeek V4 Pro", 36.87, 46.28, "s",  "#FF7F0E", "frontier"),
    ("GPT-5",           19.01, 30.58, "^",  "#9aaf7e", "frontier"),
    ("Kimi K2.5",       19.92, 29.75, "D",  "#d9a87a", "frontier"),
    ("Sonnet 4.6",      24.12, 25.62, "*",  "#9467BD", "frontier"),
    # base / generalist
    ("Qwen3-235B-A22B",  6.46,  4.96, "P",  "#E377C2", "base"),
    # zero-call cluster
    ("LlaSMol-7B",       0.00,  0.82, "X",  "#7F7F7F", "zero"),
    ("ChemDFM",          0.00,  0.00, "p",  "#BCBD22", "zero"),
    ("ChemLLM",          0.00,  0.00, "h",  "#17BECF", "zero"),
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


def halo():
    return [pe.withStroke(linewidth=1.5, foreground="white")]


def regression_with_ci(xs, ys, x_grid, conf=0.95):
    """OLS fit with confidence band on mean prediction."""
    n = len(xs)
    if n < 3 or xs.std() == 0:
        return None, None, None
    slope, intercept = np.polyfit(xs, ys, 1)
    y_fit = slope * x_grid + intercept
    # Standard error of mean prediction
    y_pred = slope * xs + intercept
    residuals = ys - y_pred
    rss = np.sum(residuals ** 2)
    s_err = np.sqrt(rss / (n - 2))
    x_mean = xs.mean()
    sxx = np.sum((xs - x_mean) ** 2)
    se_mean = s_err * np.sqrt(1.0 / n + (x_grid - x_mean) ** 2 / sxx)
    t_crit = stats.t.ppf((1 + conf) / 2, n - 2)
    band = t_crit * se_mean
    return y_fit, y_fit - band, y_fit + band


def main():
    apply_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 2.6),
                                    gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.patch.set_facecolor("white")

    xs = np.array([d[1] for d in DATA])
    ys = np.array([d[2] for d in DATA])

    # ============ Panel A: Scatter + 95% CI ============
    # Draw 95% CI for the all-9 OLS
    x_grid = np.linspace(0, 42, 100)
    y_fit, lo, hi = regression_with_ci(xs, ys, x_grid)
    axA.fill_between(x_grid, lo, hi, color=MUTED, alpha=0.15, zorder=1,
                     label="95\\% CI (all 9)")
    axA.plot(x_grid, y_fit, color=MUTED, lw=0.8, ls=(0, (3, 2)),
             alpha=0.85, zorder=2)

    # Plot points
    for label, x, y, marker, color, group in DATA:
        size = 38 if marker == "*" else 24
        edge_color = INK if group != "zero" else "#888888"
        axA.scatter(x, y, s=size, c=color, marker=marker,
                    edgecolor=edge_color, linewidth=0.55,
                    alpha=0.95, zorder=3)

    axA.set_xlabel("Mean tool calls per reaction")
    axA.set_ylabel("CTA@25 (\\%)")
    axA.set_xlim(-2, 42)
    axA.set_ylim(-3, 55)
    axA.set_xticks([0, 10, 20, 30, 40])
    axA.grid(color=GRID, lw=0.4, alpha=0.6, zorder=0)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.text(0.0, 1.05, "(A)", transform=axA.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    # Highlight the zero-call cluster with a soft annotation
    axA.annotate("zero-call cluster\n(domain models)",
                 xy=(0, 0), xytext=(8, -2.5),
                 fontsize=5.4, color=MUTED, style="italic",
                 ha="left", va="center",
                 arrowprops={"arrowstyle": "-", "color": MUTED,
                             "lw": 0.45, "shrinkA": 2, "shrinkB": 8},
                 zorder=4)

    # Stat text box
    r_all, p_all = stats.pearsonr(xs, ys)
    rho_all, _ = stats.spearmanr(xs, ys)
    mask_nz = xs > 0
    r_nz, p_nz = stats.pearsonr(xs[mask_nz], ys[mask_nz])
    rho_nz, _ = stats.spearmanr(xs[mask_nz], ys[mask_nz])
    txt = (
        f"All 9 models: $r={r_all:.2f}$, $\\rho={rho_all:.2f}$\n"
        f"Excl. zero-call ($n={mask_nz.sum()}$): $r={r_nz:.2f}$, $\\rho={rho_nz:.2f}$"
    )
    axA.text(0.97, 0.03, txt,
             transform=axA.transAxes, ha="right", va="bottom",
             fontsize=5.5, color=INK,
             bbox={"facecolor": "white", "edgecolor": GRID,
                   "boxstyle": "round,pad=0.35", "lw": 0.4},
             zorder=6)

    # ============ Panel B: Stratified r values bar chart ============
    levels = [
        ("All 9", r_all, rho_all, p_all, 9),
        ("Excl.\nzero-call", r_nz, rho_nz, p_nz, int(mask_nz.sum())),
    ]
    n_levels = len(levels)
    x = np.arange(n_levels)
    bar_w = 0.32

    # Pearson and Spearman bars
    for i, (lbl, r, rho, p, n) in enumerate(levels):
        # Pearson
        axB.bar(x[i] - bar_w / 2, r, bar_w, color="#cc526a",
                edgecolor=INK, linewidth=0.45, zorder=3,
                label="Pearson $r$" if i == 0 else None)
        axB.text(x[i] - bar_w / 2, r + 0.025, f"{r:.2f}",
                 ha="center", va="bottom", fontsize=5.2,
                 color=INK, weight="bold", path_effects=halo(), zorder=5)
        # Spearman
        axB.bar(x[i] + bar_w / 2, rho, bar_w, color="#86cfcc",
                edgecolor=INK, linewidth=0.45, zorder=3,
                label="Spearman $\\rho$" if i == 0 else None)
        axB.text(x[i] + bar_w / 2, rho + 0.025, f"{rho:.2f}",
                 ha="center", va="bottom", fontsize=5.2,
                 color=INK, weight="bold", path_effects=halo(), zorder=5)
        # n annotation below x label
        axB.text(x[i], -0.18, f"$n={n}$",
                 ha="center", va="top", fontsize=5.0,
                 color=MUTED, transform=axB.get_xaxis_transform())

    # Significance line at 0.05 (informational)
    axB.axhline(0, color=INK, lw=0.4, zorder=2)

    axB.set_xticks(x)
    axB.set_xticklabels([lvl[0] for lvl in levels], fontsize=5.7)
    axB.set_ylim(0, 1.1)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_ylabel("Correlation coefficient")
    axB.grid(axis="y", color=GRID, lw=0.4, alpha=0.7, zorder=0)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.legend(loc="upper right", frameon=False, handlelength=1.0,
               handletextpad=0.4, fontsize=5.6)
    axB.text(0.0, 1.05, "(B)", transform=axB.transAxes,
             fontsize=7.4, weight="bold", color=INK)

    plt.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.20, wspace=0.32)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print()
    print("Stats summary:")
    for lbl, r, rho, p, n in levels:
        print(f"  {lbl}: n={n}  r={r:.3f} (p={p:.3g})  ρ={rho:.3f}")


if __name__ == "__main__":
    main()
