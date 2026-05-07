#!/usr/bin/env python3
"""Tool calling patterns: success vs failure trajectories.

Aggregates 5 frontier ReAct agents x 121 reactions = 605 trajectories.
A reaction is a SUCCESS if TCRE <= 0.25 (within 25% tolerance), otherwise
FAILURE (includes abstentions).

Panel A: Mean tool calls per reaction by tool type, grouped by outcome.
         Failure trajectories make MORE search and quote calls (re-trying
         queries that the supplier index does not match) but FEWER calculate
         calls (do not finish the pipeline).

Panel B: Per-reaction quote coverage cumulative distribution. Coverage =
         fraction of truth non-solvent components for which the agent issued
         a get_supplier_quotes call that returned at least one quote.
         Success reactions concentrate at full coverage; failure reactions
         plateau at partial coverage.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chemcost.evaluation.metrics import component_names_equivalent  # noqa: E402

R = ROOT / "results"
OUT = ROOT / "manuscript/neurips_2025/figures/traj_patterns.pdf"

INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE9"

COL_SUCCESS = "#0072B2"   # Wong blue
COL_FAILURE = "#D55E00"   # Wong vermillion

TOOL_ORDER = ["search_chemical", "get_supplier_quotes",
              "compute_molar_mass", "calculate"]
TOOL_LABEL = {
    "search_chemical":     "search\n_chemical",
    "get_supplier_quotes": "get\n_supplier_quotes",
    "compute_molar_mass":  "compute\n_molar_mass",
    "calculate":           "calculate",
}

MODELS = [
    ("Qwen3.5-Plus",   "dev_react_qwen35plus_clean.json",      "dev_multi_qwen35plus_clean.json"),
    ("DeepSeek V4 Pro","dev_react_deepseek_v4_pro_clean.json", "dev_multi40_deepseek_v4_pro_clean.json"),
    ("GPT-5",          "dev_react_gpt5_traj.json",             "dev_multi40_gpt5_clean.json"),
    ("Kimi K2.5",      "dev_react_kimi_k25_clean.json",        "dev_multi40_kimi_k25_clean.json"),
    ("Sonnet 4.6",     "dev_react_sonnet46_traj.json",         "dev_multi40_sonnet46_clean.json"),
]

DEV_S = ROOT / "data/processed/splits/dev.jsonl"
DEV_M = ROOT / "data/processed/splits/dev_multistep_v2.jsonl"


def safe_str(x):
    return x if isinstance(x, str) else ""


def loose_match(t, q):
    t = safe_str(t); q = safe_str(q)
    if not t or not q:
        return False
    try:
        if component_names_equivalent(t, q):
            return True
    except Exception:
        pass
    return t.lower().strip() in q.lower().strip() or q.lower().strip() in t.lower().strip()


def load_truth():
    truth = {}
    for fp in (DEV_S, DEV_M):
        with fp.open() as f:
            for line in f:
                r = json.loads(line)
                truth[r["reaction_id"]] = r
    return truth


def collect():
    truth = load_truth()
    n_succ = n_fail = 0
    tc = {"success": Counter(), "failure": Counter()}
    cov = {"success": [], "failure": []}
    for _, sfn, mfn in MODELS:
        for fn in (sfn, mfn):
            p = R / fn
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for pred in d.get("predictions", []):
                t = truth.get(pred["reaction_id"])
                if not t:
                    continue
                tcre = pred.get("tcre")
                outcome = "success" if (tcre is not None and tcre <= 0.25) else "failure"
                if outcome == "success":
                    n_succ += 1
                else:
                    n_fail += 1
                tcs = pred.get("tool_calls") or []
                for c in tcs:
                    tn = safe_str(c.get("tool_name"))
                    if tn:
                        tc[outcome][tn] += 1
                # quote coverage
                quote_ok = {}
                for c in tcs:
                    if safe_str(c.get("tool_name")) != "get_supplier_quotes":
                        continue
                    args = c.get("arguments") or {}
                    if not isinstance(args, dict):
                        continue
                    q = args.get("smiles_or_name") or ""
                    if not isinstance(q, str):
                        continue
                    got = False
                    try:
                        res = c.get("result", "")
                        rj = json.loads(res) if isinstance(res, str) else res
                        if isinstance(rj, dict) and (
                            rj.get("quotes")
                            or rj.get("tier") == "pack_based"
                        ):
                            got = True
                    except Exception:
                        pass
                    key = q.lower()
                    quote_ok[key] = quote_ok.get(key, False) or got
                tnonsolv = [c for c in t["components"]
                            if safe_str(c.get("role")).lower() != "solvent"]
                if not tnonsolv:
                    cov[outcome].append(1.0)
                    continue
                hits = 0
                for tc_truth in tnonsolv:
                    tname = safe_str(tc_truth.get("name")) or safe_str(tc_truth.get("smiles"))
                    tsmi = safe_str(tc_truth.get("smiles"))
                    for q, ok in quote_ok.items():
                        if ok and (loose_match(tname, q) or (tsmi and loose_match(tsmi, q))):
                            hits += 1
                            break
                cov[outcome].append(hits / len(tnonsolv))
    return n_succ, n_fail, tc, cov


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
        "font.size": 7.4,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 0,
        "ytick.major.size": 2.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def halo():
    return [pe.withStroke(linewidth=1.7, foreground="white")]


def main():
    apply_style()
    n_succ, n_fail, tc, cov = collect()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.85),
                                    gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.patch.set_facecolor("white")

    # ===== Panel A: grouped bar — tool calls per reaction by outcome =====
    x = np.arange(len(TOOL_ORDER))
    bw = 0.36
    succ_vals = [tc["success"][t] / max(n_succ, 1) for t in TOOL_ORDER]
    fail_vals = [tc["failure"][t] / max(n_fail, 1) for t in TOOL_ORDER]

    axA.bar(x - bw/2, succ_vals, bw, color=COL_SUCCESS, linewidth=0,
            label=f"Success (TCRE $\\leq$ 0.25, $n={n_succ}$)", zorder=3)
    axA.bar(x + bw/2, fail_vals, bw, color=COL_FAILURE, linewidth=0,
            label=f"Failure ($n={n_fail}$)", zorder=3)

    for i, (s, f) in enumerate(zip(succ_vals, fail_vals)):
        axA.text(i - bw/2, s + 0.3, f"{s:.1f}", ha="center", va="bottom",
                 fontsize=6.0, color=COL_SUCCESS, weight="bold",
                 path_effects=halo(), zorder=5)
        axA.text(i + bw/2, f + 0.3, f"{f:.1f}", ha="center", va="bottom",
                 fontsize=6.0, color=COL_FAILURE, weight="bold",
                 path_effects=halo(), zorder=5)

    axA.set_xticks(x)
    axA.set_xticklabels([TOOL_LABEL[t] for t in TOOL_ORDER], fontsize=6.0,
                         linespacing=0.95)
    axA.set_ylabel("Mean calls per reaction")
    axA.set_ylim(0, 18.5)
    axA.set_yticks([0, 5, 10, 15])
    axA.grid(axis="y", color=GRID, lw=0.5, alpha=0.7, zorder=0)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.legend(loc="upper left", frameon=False, handlelength=1.0,
               handletextpad=0.4, labelspacing=0.25, borderpad=0.2)
    axA.text(0.0, 1.05, "(A)", transform=axA.transAxes,
             fontsize=8.4, weight="bold", color=INK)

    # ===== Panel B: CDF of per-reaction quote coverage =====
    for outcome, color, label in (("success", COL_SUCCESS, "Success"),
                                  ("failure", COL_FAILURE, "Failure")):
        arr = np.sort(np.asarray(cov[outcome]))
        if len(arr) == 0:
            continue
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        # Step CDF
        axB.step(arr, cdf, where="post", color=color, lw=1.7,
                 label=label, zorder=3)
        # mark median
        med = float(np.median(arr))
        axB.scatter([med], [0.5], s=22, color=color, edgecolor="white",
                    linewidth=0.6, zorder=5)

    axB.set_xlabel("Quote coverage (fraction of truth components priced)")
    axB.set_ylabel("Cumulative reactions")
    axB.set_xlim(-0.02, 1.02)
    axB.set_ylim(0, 1.02)
    axB.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    # Annotate the gap at coverage = 1.0 (full coverage)
    succ = np.asarray(cov["success"]); fail = np.asarray(cov["failure"])
    succ_full = (succ == 1.0).mean()
    fail_full = (fail == 1.0).mean()
    # Bars at x=1.0 visualizing the % NOT at full coverage
    axB.scatter([1.0], [1 - succ_full], s=22, color=COL_SUCCESS,
                edgecolor="white", linewidth=0.6, zorder=5)
    axB.scatter([1.0], [1 - fail_full], s=22, color=COL_FAILURE,
                edgecolor="white", linewidth=0.6, zorder=5)
    axB.text(0.985, 1 - succ_full - 0.04, f"{succ_full*100:.0f}% full",
             ha="right", va="top", fontsize=6.0, color=COL_SUCCESS,
             weight="bold", path_effects=halo())
    axB.text(0.985, 1 - fail_full + 0.04, f"{fail_full*100:.0f}% full",
             ha="right", va="bottom", fontsize=6.0, color=COL_FAILURE,
             weight="bold", path_effects=halo())
    axB.legend(loc="upper left", frameon=False, handlelength=1.4,
               handletextpad=0.4, labelspacing=0.3, borderpad=0.2)
    axB.text(0.0, 1.05, "(B)", transform=axB.transAxes,
             fontsize=8.4, weight="bold", color=INK)

    plt.subplots_adjust(left=0.075, right=0.93, top=0.92, bottom=0.18, wspace=0.30)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")
    print()
    print(f"n_success = {n_succ},  n_failure = {n_fail}")
    print("Mean tool calls per reaction:")
    for t in TOOL_ORDER:
        s = tc["success"][t] / max(n_succ, 1)
        f = tc["failure"][t] / max(n_fail, 1)
        print(f"  {t:<22s}  success={s:5.2f}  failure={f:5.2f}  (s/f = {s/f if f>0 else float('inf'):.2f}x)")
    print("Quote coverage:")
    for outcome in ("success", "failure"):
        arr = np.asarray(cov[outcome])
        print(f"  {outcome}: mean={arr.mean()*100:.1f}%  full_cov={(arr==1.0).mean()*100:.0f}%  "
              f"zero_cov={(arr==0).mean()*100:.0f}%  (n={len(arr)})")


if __name__ == "__main__":
    main()
