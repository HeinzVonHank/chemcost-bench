#!/usr/bin/env python3
"""Compute per-step procurement costs for multi-step reactions.

Outputs a JSON sidecar with per-step cost breakdown, used by
plot_multistep_cascade.py for the appendix figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemcost.cost_calculator import calculate_multistep_procurement_cost
from chemcost.pricing.pricing_db import PricingDB

DEV_MULTI = ROOT / "data/processed/splits/dev_multistep_v2.jsonl"
DB_PATH = ROOT / "data/processed/pricing_db.sqlite"
OUT_JSON = ROOT / "results/multistep_cascade_analysis.json"


def main():
    db = PricingDB(str(DB_PATH))
    records = [json.loads(line) for line in open(DEV_MULTI)]
    print(f"Loaded {len(records)} multi-step records")

    out = []
    n_complete = 0  # reactions with all per-step costs available

    for rec in records:
        rid = rec["reaction_id"]
        n_steps = rec["n_steps"]
        try:
            res = calculate_multistep_procurement_cost(rec["steps"], db)
        except Exception as e:
            print(f"  {rid}: FAILED ({e})")
            continue

        step_costs = [sr.result.procurement_cost_usd_per_g_product
                      for sr in res.step_results]
        intermediates = [sr.intermediate_cost_per_g for sr in res.step_results]
        all_priced = all(c is not None for c in step_costs)
        if all_priced:
            n_complete += 1

        # Compute per-step *delta* (added cost at this step) when we have
        # both this step's total cost AND last step's intermediate input.
        # delta_i = cost_i - cost_{i-1} (cost added by this step).
        # For step 1 the delta == cost_1 (no upstream intermediate).
        deltas: list[float | None] = []
        for i, sr in enumerate(res.step_results):
            cost = sr.result.procurement_cost_usd_per_g_product
            if cost is None:
                deltas.append(None)
                continue
            if i == 0:
                deltas.append(cost)
            else:
                prev = res.step_results[i - 1].result.procurement_cost_usd_per_g_product
                deltas.append(cost - prev if prev is not None else None)

        out.append({
            "reaction_id": rid,
            "n_steps": n_steps,
            "final_cost": res.procurement_cost_usd_per_g_product,
            "step_costs": step_costs,           # cost at end of each step
            "step_deltas": deltas,              # cost ADDED by each step
            "intermediates": intermediates,     # cost/g of intermediate input
            "cost_tier": res.cost_tier,
            "all_steps_priced": all_priced,
        })

    print(f"\nReactions with ALL per-step costs available: {n_complete}/{len(records)}")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved analysis -> {OUT_JSON}")

    # Summary stats
    print("\n=== Step-count distribution (analyzed) ===")
    from collections import Counter
    sc = Counter(r["n_steps"] for r in out)
    for k, v in sorted(sc.items()):
        print(f"  {k} steps: {v} reactions")

    # Last-step / final ratio (only on fully-priced)
    ratios = []
    for r in out:
        if not r["all_steps_priced"]: continue
        if r["final_cost"] is None or r["final_cost"] == 0: continue
        last_delta = r["step_deltas"][-1]
        if last_delta is None: continue
        ratios.append(last_delta / r["final_cost"])
    if ratios:
        import statistics
        print(f"\nLast-step delta / final ratio (n={len(ratios)}):")
        print(f"  mean   = {statistics.mean(ratios):.2%}")
        print(f"  median = {statistics.median(ratios):.2%}")
        print(f"  min/max = {min(ratios):.2%} / {max(ratios):.2%}")


if __name__ == "__main__":
    main()
