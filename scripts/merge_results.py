#!/usr/bin/env python3
"""Merge staged result JSONs into a single file and recompute metrics."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from chemcost.evaluation.metrics import ComponentMatch, CostPrediction, evaluate_stratified


def main():
    parser = argparse.ArgumentParser(description="Merge staged ChemCost results")
    parser.add_argument("inputs", nargs="+", help="Input JSON result files (in order)")
    parser.add_argument("--output", required=True, help="Output merged JSON path")
    args = parser.parse_args()

    all_predictions = []
    seen_ids = set()

    for path in args.inputs:
        with open(path) as f:
            data = json.load(f)
        n_before = len(all_predictions)
        for p in data["predictions"]:
            if p["reaction_id"] not in seen_ids:
                seen_ids.add(p["reaction_id"])
                all_predictions.append(p)
        print(f"{path}: +{len(all_predictions) - n_before} predictions")

    print(f"Total unique predictions: {len(all_predictions)}")

    # Rebuild CostPrediction objects for metric computation
    cost_preds = [
        CostPrediction(
            reaction_id=p["reaction_id"],
            predicted_cost=p["predicted_cost"],
            true_cost=p["true_cost"],
            predicted_components=[],
            true_component_names=[],
            cost_tier=p.get("cost_tier", "unknown"),
        )
        for p in all_predictions
    ]

    stratified = evaluate_stratified(cost_preds)
    results = stratified["all"]

    output = {
        "metrics": results.to_dict(),
        "metrics_by_tier": {tier: r.to_dict() for tier, r in stratified.items()},
        "n_evaluable_records": len(all_predictions),
        "predictions": all_predictions,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Merged Results ===")
    for k, v in results.to_dict().items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
