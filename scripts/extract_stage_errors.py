#!/usr/bin/env python3
"""Extract stage-level error attribution from evaluation results with trajectories."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def classify_prediction(pred, gt_record=None):
    """Classify a prediction into a failure stage."""
    predicted_cost = pred.get("predicted_cost")
    true_cost = pred.get("true_cost")
    tcre = pred.get("tcre")

    if predicted_cost is None:
        return "abstention"

    if tcre is not None and tcre <= 0.25:
        return "correct"

    tool_calls = pred.get("tool_calls", [])
    if not tool_calls:
        return "unknown_error"

    searches = [tc for tc in tool_calls if tc["tool_name"] == "search_chemical"]
    quotes = [tc for tc in tool_calls if tc["tool_name"] == "get_supplier_quotes"]

    if not searches and not quotes:
        return "chemical_id_error"

    search_failures = sum(1 for s in searches if not s.get("success", True))
    if searches and search_failures == len(searches):
        return "chemical_id_error"

    if not quotes:
        return "price_retrieval_error"

    quote_failures = sum(1 for q in quotes if not q.get("success", True))
    if quotes and quote_failures == len(quotes):
        return "price_retrieval_error"

    return "arithmetic_error"


def analyze_results(results_path):
    with open(results_path) as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    stage_counts = Counter()
    details = []

    for pred in predictions:
        stage = classify_prediction(pred)
        stage_counts[stage] += 1
        details.append({
            "reaction_id": pred["reaction_id"],
            "stage": stage,
            "predicted_cost": pred.get("predicted_cost"),
            "true_cost": pred.get("true_cost"),
            "tcre": pred.get("tcre"),
            "n_tool_calls": len(pred.get("tool_calls", [])),
        })

    total = len(predictions)
    print(f"\n{'='*60}")
    print(f"Stage-Level Error Analysis: {results_path}")
    print(f"{'='*60}")
    print(f"Total predictions: {total}\n")

    order = ["correct", "abstention", "chemical_id_error",
             "price_retrieval_error", "arithmetic_error", "unknown_error"]
    for stage in order:
        count = stage_counts.get(stage, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {stage:<25} {count:>4} ({pct:>5.1f}%)")

    all_tool_calls = []
    for pred in predictions:
        all_tool_calls.extend(pred.get("tool_calls", []))

    if all_tool_calls:
        tool_counts = Counter(tc["tool_name"] for tc in all_tool_calls)
        n_with_traj = sum(1 for p in predictions if p.get("tool_calls"))
        avg_calls = len(all_tool_calls) / n_with_traj if n_with_traj else 0

        print(f"\nTool usage ({n_with_traj} reactions with trajectories):")
        print(f"  Avg tool calls per reaction: {avg_calls:.1f}")
        for tool, count in tool_counts.most_common():
            print(f"  {tool}: {count}")

    return {"stage_counts": dict(stage_counts), "total": total, "details": details}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results.json> [results2.json ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        result = analyze_results(path)
        out_path = path.replace(".json", "_stages.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {out_path}")
