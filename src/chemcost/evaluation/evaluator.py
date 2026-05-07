"""Evaluation pipeline: run an agent on the benchmark and compute metrics."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Literal, Protocol

from .metrics import (
    BenchmarkResults,
    ComponentMatch,
    CostPrediction,
    component_names_equivalent,
    evaluate_stratified,
    price_optimization_score,
    token_efficiency,
)
from .tool_tracker import ToolTracker

logger = logging.getLogger(__name__)


class CostEstimationAgent(Protocol):
    """Protocol for agents that estimate reaction costs."""

    def estimate_cost(self, reaction: dict) -> dict:
        """Given a reaction record, return a cost estimate.

        Input: reaction dict with keys like reaction_name, product_mw,
        and components (names only, with SMILES intentionally withheld).
        Output: dict with predicted_cost_per_gram and predicted_components
        (list of {name, price_per_gram}).
        """
        ...


def _ground_truth_cost(reaction: dict) -> float | None:
    """Return the active benchmark truth cost for evaluation.

    Procurement cost is authoritative when the field exists. Older dataset
    snapshots may only carry the legacy median-based field.
    """
    if "procurement_cost_usd_per_g_product" in reaction:
        return reaction.get("procurement_cost_usd_per_g_product")
    return reaction.get("total_cost_per_gram_product_usd")


def load_dataset(path: str | Path) -> list[dict]:
    """Load a JSONL dataset split."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_product_component(component: dict) -> bool:
    """Return whether a component represents the target product, not an input."""
    return component.get("role") == "product" or component.get("original_role") == "product"


def _agent_visible_components(reaction: dict) -> list[dict]:
    """Return the component rows that should be visible to an evaluated agent."""
    return [
        component
        for component in reaction.get("components", [])
        if not _is_product_component(component)
    ]


def prepare_agent_input(reaction: dict) -> dict:
    """Strip ground-truth pricing from a reaction record for agent input."""
    component_rows = []
    for c in _agent_visible_components(reaction):
        name = c.get("name") or c.get("smiles") or "UNKNOWN_COMPONENT"
        row = {
            "name": name,
            "role": c.get("role", "reactant"),
            "equivalents": c.get("equivalents"),
            "mw": c.get("mw"),  # MW provided; SMILES withheld (official eval setting)
        }
        if c.get("quantity_description"):
            row["quantity_description"] = c["quantity_description"]
        component_rows.append(row)

    product = reaction.get("product", {})
    product_mw = (product.get("mw") if isinstance(product, dict) else None) \
        or reaction.get("product_mw")

    agent_input = {
        "reaction_id": reaction["reaction_id"],
        "reaction_name": reaction["reaction_name"],
        "reaction_smiles": "",
        "product_smiles": "",
        "product_mw": product_mw,
        "yield_percent": reaction.get("yield_percent"),
        "components": component_rows,
    }
    if reaction.get("description"):
        agent_input["description"] = reaction["description"]
    if reaction.get("mixed_format"):
        agent_input["mixed_format"] = reaction["mixed_format"]
    return agent_input


def _format_equivalents(equiv: float | None, role: str) -> str:
    """Human-readable equivalents string, handling catalytic mol%."""
    if equiv is None:
        return ""
    if role == "catalyst" and equiv < 1.0:
        mol_pct = equiv * 100
        if mol_pct == int(mol_pct):
            return f"{int(mol_pct)} mol%"
        return f"{mol_pct:.1f} mol%"
    if equiv == int(equiv):
        return f"{int(equiv)} equivalent{'s' if equiv != 1 else ''}"
    return f"{equiv:.2f} equivalents"


def _component_phrase(comp: dict) -> str:
    """Build a natural-language phrase for a single component."""
    name = comp.get("name") or comp.get("smiles") or "UNKNOWN_COMPONENT"
    role = comp.get("role", "reactant")
    equiv = comp.get("equivalents")
    mw = comp.get("mw")
    quantity_description = comp.get("quantity_description")

    parts = []
    if quantity_description:
        parts.append(str(quantity_description))
    else:
        eq_str = _format_equivalents(equiv, role)
        if eq_str:
            parts.append(eq_str)
    parts.append(f"of {name}")
    if mw is not None:
        parts.append(f"(MW {mw:.2f} g/mol)")
    parts.append(f"as {role}")
    return " ".join(parts)


def prepare_agent_input_natural_language(reaction: dict) -> dict:
    """Convert a reaction record into natural-language prose for agent input.

    Returns the same dict schema as ``prepare_agent_input`` but with an
    additional ``description`` field containing a paragraph that encodes all
    component, yield, and product information in prose form.  The structured
    ``components`` list is **omitted** so the agent must parse the paragraph.

    If the record already has a ``description`` field (e.g. from noise
    injection Stage 4 format transform), it is preserved instead of
    regenerating.
    """
    product = reaction.get("product", {})
    product_mw = (product.get("mw") if isinstance(product, dict) else None) \
        or reaction.get("product_mw")
    yield_pct = reaction.get("yield_percent")

    # If noise injection already produced a description, keep it
    if reaction.get("description"):
        return {
            "reaction_id": reaction["reaction_id"],
            "reaction_name": reaction.get("reaction_name", ""),
            "reaction_smiles": "",
            "product_smiles": "",
            "product_mw": product_mw,
            "yield_percent": yield_pct,
            "description": reaction["description"],
        }

    # ── Build component list (same stripping as prepare_agent_input) ────────
    components = []
    for c in _agent_visible_components(reaction):
        name = c.get("name") or c.get("smiles") or "UNKNOWN_COMPONENT"
        component = {
            "name": name,
            "role": c.get("role", "reactant"),
            "equivalents": c.get("equivalents"),
            "mw": c.get("mw"),
        }
        if c.get("quantity_description"):
            component["quantity_description"] = c["quantity_description"]
        components.append(component)

    # ── Group components by role for readable prose ─────────────────────────
    role_order = ["reactant", "catalyst", "reagent", "base", "solvent"]
    by_role: dict[str, list[dict]] = {}
    for comp in components:
        r = comp.get("role", "reactant")
        by_role.setdefault(r, []).append(comp)

    # ── Assemble prose paragraphs ───────────────────────────────────────────
    reaction_name = reaction.get("reaction_name", "unnamed reaction")
    sentences: list[str] = []
    sentences.append(
        f"The following is a {reaction_name} reaction."
    )

    # Reactants first
    for role in role_order:
        comps = by_role.pop(role, [])
        if not comps:
            continue
        if len(comps) == 1:
            sentences.append(f"Use {_component_phrase(comps[0])}.")
        else:
            phrases = [_component_phrase(c) for c in comps]
            joined = ", and ".join(
                [", ".join(phrases[:-1]), phrases[-1]]
            ) if len(phrases) > 2 else " and ".join(phrases)
            sentences.append(f"Use {joined}.")

    # Any remaining roles not in role_order
    for role, comps in by_role.items():
        for c in comps:
            sentences.append(f"Use {_component_phrase(c)}.")

    # Product info
    product_parts = []
    if product_mw is not None:
        product_parts.append(f"molecular weight {product_mw:.2f} g/mol")
    if product_parts:
        sentences.append(f"The target product has {', '.join(product_parts)}.")

    # Yield
    if yield_pct is not None:
        sentences.append(f"The expected yield is {yield_pct}%.")

    description = " ".join(sentences)

    return {
        "reaction_id": reaction["reaction_id"],
        "reaction_name": reaction.get("reaction_name", ""),
        "reaction_smiles": "",
        "product_smiles": "",
        "product_mw": product_mw,
        "yield_percent": yield_pct,
        "description": description,
    }


def run_evaluation(
    agent: CostEstimationAgent,
    dataset_path: str | Path,
    output_path: str | Path | None = None,
    max_workers: int = 8,
    start: int = 0,
    end: int | None = None,
    input_format: Literal["structured", "natural_language"] = "structured",
    record_transform: Callable[[dict], dict] | None = None,
    save_trajectories: bool = False,
) -> BenchmarkResults:
    """Run an agent on the dataset and compute metrics.

    start/end slice the raw dataset lines (0-indexed, end exclusive) before
    filtering for evaluable reactions — useful for staged runs.

    input_format controls how reaction data is presented to the agent:
      - ``"structured"`` (default): tabular dict with a ``components`` list
      - ``"natural_language"``: prose paragraph; ``components`` replaced by
        a ``description`` string

    record_transform is an optional callable applied to each reaction record
    *before* it is passed to ``prepare_agent_input``.  This is used for noise
    injection — the transform modifies component names while preserving the
    ground-truth cost for evaluation.
    """
    _prepare = (
        prepare_agent_input_natural_language
        if input_format == "natural_language"
        else prepare_agent_input
    )
    dataset = load_dataset(dataset_path)
    if end is not None:
        dataset = dataset[start:end]
    elif start:
        dataset = dataset[start:]
    predictions: list[CostPrediction] = []
    n_skipped_missing_truth = 0

    evaluable = []
    for reaction in dataset:
        true_cost = _ground_truth_cost(reaction)
        if true_cost is None or true_cost <= 0:
            n_skipped_missing_truth += 1
        else:
            evaluable.append(reaction)

    trajectories_by_id: dict[str, list[dict]] = {}
    effective_workers = max_workers

    cache_path: Path | None = None
    results_map: dict[str, tuple] = {}
    if output_path:
        cache_path = Path(output_path).with_suffix(".partial.jsonl")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            eval_ids = {r["reaction_id"] for r in evaluable}
            with open(cache_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rid = rec["reaction_id"]
                    if rid not in eval_ids:
                        continue
                    reaction = next(r for r in evaluable if r["reaction_id"] == rid)
                    results_map[rid] = (reaction, rec["result"])
                    if rec.get("tool_calls"):
                        trajectories_by_id[rid] = rec["tool_calls"]
            logger.info(f"Resumed {len(results_map)} cached reactions from {cache_path}")

    cache_lock = threading.Lock()

    def _append_cache(rid: str, result: dict, tool_calls: list[dict]):
        if cache_path is None:
            return
        line = json.dumps({
            "reaction_id": rid,
            "result": result,
            "tool_calls": tool_calls,
        })
        with cache_lock, open(cache_path, "a") as fh:
            fh.write(line + "\n")

    def _run_one(reaction):
        transformed = record_transform(reaction) if record_transform else reaction
        agent_input = _prepare(transformed)

        reaction_tracker = None
        run_agent = agent
        # ReAct-style agent: clone + wrap TOOL_REGISTRY
        if save_trajectories and hasattr(agent, '_tools_override'):
            from copy import copy

            from ..tools.agent_tools import TOOL_REGISTRY as _base_tools
            run_agent = copy(agent)
            reaction_tracker = ToolTracker()
            run_agent._tools_override = reaction_tracker.wrap_tools(_base_tools)
        # SDK agent: per-instance tool_calls log, needs serial execution
        elif save_trajectories and hasattr(agent, '_last_tool_calls'):
            from copy import copy
            run_agent = copy(agent)
            run_agent._last_tool_calls = []

        try:
            result = run_agent.estimate_cost(agent_input)
        except Exception as e:
            logger.error(f"Agent failed on {reaction['reaction_id']}: {e}")
            result = {"predicted_cost_per_gram": None, "predicted_components": []}

        tool_calls = []
        if reaction_tracker is not None:
            for tc in reaction_tracker.get_calls():
                tool_calls.append({
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "success": tc.success,
                    "step": tc.step_number,
                })
        elif save_trajectories and hasattr(run_agent, 'tool_calls'):
            tool_calls = list(run_agent.tool_calls)

        return reaction, result, tool_calls

    pending = [r for r in evaluable if r["reaction_id"] not in results_map]
    if pending:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_run_one, rxn): rxn["reaction_id"] for rxn in pending}
            done = 0
            for future in as_completed(futures):
                reaction, result, tool_calls = future.result()
                rid = reaction["reaction_id"]
                results_map[rid] = (reaction, result)
                if tool_calls:
                    trajectories_by_id[rid] = tool_calls
                _append_cache(rid, result, tool_calls)
                done += 1
                if done % 10 == 0:
                    logger.info(f"  [{done}/{len(pending)}] reactions completed")

    # Preserve raw agent results for token tracking
    raw_results_by_id: dict[str, dict] = {}
    raw_metric_results: list[dict] = []
    for reaction in evaluable:
        reaction, result = results_map[reaction["reaction_id"]]
        raw_results_by_id[reaction["reaction_id"]] = result
        true_cost = _ground_truth_cost(reaction)
        raw_metric_results.append(
            {
                "reaction_id": reaction["reaction_id"],
                "predicted_cost_per_gram": result.get("predicted_cost_per_gram"),
                "true_cost": true_cost,
                "token_usage": result.get("token_usage"),
                "predicted_components": result.get("predicted_components", []),
                "min_prices": result.get("min_prices"),
            }
        )

        # Build component matches
        true_component_names = [
            c.get("name") or c.get("smiles") or "UNKNOWN_COMPONENT"
            for c in _agent_visible_components(reaction)
        ]
        pred_components = result.get("predicted_components", [])

        matches = []
        for pc in pred_components:
            pc_name = pc.get("name", "")
            found = any(
                component_names_equivalent(pc_name, true_name)
                for true_name in true_component_names
            )
            matches.append(
                ComponentMatch(
                    name=pc.get("name", ""),
                    found=found,
                )
            )

        predictions.append(
            CostPrediction(
                reaction_id=reaction["reaction_id"],
                predicted_cost=result.get("predicted_cost_per_gram"),
                true_cost=true_cost,
                predicted_components=matches,
                true_component_names=true_component_names,
                cost_tier=reaction.get("cost_tier", "unknown"),
            )
        )

    stratified = evaluate_stratified(predictions)
    results = stratified["all"]
    supplementary_metrics = {
        "token_efficiency": token_efficiency(raw_metric_results),
        "price_optimization_score": price_optimization_score(raw_metric_results),
    }

    # Save detailed results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "metrics": results.to_dict(),
                    "metrics_by_tier": {
                        tier: r.to_dict() for tier, r in stratified.items()
                    },
                    "supplementary_metrics": supplementary_metrics,
                    "n_input_records": len(dataset),
                    "n_skipped_missing_truth": n_skipped_missing_truth,
                    "n_evaluable_records": len(predictions),
                    "predictions": [
                        {
                            "reaction_id": p.reaction_id,
                            "predicted_cost": p.predicted_cost,
                            "true_cost": p.true_cost,
                            "cost_tier": p.cost_tier,
                            "tcre": abs(p.predicted_cost - p.true_cost) / p.true_cost
                            if p.predicted_cost is not None
                            and p.true_cost is not None
                            and p.true_cost > 0
                            else None,
                            "token_usage": raw_results_by_id.get(
                                p.reaction_id, {}
                            ).get("token_usage"),
                            "predicted_components": [
                                {
                                    "name": c.get("name", ""),
                                    "price_per_gram": c.get("price_per_gram"),
                                }
                                for c in raw_results_by_id.get(
                                    p.reaction_id, {}
                                ).get("predicted_components", [])
                            ],
                            **({"tool_calls": trajectories_by_id[p.reaction_id]}
                               if p.reaction_id in trajectories_by_id else {}),
                        }
                        for p in predictions
                    ],
                },
                f,
                indent=2,
            )

    return results
