"""Ablation experiment runner for ChemCost benchmark.

Allows systematically disabling subsets of tools to measure
each tool's contribution to agent performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .evaluator import _agent_visible_components, _ground_truth_cost, prepare_agent_input
from .metrics import (
    BenchmarkResults,
    ComponentMatch,
    CostPrediction,
    component_names_equivalent,
    evaluate,
)
from .tool_tracker import ToolTracker, ToolUsageStats

logger = logging.getLogger(__name__)


def _visible_component_name(component: dict) -> str:
    return component.get("name") or component.get("smiles") or "UNKNOWN_COMPONENT"


def _find_true_component(name: str, components: list[dict]) -> dict | None:
    for component in components:
        if component_names_equivalent(name, _visible_component_name(component)):
            return component
    return None


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    disabled_tools: list[str]
    description: str


# Pre-defined ablation configurations covering common experiments.
ABLATIONS: list[AblationConfig] = [
    AblationConfig("full", [], "All tools available"),
    AblationConfig("no_search", ["search_chemical"], "No chemical search"),
    AblationConfig("no_price", ["get_supplier_quotes"], "No price lookup"),
    AblationConfig("no_mw", ["compute_molar_mass"], "No MW computation"),
    AblationConfig("no_calc", ["calculate"], "No calculator"),
    AblationConfig(
        "search_only",
        ["get_supplier_quotes", "compute_molar_mass", "calculate"],
        "Only search",
    ),
    AblationConfig(
        "no_tools",
        ["search_chemical", "get_supplier_quotes", "compute_molar_mass", "calculate"],
        "Zero-shot",
    ),
]


@dataclass
class AblationRun:
    """Results from a single ablation configuration."""

    config: AblationConfig
    metrics: BenchmarkResults
    tool_stats: ToolUsageStats
    predictions: list[CostPrediction] = field(default_factory=list)


@dataclass
class AblationResults:
    """Collected results from all ablation runs."""

    runs: list[AblationRun] = field(default_factory=list)

    def summary_table(self) -> list[dict]:
        """Return a list of dicts suitable for tabular display."""
        rows = []
        for run in self.runs:
            row = {
                "ablation": run.config.name,
                "description": run.config.description,
                "disabled_tools": run.config.disabled_tools,
                **run.metrics.to_dict(),
                "total_tool_calls": run.tool_stats.total_calls,
                "redundant_calls": run.tool_stats.redundant_calls,
            }
            rows.append(row)
        return rows


def apply_ablation(tools: dict, config: AblationConfig) -> dict:
    """Remove disabled tools from a tool registry.

    Args:
        tools: Full tool registry dict.
        config: Ablation config specifying which tools to disable.

    Returns:
        Filtered copy of the tool registry.
    """
    return {
        name: info
        for name, info in tools.items()
        if name not in config.disabled_tools
    }


def run_ablation(
    agent_factory,
    dataset: list[dict],
    ablations: list[AblationConfig] | None = None,
    tools: dict | None = None,
) -> AblationResults:
    """Run an agent under each ablation config and collect metrics.

    Args:
        agent_factory: Callable ``(tools_dict) -> agent`` that creates an
            agent configured to use the given tools.  The returned agent
            must have an ``estimate_cost(reaction: dict) -> dict`` method.
        dataset: List of reaction dicts (with ground-truth fields).
        ablations: Ablation configs to run. Defaults to :data:`ABLATIONS`.
        tools: Base tool registry. Defaults to ``TOOL_REGISTRY`` from
            ``chemcost.tools.agent_tools``.

    Returns:
        :class:`AblationResults` with one :class:`AblationRun` per config.
    """
    if ablations is None:
        ablations = ABLATIONS

    if tools is None:
        from ..tools.agent_tools import TOOL_REGISTRY
        tools = TOOL_REGISTRY

    results = AblationResults()

    for config in ablations:
        logger.info("Running ablation: %s (%s)", config.name, config.description)

        # Apply ablation: remove disabled tools
        ablated_tools = apply_ablation(tools, config)

        # Set up tracking
        tracker = ToolTracker()
        tracked_tools = tracker.wrap_tools(ablated_tools)

        # Create agent with the ablated+tracked tools
        agent = agent_factory(tracked_tools)

        predictions: list[CostPrediction] = []

        for reaction in dataset:
            tracker.mark_reaction_boundary()

            # Strip ground-truth for agent input (consistent with evaluator)
            agent_input = prepare_agent_input(reaction)

            try:
                result = agent.estimate_cost(agent_input)
            except Exception as e:
                logger.error(
                    "Agent failed on %s (ablation=%s): %s",
                    reaction["reaction_id"],
                    config.name,
                    e,
                )
                result = {"predicted_cost_per_gram": None, "predicted_components": []}

            true_components = _agent_visible_components(reaction)
            pred_components = result.get("predicted_components", [])

            matches = []
            for pc in pred_components:
                pc_name = pc.get("name", "")
                true_component = _find_true_component(pc_name, true_components)
                found = true_component is not None
                matches.append(
                    ComponentMatch(
                        name=pc_name,
                        found=found,
                        price_pred=pc.get("price_per_gram"),
                        price_true=(
                            true_component.get("price_per_gram_usd")
                            if true_component is not None
                            else None
                        ),
                        mw_pred=pc.get("mw"),
                        mw_true=(
                            true_component.get("mw")
                            if true_component is not None
                            else None
                        ),
                    )
                )

            predictions.append(
                CostPrediction(
                    reaction_id=reaction["reaction_id"],
                    predicted_cost=result.get("predicted_cost_per_gram"),
                    true_cost=_ground_truth_cost(reaction),
                    predicted_components=matches,
                    true_component_names=[
                        _visible_component_name(c) for c in true_components
                    ],
                )
            )

        metrics = evaluate(predictions)
        tool_stats = tracker.get_stats()

        results.runs.append(
            AblationRun(
                config=config,
                metrics=metrics,
                tool_stats=tool_stats,
                predictions=predictions,
            )
        )

    return results
