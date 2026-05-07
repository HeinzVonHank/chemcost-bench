#!/usr/bin/env python3
"""Run baseline agents on a dataset split and compute metrics."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Allow claude -p to run inside a Claude Code session
os.environ.pop("CLAUDECODE", None)

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from chemcost.baselines.prompting_baselines import (
    CoTBaseline,
    FewShotBaseline,
    FewShotReActBaseline,
)
from chemcost.baselines.react_agent import AgentSDKBaseline, ReActAgent, ZeroShotBaseline
from chemcost.evaluation.evaluator import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run ChemCost evaluation")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument(
        "--agent",
        choices=["react", "zero-shot", "cot", "few-shot", "few-shot-react", "sdk"],
        default="react",
        help="Agent type (sdk = claude-agent-sdk, no API key required)",
    )
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="LLM model name")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "qwen", "deepseek", "kimi", "glm", "openrouter"],
        default="anthropic",
    )
    parser.add_argument("--output", default=None, help="Output JSON path for detailed results")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based) into dataset lines")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) into dataset lines")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4 for SDK)")
    parser.add_argument(
        "--input-format",
        choices=["structured", "natural_language"],
        default="structured",
        help="Input format for agent: structured (default) or natural_language (prose)",
    )
    parser.add_argument(
        "--noise",
        choices=["none", "low", "medium", "high", "rich"],
        default="none",
        help="Noise level controlling per-stage intensity (default: none)",
    )
    parser.add_argument(
        "--noise-stages",
        default=None,
        help="Comma-separated stages to enable: name,quantity,missing,format. "
             "If omitted, stages are selected by --noise level.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="Random seed for noise injection (default: 42)",
    )
    parser.add_argument(
        "--format-kind",
        choices=["nl_only", "ocr_only", "nl_plus_ocr"],
        default=None,
        help="Override the +Fmt sub-type. Only meaningful when --noise-stages includes 'format'.",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Save per-reaction tool-call trajectories in output JSON",
    )
    args = parser.parse_args()

    if args.agent == "react":
        agent = ReActAgent(model=args.model, provider=args.provider)
    elif args.agent == "cot":
        agent = CoTBaseline(model=args.model, provider=args.provider)
    elif args.agent == "few-shot":
        agent = FewShotBaseline(model=args.model, provider=args.provider)
    elif args.agent == "few-shot-react":
        agent = FewShotReActBaseline(model=args.model, provider=args.provider)
    elif args.agent == "sdk":
        agent = AgentSDKBaseline(model=args.model)
    else:
        agent = ZeroShotBaseline(model=args.model, provider=args.provider)

    if args.output:
        output_path = args.output
    else:
        model_tag = args.model.split("/")[-1]
        stage = f"_s{args.start}-{args.end}" if (args.start or args.end) else ""
        output_path = f"results/test_{args.agent}_{model_tag}{stage}.json"

    suffix = f" [lines {args.start}:{args.end}]" if (args.start or args.end) else ""
    fmt_tag = f" [{args.input_format}]" if args.input_format != "structured" else ""
    noise_tag = ""
    if args.noise_stages:
        noise_tag = f" [noise-stages={args.noise_stages}]"
    elif args.noise != "none":
        noise_tag = f" [noise={args.noise}]"
    print(
        f"Running {args.agent} agent ({args.model}) on "
        f"{args.dataset}{suffix}{fmt_tag}{noise_tag}..."
    )

    # Apply noise injection if requested
    noise_fn = None
    if args.noise != "none" or args.noise_stages:
        from chemcost.noise import default_noise_types_for_level, inject_noise

        noise_level = args.noise if args.noise != "none" else "medium"
        noise_seed = args.noise_seed

        # Stage mapping: short names -> internal noise type names
        stage_name_map = {
            "name": ["isomer", "name_variation"],
            "quantity": ["quantity"],
            "missing": ["missing_info"],
            "format": ["format"],
        }

        if args.noise_stages:
            # Explicit stage selection (controlled single-factor experiment)
            noise_types = []
            for s in args.noise_stages.split(","):
                s = s.strip()
                if s in stage_name_map:
                    noise_types.extend(stage_name_map[s])
                else:
                    noise_types.append(s)
        else:
            noise_types = default_noise_types_for_level(noise_level)

        format_kind_arg = args.format_kind

        def noise_fn(record):
            reaction_id = record.get("reaction_id", "")
            seed_material = f"{noise_seed}:{reaction_id}".encode("utf-8")
            record_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
            return inject_noise(
                record, noise_types=noise_types,
                noise_level=noise_level, seed=record_seed,
                format_kind=format_kind_arg,
            )

    # Enable trajectory saving: set tools override on agent so tracker can wrap them
    if args.save_trajectories and hasattr(agent, '_tools_override'):
        agent._tools_override = None  # Will be set per-reaction by evaluator

    results = run_evaluation(
        agent,
        args.dataset,
        output_path,
        max_workers=args.workers,
        start=args.start,
        end=args.end,
        input_format=args.input_format,
        record_transform=noise_fn,
        save_trajectories=args.save_trajectories,
    )

    print("\n=== Results ===")
    for key, value in results.to_dict().items():
        print(f"  {key}: {value}")
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
