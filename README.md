# ChemCost --- A Benchmark for Chemical Reaction Procurement Cost Estimation by LLM Agents

**Anonymous release for double-blind review at NeurIPS 2026 (Datasets and Benchmarks Track).**

ChemCost evaluates LLM agents on a chemistry-specific tool-use task: given a reaction description, recover the US-dollar cost of procuring enough reagent material to produce one gram of product. Ground truth is computed deterministically from a frozen snapshot of pack-level supplier quotes together with a fixed smallest-qualifying-pack rule. Evaluation is fully judge-free; every intermediate quantity along the agent's reasoning chain can be checked automatically against the frozen database and procurement rules. The benchmark contains 1,427 evaluable reactions (1,300 single-step and 127 multi-step routes), grounded to 230,755 pack-level supplier quotes covering 2,261 chemicals across 157 suppliers.

This repository contains the code and data needed to **run evaluations and reproduce the paper's tables and figures**. Dataset construction code (PDF parsing, ORD download, supplier scraping, pricing-DB build) is intentionally not included; the pricing snapshot ships pre-built so reviewers can run evaluations without any external data-collection credentials.

## Repository Contents

```
src/chemcost/             Core package
  baselines/                ReAct, ZeroShot, CoT, FewShot, FewShotReAct, AgentSDK agents
  evaluation/               Evaluator, metrics, tool tracker, ablation utilities
  noise/                    Four-stage noise-injection pipeline (name/quantity/missing/format)
  pricing/                  Read-only interface to the frozen pricing SQLite DB
  tools/                    Four agent tools: search_chemical, get_supplier_quotes,
                              compute_molar_mass, calculate
  cost_calculator.py        Deterministic ground-truth cost rule (smallest qualifying pack)

scripts/                  Evaluation entrypoint and analysis utilities
  run_evaluation.py         Single CLI to run any agent on any split with optional noise
  plot_*.py / analyze_*.py  Reproduce paper figures and tables from result JSONs
  score_human_eval.py       Score the human-reference annotations (provided separately)
  download_data.py          One-shot fetch of the dataset and pricing DB from Hugging Face

tests/                    Pytest unit tests for the cost calculator, metrics, noise
                            pipeline, prompting baselines, and tool evaluation.

pyproject.toml            Package metadata and dependency groups.
LICENSE                   MIT (code) and CC BY 4.0 (dataset content).
```

The dataset (1,427-reaction JSONL files and the 22 MB frozen supplier-quote
SQLite) is **not bundled in this repository**. After installation, run
`python3 scripts/download_data.py` to fetch the canonical artefact from the
anonymous Hugging Face release at
`https://huggingface.co/datasets/nips2026-chemcost/chemcost-bench`. The
download script writes to `data/processed/splits/` and `data/processed/`,
which is what the evaluation entrypoint expects.

## Installation

Python 3.10 or newer is required.

```bash
git clone <anonymous-repo-url>
cd chemcost-anon
pip install -e ".[eval,analysis]"
python3 scripts/download_data.py
```

Optional extras:

- `dev`: `pytest`, `ruff` for testing and linting.
- `eval`: clients used by the evaluation entrypoint (`anthropic`, `openai`, `claude-agent-sdk`, `httpx`, `pubchempy`, `rdkit`).

API credentials are read from environment variables at run time. Set the variables for the providers you intend to evaluate:

```bash
export ANTHROPIC_API_KEY=...    # for --provider anthropic
export OPENAI_API_KEY=...       # for --provider openai
export DASHSCOPE_API_KEY=...    # for --provider qwen
export DEEPSEEK_API_KEY=...     # for --provider deepseek
export KIMI_API_KEY=...         # for --provider kimi
export ZHIPU_API_KEY=...        # for --provider glm
```

The `sdk` agent uses Claude Code OAuth (no key required) and is included as a no-key reference path.

## Quick Start

Three example commands covering the main evaluation modes:

**1) Tool-augmented ReAct on the dev set (Sonnet 4.6):**

```bash
python3 scripts/run_evaluation.py \
  --dataset data/processed/splits/dev_eval_ready.jsonl \
  --agent react --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --output results/dev_react_sonnet46.json
```

**2) No-tool Chain-of-Thought baseline:**

```bash
python3 scripts/run_evaluation.py \
  --dataset data/processed/splits/dev_eval_ready.jsonl \
  --agent cot --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --output results/dev_cot_sonnet46.json
```

**3) ReAct under combined "rich" noise:**

```bash
python3 scripts/run_evaluation.py \
  --dataset data/processed/splits/dev_eval_ready.jsonl \
  --agent react --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --noise rich --noise-stages name,quantity,missing,format \
  --noise-seed 42 \
  --output results/dev_react_sonnet46_noise_rich_all.json
```

Output JSON shape (subset of fields):

```json
{
  "metrics": {
    "n_total": 81, "n_predicted": 70,
    "median_tcre": 0.21,
    "cta@10": 0.247, "cta@25": 0.346, "cta@50": 0.481,
    "component_precision": 0.602, "component_recall": 0.517
  },
  "predictions": [
    { "reaction_id": "CHEMCOST-0351",
      "predicted_cost": 17199.18, "true_cost": 17528.94,
      "tcre": 0.019,
      "predicted_components": [...],
      "token_usage": { "input_tokens": 26578, "output_tokens": 2129 } }
  ]
}
```

## Dataset

The dataset is hosted as a public Hugging Face release at

`https://huggingface.co/datasets/nips2026-chemcost/chemcost-bench`

published under the same anonymous account and accompanied by a Croissant 1.1 metadata file with the NeurIPS Responsible-AI fields. The release contains:

- `chemcost.jsonl` -- unified 1,427-reaction dataset (1,300 single-step + 127 multi-step routes).
- `splits/dev_eval_ready.jsonl` -- 90 reactions used as the "all set" in the main paper.
- `splits/test_eval_ready.jsonl` -- 1,298 reactions held out for the test split.
- `splits/dev_multistep.jsonl` -- 40 multi-step routes for reaction-type analysis.
- `pricing_db.sqlite` -- frozen supplier-quote database (~22 MB, 230,755 quotes across 2,261 chemicals and 157 suppliers).

After cloning this repository, run `python3 scripts/download_data.py` to pull all five files into `data/processed/`. The evaluator reads from those paths and never issues live network calls to suppliers.

## Evaluation Protocol

Each reaction in a split provides component names, component roles, stoichiometric equivalents, molecular weights, the reported yield, and the product identity. **Supplier prices and canonical molecular identifiers (SMILES, CAS) are withheld** from the agent input so that correct prediction requires explicit chemical resolution and multi-step procurement reasoning.

ReAct agents are equipped with four tools:

- `search_chemical(query)`: PubChem name lookup returning SMILES and molecular weight.
- `get_supplier_quotes(smiles_or_name)`: pack-level rows `(quantity_g, price_usd, purity)` from the frozen pricing database, filtered to purity ≥ 95%; the agent must select the right pack itself.
- `compute_molar_mass(smiles)`: RDKit-validated molecular weight.
- `calculate(expression)`: safe arithmetic.

Ground truth fixes the reaction scale at 1 mmol of the limiting reagent. For each non-solvent component `i` with equivalents `e_i` and molecular weight `M_i`, the required mass is `m_i = e_i × M_i × 0.001` grams. The component purchase cost `q_i` is the price of the smallest qualifying pack with quantity_g ≥ `m_i` and purity ≥ 95%; if `m_i` exceeds every pack, the cost scales by the number of largest-pack units required. If any required non-solvent component has no qualifying pack quote, the reaction is marked `unpriced` and receives no scalar procurement-cost label. Solvents are excluded. For priced reactions, the label is

```
c = (sum of q_i over non-solvent components)
    / (product_MW * 0.001 * yield_percent / 100)
```

The headline metric is **CTA@k** (cost tolerance accuracy at relative error `k/100`): the fraction of reactions whose predicted procurement cost falls within `k%` of the deterministic ground truth, with abstentions counted as failures. We report CTA@10 and CTA@25 in the Clean setting and CTA@10 across noise settings, alongside component precision and component recall computed via an alias-aware string match against the ground-truth component set.

## Reproducing Paper Tables

The main result table is produced by running each agent on the dev split under one Clean view and five perturbed views:

```bash
# Clean
python3 scripts/run_evaluation.py --dataset data/processed/splits/dev_eval_ready.jsonl \
  --agent react --provider <prov> --model <model> \
  --output results/dev_react_<model>.json

# Each individual noise stage
for stage in name quantity missing format; do
  python3 scripts/run_evaluation.py --dataset data/processed/splits/dev_eval_ready.jsonl \
    --agent react --provider <prov> --model <model> \
    --noise medium --noise-stages $stage --noise-seed 42 \
    --output results/dev_react_<model>_noise_${stage}.json
done

# Joint application (All Noise)
python3 scripts/run_evaluation.py --dataset data/processed/splits/dev_eval_ready.jsonl \
  --agent react --provider <prov> --model <model> \
  --noise medium --noise-stages name,quantity,missing,format --noise-seed 42 \
  --output results/dev_react_<model>_noise_all.json
```

After all twelve runs complete, the figure-generation scripts under `scripts/plot_*.py` ingest the result JSONs and emit the tool-usage, abstention, reaction-type, and stage-pipeline figures used in the paper. The tabular columns in the main results table are read directly from each result JSON's `metrics` block (component precision, component recall, CTA@10, CTA@25).

For the no-tool baselines (Chain-of-Thought, FewShot, ZeroShot), substitute `--agent cot`, `--agent few-shot`, or `--agent zero-shot`. For the AgentSDK reference path that uses Claude Code OAuth, use `--agent sdk` and omit `--provider`.

Multi-step results in the reaction-type analysis are produced by passing `data/processed/splits/dev_multistep.jsonl` as the dataset.

## Anonymity Notice

This repository is an anonymous release prepared for the NeurIPS 2026 Datasets and Benchmarks Track double-blind review. Author names, institutions, GitHub handles, internal documentation, build-pipeline scripts that reference proprietary credentials, and dataset-construction utilities have been intentionally removed from this snapshot. Reviewers who notice any remaining identifying string are kindly asked to contact program chairs rather than circulating it.

## License

Code is released under the MIT License; dataset content (the JSONL splits and the pricing database) is released under CC BY 4.0. Full text of both licenses is in `LICENSE`.

For citation while under review, please use:

```bibtex
@unpublished{chemcost2026anon,
  title  = {ChemCost: A Benchmark for Chemical Reaction Procurement Cost Estimation by LLM Agents},
  author = {Anonymous},
  year   = {2026},
  note   = {Under double-blind review at NeurIPS 2026 Datasets and Benchmarks Track}
}
```
