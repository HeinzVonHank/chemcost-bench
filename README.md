# ChemCost --- A Benchmark for Chemical Reaction Procurement Cost Estimation by LLM Agents

**Anonymous release for double-blind review at NeurIPS 2026 (Datasets and Benchmarks Track).**

ChemCost evaluates LLM agents on a chemistry-specific tool-use task: given a reaction description, recover the US-dollar cost of procuring enough reagent material to produce one gram of product. Ground truth is computed deterministically from a frozen snapshot of pack-level supplier quotes together with a fixed smallest-qualifying-pack rule. Evaluation is fully judge-free; every intermediate quantity along the agent's reasoning chain can be checked automatically against the frozen database and procurement rules. The benchmark contains 1,427 evaluable reactions (1,300 single-step and 127 multi-step routes), grounded to 230,755 pack-level supplier quotes covering 2,261 chemicals across 157 suppliers.

This repository provides the **core evaluation package** so reviewers can inspect the agent baselines, the deterministic cost rule, the tool definitions, the noise pipeline, and the metric implementations referenced in the paper. The dataset and the frozen supplier-quote database are published separately on Hugging Face under the same anonymous account.

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

tests/                    Pytest unit tests for the cost calculator, metrics,
                            noise pipeline, prompting baselines, and tool evaluation.

pyproject.toml            Package metadata and dependency groups.
LICENSE                   MIT (code) and CC BY 4.0 (dataset content).
```

The dataset (1,427-reaction JSONL files and the 22 MB frozen supplier-quote SQLite) lives on Hugging Face at

`https://huggingface.co/datasets/nips2026-chemcost/chemcost-bench`

published under the same anonymous account and accompanied by a Croissant 1.1 metadata file with the NeurIPS Responsible-AI fields.

## Installation

Python 3.10 or newer is required.

```bash
pip install -e ".[eval,analysis]"
```

Optional extras:

- `eval`: client libraries used by the agent baselines (`openai`, `anthropic`, `claude-agent-sdk`, `httpx`, `rdkit`).
- `analysis`: numerics and plotting (`numpy`, `matplotlib`, `scipy`).
- `dev`: `pytest` and `ruff`.

API credentials are read from environment variables at run time. Set the variables for the providers you intend to evaluate:

```bash
export ANTHROPIC_API_KEY=...    # Claude family via the native Anthropic API
export OPENAI_API_KEY=...       # GPT family via the native OpenAI API
export DASHSCOPE_API_KEY=...    # Qwen family
export DEEPSEEK_API_KEY=...     # DeepSeek family
export KIMI_API_KEY=...         # Kimi K2.x
export ZHIPU_API_KEY=...        # GLM-4.x
```

The `sdk` agent uses Claude Code OAuth (no key required) and is included as a no-key reference path.

## Programmatic Use

The package exposes a single evaluator function. Given a list of reactions and an agent instance, it returns a metrics dict and per-reaction predictions. Minimal example:

```python
from chemcost.baselines.react_agent import ReActAgent
from chemcost.evaluation.evaluator import run_evaluation

# Reactions are dicts loaded from the released JSONL files; see the dataset card
# on Hugging Face for the schema.
reactions = [...]   # e.g. json.loads(line) for line in open("dev_eval_ready.jsonl")

agent = ReActAgent(
    model="claude-sonnet-4-20250514",
    provider="anthropic",
    max_steps=40,
)
result = run_evaluation(agent, reactions)
print(result["metrics"])     # cta@10, cta@25, component_recall, ...
```

To use the no-tool baselines, swap `ReActAgent` for `CoTBaseline`, `FewShotBaseline`, or `ZeroShotBaseline` from `chemcost.baselines`. To run under noise, transform the reactions with `chemcost.noise.noise_injector.inject_noise(reaction, level="medium", stages=["name", "quantity"], seed=42)` before passing them to `run_evaluation`.

The deterministic ground-truth cost is computed by `chemcost.cost_calculator.calculate_procurement_cost`; the read-only pricing interface is `chemcost.pricing.pricing_db.PricingDB`.

## Evaluation Protocol

Each reaction provides component names, component roles, stoichiometric equivalents, molecular weights, the reported yield, and the product identity. **Supplier prices and canonical molecular identifiers (SMILES, CAS) are withheld** from the agent input, so correct prediction requires explicit chemical resolution and multi-step procurement reasoning.

ReAct agents are equipped with four tools:

- `search_chemical(query)`: PubChem name lookup returning SMILES and molecular weight.
- `get_supplier_quotes(smiles_or_name)`: pack-level rows `(quantity_g, price_usd, purity)` from the frozen pricing database, filtered to purity >= 95%; the agent must select the right pack itself.
- `compute_molar_mass(smiles)`: RDKit-validated molecular weight.
- `calculate(expression)`: safe arithmetic.

Ground truth fixes the reaction scale at 1 mmol of the limiting reagent. For each non-solvent component `i` with equivalents `e_i` and molecular weight `M_i`, the required mass is `m_i = e_i * M_i * 0.001` grams. The component purchase cost `q_i` is the price of the smallest qualifying pack with `quantity_g >= m_i` and `purity >= 95%`; if `m_i` exceeds every pack, the cost scales by the number of largest-pack units required. Solvents are excluded. The label is

```
c = sum(q_i for non-solvent i)
    / (product_MW * 0.001 * yield_percent / 100)
```

The headline metric is **CTA@k** (cost tolerance accuracy at relative error `k/100`): the fraction of reactions whose predicted procurement cost falls within `k%` of the deterministic ground truth, with abstentions counted as failures. We report CTA@10 and CTA@25 in the Clean setting and CTA@10 across noise settings, alongside component precision and component recall computed via an alias-aware string match against the ground-truth component set.

## Anonymity Notice

This repository is an anonymous release prepared for the NeurIPS 2026 Datasets and Benchmarks Track double-blind review. Author names, institutions, GitHub handles, internal documentation, build-pipeline scripts that reference proprietary credentials, and dataset-construction utilities have been intentionally removed from this snapshot.

## License

Code is released under the MIT License; dataset content (the JSONL splits and the pricing database hosted on Hugging Face) is released under CC BY 4.0. Full text of both licenses is in `LICENSE`.

For citation while under review, please use:

```bibtex
@unpublished{chemcost2026anon,
  title  = {ChemCost: A Benchmark for Chemical Reaction Procurement Cost Estimation by LLM Agents},
  author = {Anonymous},
  year   = {2026},
  note   = {Under double-blind review at NeurIPS 2026 Datasets and Benchmarks Track}
}
```
