#!/usr/bin/env python3
"""Stage-level error decomposition of ReAct predictions.

Decomposes each agent's failure into 4 distinct pipeline stages:

  Stage 1 (Grounding):    truth chemical canonically matched in
                           predicted_components.
  Stage 2 (Retrieval):    agent issued a get_supplier_quotes call whose
                           argument resolves to the same DB chemical_id
                           as the truth.
  Stage 3 (Pack selection): agent's reported price_per_gram is within 5%
                           of the oracle pack's $/g (smallest pack with
                           quantity_g >= required_mass_g, purity ≥ 95%).
  Stage 5 (Aggregation):  given the agent's predicted_components and
                           prices, redoing the procurement arithmetic with
                           truth equiv/MW/yield reproduces the agent's
                           reported predicted_cost_per_gram (within 10%).

Stage 4 (mass conversion) is implicitly checked via Stage 3 (selecting the
right pack requires correct required_mass_g) — we do not separate it out.

Output: results/stage_analysis_{model}.json with per-reaction, per-component
stage scores; aggregate stats printed to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemcost.pricing.pricing_db import PricingDB
from chemcost.evaluation.metrics import component_names_equivalent

DB = PricingDB(str(ROOT / "data/processed/pricing_db.sqlite"))

MODELS = [
    ("DS V4 Pro",      "results/dev_react_deepseek_v4_pro_clean.json",
                       "results/dev_multi40_deepseek_v4_pro_clean.json"),
    ("Qwen3.5-Plus",   "results/dev_react_qwen35plus_clean.json",
                       "results/dev_multi_qwen35plus_clean.json"),
    ("Sonnet 4.6",     "results/dev_react_sonnet46_traj.json",
                       "results/dev_multi40_sonnet46_clean.json"),
    ("Kimi K2.5",      "results/dev_react_kimi_k25_clean.json",
                       "results/dev_multi40_kimi_k25_clean.json"),
    ("GPT-5",          "results/dev_react_gpt5_traj.json",
                       "results/dev_multi40_gpt5_clean.json"),
]

PACK_TOLERANCE = 0.05    # 5% relative error on pack $/g
AGG_TOLERANCE  = 0.10    # 10% relative error on aggregation arithmetic
SCALE_MMOL = 1.0


def chem_id(smiles: str | None = None, name: str | None = None) -> int | None:
    """Lookup chemical_id by SMILES or name, returning None if not found."""
    if smiles:
        cid = DB._lookup_chemical_id(smiles=smiles)
        if cid is not None:
            return cid
    if name:
        cid = DB._lookup_chemical_id(name=name)
        if cid is not None:
            return cid
    return None


def oracle_pack(truth_smiles: str | None, truth_name: str | None,
                required_mass_g: float) -> dict | None:
    """Return the smallest pack covering required_mass_g (≥95% purity)."""
    quotes = DB.get_pack_quotes(smiles=truth_smiles, name=truth_name,
                                  min_purity=95.0)
    if not quotes:
        return None
    # Smallest pack with quantity_g >= required_mass_g
    for q in quotes:
        if q["quantity_g"] >= required_mass_g:
            return q
    # If none covers, return the largest (we'd buy multiple of these)
    return quotes[-1]


def is_non_solvent(c: dict) -> bool:
    return (c.get("role") or "").lower() not in {"solvent", "product"}


def find_predicted(truth_name: str, predicted: list[dict]) -> dict | None:
    """First predicted component whose name matches the truth (alias-aware)."""
    for pc in predicted or []:
        pname = (pc.get("name") or "").strip()
        if pname and component_names_equivalent(truth_name, pname):
            return pc
    return None


def get_quotes_arg(tc: dict) -> str | None:
    args = tc.get("arguments") or {}
    return args.get("smiles_or_name") or args.get("smiles") or args.get("name")


def load_records(path: str) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rid = r.get("reaction_id")
            if rid:
                out[rid] = r
    return out


def analyze_one(rec: dict, pred: dict) -> dict:
    """Stage-level breakdown for a single reaction-prediction pair."""
    rid = rec["reaction_id"]
    yield_pct = rec.get("yield_percent") or 0
    product_mw = rec.get("product_mw") or 0

    truth_components = [c for c in (rec.get("components") or [])
                        if is_non_solvent(c) and (c.get("name") or "").strip()]
    predicted = pred.get("predicted_components") or []
    tool_calls = pred.get("tool_calls") or []

    # Pre-resolve agent's get_supplier_quotes args to chemical_ids
    retrieved_chem_ids: set[int] = set()
    for tc in tool_calls:
        if tc.get("tool_name") != "get_supplier_quotes":
            continue
        arg = get_quotes_arg(tc)
        if not arg:
            continue
        cid = chem_id(smiles=arg, name=arg)
        if cid is not None:
            retrieved_chem_ids.add(cid)

    component_results = []
    for tc in truth_components:
        truth_smi = (tc.get("smiles") or "").strip() or None
        truth_name = (tc.get("name") or "").strip()
        truth_equiv = tc.get("equivalents")
        truth_mw = tc.get("mw")

        # Stage 1: Grounding
        pc = find_predicted(truth_name, predicted)
        grounded = pc is not None

        # Stage 2: Retrieval (only meaningful if grounded)
        truth_cid = chem_id(smiles=truth_smi, name=truth_name)
        retrieved = (truth_cid is not None) and (truth_cid in retrieved_chem_ids)

        # Stage 3: Pack selection
        pack_correct = None  # None = N/A (not grounded or no agent price)
        if grounded and pc is not None:
            agent_price = pc.get("price_per_gram")
            if agent_price is not None and truth_equiv is not None and truth_mw is not None:
                req_mass_g = float(truth_equiv) * float(truth_mw) * 0.001 * SCALE_MMOL
                op = oracle_pack(truth_smi, truth_name, req_mass_g)
                if op is not None and op["quantity_g"] > 0:
                    oracle_ppg = op["price_usd"] / op["quantity_g"]
                    rel_err = abs(agent_price - oracle_ppg) / max(oracle_ppg, 1e-9)
                    pack_correct = bool(rel_err <= PACK_TOLERANCE)

        component_results.append({
            "name": truth_name,
            "grounded": grounded,
            "retrieved": retrieved,
            "pack_correct": pack_correct,
        })

    # Stage 5: Aggregation arithmetic — given agent's components × prices,
    # would the truth-mass / truth-yield arithmetic reproduce predicted_cost?
    pred_cost = pred.get("predicted_cost")
    agg_check = None
    expected_cost = None
    if pred_cost is not None and product_mw and yield_pct:
        total_purchase = 0.0
        n_priced = 0
        for tc in truth_components:
            truth_name = (tc.get("name") or "").strip()
            equiv = tc.get("equivalents")
            mw = tc.get("mw")
            if equiv is None or mw is None:
                continue
            pc = find_predicted(truth_name, predicted)
            if pc is None or pc.get("price_per_gram") is None:
                continue
            req_mass_g = float(equiv) * float(mw) * 0.001 * SCALE_MMOL
            total_purchase += float(pc["price_per_gram"]) * req_mass_g
            n_priced += 1
        if n_priced > 0:
            grams_product = float(product_mw) * 0.001 * SCALE_MMOL * (float(yield_pct) / 100.0)
            if grams_product > 0:
                expected_cost = total_purchase / grams_product
                rel_err = abs(pred_cost - expected_cost) / max(expected_cost, 1e-9)
                agg_check = bool(rel_err <= AGG_TOLERANCE)

    return {
        "reaction_id": rid,
        "n_truth_components": len(truth_components),
        "components": component_results,
        "agg_check": agg_check,
        "predicted_cost": pred_cost,
        "expected_cost_from_agent_prices": expected_cost,
    }


def aggregate(model_name: str, per_reaction: list[dict]) -> dict:
    """Aggregate per-reaction stage scores into per-stage hit rates."""
    n_truth = n_grounded = n_retrieved = n_pack = n_pack_eligible = 0
    n_agg_checked = n_agg_pass = 0
    for r in per_reaction:
        for c in r["components"]:
            n_truth += 1
            if c["grounded"]:
                n_grounded += 1
                if c["retrieved"]:
                    n_retrieved += 1
                    if c["pack_correct"] is not None:
                        n_pack_eligible += 1
                        if c["pack_correct"]:
                            n_pack += 1
        if r["agg_check"] is not None:
            n_agg_checked += 1
            if r["agg_check"]:
                n_agg_pass += 1
    return {
        "model": model_name,
        "n_components": n_truth,
        "stage1_grounded_rate":  n_grounded / n_truth if n_truth else 0.0,
        "stage2_retrieved_rate": n_retrieved / max(n_grounded, 1),  # given grounded
        "stage3_pack_rate":      n_pack / max(n_pack_eligible, 1), # given retrieved+priced
        "stage5_agg_rate":       n_agg_pass / max(n_agg_checked, 1),
        "n_grounded":  n_grounded,
        "n_retrieved": n_retrieved,
        "n_pack_eligible": n_pack_eligible,
        "n_pack_correct": n_pack,
        "n_agg_checked": n_agg_checked,
        "n_agg_pass": n_agg_pass,
    }


def main():
    # Load truth records (single + multi)
    truth = {}
    for path in [ROOT / "data/processed/splits/dev.jsonl",
                 ROOT / "data/processed/splits/dev_multistep_v2.jsonl"]:
        truth.update(load_records(str(path)))
    print(f"Loaded {len(truth)} truth records")

    summary = []
    for model_name, single_path, multi_path in MODELS:
        all_preds: list[dict] = []
        for p in [single_path, multi_path]:
            full = ROOT / p
            if full.exists():
                d = json.load(open(full))
                all_preds.extend(d.get("predictions") or [])
        if not all_preds:
            print(f"[{model_name}] no predictions, skipping")
            continue

        per_rxn = []
        for pred in all_preds:
            rid = pred.get("reaction_id")
            rec = truth.get(rid)
            if rec is None:
                continue
            per_rxn.append(analyze_one(rec, pred))

        agg = aggregate(model_name, per_rxn)
        summary.append(agg)

        # Save per-reaction breakdown
        out_path = ROOT / f"results/stage_analysis_{model_name.lower().replace(' ','_').replace('.','')}.json"
        with open(out_path, "w") as f:
            json.dump({"model": model_name, "aggregate": agg,
                       "per_reaction": per_rxn}, f, indent=2)
        print(f"[{model_name}] saved {out_path}")

    print()
    print(f"{'Model':<14} {'n_comp':>7} {'S1':>7} {'S2|S1':>7} {'S3|S2':>7} {'S5':>7}")
    print("-" * 56)
    for a in summary:
        print(f"{a['model']:<14} "
              f"{a['n_components']:>7} "
              f"{100*a['stage1_grounded_rate']:>6.1f}% "
              f"{100*a['stage2_retrieved_rate']:>6.1f}% "
              f"{100*a['stage3_pack_rate']:>6.1f}% "
              f"{100*a['stage5_agg_rate']:>6.1f}%")

    # Save summary
    with open(ROOT / "results/stage_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: results/stage_analysis_summary.json")


if __name__ == "__main__":
    main()
