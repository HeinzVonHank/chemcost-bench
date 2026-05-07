#!/usr/bin/env python3
"""Post-hoc slice noise-experiment results by noise sub-type.

Re-injects noise on the dev split using the same parameters as the
original eval run, builds a (reaction_id, component_name) -> sub_type
map, then aggregates Recall / CTA@10 / CTA@25 / Abstention per sub-type.

Example
-------
    python3 scripts/analyze_noise_subtypes.py \\
        --predictions results/dev_react_deepseek_v4_pro_noise_quantity.json \\
        --stage quantity \\
        --level medium \\
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemcost.evaluation.metrics import component_names_equivalent  # noqa: E402
from chemcost.noise import inject_noise  # noqa: E402

# Stage short-name -> noise_types passed to inject_noise.  Mirrors the
# mapping in scripts/run_evaluation.py so post-hoc analysis matches what
# the agent originally saw.
STAGE_TO_TYPES: dict[str, list[str]] = {
    "name": ["isomer", "name_variation"],
    "quantity": ["quantity"],
    "missing": ["missing_info"],
    "format": ["format"],
}

# Sub-type field names per stage (component-level).  Format is special-
# cased: format_kind lives at the record level.
SUBTYPE_FIELD_BY_STAGE: dict[str, str] = {
    "name": "noise_type",
    "quantity": "quantity_noise_kind",
    "missing": "noise_type",
    "format": "noise_type",  # for OCR sub-tagging
}


def _record_seed(noise_seed: int, reaction_id: str) -> int:
    """Replicate run_evaluation.py per-record seed derivation."""
    seed_material = f"{noise_seed}:{reaction_id}".encode("utf-8")
    return int(hashlib.sha256(seed_material).hexdigest()[:8], 16)


def _load_predictions(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    preds = data.get("predictions") or []
    if not preds:
        raise SystemExit(f"No predictions in {path}")
    return preds


def _load_split(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Split file not found: {path}")
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _is_non_solvent(comp: dict) -> bool:
    role = (comp.get("role") or comp.get("original_role") or "").lower()
    return role not in {"solvent", "product"}


def _component_predicted(predicted_components: list[dict], true_name: str) -> bool:
    if not true_name:
        return False
    for m in predicted_components or []:
        pname = m.get("name") or ""
        if not pname:
            continue
        if component_names_equivalent(true_name, pname):
            return True
    return False


def _format_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100.0 * num / denom:.0f}%"


def _build_subtype_map(
    split_records: list[dict],
    stage: str,
    level: str,
    noise_seed: int,
) -> tuple[dict[tuple[str, str], set[str]], dict[str, set[str]]]:
    """Re-inject noise; return per-component subtype map and per-record format_kind.

    Returns
    -------
    comp_subtype : {(reaction_id, normalized_true_name): {sub_type, ...}}
        A component may carry multiple sub-types when stages compose
        (e.g. name_variation + ocr).  We track all of them.
    record_format_kind : {sub_type: {reaction_id, ...}}
        Reactions touched by a record-level format_kind (only used for
        the format stage).
    """
    noise_types = STAGE_TO_TYPES.get(stage)
    if noise_types is None:
        raise SystemExit(
            f"Unknown stage {stage!r}; choose from {list(STAGE_TO_TYPES)}"
        )

    comp_subtype: dict[tuple[str, str], set[str]] = defaultdict(set)
    record_format_kind: dict[str, set[str]] = defaultdict(set)

    for rec in split_records:
        rid = rec.get("reaction_id", "")
        if not rid:
            continue
        rec_seed = _record_seed(noise_seed, rid)
        try:
            noisy = inject_noise(
                rec, noise_types=noise_types,
                noise_level=level, seed=rec_seed,
            )
        except Exception as exc:  # defensive: don't let one bad record kill all
            print(f"warning: inject_noise failed for {rid}: {exc}", file=sys.stderr)
            continue

        # Record-level format_kind (added by parallel agent; may not exist yet)
        fmt_kind = noisy.get("format_kind")
        if stage == "format" and fmt_kind:
            record_format_kind[fmt_kind].add(rid)

        for comp in noisy.get("components") or []:
            true_name = (comp.get("original_name") or comp.get("name") or "").strip()
            if not true_name:
                continue
            sub = comp.get(SUBTYPE_FIELD_BY_STAGE[stage])
            if not sub:
                continue
            comp_subtype[(rid, true_name.lower())].add(sub)

    return comp_subtype, record_format_kind


def _aggregate(
    predictions: list[dict],
    split_index: dict[str, dict],
    comp_subtype: dict[tuple[str, str], set[str]],
    record_format_kind: dict[str, set[str]],
) -> tuple[dict[str, dict], dict]:
    """Compute per-sub-type and overall aggregates."""
    # Per-sub-type accumulators
    bins: dict[str, dict] = defaultdict(lambda: {
        "n_components_affected": 0,
        "n_components_recalled": 0,
        "n_predicted_total": 0,      # for precision (total predicted across reactions)
        "n_predicted_correct": 0,    # for precision (matched ground truth)
        "reactions": set(),
        "reactions_predicted": set(),
        "reactions_within_10": set(),
        "reactions_within_25": set(),
        "reactions_abstained": set(),
    })
    overall = {
        "n_components_affected": 0,
        "n_components_recalled": 0,
        "n_predicted_total": 0,
        "n_predicted_correct": 0,
        "reactions": set(),
        "reactions_predicted": set(),
        "reactions_within_10": set(),
        "reactions_within_25": set(),
        "reactions_abstained": set(),
    }

    for pred in predictions:
        rid = pred.get("reaction_id", "")
        if not rid:
            continue
        rec = split_index.get(rid)
        if rec is None:
            continue
        predicted_cost = pred.get("predicted_cost")
        tcre = pred.get("tcre")
        predicted_components = pred.get("predicted_components") or []

        # Sub-types touching this reaction (component-level + record-level)
        rxn_subtypes: set[str] = set()
        for fmt_kind, rids in record_format_kind.items():
            if rid in rids:
                rxn_subtypes.add(fmt_kind)

        for comp in rec.get("components") or []:
            if not _is_non_solvent(comp):
                continue
            true_name = (comp.get("name") or "").strip()
            if not true_name:
                continue
            subs = comp_subtype.get((rid, true_name.lower()), set())
            if not subs:
                continue
            recalled = _component_predicted(predicted_components, true_name)
            for sub in subs:
                rxn_subtypes.add(sub)
                b = bins[sub]
                b["n_components_affected"] += 1
                if recalled:
                    b["n_components_recalled"] += 1

            overall["n_components_affected"] += 1
            if recalled:
                overall["n_components_recalled"] += 1

        # Precision @ reaction level: for each reaction touched by a
        # sub-type, count predicted components and how many of them
        # match a non-solvent ground-truth name.
        true_nonsolv_names = [
            (c.get("name") or "").strip() for c in (rec.get("components") or [])
            if _is_non_solvent(c) and (c.get("name") or "").strip()
        ]
        n_pred = len(predicted_components or [])
        n_pred_correct = 0
        for pc in predicted_components or []:
            pname = (pc.get("name") or "").strip()
            if not pname:
                continue
            for tn in true_nonsolv_names:
                if component_names_equivalent(tn, pname):
                    n_pred_correct += 1
                    break
        for sub in rxn_subtypes:
            b = bins[sub]
            b["n_predicted_total"] += n_pred
            b["n_predicted_correct"] += n_pred_correct
        if rxn_subtypes:
            overall["n_predicted_total"] += n_pred
            overall["n_predicted_correct"] += n_pred_correct

        # Reaction-level metrics: assign to every sub-type touching this rxn
        within10 = (
            predicted_cost is not None and tcre is not None and tcre <= 0.10
        )
        within25 = (
            predicted_cost is not None and tcre is not None and tcre <= 0.25
        )
        abstained = predicted_cost is None

        for sub in rxn_subtypes:
            b = bins[sub]
            b["reactions"].add(rid)
            if predicted_cost is not None:
                b["reactions_predicted"].add(rid)
            if within10:
                b["reactions_within_10"].add(rid)
            if within25:
                b["reactions_within_25"].add(rid)
            if abstained:
                b["reactions_abstained"].add(rid)

        if rxn_subtypes:
            overall["reactions"].add(rid)
            if predicted_cost is not None:
                overall["reactions_predicted"].add(rid)
            if within10:
                overall["reactions_within_10"].add(rid)
            if within25:
                overall["reactions_within_25"].add(rid)
            if abstained:
                overall["reactions_abstained"].add(rid)

    return bins, overall


def _row(name: str, b: dict) -> str:
    n_comp = b["n_components_affected"]
    n_rec = len(b["reactions"])
    recall = _format_pct(b["n_components_recalled"], n_comp)
    precision = _format_pct(b["n_predicted_correct"], b["n_predicted_total"])
    cta10 = _format_pct(len(b["reactions_within_10"]), n_rec)
    cta25 = _format_pct(len(b["reactions_within_25"]), n_rec)
    abst = _format_pct(len(b["reactions_abstained"]), n_rec)
    return (
        f"| {name:<20} | {n_comp:>6} | {n_rec:>7} | "
        f"{recall:>6} | {precision:>6} | {cta10:>6} | {cta25:>6} | {abst:>5} |"
    )


def export_raw_counts(bins: dict[str, dict]) -> dict[str, dict]:
    """Return JSON-serializable raw counts per sub-type, for pooling."""
    out = {}
    for sub, b in bins.items():
        out[sub] = {
            "n_components_affected": b["n_components_affected"],
            "n_components_recalled": b["n_components_recalled"],
            "n_predicted_total": b["n_predicted_total"],
            "n_predicted_correct": b["n_predicted_correct"],
            "n_reactions": len(b["reactions"]),
            "n_reactions_within_10": len(b["reactions_within_10"]),
            "n_reactions_within_25": len(b["reactions_within_25"]),
            "n_reactions_abstained": len(b["reactions_abstained"]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--stage", required=True, choices=list(STAGE_TO_TYPES),
        help="Noise stage to slice by",
    )
    parser.add_argument(
        "--level", default="medium",
        choices=["low", "medium", "high", "rich"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dev-split",
        default=ROOT / "data/processed/splits/dev.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--multi-split", default=None, type=Path,
        help="Optional multi-step split to merge in",
    )
    args = parser.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"Predictions file not found: {args.predictions}")

    predictions = _load_predictions(args.predictions)
    records = _load_split(args.dev_split)
    if args.multi_split:
        records = records + _load_split(args.multi_split)
    split_index = {r.get("reaction_id", ""): r for r in records if r.get("reaction_id")}

    comp_subtype, record_format_kind = _build_subtype_map(
        records, args.stage, args.level, args.seed,
    )

    bins, overall = _aggregate(
        predictions, split_index, comp_subtype, record_format_kind,
    )

    title_stage = {
        "name": "+Name (Aliasing)",
        "quantity": "+Qty (Requantification)",
        "missing": "+Miss (Omission)",
        "format": "+Fmt (Reformatting)",
    }[args.stage]

    print(
        f"## {args.predictions.name} x {title_stage}, "
        f"level={args.level}, seed={args.seed}"
    )
    print()
    header = (
        "| Sub-type             | n_comp | n_react | Recall | Precis | CTA@10 | CTA@25 | Abst |"
    )
    sep = (
        "|----------------------|-------:|--------:|-------:|-------:|-------:|-------:|-----:|"
    )
    print(header)
    print(sep)

    if not bins:
        print("| (no sub-types observed — re-injection produced no tags) |")
    else:
        for sub in sorted(bins):
            print(_row(sub, bins[sub]))

    print(_row("(overall)", overall))


if __name__ == "__main__":
    main()
