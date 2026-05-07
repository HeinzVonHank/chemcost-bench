"""Compare tool-call patterns of successful vs failed trajectories within each model.

Defines:
  success  = predicted_cost not null AND TCRE <= 0.25 (CTA@25 hit)
  fail     = predicted_cost not null AND TCRE > 0.25
  abstain  = predicted_cost is null

For each trajectory, extracts:
  total_calls, unique_calls, retry_rate
  calls per channel: search / quote / mw / calc
  empty-return rate per channel
  quote-grounded rate (quote call whose query already returned a valid search result)
  first-quote step (turn at which the agent first issues a quote call)
  trajectory length in turns

Reports group means + Mann-Whitney U + Cliff's delta (effect size).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

CHANNELS = ("search", "quote", "mw", "calc", "other")


def channel_of(name: str) -> str:
    n = (name or "").lower()
    if "search" in n:
        return "search"
    if "quote" in n or "price" in n or "supplier" in n:
        return "quote"
    if "mass" in n or "mw" in n or "molar" in n:
        return "mw"
    if "calc" in n:
        return "calc"
    return "other"


def is_empty_result(result) -> bool:
    if result is None:
        return True
    if isinstance(result, str):
        s = result.strip().lower()
        if not s or s in ("null", "none", "[]", "{}", "[\n]", "{\n}"):
            return True
        try:
            j = json.loads(result)
        except Exception:
            return False
        return is_empty_result(j)
    if isinstance(result, dict):
        if not result:
            return True
        if "error" in result and not any(k for k in result if k != "error"):
            return True
        for marker in ("not_found", "empty", "no_results"):
            if result.get(marker):
                return True
        if "tier" in result and result.get("tier") == "unpriced":
            return True
        return all(v in (None, "", [], {}) for v in result.values())
    if isinstance(result, list):
        return len(result) == 0
    return False


_NORM = re.compile(r"[^A-Za-z0-9@\-=#\(\)\[\]]+")


def normalize_query(s) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = json.dumps(s, sort_keys=True)
    return _NORM.sub("", s).lower()


def query_string(args: dict) -> str:
    if not isinstance(args, dict):
        return normalize_query(args)
    for k in ("query", "smiles_or_name", "name", "smiles", "expression"):
        if k in args and args[k]:
            return normalize_query(args[k])
    return normalize_query(args)


def extract_features(tool_calls: list[dict]) -> dict:
    n_total = len(tool_calls)
    seen = set()
    retries = 0
    by_ch = Counter()
    empty_by_ch = Counter()
    valid_search_queries = set()
    quote_grounded = 0
    first_quote_step = None
    for i, tc in enumerate(tool_calls):
        name = tc.get("tool_name") or tc.get("name") or ""
        args = tc.get("arguments") or tc.get("input") or {}
        result = tc.get("result")
        ch = channel_of(name)
        by_ch[ch] += 1
        empty = is_empty_result(result)
        if empty:
            empty_by_ch[ch] += 1
        sig = (name, query_string(args))
        if sig in seen:
            retries += 1
        else:
            seen.add(sig)
        q = query_string(args)
        if ch == "search" and not empty:
            valid_search_queries.add(q)
        if ch == "quote":
            if first_quote_step is None:
                first_quote_step = i
            if q in valid_search_queries:
                quote_grounded += 1
    f = {"total_calls": n_total, "unique_calls": len(seen),
         "retries": retries,
         "retry_rate": retries / n_total if n_total else 0.0,
         "first_quote_step": first_quote_step if first_quote_step is not None else math.nan}
    for ch in CHANNELS:
        f[f"n_{ch}"] = by_ch.get(ch, 0)
        f[f"empty_{ch}"] = (empty_by_ch.get(ch, 0) / by_ch[ch]) if by_ch.get(ch) else math.nan
    n_quote = by_ch.get("quote", 0)
    f["quote_grounded_rate"] = quote_grounded / n_quote if n_quote else math.nan
    return f


def classify(rec: dict) -> str:
    pc = rec.get("predicted_cost")
    tcre = rec.get("tcre")
    if pc is None or tcre is None:
        return "abstain"
    return "success" if tcre <= 0.25 else "fail"


def mann_whitney(xs, ys):
    """Two-sided Mann-Whitney U on simple lists, no scipy."""
    xs = [v for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))]
    ys = [v for v in ys if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return math.nan, math.nan
    combined = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    combined.sort(key=lambda t: t[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    R1 = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mu) / sigma if sigma else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return U, p


def cliffs_delta(xs, ys):
    xs = [v for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))]
    ys = [v for v in ys if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs or not ys:
        return math.nan
    g = l = 0
    for x in xs:
        for y in ys:
            if x > y:
                g += 1
            elif x < y:
                l += 1
    return (g - l) / (len(xs) * len(ys))


def fmt(x, w=7, p=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'nan':>{w}}"
    return f"{x:>{w}.{p}f}"


def report(file_path: Path):
    d = json.load(open(file_path))
    recs = d.get("predictions") or d.get("results") or []
    by_class = defaultdict(list)
    for r in recs:
        tcs = r.get("tool_calls") or []
        f = extract_features(tcs)
        f["_class"] = classify(r)
        f["_tcre"] = r.get("tcre")
        by_class[f["_class"]].append(f)

    name = file_path.stem
    counts = {k: len(v) for k, v in by_class.items()}
    print()
    print(f"=== {name} ===")
    print(f"n_success={counts.get('success',0)}  n_fail={counts.get('fail',0)}  "
          f"n_abstain={counts.get('abstain',0)}  total={sum(counts.values())}")
    if not by_class.get("success") or not by_class.get("fail"):
        print("(skipping: need both success and fail trajectories)")
        return

    feats = ["total_calls", "unique_calls", "retry_rate",
             "n_search", "n_quote", "n_mw", "n_calc",
             "empty_search", "empty_quote",
             "quote_grounded_rate", "first_quote_step"]

    succ = by_class["success"]
    fail = by_class["fail"]
    abst = by_class.get("abstain", [])

    header = f"{'feature':<22} {'succ_med':>9} {'fail_med':>9} {'abst_med':>9} " \
             f"{'succ_mean':>10} {'fail_mean':>10} {'p(MWU)':>8} {'cliff':>7}"
    print(header)
    print("-" * len(header))
    for k in feats:
        s_vals = [r[k] for r in succ]
        f_vals = [r[k] for r in fail]
        a_vals = [r[k] for r in abst] if abst else []
        s_clean = [v for v in s_vals if not (isinstance(v, float) and math.isnan(v))]
        f_clean = [v for v in f_vals if not (isinstance(v, float) and math.isnan(v))]
        a_clean = [v for v in a_vals if not (isinstance(v, float) and math.isnan(v))]
        med_s = st.median(s_clean) if s_clean else math.nan
        med_f = st.median(f_clean) if f_clean else math.nan
        med_a = st.median(a_clean) if a_clean else math.nan
        mean_s = st.mean(s_clean) if s_clean else math.nan
        mean_f = st.mean(f_clean) if f_clean else math.nan
        _, p = mann_whitney(s_vals, f_vals)
        cd = cliffs_delta(s_vals, f_vals)
        print(f"{k:<22} {fmt(med_s,9)} {fmt(med_f,9)} {fmt(med_a,9)} "
              f"{fmt(mean_s,10)} {fmt(mean_f,10)} {fmt(p,8,4)} {fmt(cd,7)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="result JSONs")
    args = ap.parse_args()
    for f in args.files:
        report(Path(f))


if __name__ == "__main__":
    main()
