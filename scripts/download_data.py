"""Download the ChemCost dataset and frozen pricing database from Hugging Face.

Run once after cloning the anonymous code release:

    python3 scripts/download_data.py

Files placed under ``data/`` (relative to repo root):

    data/chemcost.jsonl                       Unified 1,427-reaction dataset.
    data/processed/pricing_db.sqlite          Frozen supplier-quote database.
    data/processed/splits/dev_eval_ready.jsonl
    data/processed/splits/test_eval_ready.jsonl
    data/processed/splits/dev_multistep.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

REPO = "nips2026-chemcost/chemcost-bench"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

# (remote_path_in_hf_repo, local_path_relative_to_repo_root)
FILES = [
    ("chemcost.jsonl",                            "data/chemcost.jsonl"),
    ("splits/dev_eval_ready.jsonl",               "data/processed/splits/dev_eval_ready.jsonl"),
    ("splits/test_eval_ready.jsonl",              "data/processed/splits/test_eval_ready.jsonl"),
    ("splits/dev_multistep.jsonl",                "data/processed/splits/dev_multistep.jsonl"),
    ("pricing_db.sqlite",                         "data/processed/pricing_db.sqlite"),
]


def fetch(remote: str, local: Path, force: bool = False) -> None:
    if local.exists() and not force:
        print(f"  skip {local} (already exists)")
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{remote}"
    print(f"  fetch {url}")
    urlretrieve(url, local)
    print(f"    -> {local} ({local.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the local file already exists.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Repository root (default: parent of scripts/).")
    args = parser.parse_args()

    print(f"Downloading ChemCost from {REPO} into {args.root}")
    for remote, rel in FILES:
        fetch(remote, args.root / rel, force=args.force)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
