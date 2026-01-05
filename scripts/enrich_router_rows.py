#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

# Allow running as: python scripts/<script>.py from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.anytime_postprocess import postprocess_checkpoint, add_step_deltas


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Add parseable/conf clamp + delta features to router rows JSONL")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--check_correct", action="store_true", help="Recompute correctness using gold.")
    args = ap.parse_args()

    by_uid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in tqdm(iter_jsonl(args.in_path), desc="read"):
        uid = str(row.get("uid"))
        by_uid[uid].append(row)

    out_rows: List[Dict[str, Any]] = []
    for uid, rows in tqdm(by_uid.items(), desc="enrich"):
        rows_sorted = sorted(rows, key=lambda r: int(r.get("t", 0)))
        gold = rows_sorted[0].get("gold")

        for r in rows_sorted:
            # Map router-row keys to checkpoint convention.
            if "raw" not in r and "raw_text" in r:
                r["raw"] = r["raw_text"]

            postprocess_checkpoint(
                r,
                gold=gold if args.check_correct else None,
                recompute_correct=args.check_correct,
            )

            # Keep raw_text as canonical key for downstream router scripts
            if "raw_text" not in r and "raw" in r:
                r["raw_text"] = r["raw"]

        add_step_deltas(rows_sorted)
        out_rows.extend(rows_sorted)

    write_jsonl(args.out_path, out_rows)
    print(f"Wrote {len(out_rows)} rows to {args.out_path}")


if __name__ == "__main__":
    main()
