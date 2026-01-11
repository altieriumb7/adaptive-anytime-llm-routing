#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

# Allow running as: python scripts/<script>.py from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.anytime_postprocess import postprocess_trajectory_record


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]], *, mode: str = "w") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize/enrich trajectory JSONL files.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input trajectories JSONL")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL")
    ap.add_argument(
        "--expected_ts",
        default="1,2,3,4",
        help="Comma-separated expected t values. Use '' to disable.",
    )
    ap.add_argument(
        "--monotone_best_so_far",
        action="store_true",
        help="Training-time only: if a prefix checkpoint is correct, later checkpoints inherit its final answer.",
    )
    ap.add_argument(
        "--prefer_high_conf",
        action="store_true",
        help="When multiple prefix correct answers exist, pick the one with highest confidence (default keeps earliest).",
    )
    ap.add_argument(
        "--drop_if_missing_ts",
        action="store_true",
        help="Drop examples that miss any expected t.",
    )
    ap.add_argument(
        "--drop_if_any_unparseable",
        action="store_true",
        help="Drop examples where any checkpoint has no parseable final answer.",
    )
    ap.add_argument("--max_rows", type=int, default=None, help="Process at most N rows (for quick tests).")
    args = ap.parse_args()

    expected_ts: Optional[List[int]]
    if args.expected_ts.strip() == "":
        expected_ts = None
    else:
        expected_ts = [int(x.strip()) for x in args.expected_ts.split(",") if x.strip()]

    buf: List[Dict[str, Any]] = []
    kept = 0
    dropped = 0

    for i, rec in enumerate(tqdm(iter_jsonl(args.in_path), desc="clean")):
        if args.max_rows is not None and i >= args.max_rows:
            break

        postprocess_trajectory_record(
            rec,
            expected_ts=tuple(expected_ts) if expected_ts is not None else None,
            monotone_best_so_far=args.monotone_best_so_far,
            prefer_high_conf=args.prefer_high_conf,
        )

        if args.drop_if_missing_ts and expected_ts is not None:
            if rec.get("missing_ts"):
                dropped += 1
                continue

        if args.drop_if_any_unparseable:
            cps = rec.get("checkpoints") or []
            if any(isinstance(cp, dict) and not cp.get("parseable", True) for cp in cps):
                dropped += 1
                continue

        buf.append(rec)
        kept += 1
        if len(buf) >= 2000:
            write_jsonl(args.out_path, buf, mode="a" if os.path.exists(args.out_path) else "w")
            buf = []

    if buf:
        write_jsonl(args.out_path, buf, mode="a" if os.path.exists(args.out_path) else "w")

    print(f"Wrote {kept} rows to {args.out_path} (dropped {dropped}).")


if __name__ == "__main__":
    main()
