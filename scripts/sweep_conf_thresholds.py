#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np

from eval_depth_router import read_jsonl_grouped, evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True, help="Dev JSONL (flat preds or trajectory-per-line).")
    ap.add_argument("--threshold_min", type=float, default=0.50)
    ap.add_argument("--threshold_max", type=float, default=0.99)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--target_mean_steps", type=str, default="1,2,3,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_grid", type=str, default=None, help="Optional: write sweep grid JSON.")
    args = ap.parse_args()

    dev = read_jsonl_grouped(args.dev)
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.n)

    grid = []
    for th in thresholds:
        res = evaluate(dev, policy="conf", threshold=float(th), seed=args.seed)
        grid.append(
            {
                "threshold": float(th),
                "acc": float(res["acc"]),
                "mean_steps": float(res["mean_steps"]),
                "mean_tokens": float(res["mean_tokens"]),
            }
        )

    targets = [float(x.strip()) for x in args.target_mean_steps.split(",") if x.strip()]
    for B in targets:
        best = min(grid, key=lambda r: abs(r["mean_steps"] - B))
        print(json.dumps({
            "target_mean_steps": B,
            "best_threshold": best["threshold"],
            "mean_steps": best["mean_steps"],
            "acc": best["acc"]
        }))

    if args.out_grid:
        with open(args.out_grid, "w", encoding="utf-8") as f:
            json.dump(grid, f, indent=2)


if __name__ == "__main__":
    main()
