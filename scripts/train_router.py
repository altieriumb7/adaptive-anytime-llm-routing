#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import numpy as np

from src.router.features import (
    FEATURE_NAMES,
    build_expected_improvement_dataset,
    build_learned_stop_dataset,
)
from src.router.io import read_jsonl_grouped
from src.router.logreg import LogisticRegression


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dev JSONL (flat preds or trajectory-per-line).")
    ap.add_argument("--out", required=True, help="Output JSON model path.")
    ap.add_argument("--mode", choices=["learned_stop", "expected_improvement"], required=True)

    ap.add_argument("--lambda_cost", type=float, default=0.0, help="Only for learned_stop target.")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    trajs = read_jsonl_grouped(args.data)
    if not trajs:
        raise SystemExit("No trajectories loaded.")

    trainer = LogisticRegression(l2=args.l2, lr=args.lr, epochs=args.epochs, seed=args.seed)

    if args.mode == "learned_stop":
        X, y = build_learned_stop_dataset(trajs, lambda_cost=args.lambda_cost)
        if X.shape[0] == 0:
            raise SystemExit("Empty learned_stop dataset after preprocessing.")

        model = trainer.fit(X, y, feature_names=FEATURE_NAMES)
        out_obj: Dict[str, Any] = model.to_json()
        out_obj["mode"] = "learned_stop"
        out_obj["lambda_cost"] = float(args.lambda_cost)

    else:
        X, y_now, y_next = build_expected_improvement_dataset(trajs)
        if X.shape[0] == 0:
            raise SystemExit("Empty expected_improvement dataset (need at least 2 steps per example).")

        model_now = trainer.fit(X, y_now, feature_names=FEATURE_NAMES)
        model_next = trainer.fit(X, y_next, feature_names=FEATURE_NAMES)

        out_obj = {
            "mode": "expected_improvement",
            "model_now": model_now.to_json(),
            "model_next": model_next.to_json(),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
