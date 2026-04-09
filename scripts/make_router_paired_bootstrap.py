#!/usr/bin/env python3
"""Lightweight paired uncertainty support from bundled Option-B per-seed table.

Computes paired routed-vs-fixed deltas and bootstrap CIs by resampling split seeds.
This does NOT estimate retraining/model-seed variance.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _paired_values(rows: List[Dict[str, str]], budget_tag: str, routed_policy: str) -> Tuple[List[int], List[float], List[float]]:
    idx = {}
    for r in rows:
        if r["split"] != "test":
            continue
        key = (int(r["seed"]), r["budget_tag"], r["policy"])
        idx[key] = r

    seeds = sorted({int(r["seed"]) for r in rows if r["split"] == "test"})
    d_acc, d_tok = [], []
    for s in seeds:
        fixed = idx[(s, budget_tag, "fixed")]
        routed = idx[(s, budget_tag, routed_policy)]
        d_acc.append(float(routed["acc"]) - float(fixed["acc"]))
        d_tok.append(float(routed["mean_tokens"]) - float(fixed["mean_tokens"]))
    return seeds, d_acc, d_tok


def _bootstrap_mean_ci(values: List[float], n_boot: int, rng: random.Random) -> Tuple[float, float, float]:
    n = len(values)
    point = statistics.mean(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return point, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="artifacts/router_optionB/paper_table_test_full_per_seed.csv")
    ap.add_argument("--out_csv", default="artifacts/paper/tables/router_paired_bootstrap_gsm8k.csv")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--rng_seed", type=int, default=12345)
    args = ap.parse_args()

    rows = _load_rows(args.in_csv)
    rng = random.Random(args.rng_seed)

    comparisons = [
        ("conf_B2_minus_fixed_B2", "B2", "conf"),
        ("stability_B3_minus_fixed_B3", "B3", "stability"),
    ]

    out_rows = []
    for label, budget, routed in comparisons:
        seeds, d_acc, d_tok = _paired_values(rows, budget, routed)
        point_acc, lo_acc, hi_acc = _bootstrap_mean_ci(d_acc, args.n_boot, rng)
        point_tok, lo_tok, hi_tok = _bootstrap_mean_ci(d_tok, args.n_boot, rng)
        out_rows.append(
            {
                "comparison": label,
                "seeds": ",".join(str(s) for s in seeds),
                "paired_seed_deltas_acc": "|".join(f"{x:.6f}" for x in d_acc),
                "paired_seed_deltas_tokens": "|".join(f"{x:.6f}" for x in d_tok),
                "n_boot": str(args.n_boot),
                "point_delta_acc": f"{point_acc:.6f}",
                "ci95_low": f"{lo_acc:.6f}",
                "ci95_high": f"{hi_acc:.6f}",
                "point_delta_tokens": f"{point_tok:.6f}",
                "tokens_ci95_low": f"{lo_tok:.6f}",
                "tokens_ci95_high": f"{hi_tok:.6f}",
                "notes": "paired bootstrap over split seeds only (n=3), not retraining variance",
            }
        )

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
