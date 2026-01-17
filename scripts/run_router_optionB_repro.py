#!/usr/bin/env python3
"""Option-B router evaluation (Strada A: compute-matched in expectation).

Strada A: define compute tiers B1..B4 by fixed-depth baselines (k=1..4).
For each tier and seed, tune adaptive routers on dev to match the fixed baseline
in *expected* compute (mean tokens), then evaluate once on test.

Outputs:
  - artifacts/router_optionB_seed{seed}/summary.csv (dev+test rows)
  - artifacts/router_optionB/paper_table_test_full_per_seed.csv (test rows, long form)
  - artifacts/router_optionB/paper_table_test_acc_tokens.csv (compact)

Run from repo root:
  python scripts/run_router_optionB_repro.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- IMPORTANT: add repo root to sys.path BEFORE importing src.* or scripts.* ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.conf_calibrator import ConfidenceCalibrator
from scripts import eval_depth_router as _edr


# ---------------- Config ----------------
SEEDS = [0, 1, 2]
TIERS = [1, 2, 3, 4]  # B1..B4
THS = [i * 0.99 / 499 for i in range(500)]  # 0.00..0.99 finer grid
STAB_GRID = [(m, min_step) for m in [1, 2, 3, 4] for min_step in [1, 2, 3, 4]]

OUTROOT = Path("artifacts/router_optionB")
OUTROOT.mkdir(parents=True, exist_ok=True)


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def load_calibrator_for_seed(seed: int) -> Optional[ConfidenceCalibrator]:
    p = Path(f"artifacts/calibration/platt_seed{seed}.json")
    if p.exists():
        return ConfidenceCalibrator.from_json(str(p))
    return None


def choose_conf_threshold(
    dev_examples: List[Dict[str, Any]],
    target_mean_tokens: float,
    calibrator: Optional[ConfidenceCalibrator],
) -> Tuple[float, Dict[str, Any]]:
    best = None
    for th in THS:
        r = _edr.evaluate(dev_examples, policy="conf", threshold=th, seed=0, calibrator=calibrator)
        score = (abs(float(r["mean_tokens"]) - target_mean_tokens), -float(r["acc"]))
        if best is None or score < best[0]:
            best = (score, th, r)
    assert best is not None
    return float(best[1]), best[2]


def choose_stability(
    dev_examples: List[Dict[str, Any]],
    target_mean_tokens: float,
    calibrator: Optional[ConfidenceCalibrator],
) -> Tuple[int, int, Dict[str, Any]]:
    best = None
    for m, min_step in STAB_GRID:
        r = _edr.evaluate(dev_examples, policy="stability", m=m, min_step=min_step, seed=0, calibrator=calibrator)
        score = (abs(float(r["mean_tokens"]) - target_mean_tokens), -float(r["acc"]))
        if best is None or score < best[0]:
            best = (score, m, min_step, r)
    assert best is not None
    _, m, min_step, r = best
    return int(m), int(min_step), r


def evaluate_oracle(examples: List[Dict[str, Any]], calibrator: Optional[ConfidenceCalibrator]) -> Dict[str, Any]:
    """Offline oracle: stop at earliest correct step in trajectory; if never correct, stop at last."""
    stop_steps: List[int] = []
    stop_tokens: List[int] = []
    correct: List[int] = []

    for ex in examples:
        gold, steps = _edr.extract_steps(ex)
        if calibrator is not None:
            steps = [
                _edr.Step(ans=st.ans, conf=calibrator.calibrate(st.t, st.conf), tokens=st.tokens, t=st.t)
                for st in steps
            ]
        cum_tokens = _edr.compute_prefix_tokens(steps)
        T = len(steps)

        s = T
        for i in range(1, T + 1):
            if steps[i - 1].ans == gold:
                s = i
                break

        stop_steps.append(s)
        stop_tokens.append(cum_tokens[s - 1])
        correct.append(int(steps[s - 1].ans == gold))

    out = {
        "n": len(examples),
        "policy": "oracle",
        "acc": sum(correct) / max(1, len(correct)),
        "mean_steps": sum(stop_steps) / max(1, len(stop_steps)),
        "p95_steps": _edr.p95([float(x) for x in stop_steps]),
        "mean_tokens": sum(stop_tokens) / max(1, len(stop_tokens)),
        "p95_tokens": _edr.p95([float(x) for x in stop_tokens]),
        "stop_histogram": _edr.build_stop_histogram(stop_steps, max(stop_steps)) if stop_steps else [],
    }
    return out


def row_from_eval(
    *,
    seed: int,
    split: str,
    policy: str,
    budget_tag: str,
    params: str,
    r: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "seed": seed,
        "split": split,
        "policy": policy,
        "budget_tag": budget_tag,
        "params": params,
        "acc": float(r.get("acc", float("nan"))),
        "mean_tokens": float(r.get("mean_tokens", float("nan"))),
        "mean_steps": float(r.get("mean_steps", float("nan"))),
        "p95_tokens": float(r.get("p95_tokens", float("nan"))),
        "p95_steps": float(r.get("p95_steps", float("nan"))),
        "flip_rate": float(r.get("flip_rate", float("nan"))),
        "regress_at_stop_rate": float(r.get("regress_at_stop_rate", float("nan"))),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------- Run ----------------
all_rows: List[Dict[str, Any]] = []

for seed in SEEDS:
    dev_path = f"data/router_splits_seeds/seed{seed}/dev.jsonl"
    test_path = f"data/router_splits_seeds/seed{seed}/test.jsonl"

    dev_ex = _edr.read_jsonl_grouped(dev_path)
    test_ex = _edr.read_jsonl_grouped(test_path)

    calibrator = load_calibrator_for_seed(seed)

    out_seed_dir = Path(f"artifacts/router_optionB_seed{seed}")
    out_seed_dir.mkdir(parents=True, exist_ok=True)

    rows_seed: List[Dict[str, Any]] = []

    # Oracle is the same across tiers, but we emit it per tier for easy table formatting.
    rdev_oracle = evaluate_oracle(dev_ex, calibrator)
    rtest_oracle = evaluate_oracle(test_ex, calibrator)

    for B in TIERS:
        budget_tag = f"B{B}"

        # (1) Fixed defines the tier compute (target mean tokens on dev)
        rdev_fixed = _edr.evaluate(dev_ex, policy="fixed", k=B, seed=0, calibrator=calibrator)
        rtest_fixed = _edr.evaluate(test_ex, policy="fixed", k=B, seed=0, calibrator=calibrator)
        target_tokens = float(rdev_fixed["mean_tokens"])

        rows_seed.append(row_from_eval(seed=seed, split="dev", policy="fixed", budget_tag=budget_tag, params=f"k={B}", r=rdev_fixed))
        rows_seed.append(row_from_eval(seed=seed, split="test", policy="fixed", budget_tag=budget_tag, params=f"k={B}", r=rtest_fixed))

        # (2) Confidence tuned to match mean tokens
        th, rdev_conf = choose_conf_threshold(dev_ex, target_tokens, calibrator)
        rtest_conf = _edr.evaluate(test_ex, policy="conf", threshold=th, seed=0, calibrator=calibrator)
        rows_seed.append(row_from_eval(seed=seed, split="dev", policy="conf", budget_tag=budget_tag, params=f"thr={th:.4f}", r=rdev_conf))
        rows_seed.append(row_from_eval(seed=seed, split="test", policy="conf", budget_tag=budget_tag, params=f"thr={th:.4f}", r=rtest_conf))

        # histogram from dev conf to define random-matched control
        hist_path = out_seed_dir / f"hist_{budget_tag}.json"
        with hist_path.open("w", encoding="utf-8") as f:
            json.dump(rdev_conf.get("stop_histogram", []), f)

        rdev_rand = _edr.evaluate(dev_ex, policy="random", random_hist_path=str(hist_path), seed=123, calibrator=calibrator)
        rtest_rand = _edr.evaluate(test_ex, policy="random", random_hist_path=str(hist_path), seed=123, calibrator=calibrator)
        rows_seed.append(row_from_eval(seed=seed, split="dev", policy="random", budget_tag=budget_tag, params="matched", r=rdev_rand))
        rows_seed.append(row_from_eval(seed=seed, split="test", policy="random", budget_tag=budget_tag, params="matched", r=rtest_rand))

        # (3) Stability tuned to match mean tokens
        m, min_step, rdev_stab = choose_stability(dev_ex, target_tokens, calibrator)
        rtest_stab = _edr.evaluate(test_ex, policy="stability", m=m, min_step=min_step, seed=0, calibrator=calibrator)
        rows_seed.append(row_from_eval(seed=seed, split="dev", policy="stability", budget_tag=budget_tag, params=f"m={m};min={min_step}", r=rdev_stab))
        rows_seed.append(row_from_eval(seed=seed, split="test", policy="stability", budget_tag=budget_tag, params=f"m={m};min={min_step}", r=rtest_stab))

        # Oracle rows (duplicated per tier for table layout)
        rows_seed.append(row_from_eval(seed=seed, split="dev", policy="oracle", budget_tag=budget_tag, params="-", r=rdev_oracle))
        rows_seed.append(row_from_eval(seed=seed, split="test", policy="oracle", budget_tag=budget_tag, params="-", r=rtest_oracle))

    summary_path = out_seed_dir / "summary.csv"
    write_csv(summary_path, rows_seed)
    print("Wrote", summary_path)

    all_rows.extend(rows_seed)

# Long-form per-seed (test only)
full_per_seed = [r for r in all_rows if r["split"] == "test"]
full_csv = OUTROOT / "paper_table_test_full_per_seed.csv"
write_csv(full_csv, full_per_seed)
print("Wrote", full_csv)

# Compact table (budget row, policy columns: acc (tokens))
out_table = OUTROOT / "paper_table_test_acc_tokens.csv"
policies = ["fixed", "conf", "random", "stability", "oracle"]

with out_table.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["budget_tag"] + policies)
    for B in TIERS:
        tag = f"B{B}"
        row = [tag]
        for pol in policies:
            accs = [float(r["acc"]) for r in full_per_seed if r["budget_tag"] == tag and r["policy"] == pol]
            toks = [float(r["mean_tokens"]) for r in full_per_seed if r["budget_tag"] == tag and r["policy"] == pol]
            ma, _ = mean_std(accs)
            mt, _ = mean_std(toks)
            row.append(f"{ma:.4f} ({mt:.0f})")
        w.writerow(row)

print("Wrote", out_table)
