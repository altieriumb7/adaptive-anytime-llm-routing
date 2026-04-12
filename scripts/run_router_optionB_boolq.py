#!/usr/bin/env python3
"""
Option-B router evaluation (Strada A / Option B): compute-matched *in expectation*.

BoolQ paper-facing semantics:
- The held-out evaluation split is exported with split label "validation" (not "test").
- Output filenames also use "validation" consistently.

We define compute tiers B1..B4 by fixed-depth baselines (k=1..4).
For each tier and seed, we tune each adaptive policy on DEV to match the fixed baseline's
expected compute (mean tokens). When exact matching is not attainable with a single setting,
we use a 2-point mixture (low/hi) and report expected metrics (no Monte Carlo noise).

Outputs:
  - <out_dir>/seed{seed}/summary.csv (dev + validation rows)
  - <out_dir>/paper_table_validation_full_per_seed.csv (validation rows, long form)
  - <out_dir>/paper_table_validation_acc_tokens.csv (compact)

Std-dev convention:
- All cross-seed summaries use sample standard deviation (statistics.stdev, ddof=1).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import scripts.eval_depth_router as edr
except ModuleNotFoundError:
    import eval_depth_router as edr

from src.calibration.conf_calibrator import ConfidenceCalibrator
from scripts.lfs_guard import assert_materialized


DEFAULT_OUTROOT = "artifacts/router_optionB_boolq"
DEFAULT_SEEDS = [int(s) for s in os.environ.get("ROUTER_SEEDS", "0 1 2").replace(",", " ").split()]
TARGETS = [1, 2, 3, 4]
T_MAX = 4
EVAL_SPLIT_LABEL = "validation"


def sample_std(xs: List[float]) -> float:
    return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


def make_threshold_grid() -> List[float]:
    grid = [i * 0.999 / 999 for i in range(1000)]
    grid += [1.0 - 10.0 ** (-k) for k in range(2, 13)]
    grid += [1.0 - j * 1e-6 for j in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]]
    grid = [max(0.0, min(1.0 - 1e-15, float(x))) for x in grid]
    return sorted(set(grid))


THS = make_threshold_grid()
STAB_GRID = [(m, min_step) for m in [1, 2, 3, 4] for min_step in [1, 2, 3, 4]]


def pad_hist(hist: List[float], T: int = T_MAX) -> List[float]:
    h = list(hist)
    if len(h) < T:
        h = h + [0.0] * (T - len(h))
    if len(h) > T:
        h = h[:T]
    s = sum(h)
    if s <= 0:
        return [1.0 / T] * T
    return [x / s for x in h]


def random_expected_metrics(examples, stop_hist):
    stop_hist = pad_hist(stop_hist, T_MAX)
    n = len(examples)
    acc = steps = toks = 0.0
    for ex in examples:
        gold, stps = edr.extract_steps(ex)
        cum = edr.compute_prefix_tokens(stps)
        for i, p in enumerate(stop_hist, start=1):
            steps += p * i
            toks += p * cum[i - 1]
            acc += p * (1.0 if stps[i - 1].ans == gold else 0.0)
    return {"acc": acc / n, "mean_steps": steps / n, "mean_tokens": toks / n, "stop_histogram": stop_hist}


@dataclass
class Mix:
    mode: str
    th: Optional[float] = None
    m: Optional[int] = None
    min_step: Optional[int] = None
    p_low: Optional[float] = None
    low: Optional[Dict[str, Any]] = None
    high: Optional[Dict[str, Any]] = None
    desc: str = ""


def mix_metrics(p_low: float, low: Dict[str, Any], high: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(low)
    out["acc"] = p_low * float(low["acc"]) + (1 - p_low) * float(high["acc"])
    out["mean_tokens"] = p_low * float(low["mean_tokens"]) + (1 - p_low) * float(high["mean_tokens"])
    out["mean_steps"] = p_low * float(low["mean_steps"]) + (1 - p_low) * float(high["mean_steps"])
    h_low = pad_hist(low.get("stop_histogram", []), T_MAX)
    h_high = pad_hist(high.get("stop_histogram", []), T_MAX)
    h = [p_low * a + (1 - p_low) * b for a, b in zip(h_low, h_high)]
    out["stop_histogram"] = pad_hist(h, T_MAX)
    return out


def pick_bracket_by_tokens(cands: List[Tuple[float, Dict[str, Any]]], target: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cands = sorted(cands, key=lambda x: x[0])
    below = [c for c in cands if c[0] <= target]
    above = [c for c in cands if c[0] >= target]
    if not below or not above:
        cands2 = sorted(cands, key=lambda x: abs(x[0] - target))
        low = cands2[0][1]
        high = cands2[1][1] if len(cands2) > 1 else cands2[0][1]
        return low, high
    return below[-1][1], above[0][1]


def choose_conf_mix(dev_examples, target_tokens: float, calibrator: Optional[ConfidenceCalibrator], seed: int) -> Mix:
    cands: List[Tuple[float, Dict[str, Any]]] = []
    for th in THS:
        r = edr.evaluate(dev_examples, policy="conf", threshold=float(th), seed=seed, calibrator=calibrator)
        cands.append((float(r["mean_tokens"]), {"th": float(th), "res": r}))

    low_w, high_w = pick_bracket_by_tokens([(mt, w) for mt, w in cands], target_tokens)
    best_single = min([w for _, w in cands], key=lambda w: abs(float(w["res"]["mean_tokens"]) - target_tokens))
    if abs(float(best_single["res"]["mean_tokens"]) - target_tokens) < 1e-6:
        return Mix(mode="single", th=float(best_single["th"]))

    low = low_w["res"]
    high = high_w["res"]
    mu_low = float(low["mean_tokens"])
    mu_high = float(high["mean_tokens"])
    if abs(mu_high - mu_low) < 1e-12:
        return Mix(mode="single", th=float(best_single["th"]))

    p_low = (mu_high - target_tokens) / (mu_high - mu_low)
    p_low = max(0.0, min(1.0, float(p_low)))
    return Mix(mode="mix", p_low=p_low, low=low_w, high=high_w)


def stability_is_trivial_fixed(m: int, min_step: int, tier_k: int) -> bool:
    return (m == 1 and min_step == tier_k)


def choose_stability_mix(dev_examples, target_tokens: float, calibrator: Optional[ConfidenceCalibrator], seed: int, tier_k: int) -> Mix:
    def eval_grid(exclude_trivial: bool) -> List[Tuple[float, Dict[str, Any]]]:
        out = []
        for m, min_step in STAB_GRID:
            if exclude_trivial and tier_k >= 2 and stability_is_trivial_fixed(m, min_step, tier_k):
                continue
            r = edr.evaluate(dev_examples, policy="stability", m=m, min_step=min_step, seed=seed, calibrator=calibrator)
            out.append((float(r["mean_tokens"]), {"m": m, "min_step": min_step, "res": r}))
        return out

    cands = eval_grid(exclude_trivial=True) or eval_grid(exclude_trivial=False)
    best_single = min([w for _, w in cands], key=lambda w: abs(float(w["res"]["mean_tokens"]) - target_tokens))
    if abs(float(best_single["res"]["mean_tokens"]) - target_tokens) < 1e-6:
        return Mix(mode="single", m=int(best_single["m"]), min_step=int(best_single["min_step"]))

    low_w, high_w = pick_bracket_by_tokens([(mt, w) for mt, w in cands], target_tokens)
    low = low_w["res"]
    high = high_w["res"]
    mu_low = float(low["mean_tokens"])
    mu_high = float(high["mean_tokens"])
    if abs(mu_high - mu_low) < 1e-12:
        return Mix(mode="single", m=int(best_single["m"]), min_step=int(best_single["min_step"]))

    p_low = (mu_high - target_tokens) / (mu_high - mu_low)
    p_low = max(0.0, min(1.0, float(p_low)))
    return Mix(mode="mix", p_low=p_low, low=low_w, high=high_w)


def oracle_metrics(examples, seed: int = 0, calibrator: Optional[ConfidenceCalibrator] = None) -> Dict[str, Any]:
    stop_steps, stop_tokens, correct = [], [], []
    for ex in examples:
        gold, steps = edr.extract_steps(ex)
        cum_tokens = edr.compute_prefix_tokens(steps)
        T = len(steps)
        best = None
        for i in range(1, T + 1):
            if steps[i - 1].ans == gold:
                best = i
                break
        if best is None:
            best = T
        stop_steps.append(best)
        stop_tokens.append(cum_tokens[best - 1])
        correct.append(1 if steps[best - 1].ans == gold else 0)

    n = len(examples)
    return {
        "n": n,
        "policy": "oracle",
        "seed": seed,
        "acc": sum(correct) / max(1, n),
        "mean_steps": sum(stop_steps) / max(1, n),
        "mean_tokens": sum(stop_tokens) / max(1, n),
        "stop_histogram": pad_hist(edr.build_stop_histogram(stop_steps, T_MAX), T_MAX),
    }


def save_summary(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parse_seeds(s: str) -> List[int]:
    return [int(x) for x in str(s).replace(",", " ").split() if str(x).strip()]


def _default_seed_split_paths(seed: int) -> Tuple[str, str]:
    return (
        f"data/router_splits_boolq_seeds/seed{seed}/dev.jsonl",
        f"data/router_splits_boolq_seeds/seed{seed}/test.jsonl",
    )


def run_router_boolq(out_dir: str, seeds: List[int], dev_jsonl: Optional[str], test_jsonl: Optional[str], calibrator_json: Optional[str]) -> None:
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    all_seed_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        dev_path, eval_path = (dev_jsonl, test_jsonl) if (dev_jsonl and test_jsonl) else _default_seed_split_paths(seed)
        assert_materialized(Path(dev_path), role=f"BoolQ dev split for seed {seed}")
        assert_materialized(Path(eval_path), role=f"BoolQ validation split for seed {seed}")
        out_seed = str(Path(out_dir) / f"seed{seed}")
        Path(out_seed).mkdir(parents=True, exist_ok=True)

        dev_ex = edr.read_jsonl_grouped(dev_path)
        eval_ex = edr.read_jsonl_grouped(eval_path)

        calibrator = ConfidenceCalibrator.from_json(calibrator_json) if calibrator_json else None
        rows: List[Dict[str, Any]] = []

        dev_or = oracle_metrics(dev_ex, seed=seed, calibrator=calibrator)
        eval_or = oracle_metrics(eval_ex, seed=seed, calibrator=calibrator)

        for B in TARGETS:
            budget_tag = f"B{B}"
            dev_fixed = edr.evaluate(dev_ex, policy="fixed", k=B, seed=seed, calibrator=calibrator)
            eval_fixed = edr.evaluate(eval_ex, policy="fixed", k=B, seed=seed, calibrator=calibrator)
            target_tokens = float(dev_fixed["mean_tokens"])

            conf_mix = choose_conf_mix(dev_ex, target_tokens, calibrator, seed)
            if conf_mix.mode == "single":
                th = float(conf_mix.th)
                dev_conf = edr.evaluate(dev_ex, policy="conf", threshold=th, seed=seed, calibrator=calibrator)
                eval_conf = edr.evaluate(eval_ex, policy="conf", threshold=th, seed=seed, calibrator=calibrator)
                conf_hist = pad_hist(dev_conf.get("stop_histogram", []), T_MAX)
            else:
                assert conf_mix.low is not None and conf_mix.high is not None and conf_mix.p_low is not None
                low_w, high_w = conf_mix.low, conf_mix.high
                dev_conf = mix_metrics(float(conf_mix.p_low), low_w["res"], high_w["res"])
                eval_low = edr.evaluate(eval_ex, policy="conf", threshold=float(low_w["th"]), seed=seed, calibrator=calibrator)
                eval_high = edr.evaluate(eval_ex, policy="conf", threshold=float(high_w["th"]), seed=seed, calibrator=calibrator)
                eval_conf = mix_metrics(float(conf_mix.p_low), eval_low, eval_high)
                conf_hist = pad_hist(dev_conf.get("stop_histogram", []), T_MAX)

            dev_rand = random_expected_metrics(dev_ex, conf_hist)
            eval_rand = random_expected_metrics(eval_ex, conf_hist)
            with open(f"{out_seed}/conf_hist_{budget_tag}.json", "w", encoding="utf-8") as f:
                json.dump(conf_hist, f)

            stab_mix = choose_stability_mix(dev_ex, target_tokens, calibrator, seed, tier_k=B)
            if stab_mix.mode == "single":
                m, min_step = int(stab_mix.m), int(stab_mix.min_step)
                dev_stab = edr.evaluate(dev_ex, policy="stability", m=m, min_step=min_step, seed=seed, calibrator=calibrator)
                eval_stab = edr.evaluate(eval_ex, policy="stability", m=m, min_step=min_step, seed=seed, calibrator=calibrator)
            else:
                assert stab_mix.low is not None and stab_mix.high is not None and stab_mix.p_low is not None
                low_w, high_w = stab_mix.low, stab_mix.high
                dev_stab = mix_metrics(float(stab_mix.p_low), low_w["res"], high_w["res"])
                eval_low = edr.evaluate(eval_ex, policy="stability", m=int(low_w["m"]), min_step=int(low_w["min_step"]), seed=seed, calibrator=calibrator)
                eval_high = edr.evaluate(eval_ex, policy="stability", m=int(high_w["m"]), min_step=int(high_w["min_step"]), seed=seed, calibrator=calibrator)
                eval_stab = mix_metrics(float(stab_mix.p_low), eval_low, eval_high)

            def pack(split: str, policy: str, res: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "seed": seed,
                    "split": split,
                    "budget_tag": budget_tag,
                    "policy": policy,
                    "acc": float(res["acc"]),
                    "mean_tokens": float(res["mean_tokens"]),
                    "mean_steps": float(res["mean_steps"]),
                }

            rows.extend(
                [
                    pack("dev", "fixed", dev_fixed),
                    pack(EVAL_SPLIT_LABEL, "fixed", eval_fixed),
                    pack("dev", "conf", dev_conf),
                    pack(EVAL_SPLIT_LABEL, "conf", eval_conf),
                    pack("dev", "random", dev_rand),
                    pack(EVAL_SPLIT_LABEL, "random", eval_rand),
                    pack("dev", "stability", dev_stab),
                    pack(EVAL_SPLIT_LABEL, "stability", eval_stab),
                    pack("dev", "oracle", dev_or),
                    pack(EVAL_SPLIT_LABEL, "oracle", eval_or),
                ]
            )

        save_summary(f"{out_seed}/summary.csv", rows)
        print(f"Wrote {out_seed}/summary.csv")
        all_seed_rows.extend([r for r in rows if r["split"] == EVAL_SPLIT_LABEL])

    long_path = f"{out_dir}/paper_table_validation_full_per_seed.csv"
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_seed_rows[0].keys()))
        w.writeheader()
        for r in all_seed_rows:
            w.writerow(r)
    print(f"Wrote {long_path}")

    from collections import defaultdict

    group = defaultdict(list)
    for r in all_seed_rows:
        group[(r["policy"], r["budget_tag"])].append(r)

    compact_rows = []
    for (policy, budget_tag), items in sorted(group.items(), key=lambda x: (x[0][0], int(x[0][1][1:]))):
        accs = [float(it["acc"]) for it in items]
        toks = [float(it["mean_tokens"]) for it in items]
        stps = [float(it["mean_steps"]) for it in items]
        compact_rows.append(
            {
                "policy": policy,
                "budget_tag": budget_tag,
                "acc_mean": statistics.mean(accs),
                "acc_std": sample_std(accs),
                "tokens_mean": statistics.mean(toks),
                "tokens_std": sample_std(toks),
                "steps_mean": statistics.mean(stps),
                "steps_std": sample_std(stps),
            }
        )

    compact_path = f"{out_dir}/paper_table_validation_acc_tokens.csv"
    with open(compact_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(compact_rows[0].keys()))
        w.writeheader()
        for r in compact_rows:
            w.writerow(r)
    print(f"Wrote {compact_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.environ.get("ROUTER_BOOLQ_OUTDIR", DEFAULT_OUTROOT))
    ap.add_argument("--seeds", default=None, help="Comma/space separated seeds. Overrides ROUTER_SEEDS env.")
    ap.add_argument("--dev_jsonl", default=None, help="Path to BoolQ dev jsonl (overrides seed-based defaults).")
    ap.add_argument("--test_jsonl", default=None, help="Path to BoolQ held-out jsonl (exported as split=validation).")
    ap.add_argument("--calibrator", default=None, help="Optional calibrator json path applied to all seeds.")
    args = ap.parse_args()

    seeds = DEFAULT_SEEDS if args.seeds is None else _parse_seeds(args.seeds)
    run_router_boolq(
        out_dir=args.out_dir,
        seeds=seeds,
        dev_jsonl=args.dev_jsonl,
        test_jsonl=args.test_jsonl,
        calibrator_json=args.calibrator,
    )


if __name__ == "__main__":
    main()
