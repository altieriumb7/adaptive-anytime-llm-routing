#!/usr/bin/env python3
"""
Option-B router evaluation (Strada A / Option B): compute-matched *in expectation*.

We define compute tiers B1..B4 by fixed-depth baselines (k=1..4).
For each tier and seed, we tune each adaptive policy on DEV to match the fixed baseline's
expected compute (mean tokens). When exact matching is not attainable with a single setting,
we use a 2-point mixture (low/hi) and report expected metrics (no Monte Carlo noise).

Outputs:
  - <out_dir>/seed{seed}/summary.csv (dev+test rows)
  - <out_dir>/paper_table_test_full_per_seed.csv (test rows, long form)
  - <out_dir>/paper_table_test_acc_tokens.csv (compact)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import scripts.eval_depth_router as edr
from src.calibration.conf_calibrator import ConfidenceCalibrator


DEFAULT_OUTDIR = "artifacts/router_optionB"
DEFAULT_SEEDS = [int(s) for s in os.environ.get("ROUTER_SEEDS", "0 1 2").replace(",", " ").split()]
TARGETS = [1, 2, 3, 4]  # B1..B4
T_MAX = 4              # anytime trajectory length


def make_threshold_grid() -> List[float]:
    # Dense grid + very high thresholds
    grid = [i * 0.999 / 999 for i in range(1000)]  # 0..0.999
    grid += [1.0 - 10.0 ** (-k) for k in range(2, 13)]  # 0.99, 0.999, ..., 1-1e-12
    grid += [1.0 - j * 1e-6 for j in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]]
    # clamp below 1
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
    """Deterministic expected metrics under stop_hist (no sampling)."""
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
    mode: str  # "single" or "mix"
    # for single:
    th: Optional[float] = None
    m: Optional[int] = None
    min_step: Optional[int] = None
    # for mix:
    p_low: Optional[float] = None
    low: Optional[Dict[str, Any]] = None   # wrapper dict
    high: Optional[Dict[str, Any]] = None  # wrapper dict
    desc: str = ""


def mix_metrics(p_low: float, low: Dict[str, Any], high: Dict[str, Any]) -> Dict[str, Any]:
    """Expected values for averages (acc/mean_tokens/mean_steps), plus a mixed stop histogram."""
    out = dict(low)
    out["acc"] = p_low * float(low["acc"]) + (1 - p_low) * float(high["acc"])
    out["mean_tokens"] = p_low * float(low["mean_tokens"]) + (1 - p_low) * float(high["mean_tokens"])
    out["mean_steps"] = p_low * float(low["mean_steps"]) + (1 - p_low) * float(high["mean_steps"])
    h_low = pad_hist(low.get("stop_histogram", []), T_MAX)
    h_high = pad_hist(high.get("stop_histogram", []), T_MAX)
    h = [p_low * a + (1 - p_low) * b for a, b in zip(h_low, h_high)]
    out["stop_histogram"] = pad_hist(h, T_MAX)
    return out


def pick_bracket_by_tokens(
    cands: List[Tuple[float, Dict[str, Any]]], target: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """cands: list of (mean_tokens, wrapper). Returns (low_wrapper, high_wrapper)."""
    cands = sorted(cands, key=lambda x: x[0])
    below = [c for c in cands if c[0] <= target]
    above = [c for c in cands if c[0] >= target]
    if not below or not above:
        # fallback: closest two by abs delta (may not bracket)
        cands2 = sorted(cands, key=lambda x: abs(x[0] - target))
        low = cands2[0][1]
        high = cands2[0][1]
        if len(cands2) > 1:
            high = cands2[1][1]
        return low, high
    low = below[-1][1]   # closest below
    high = above[0][1]   # closest above
    return low, high


def choose_conf_mix(dev_examples, target_tokens: float, calibrator: Optional[ConfidenceCalibrator], seed: int) -> Mix:
    cands: List[Tuple[float, Dict[str, Any]]] = []
    for th in THS:
        r = edr.evaluate(dev_examples, policy="conf", threshold=float(th), seed=seed, calibrator=calibrator)
        cands.append((float(r["mean_tokens"]), {"th": float(th), "res": r}))

    low_w, high_w = pick_bracket_by_tokens([(mt, w) for mt, w in cands], target_tokens)

    best_single = min([w for _, w in cands], key=lambda w: abs(float(w["res"]["mean_tokens"]) - target_tokens))
    if abs(float(best_single["res"]["mean_tokens"]) - target_tokens) < 1e-6:
        return Mix(mode="single", th=float(best_single["th"]), desc=f"single th={best_single['th']:.6g}")

    low = low_w["res"]
    high = high_w["res"]
    mu_low = float(low["mean_tokens"])
    mu_high = float(high["mean_tokens"])
    if abs(mu_high - mu_low) < 1e-12:
        return Mix(mode="single", th=float(best_single["th"]), desc=f"fallback single th={best_single['th']:.6g}")

    p_low = (mu_high - target_tokens) / (mu_high - mu_low)
    p_low = max(0.0, min(1.0, float(p_low)))
    return Mix(mode="mix", p_low=p_low, low=low_w, high=high_w, desc=f"mix conf p_low={p_low:.3f}")


def stability_is_trivial_fixed(m: int, min_step: int, tier_k: int) -> bool:
    # With m=1, policy_stability returns at i=min_step always => identical to fixed depth=min_step.
    return (m == 1 and min_step == tier_k)


def choose_stability_mix(
    dev_examples, target_tokens: float, calibrator: Optional[ConfidenceCalibrator], seed: int, tier_k: int
) -> Mix:
    def eval_grid(exclude_trivial: bool) -> List[Tuple[float, Dict[str, Any]]]:
        out = []
        for m, min_step in STAB_GRID:
            if exclude_trivial and tier_k >= 2 and stability_is_trivial_fixed(m, min_step, tier_k):
                continue
            r = edr.evaluate(dev_examples, policy="stability", m=m, min_step=min_step, seed=seed, calibrator=calibrator)
            out.append((float(r["mean_tokens"]), {"m": m, "min_step": min_step, "res": r}))
        return out

    cands = eval_grid(exclude_trivial=True)
    if not cands:
        cands = eval_grid(exclude_trivial=False)

    best_single = min([w for _, w in cands], key=lambda w: abs(float(w["res"]["mean_tokens"]) - target_tokens))
    if abs(float(best_single["res"]["mean_tokens"]) - target_tokens) < 1e-6:
        return Mix(
            mode="single",
            m=int(best_single["m"]),
            min_step=int(best_single["min_step"]),
            desc=f"single stab m={best_single['m']} min_step={best_single['min_step']}",
        )

    low_w, high_w = pick_bracket_by_tokens([(mt, w) for mt, w in cands], target_tokens)

    low = low_w["res"]
    high = high_w["res"]
    mu_low = float(low["mean_tokens"])
    mu_high = float(high["mean_tokens"])
    if abs(mu_high - mu_low) < 1e-12:
        return Mix(
            mode="single",
            m=int(best_single["m"]),
            min_step=int(best_single["min_step"]),
            desc=f"fallback single stab m={best_single['m']} min_step={best_single['min_step']}",
        )

    p_low = (mu_high - target_tokens) / (mu_high - mu_low)
    p_low = max(0.0, min(1.0, float(p_low)))

    return Mix(
        mode="mix",
        p_low=p_low,
        low=low_w,
        high=high_w,
        desc=f"mix stab p_low={p_low:.3f} (low m={low_w['m']},min_step={low_w['min_step']} / high m={high_w['m']},min_step={high_w['min_step']})",
    )


def oracle_metrics(examples, seed: int = 0, calibrator: Optional[ConfidenceCalibrator] = None) -> Dict[str, Any]:
    # Oracle: earliest correct step if any, else last step. Ignore conf.
    stop_steps = []
    stop_tokens = []
    correct = []
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
        f"data/router_splits_seeds/seed{seed}/dev.jsonl",
        f"data/router_splits_seeds/seed{seed}/test.jsonl",
    )


def _dataset_paths(dataset: str) -> Tuple[str, str]:
    return (
        f"data/router_{dataset}/dev.jsonl",
        f"data/router_{dataset}/test.jsonl",
    )


def run_router(
    *,
    out_dir: str,
    seeds: List[int],
    dataset: Optional[str],
    dev_jsonl: Optional[str],
    test_jsonl: Optional[str],
    calibrator_json: Optional[str],
) -> None:
    """Run Option-B router evaluation and emit CSVs + LaTeX-ready inputs."""
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_seed_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        # Resolve input paths
        if dev_jsonl and test_jsonl:
            dev_path, test_path = dev_jsonl, test_jsonl
        elif dataset:
            dev_path, test_path = _dataset_paths(dataset)
        else:
            dev_path, test_path = _default_seed_split_paths(seed)

        out_seed = str(Path(out_dir) / f"seed{seed}")
        Path(out_seed).mkdir(parents=True, exist_ok=True)

        dev_ex = edr.read_jsonl_grouped(dev_path)
        test_ex = edr.read_jsonl_grouped(test_path)

        # Calibrator: explicit path > per-seed default > none
        if calibrator_json:
            calibrator = ConfidenceCalibrator.from_json(calibrator_json)
        else:
            calib_path = f"artifacts/calibration/platt_seed{seed}.json"
            calibrator = ConfidenceCalibrator.from_json(calib_path) if Path(calib_path).exists() else None

        rows: List[Dict[str, Any]] = []

        dev_or = oracle_metrics(dev_ex, seed=seed, calibrator=calibrator)
        test_or = oracle_metrics(test_ex, seed=seed, calibrator=calibrator)

        for B in TARGETS:
            budget_tag = f"B{B}"

            dev_fixed = edr.evaluate(dev_ex, policy="fixed", k=B, seed=seed, calibrator=calibrator)
            test_fixed = edr.evaluate(test_ex, policy="fixed", k=B, seed=seed, calibrator=calibrator)
            target_tokens = float(dev_fixed["mean_tokens"])

            # ---- Confidence (mixture)
            conf_mix = choose_conf_mix(dev_ex, target_tokens, calibrator, seed)
            if conf_mix.mode == "single":
                th = float(conf_mix.th)
                dev_conf = edr.evaluate(dev_ex, policy="conf", threshold=th, seed=seed, calibrator=calibrator)
                test_conf = edr.evaluate(test_ex, policy="conf", threshold=th, seed=seed, calibrator=calibrator)
                conf_hist = pad_hist(dev_conf.get("stop_histogram", []), T_MAX)
            else:
                assert conf_mix.low is not None and conf_mix.high is not None and conf_mix.p_low is not None
                low_w = conf_mix.low
                high_w = conf_mix.high
                dev_low = low_w["res"]
                dev_high = high_w["res"]
                th_low = float(low_w["th"])
                th_high = float(high_w["th"])

                test_low = edr.evaluate(test_ex, policy="conf", threshold=th_low, seed=seed, calibrator=calibrator)
                test_high = edr.evaluate(test_ex, policy="conf", threshold=th_high, seed=seed, calibrator=calibrator)

                dev_conf = mix_metrics(float(conf_mix.p_low), dev_low, dev_high)
                test_conf = mix_metrics(float(conf_mix.p_low), test_low, test_high)
                conf_hist = pad_hist(dev_conf.get("stop_histogram", []), T_MAX)

            # ---- Random matched (deterministic expectation)
            hist_path = str(Path(out_seed) / f"conf_hist_{budget_tag}.json")
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(conf_hist, f)
            dev_rand = random_expected_metrics(dev_ex, conf_hist)
            test_rand = random_expected_metrics(test_ex, conf_hist)

            # ---- Stability (mixture)
            stab_mix = choose_stability_mix(dev_ex, target_tokens, calibrator, seed, tier_k=B)
            if stab_mix.mode == "single":
                m = int(stab_mix.m)
                min_step = int(stab_mix.min_step)
                dev_stab = edr.evaluate(dev_ex, policy="stability", m=m, min_step=min_step, seed=seed, calibrator=calibrator)
                test_stab = edr.evaluate(test_ex, policy="stability", m=m, min_step=min_step, seed=seed, calibrator=calibrator)
            else:
                assert stab_mix.low is not None and stab_mix.high is not None and stab_mix.p_low is not None
                low_w = stab_mix.low
                high_w = stab_mix.high

                dev_low = low_w["res"]
                dev_high = high_w["res"]

                m_low, ms_low = int(low_w["m"]), int(low_w["min_step"])
                m_high, ms_high = int(high_w["m"]), int(high_w["min_step"])

                test_low = edr.evaluate(test_ex, policy="stability", m=m_low, min_step=ms_low, seed=seed, calibrator=calibrator)
                test_high = edr.evaluate(test_ex, policy="stability", m=m_high, min_step=ms_high, seed=seed, calibrator=calibrator)

                dev_stab = mix_metrics(float(stab_mix.p_low), dev_low, dev_high)
                test_stab = mix_metrics(float(stab_mix.p_low), test_low, test_high)

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
                    pack("test", "fixed", test_fixed),
                    pack("dev", "conf", dev_conf),
                    pack("test", "conf", test_conf),
                    pack("dev", "random", dev_rand),
                    pack("test", "random", test_rand),
                    pack("dev", "stability", dev_stab),
                    pack("test", "stability", test_stab),
                    pack("dev", "oracle", dev_or),
                    pack("test", "oracle", test_or),
                ]
            )

        save_summary(str(Path(out_seed) / "summary.csv"), rows)
        print(f"Wrote {out_seed}/summary.csv")
        all_seed_rows.extend([r for r in rows if r["split"] == "test"])

    if not all_seed_rows:
        raise SystemExit("No test rows produced. Check your input JSONLs.")

    # Save per-seed long-form test table
    long_path = str(Path(out_dir) / "paper_table_test_full_per_seed.csv")
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_seed_rows[0].keys()))
        w.writeheader()
        for r in all_seed_rows:
            w.writerow(r)
    print(f"Wrote {long_path}")

    # Save compact test table with mean over seeds (acc/tokens)
    from collections import defaultdict
    import statistics

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
                "acc_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "tokens_mean": statistics.mean(toks),
                "tokens_std": statistics.pstdev(toks) if len(toks) > 1 else 0.0,
                "steps_mean": statistics.mean(stps),
                "steps_std": statistics.pstdev(stps) if len(stps) > 1 else 0.0,
            }
        )

    compact_path = str(Path(out_dir) / "paper_table_test_acc_tokens.csv")
    with open(compact_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(compact_rows[0].keys()))
        w.writeheader()
        for r in compact_rows:
            w.writerow(r)
    print(f"Wrote {compact_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Dataset name (e.g., gsm8k, boolq, svamp). Used to infer data/router_<dataset>/{dev,test}.jsonl")
    ap.add_argument("--dev_jsonl", default=None, help="Path to dev jsonl (overrides --dataset inference)")
    ap.add_argument("--test_jsonl", default=None, help="Path to test jsonl (overrides --dataset inference)")
    ap.add_argument("--calibrator", default=None, help="Path to a calibrator json. If omitted, uses artifacts/calibration/platt_seed{seed}.json when available.")
    ap.add_argument("--out_dir", default=os.environ.get("ROUTER_OUTDIR", DEFAULT_OUTDIR))
    ap.add_argument("--seeds", default=None, help="Comma/space separated seeds. Overrides ROUTER_SEEDS env.")
    args = ap.parse_args()

    seeds = DEFAULT_SEEDS if args.seeds is None else _parse_seeds(args.seeds)
    run_router(
        out_dir=args.out_dir,
        seeds=seeds,
        dataset=args.dataset,
        dev_jsonl=args.dev_jsonl,
        test_jsonl=args.test_jsonl,
        calibrator_json=args.calibrator,
    )


if __name__ == "__main__":
    main()
