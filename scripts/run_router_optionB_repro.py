#!/usr/bin/env python3
"""
Option-B router evaluation (Strada A / Option B): compute-matched *in expectation*.

We define compute tiers B1..B4 by fixed-depth baselines (k=1..4).
For each tier and seed, we tune each adaptive policy on DEV to match the fixed baseline's
expected compute (mean tokens). When exact matching is not attainable with a single setting,
we use a 2-point mixture (low/hi) and report expected metrics (no Monte Carlo noise).

Outputs:
  - artifacts/router_optionB_seed{seed}/summary.csv (dev+test rows)
  - artifacts/router_optionB/paper_table_test_full_per_seed.csv (test rows, long form)
  - artifacts/router_optionB/paper_table_test_acc_tokens.csv (compact)
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import scripts.eval_depth_router as edr
from src.calibration.conf_calibrator import ConfidenceCalibrator


OUTROOT = "artifacts/router_optionB"
SEEDS = [int(s) for s in os.environ.get("ROUTER_SEEDS", "0 1 2").replace(",", " ").split()]
TARGETS = [1, 2, 3, 4]  # B1..B4
T_MAX = 4              # anytime trajectory length


def make_threshold_grid() -> List[float]:
    # Dense grid + very high thresholds
    grid = [i * 0.999 / 999 for i in range(1000)]   # 0..0.999
    grid += [1.0 - 10.0 ** (-k) for k in range(2, 13)]  # 0.99, 0.999, ..., 1-1e-12
    grid += [1.0 - j * 1e-6 for j in [1,2,5,10,20,50,100,200,500,1000]]
    # clamp below 1
    grid = [max(0.0, min(1.0 - 1e-15, float(x))) for x in grid]
    return sorted(set(grid))

def random_expected_metrics(examples, stop_hist):
    stop_hist = pad_hist(stop_hist, T_MAX)
    n = len(examples)
    acc = steps = toks = 0.0
    for ex in examples:
        gold, stps = edr.extract_steps(ex)
        cum = edr.compute_prefix_tokens(stps)
        for i, p in enumerate(stop_hist, start=1):
            steps += p * i
            toks  += p * cum[i-1]
            acc   += p * (1.0 if stps[i-1].ans == gold else 0.0)
    return {"acc": acc/n, "mean_steps": steps/n, "mean_tokens": toks/n, "stop_histogram": stop_hist}


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


@dataclass
class Mix:
    mode: str  # "single" or "mix"
    # for single:
    th: Optional[float] = None
    m: Optional[int] = None
    min_step: Optional[int] = None
    # for mix:
    p_low: Optional[float] = None
    low: Optional[Dict[str, Any]] = None
    high: Optional[Dict[str, Any]] = None
    desc: str = ""


def mix_metrics(p_low: float, low: Dict[str, Any], high: Dict[str, Any]) -> Dict[str, Any]:
    # Expected values for averages (acc/mean_tokens/mean_steps), plus a mixed histogram.
    out = dict(low)
    out["acc"] = p_low * float(low["acc"]) + (1 - p_low) * float(high["acc"])
    out["mean_tokens"] = p_low * float(low["mean_tokens"]) + (1 - p_low) * float(high["mean_tokens"])
    out["mean_steps"] = p_low * float(low["mean_steps"]) + (1 - p_low) * float(high["mean_steps"])
    # stop_histogram: mix + pad to T_MAX
    h_low = pad_hist(low.get("stop_histogram", []), T_MAX)
    h_high = pad_hist(high.get("stop_histogram", []), T_MAX)
    h = [p_low * a + (1 - p_low) * b for a, b in zip(h_low, h_high)]
    out["stop_histogram"] = pad_hist(h, T_MAX)
    return out


def pick_bracket_by_tokens(cands: List[Tuple[float, Dict[str, Any]]], target: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # cands: list of (mean_tokens, res)
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
    # build bracket (by mean_tokens)
    low_wrap, high_wrap = pick_bracket_by_tokens([(mt, w) for mt, w in cands], target_tokens)

    # If exact-ish with a single threshold, take it
    best_single = min([w for _, w in cands], key=lambda w: abs(float(w["res"]["mean_tokens"]) - target_tokens))
    if abs(float(best_single["res"]["mean_tokens"]) - target_tokens) < 1e-6:
        return Mix(mode="single", th=float(best_single["th"]), desc=f"single th={best_single['th']:.6g}")

    low = low_wrap["res"]
    high = high_wrap["res"]

    mu_low = float(low["mean_tokens"])
    mu_high = float(high["mean_tokens"])
    if abs(mu_high - mu_low) < 1e-12:
        # cannot mix; fall back to closest single
        return Mix(mode="single", th=float(best_single["th"]), desc=f"fallback single th={best_single['th']:.6g}")

    # weight for LOW to hit target in expectation
    p_low = (mu_high - target_tokens) / (mu_high - mu_low)
    p_low = max(0.0, min(1.0, float(p_low)))

    return Mix(mode="mix", p_low=p_low, low=low_wrap, high=high_wrap, desc=f"mix conf p_low={p_low:.3f}")


def stability_is_trivial_fixed(m: int, min_step: int, tier_k: int) -> bool:
    # With m=1, policy_stability returns at i=min_step always => identical to fixed depth=min_step.
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

    # First try excluding trivial fixed replica; if bracketing fails badly, allow it as fallback.
    cands = eval_grid(exclude_trivial=True)
    if not cands:
        cands = eval_grid(exclude_trivial=False)

    # closest single exact?
    best_single = min([w for _, w in cands], key=lambda w: abs(float(w["res"]["mean_tokens"]) - target_tokens))
    if abs(float(best_single["res"]["mean_tokens"]) - target_tokens) < 1e-6:
        return Mix(mode="single", m=int(best_single["m"]), min_step=int(best_single["min_step"]),
                   desc=f"single stab m={best_single['m']} min_step={best_single['min_step']}")

    low_wrap, high_wrap = pick_bracket_by_tokens([(mt, w["res"]) for mt, w in cands], target_tokens)

    # If we got here, low_wrap/high_wrap are res dicts; we need their params too.
    # Find matching wrappers:
    def find_wrap(res_obj):
        for _, w in cands:
            if w["res"] is res_obj:
                return w
        # fallback (shouldn't happen): closest by tokens
        return best_single

    low_w = find_wrap(low_wrap)
    high_w = find_wrap(high_wrap)
    low = low_w["res"]
    high = high_w["res"]

    mu_low = float(low["mean_tokens"])
    mu_high = float(high["mean_tokens"])
    if abs(mu_high - mu_low) < 1e-12:
        return Mix(mode="single", m=int(best_single["m"]), min_step=int(best_single["min_step"]),
                   desc=f"fallback single stab m={best_single['m']} min_step={best_single['min_step']}")

    p_low = (mu_high - target_tokens) / (mu_high - mu_low)
    p_low = max(0.0, min(1.0, float(p_low)))

    return Mix(
        mode="mix",
        p_low=p_low,
        low=low, high=high,
        desc=f"mix stab p_low={p_low:.3f} (low m={low_w['m']},min_step={low_w['min_step']} / high m={high_w['m']},min_step={high_w['min_step']})",
    )


def oracle_metrics(examples, seed: int = 0, calibrator: Optional[ConfidenceCalibrator] = None) -> Dict[str, Any]:
    # Oracle: earliest correct step if any, else last step.
    # Compute from raw steps; ignore conf.
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


all_seed_rows: List[Dict[str, Any]] = []

for seed in SEEDS:
    dev_path = f"data/router_splits_seeds/seed{seed}/dev.jsonl"
    test_path = f"data/router_splits_seeds/seed{seed}/test.jsonl"
    out_seed = f"{OUTROOT}_seed{seed}"
    Path(out_seed).mkdir(parents=True, exist_ok=True)

    dev_ex = edr.read_jsonl_grouped(dev_path)
    test_ex = edr.read_jsonl_grouped(test_path)

    calib_path = f"artifacts/calibration/platt_seed{seed}.json"
    calibrator = ConfidenceCalibrator.from_json(calib_path) if Path(calib_path).exists() else None

    rows: List[Dict[str, Any]] = []

    # Oracle is independent of tier
    dev_or = oracle_metrics(dev_ex, seed=seed, calibrator=calibrator)
    test_or = oracle_metrics(test_ex, seed=seed, calibrator=calibrator)

    for B in TARGETS:
        budget_tag = f"B{B}"

        # Fixed baseline defines target compute on DEV
        dev_fixed = edr.evaluate(dev_ex, policy="fixed", k=B, seed=seed, calibrator=calibrator)
        test_fixed = edr.evaluate(test_ex, policy="fixed", k=B, seed=seed, calibrator=calibrator)
        target_tokens = float(dev_fixed["mean_tokens"])

        # -------- Confidence: compute-matched in expectation (2-point mixture if needed)
        conf_mix = choose_conf_mix(dev_ex, target_tokens, calibrator, seed)
        if conf_mix.mode == "single":
            th = float(conf_mix.th)
            dev_conf = edr.evaluate(dev_ex, policy="conf", threshold=th, seed=seed, calibrator=calibrator)
            test_conf = edr.evaluate(test_ex, policy="conf", threshold=th, seed=seed, calibrator=calibrator)
            conf_hist = pad_hist(dev_conf.get("stop_histogram", []), T_MAX)
        else:
            assert conf_mix.low is not None and conf_mix.high is not None and conf_mix.p_low is not None
            # low/high are *res dicts* here
            low_w = conf_mix.low
            high_w = conf_mix.high
            assert low_w is not None and high_w is not None

            dev_low = low_w["res"]
            dev_high = high_w["res"]
            th_low = float(low_w["th"])
            th_high = float(high_w["th"])

            test_low = edr.evaluate(test_ex, policy="conf", threshold=th_low, seed=seed, calibrator=calibrator)
            test_high = edr.evaluate(test_ex, policy="conf", threshold=th_high, seed=seed, calibrator=calibrator)

            dev_conf = mix_metrics(float(conf_mix.p_low), dev_low, dev_high)
            test_conf = mix_metrics(float(conf_mix.p_low), test_low, test_high)
            conf_hist = pad_hist(dev_conf.get("stop_histogram", []), T_MAX)

        # -------- Random matched: matches CONF mixture stop histogram (on DEV)
        hist_path = f"{out_seed}/conf_hist_{budget_tag}.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(conf_hist, f)
        dev_rand = random_expected_metrics(dev_ex, conf_hist)
        test_rand = random_expected_metrics(test_ex, conf_hist)

        # -------- Stability: compute-matched in expectation via 2-point mixture (avoid trivial fixed replica when possible)
        stab_mix = choose_stability_mix(dev_ex, target_tokens, calibrator, seed, tier_k=B)
        if stab_mix.mode == "single":
            m = int(stab_mix.m)
            min_step = int(stab_mix.min_step)
            dev_stab = edr.evaluate(dev_ex, policy="stability", m=m, min_step=min_step, seed=seed, calibrator=calibrator)
            test_stab = edr.evaluate(test_ex, policy="stability", m=m, min_step=min_step, seed=seed, calibrator=calibrator)
        else:
            assert stab_mix.low is not None and stab_mix.high is not None and stab_mix.p_low is not None
            dev_low = stab_mix.low
            dev_high = stab_mix.high
            # Extract params from evaluate output fields
            m_low = int(dev_low.get("m", 2)); ms_low = int(dev_low.get("min_step", 1))
            m_high = int(dev_high.get("m", 2)); ms_high = int(dev_high.get("min_step", 1))
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

        rows.extend([
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
        ])

    save_summary(f"{out_seed}/summary.csv", rows)
    print(f"Wrote {out_seed}/summary.csv")
    all_seed_rows.extend([r for r in rows if r["split"] == "test"])


# Save per-seed long-form test table
Path(OUTROOT).mkdir(parents=True, exist_ok=True)
long_path = f"{OUTROOT}/paper_table_test_full_per_seed.csv"
with open(long_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(all_seed_rows[0].keys()))
    w.writeheader()
    for r in all_seed_rows:
        w.writerow(r)
print(f"Wrote {long_path}")


# Save compact test table with mean over seeds (acc/tokens)
# (Used by existing plotting scripts)
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
    compact_rows.append({
        "policy": policy,
        "budget_tag": budget_tag,
        "acc_mean": statistics.mean(accs),
        "acc_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        "tokens_mean": statistics.mean(toks),
        "tokens_std": statistics.pstdev(toks) if len(toks) > 1 else 0.0,
        "steps_mean": statistics.mean(stps),
        "steps_std": statistics.pstdev(stps) if len(stps) > 1 else 0.0,
    })

compact_path = f"{OUTROOT}/paper_table_test_acc_tokens.csv"
with open(compact_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(compact_rows[0].keys()))
    w.writeheader()
    for r in compact_rows:
        w.writerow(r)
print(f"Wrote {compact_path}")
