#!/usr/bin/env python3
"""
Evaluate depth-routing (early stopping) policies on anytime trajectories.

Policies:
  - fixed: stop at step k
  - conf: stop at first step with conf >= threshold
  - stability: stop when answer stays the same for m consecutive steps
  - random: sample stop steps from a provided histogram or match mean budget

Outputs routing-style metrics: accuracy, mean_steps/tokens, p95, flips, regressions-at-stop.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Step:
    ans: str
    conf: float
    tokens: Optional[int] = None


def normalize_answer(x: Any) -> str:
    """Normalize numeric answers to comparable strings."""
    if x is None:
        return ""
    s = str(x).strip()
    # Try to canonicalize numbers like "42.0" -> "42"
    try:
        f = float(s)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        # Keep a compact representation for non-integers
        return str(f).rstrip("0").rstrip(".")
    except Exception:
        return s


def extract_steps(example: Dict[str, Any]) -> Tuple[str, List[Step]]:
    """
    ADAPTER: Convert one JSON object into (gold_answer, steps[]).

    Modify this function to match your saved trajectory format.
    """
    gold = normalize_answer(example.get("gold") or example.get("answer") or example.get("target"))

    raw_steps = example.get("steps") or example.get("trajectory") or example.get("pred_steps")
    if raw_steps is None:
        raise ValueError("Example has no 'steps' / 'trajectory' / 'pred_steps' field.")

    steps: List[Step] = []
    for s in raw_steps:
        ans = normalize_answer(s.get("ans") or s.get("answer") or s.get("pred"))
        conf = s.get("conf") or s.get("confidence") or s.get("p_correct")
        if conf is None:
            raise ValueError("Step missing 'conf'/'confidence'/'p_correct'.")
        conf = float(conf)
        tok = s.get("tokens") or s.get("n_tokens") or s.get("tok")
        tok = int(tok) if tok is not None else None
        steps.append(Step(ans=ans, conf=conf, tokens=tok))

    if not steps:
        raise ValueError("Empty steps list.")
    return gold, steps


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def compute_prefix_tokens(steps: List[Step]) -> List[int]:
    """Return cumulative tokens up to each step (inclusive). If tokens missing, fallback to step index."""
    if all(st.tokens is not None for st in steps):
        cum = []
        running = 0
        for st in steps:
            running += int(st.tokens or 0)
            cum.append(running)
        return cum
    # fallback: 1 token unit per step
    return list(range(1, len(steps) + 1))


def policy_fixed(steps: List[Step], k: int) -> int:
    return max(1, min(k, len(steps)))


def policy_conf_threshold(steps: List[Step], threshold: float) -> int:
    for i, st in enumerate(steps, start=1):
        if st.conf >= threshold:
            return i
    return len(steps)


def policy_stability(steps: List[Step], m: int, min_step: int = 1) -> int:
    """
    Stop when the answer is unchanged for m consecutive steps (including current).
    E.g., m=2 stops when ans_t == ans_{t-1}.
    """
    m = max(1, m)
    min_step = max(1, min_step)
    last = None
    run = 0
    for i, st in enumerate(steps, start=1):
        if st.ans == last:
            run += 1
        else:
            run = 1
            last = st.ans
        if i >= min_step and run >= m:
            return i
    return len(steps)


def policy_random_from_hist(n_steps: int, hist: List[float], rng: random.Random) -> int:
    """
    hist is a probability distribution over stop steps 1..n_steps (len(hist)==n_steps).
    """
    if len(hist) != n_steps:
        raise ValueError("Histogram length must equal n_steps.")
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(hist, start=1):
        acc += p
        if r <= acc:
            return i
    return n_steps


def build_stop_histogram(stop_steps: List[int], max_steps: int) -> List[float]:
    counts = [0] * max_steps
    for s in stop_steps:
        counts[s - 1] += 1
    total = sum(counts)
    if total == 0:
        return [1.0 / max_steps] * max_steps
    return [c / total for c in counts]


def p95(values: List[float]) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = int(math.ceil(0.95 * len(xs))) - 1
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


def evaluate(
    examples: List[Dict[str, Any]],
    policy: str,
    k: int = 4,
    threshold: float = 0.8,
    m: int = 2,
    min_step: int = 1,
    random_hist_path: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    stop_steps: List[int] = []
    stop_tokens: List[int] = []
    correct: List[int] = []
    flips_any: List[int] = []
    regress_at_stop: List[int] = []

    # If random policy: load histogram if provided
    loaded_hist: Optional[List[float]] = None
    if policy == "random" and random_hist_path:
        with open(random_hist_path, "r", encoding="utf-8") as f:
            loaded_hist = json.load(f)
        if not isinstance(loaded_hist, list):
            raise ValueError("random_hist_path must be a JSON list.")

    for ex in examples:
        gold, steps = extract_steps(ex)
        cum_tokens = compute_prefix_tokens(steps)
        T = len(steps)

        if policy == "fixed":
            s = policy_fixed(steps, k=k)
        elif policy == "conf":
            s = policy_conf_threshold(steps, threshold=threshold)
        elif policy == "stability":
            s = policy_stability(steps, m=m, min_step=min_step)
        elif policy == "random":
            if loaded_hist is None:
                # default uniform if nothing provided
                hist = [1.0 / T] * T
            else:
                # If histogram length differs from current T, fallback to uniform
                hist = loaded_hist if len(loaded_hist) == T else [1.0 / T] * T
            s = policy_random_from_hist(T, hist, rng)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        s = max(1, min(s, T))
        pred = steps[s - 1].ans
        is_correct = int(pred == gold)

        # Flip metric: did answer ever change before stop?
        flips = 0
        prev = steps[0].ans
        for i in range(2, s + 1):
            if steps[i - 1].ans != prev:
                flips = 1
                break
            prev = steps[i - 1].ans

        # Regression-at-stop: was there an earlier correct step < s, but stopped answer is wrong?
        had_earlier_correct = 0
        if is_correct == 0:
            for i in range(1, s):
                if steps[i - 1].ans == gold:
                    had_earlier_correct = 1
                    break

        stop_steps.append(s)
        stop_tokens.append(cum_tokens[s - 1])
        correct.append(is_correct)
        flips_any.append(flips)
        regress_at_stop.append(had_earlier_correct)

    acc = sum(correct) / max(1, len(correct))
    mean_steps = sum(stop_steps) / max(1, len(stop_steps))
    mean_tokens = sum(stop_tokens) / max(1, len(stop_tokens))

    out = {
        "n": len(examples),
        "policy": policy,
        "k": k,
        "threshold": threshold,
        "m": m,
        "min_step": min_step,
        "seed": seed,
        "acc": acc,
        "mean_steps": mean_steps,
        "p95_steps": p95([float(x) for x in stop_steps]),
        "mean_tokens": mean_tokens,
        "p95_tokens": p95([float(x) for x in stop_tokens]),
        "flip_rate": sum(flips_any) / max(1, len(flips_any)),
        "regress_at_stop_rate": sum(regress_at_stop) / max(1, len(regress_at_stop)),
    }

    # Also output a histogram so you can budget-match random to this policy later
    max_steps = max(stop_steps) if stop_steps else 0
    out["stop_histogram"] = build_stop_histogram(stop_steps, max_steps) if max_steps > 0 else []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSONL with trajectories.")
    ap.add_argument("--policy", required=True, choices=["fixed", "conf", "stability", "random"])
    ap.add_argument("--k", type=int, default=4, help="Fixed policy stop step.")
    ap.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold for conf policy.")
    ap.add_argument("--m", type=int, default=2, help="Stability run length.")
    ap.add_argument("--min_step", type=int, default=1, help="Minimum step before stability can stop.")
    ap.add_argument("--random_hist_path", type=str, default=None, help="JSON list histogram for random policy.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="Write metrics JSON to this path.")

    args = ap.parse_args()
    examples = read_jsonl(args.data)
    res = evaluate(
        examples,
        policy=args.policy,
        k=args.k,
        threshold=args.threshold,
        m=args.m,
        min_step=args.min_step,
        random_hist_path=args.random_hist_path,
        seed=args.seed,
    )

    s = json.dumps(res, indent=2, sort_keys=True)
    print(s)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s + "\n")


if __name__ == "__main__":
    main()
