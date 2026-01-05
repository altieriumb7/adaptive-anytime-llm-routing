#!/usr/bin/env python3
"""
Fit a per-budget confidence calibrator.

Input JSONL can be:
  (A) flat per-step rows: keys t, conf, correct (recommended)
      e.g., data/router_splits/dev.jsonl
  (B) trajectory rows: checkpoints[] with t/conf/correct
      e.g., data/anytime_gsm8k_train_v2.jsonl

Methods:
  - platt: sigmoid(a*logit(p)+b)
  - temp:  sigmoid(logit(p)/T)
  - isotonic: monotone piecewise-linear mapping
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.calibration.conf_calibrator import ConfidenceCalibrator, PerBudgetCalibrator


def _clamp01(p: float, eps: float = 1e-6) -> float:
    if p is None:
        return 0.5
    try:
        p = float(p)
    except Exception:
        return 0.5
    if math.isnan(p) or math.isinf(p):
        return 0.5
    if p < eps:
        return eps
    if p > 1.0 - eps:
        return 1.0 - eps
    return p


def _logit(p: float, eps: float = 1e-6) -> float:
    p = _clamp01(p, eps=eps)
    return math.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temp(p: List[float], y: List[int], *, eps: float = 1e-6) -> float:
    x = np.array([_logit(pi, eps=eps) for pi in p], dtype=np.float64)
    yv = np.array(y, dtype=np.float64)
    Ts = np.exp(np.linspace(math.log(0.05), math.log(10.0), 120))
    best_T, best_loss = 1.0, float("inf")
    for T in Ts:
        pc = sigmoid(x / T)
        loss = nll(pc, yv)
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    return best_T


def fit_platt(p: List[float], y: List[int], *, eps: float = 1e-6, l2: float = 1e-3) -> Tuple[float, float]:
    x = np.array([_logit(pi, eps=eps) for pi in p], dtype=np.float64)
    yv = np.array(y, dtype=np.float64)

    a, b = 1.0, 0.0
    for _ in range(50):
        z = a * x + b
        pc = sigmoid(z)

        diff = (pc - yv)
        g_a = float(np.mean(diff * x) + l2 * a)
        g_b = float(np.mean(diff) + l2 * b)

        w = pc * (1.0 - pc)
        h_aa = float(np.mean(w * x * x) + l2)
        h_ab = float(np.mean(w * x))
        h_bb = float(np.mean(w) + l2)

        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (-h_ab * g_a + h_aa * g_b) / det

        a_new = a - step_a
        b_new = b - step_b
        if abs(a_new - a) < 1e-6 and abs(b_new - b) < 1e-6:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    return float(a), float(b)


def fit_isotonic(p: List[float], y: List[int]) -> Tuple[List[float], List[float]]:
    xs = np.array(p, dtype=np.float64)
    ys = np.array(y, dtype=np.float64)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    v = ys.tolist()
    w = [1.0] * len(v)

    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1] + 1e-12:
            i += 1
            continue
        tot_w = w[i] + w[i + 1]
        tot_v = (w[i] * v[i] + w[i + 1] * v[i + 1]) / tot_w
        v[i] = tot_v
        w[i] = tot_w
        del v[i + 1]
        del w[i + 1]
        if i > 0:
            i -= 1

    yhat = np.empty(len(xs), dtype=np.float64)
    idx = 0
    for val, wt in zip(v, w):
        cnt = int(round(wt))
        yhat[idx: idx + cnt] = val
        idx += cnt
    if idx != len(xs):
        yhat[:] = np.interp(np.linspace(0, 1, len(xs)), np.linspace(0, 1, len(xs)), yhat)

    knots_x: List[float] = []
    knots_y: List[float] = []
    j = 0
    while j < len(xs):
        k = j
        while k + 1 < len(xs) and xs[k + 1] == xs[j]:
            k += 1
        knots_x.append(float(xs[j]))
        knots_y.append(float(np.mean(yhat[j: k + 1])))
        j = k + 1

    return knots_x, knots_y


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_pairs(path: str) -> DefaultDict[int, List[Tuple[float, int]]]:
    per_t: DefaultDict[int, List[Tuple[float, int]]] = defaultdict(list)
    for obj in tqdm(iter_jsonl(path), desc="read"):
        # flat rows
        if "t" in obj and ("conf" in obj or "confidence" in obj or "p_correct" in obj):
            try:
                t = int(obj["t"])
            except Exception:
                continue
            conf = obj.get("conf", obj.get("confidence", obj.get("p_correct", None)))
            corr = obj.get("correct", None)
            if corr is None:
                continue
            try:
                c = float(conf)
            except Exception:
                c = 0.5
            per_t[t].append((_clamp01(c), int(bool(corr))))
            continue

        # trajectory rows
        cps = obj.get("checkpoints")
        if isinstance(cps, list):
            for cp in cps:
                if not isinstance(cp, dict) or "t" not in cp:
                    continue
                try:
                    t = int(cp["t"])
                except Exception:
                    continue
                conf = cp.get("conf", None)
                corr = cp.get("correct", None)
                if corr is None:
                    continue
                try:
                    c = float(conf)
                except Exception:
                    c = 0.5
                per_t[t].append((_clamp01(c), int(bool(corr))))
    return per_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--method", choices=["platt", "temp", "isotonic"], default="platt")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--min_points", type=int, default=200)
    args = ap.parse_args()

    pairs_by_t = collect_pairs(args.in_path)
    if not pairs_by_t:
        raise SystemExit("No usable (t, conf, correct) pairs found.")

    per_t_cal: Dict[int, PerBudgetCalibrator] = {}

    print("\n=== Fit confidence calibrator ===")
    for t in sorted(pairs_by_t.keys()):
        pairs = pairs_by_t[t]
        if len(pairs) < args.min_points:
            print(f"t={t}: skip (n={len(pairs)} < {args.min_points})")
            continue
        p = [pp for pp, _ in pairs]
        y = [yy for _, yy in pairs]

        if args.method == "temp":
            T = fit_temp(p, y, eps=args.eps)
            cal = PerBudgetCalibrator(method="temp", T=T, eps=args.eps)
            x = np.array([_logit(pi, eps=args.eps) for pi in p], dtype=np.float64)
            print(f"t={t}: method=temp T={T:.4f} NLL={nll(sigmoid(x/T), np.array(y, float)):.4f}")

        elif args.method == "platt":
            a, b = fit_platt(p, y, eps=args.eps)
            cal = PerBudgetCalibrator(method="platt", a=a, b=b, eps=args.eps)
            x = np.array([_logit(pi, eps=args.eps) for pi in p], dtype=np.float64)
            print(f"t={t}: method=platt a={a:.4f} b={b:.4f} NLL={nll(sigmoid(a*x+b), np.array(y, float)):.4f}")

        else:
            xs, ys = fit_isotonic(p, y)
            cal = PerBudgetCalibrator(method="isotonic", xs=xs, ys=ys, eps=args.eps)
            pc = np.array([np.interp(pi, xs, ys, left=ys[0], right=ys[-1]) for pi in p], dtype=np.float64)
            print(f"t={t}: method=isotonic knots={len(xs)} NLL={nll(pc, np.array(y, float)):.4f}")

        per_t_cal[t] = cal

    calibrator = ConfidenceCalibrator(method=args.method, per_t=per_t_cal, eps=args.eps)
    calibrator.save_json(args.out_path)
    print(f"\nSaved calibrator to: {args.out_path}")


if __name__ == "__main__":
    main()
