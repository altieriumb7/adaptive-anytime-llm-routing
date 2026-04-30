#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml  # pyyaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PDF_METADATA = {
    "Title": "paper-artifact-figure",
    "Author": "artifact-pipeline",
    "Subject": "deterministic-figure-export",
    "Keywords": "reproducible",
    "Creator": "scripts/make_paper_artifacts.py",
    "Producer": "matplotlib",
    # Fixed timestamps to avoid run-to-run hash drift from embedded dates.
    "CreationDate": datetime(2026, 1, 1, 0, 0, 0),
    "ModDate": datetime(2026, 1, 1, 0, 0, 0),
}


def save_png_pdf(base_path_no_ext: str, dpi: int = 200) -> None:
    """Save .png and .pdf variants with stable metadata for reproducible hashes."""
    plt.savefig(f"{base_path_no_ext}.png", dpi=dpi)
    plt.savefig(f"{base_path_no_ext}.pdf", metadata=_PDF_METADATA)


def clamp01(p: float, eps: float = 1e-6) -> float:
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


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def brier(confs: List[float], ys: List[int]) -> float:
    if not confs:
        return float("nan")
    p = np.asarray(confs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def ece(confs: List[float], ys: List[int], n_bins: int = 10) -> float:
    if not confs:
        return float("nan")
    p = np.asarray(confs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y[m]))
        conf = float(np.mean(p[m]))
        e += abs(acc - conf) * float(np.mean(m))
    return float(e)


def reliability_bins(confs: List[float], ys: List[int], n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(confs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    bin_cnt = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        bin_cnt[i] = int(np.sum(m))
        bin_conf[i] = float(np.mean(p[m]))
        bin_acc[i] = float(np.mean(y[m]))
    return bin_conf, bin_acc, bin_cnt


def risk_coverage_curve(confs: List[float], ys: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coverage, risk=1-acc) arrays by sorting by confidence desc."""
    if not confs:
        return np.array([]), np.array([])
    p = np.asarray(confs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    order = np.argsort(-p)
    y = y[order]
    cum = np.cumsum(y)
    idx = np.arange(1, len(y) + 1)
    acc = cum / idx
    risk = 1.0 - acc
    coverage = idx / float(len(y))
    return coverage, risk


def parse_acc_tokens(cell: str) -> Optional[Tuple[float, float]]:
    """Parse strings like '0.6505 (96)' into (acc, tokens)."""
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None
    try:
        acc_part, rest = s.split("(", 1)
        acc = float(acc_part.strip())
        tok_str = rest.split(")", 1)[0].strip()
        tok = float(tok_str)
        return acc, tok
    except Exception:
        return None


def load_router_table(path: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Return budget_tag -> policy -> (acc, tokens). Supports legacy and canonical CSVs."""
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        canonical = {"policy", "budget_tag", "acc_mean", "tokens_mean"}.issubset(fields)

        for row in reader:
            budget_tag = row.get("budget_tag") or row.get("budget") or row.get("tag")
            if not budget_tag:
                continue
            out.setdefault(budget_tag, {})

            if canonical:
                pol = str(row.get("policy", "")).strip()
                if not pol:
                    continue
                try:
                    acc = float(row.get("acc_mean"))
                    tok = float(row.get("tokens_mean"))
                except (TypeError, ValueError):
                    continue
                out[budget_tag][pol] = (acc, tok)
                continue

            for k, v in row.items():
                if k == "budget_tag":
                    continue
                parsed = parse_acc_tokens(v)
                if parsed is None:
                    continue
                out[budget_tag][k] = parsed
    return out


@dataclass
class Preds:
    conf_by_t: Dict[int, List[float]]
    y_by_t: Dict[int, List[int]]
    uid_to_correct_ts: Dict[str, List[int]]


def load_preds(path: str, budgets: List[int]) -> Preds:
    conf_by_t: Dict[int, List[float]] = {t: [] for t in budgets}
    y_by_t: Dict[int, List[int]] = {t: [] for t in budgets}
    uid_to_correct_ts: Dict[str, List[int]] = {}
    for obj in iter_jsonl(path):
        try:
            t = int(obj.get("t"))
        except Exception:
            continue
        if t not in conf_by_t:
            continue
        uid = str(obj.get("uid", ""))
        conf = clamp01(obj.get("conf", 0.5))
        corr_raw = obj.get("correct", 0)
        y = int(bool(corr_raw)) if isinstance(corr_raw, bool) else int(corr_raw)
        y = 1 if y != 0 else 0

        conf_by_t[t].append(float(conf))
        y_by_t[t].append(int(y))

        if uid:
            uid_to_correct_ts.setdefault(uid, [])
            if y == 1:
                uid_to_correct_ts[uid].append(t)

    return Preds(conf_by_t=conf_by_t, y_by_t=y_by_t, uid_to_correct_ts=uid_to_correct_ts)


def compute_ttc(uid_to_correct_ts: Dict[str, List[int]]) -> List[Optional[int]]:
    ttc: List[Optional[int]] = []
    for _, ts in uid_to_correct_ts.items():
        ttc.append(int(min(ts)) if ts else None)
    return ttc


def ttc_cdf(ttc_list: List[Optional[int]], budgets: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(ttc_list) if ttc_list else 0
    if n == 0:
        return np.array([]), np.array([])
    xs = np.array(sorted(budgets), dtype=np.int64)
    ys = []
    for t in xs:
        ys.append(sum((v is not None and v <= t) for v in ttc_list) / float(n))
    return xs.astype(np.float64), np.array(ys, dtype=np.float64)


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def fig_router_pareto(router: Dict[str, Dict[str, Tuple[float, float]]], out_dir: str) -> None:
    ensure_dir(out_dir)
    # combined
    plt.figure()
    for budget_tag, policy_map in router.items():
        xs, ys = [], []
        for _, (acc, tok) in policy_map.items():
            xs.append(tok)
            ys.append(acc)
        if xs:
            plt.scatter(xs, ys, label=budget_tag)
    plt.xlabel("Mean tokens")
    plt.ylabel("Accuracy")
    plt.title("Router trade-off (all budgets)")
    plt.legend()
    plt.tight_layout()
    save_png_pdf(os.path.join(out_dir, "router_pareto_all"), dpi=200)
    plt.close()

    # per budget
    for budget_tag, policy_map in router.items():
        plt.figure()
        for policy, (acc, tok) in policy_map.items():
            plt.scatter([tok], [acc])
            plt.annotate(policy, (tok, acc), textcoords="offset points", xytext=(5, 5))
        plt.xlabel("Mean tokens")
        plt.ylabel("Accuracy")
        plt.title(f"Router trade-off ({budget_tag})")
        plt.tight_layout()
        safe = budget_tag.replace("/", "_")
        save_png_pdf(os.path.join(out_dir, f"router_pareto_{safe}"), dpi=200)
        plt.close()


def fig_reliability(models: List[Tuple[str, Preds]], budgets: List[int], out_dir: str, n_bins: int = 10) -> List[Dict[str, Any]]:
    ensure_dir(out_dir)
    rows: List[Dict[str, Any]] = []
    for t in budgets:
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle="--")
        for name, preds in models:
            confs = preds.conf_by_t.get(t, [])
            ys = preds.y_by_t.get(t, [])
            if not confs:
                continue
            bc, ba, _ = reliability_bins(confs, ys, n_bins=n_bins)
            m = ~np.isnan(bc) & ~np.isnan(ba)
            plt.plot(bc[m], ba[m], marker="o", label=name)
            rows.append({
                "model": name,
                "t": t,
                "n": len(confs),
                "brier": brier(confs, ys),
                "ece": ece(confs, ys, n_bins=n_bins),
            })
        plt.xlabel("Mean confidence (bin)")
        plt.ylabel("Accuracy (bin)")
        plt.title(f"Reliability diagram (t={t})")
        plt.legend()
        plt.tight_layout()
        save_png_pdf(os.path.join(out_dir, f"reliability_t{t}"), dpi=200)
        plt.close()
    return rows


def fig_risk_coverage(models: List[Tuple[str, Preds]], budgets: List[int], out_dir: str) -> None:
    ensure_dir(out_dir)
    for t in budgets:
        plt.figure()
        for name, preds in models:
            confs = preds.conf_by_t.get(t, [])
            ys = preds.y_by_t.get(t, [])
            if not confs:
                continue
            cov, risk = risk_coverage_curve(confs, ys)
            if cov.size == 0:
                continue
            plt.plot(cov, risk, label=name)
        plt.xlabel("Coverage (fraction kept)")
        plt.ylabel("Risk (1 - accuracy)")
        plt.title(f"Risk–Coverage (t={t})")
        plt.legend()
        plt.tight_layout()
        save_png_pdf(os.path.join(out_dir, f"risk_coverage_t{t}"), dpi=200)
        plt.close()


def fig_ttc(models: List[Tuple[str, Preds]], budgets: List[int], out_dir: str) -> List[Dict[str, Any]]:
    ensure_dir(out_dir)
    plt.figure()
    rows: List[Dict[str, Any]] = []
    for name, preds in models:
        ttc_list = compute_ttc(preds.uid_to_correct_ts)
        xs, ys = ttc_cdf(ttc_list, budgets)
        if xs.size == 0:
            continue
        plt.plot(xs, ys, marker="o", label=name)

        solved = [t for t in ttc_list if t is not None]
        n = len(ttc_list)
        rows.append({
            "model": name,
            "n": n,
            "solved_pct": (len(solved) / float(n)) if n else float("nan"),
            "mean_ttc_solved": float(np.mean(solved)) if solved else float("nan"),
            "median_ttc_solved": float(np.median(solved)) if solved else float("nan"),
        })

    plt.xlabel("Budget t (checkpoint index)")
    plt.ylabel("Solved by ≤ t (fraction)")
    plt.title("TTC CDF")
    plt.legend()
    plt.tight_layout()
    save_png_pdf(os.path.join(out_dir, "ttc_cdf"), dpi=200)
    plt.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/paper.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = cfg.get("output_dir", "artifacts/paper")
    figs_dir = os.path.join(output_dir, "figures")
    tabs_dir = os.path.join(output_dir, "tables")
    ensure_dir(figs_dir)
    ensure_dir(tabs_dir)

    budgets = [int(x) for x in cfg.get("budgets", [1, 2, 3, 4])]
    n_bins = int(cfg.get("n_bins", 10))

    # Router
    router_csv = cfg.get("router_csv")
    if router_csv and os.path.exists(router_csv):
        router = load_router_table(router_csv)
        fig_router_pareto(router, figs_dir)
        long_rows = []
        for budget_tag, policy_map in router.items():
            for policy, (acc, tok) in policy_map.items():
                long_rows.append({"budget_tag": budget_tag, "policy": policy, "acc": acc, "tokens": tok})
        save_csv(os.path.join(tabs_dir, "router_points.csv"), long_rows, ["budget_tag", "policy", "acc", "tokens"])
    elif router_csv:
        print(f"[WARN] router_csv not found: {router_csv}")

    # Models
    models_cfg = cfg.get("models", [])
    models: List[Tuple[str, Preds]] = []
    for m in models_cfg:
        name = m.get("name")
        path = m.get("preds_jsonl")
        if not name or not path:
            continue
        if not os.path.exists(path):
            print(f"[WARN] preds_jsonl not found for {name}: {path}")
            continue
        models.append((name, load_preds(path, budgets)))

    if not models:
        print("[INFO] No models loaded. Router Pareto (if provided) is still generated.")
        return

    # TTC
    ttc_rows = fig_ttc(models, budgets, figs_dir)
    if ttc_rows:
        save_csv(os.path.join(tabs_dir, "ttc_summary.csv"), ttc_rows,
                 ["model", "n", "solved_pct", "mean_ttc_solved", "median_ttc_solved"])

    # Reliability + metrics
    cal_rows = fig_reliability(models, budgets, figs_dir, n_bins=n_bins)
    if cal_rows:
        save_csv(os.path.join(tabs_dir, "calibration_metrics.csv"), cal_rows, ["model", "t", "n", "brier", "ece"])

    # Risk–coverage
    fig_risk_coverage(models, budgets, figs_dir)

    print(f"Saved paper artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
