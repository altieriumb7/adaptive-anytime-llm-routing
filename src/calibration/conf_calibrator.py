from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _interp_piecewise(xs: List[float], ys: List[float], x: float) -> float:
    """Piecewise-linear interpolation on sorted knots."""
    if not xs:
        return x
    if len(xs) == 1:
        return float(ys[0])
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])

    lo, hi = 0, len(xs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if x < xs[mid]:
            hi = mid
        else:
            lo = mid
    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    if x1 == x0:
        return float(y0)
    w = (x - x0) / (x1 - x0)
    return float(y0 + w * (y1 - y0))


@dataclass
class PerBudgetCalibrator:
    """Supported:
      - platt: sigmoid(a*logit(p)+b)
      - temp:  sigmoid(logit(p)/T)
      - isotonic: piecewise-linear monotone mapping
    """
    method: str
    a: Optional[float] = None
    b: Optional[float] = None
    T: Optional[float] = None
    xs: Optional[List[float]] = None
    ys: Optional[List[float]] = None
    eps: float = 1e-6

    def apply(self, p: float) -> float:
        p = _clamp01(p, eps=self.eps)
        if self.method == "platt":
            a = 1.0 if self.a is None else float(self.a)
            b = 0.0 if self.b is None else float(self.b)
            x = _logit(p, eps=self.eps)
            return _sigmoid(a * x + b)
        if self.method == "temp":
            T = 1.0 if self.T is None else float(self.T)
            T = max(1e-3, T)
            x = _logit(p, eps=self.eps)
            return _sigmoid(x / T)
        if self.method == "isotonic":
            xs = self.xs or []
            ys = self.ys or []
            return float(_interp_piecewise(xs, ys, p))
        return float(p)


class ConfidenceCalibrator:
    def __init__(self, *, method: str, per_t: Dict[int, PerBudgetCalibrator], eps: float = 1e-6):
        self.method = method
        self.per_t = per_t
        self.eps = eps

    def calibrate(self, t: int, p: Optional[float]) -> float:
        if p is None:
            p = 0.5
        try:
            tt = int(t)
        except Exception:
            tt = 0
        cal = self.per_t.get(tt)
        if cal is None:
            return float(_clamp01(float(p), eps=self.eps))
        return float(cal.apply(float(p)))

    @staticmethod
    def from_json(path: str) -> "ConfidenceCalibrator":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        method = obj.get("method", "platt")
        eps = float(obj.get("eps", 1e-6))
        per_t_raw = obj.get("per_t", {})
        per_t: Dict[int, PerBudgetCalibrator] = {}
        for k, v in per_t_raw.items():
            try:
                t = int(k)
            except Exception:
                continue
            if not isinstance(v, dict):
                continue
            m = v.get("method", method)
            per_t[t] = PerBudgetCalibrator(
                method=m,
                a=v.get("a"),
                b=v.get("b"),
                T=v.get("T"),
                xs=v.get("xs"),
                ys=v.get("ys"),
                eps=eps,
            )
        return ConfidenceCalibrator(method=method, per_t=per_t, eps=eps)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"method": self.method, "eps": self.eps, "per_t": {}}
        for t, cal in sorted(self.per_t.items(), key=lambda kv: kv[0]):
            out["per_t"][str(t)] = {
                "method": cal.method,
                "a": cal.a,
                "b": cal.b,
                "T": cal.T,
                "xs": cal.xs,
                "ys": cal.ys,
            }
        return out

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

