from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .features import FEATURE_NAMES, extract_prefix_features
from .io import Trajectory
from .logreg import LogisticRegressionModel


class BasePolicy:
    def choose_step(self, traj: Trajectory) -> int:
        raise NotImplementedError


@dataclass
class FixedPolicy(BasePolicy):
    k: int

    def choose_step(self, traj: Trajectory) -> int:
        n = len(traj.steps)
        return max(1, min(int(self.k), n))


@dataclass
class ConfThresholdPolicy(BasePolicy):
    threshold: float

    def choose_step(self, traj: Trajectory) -> int:
        thr = float(self.threshold)
        for i, st in enumerate(traj.steps, start=1):
            if float(st.conf) >= thr:
                return i
        return len(traj.steps)


@dataclass
class StabilityPolicy(BasePolicy):
    m: int = 2
    min_step: int = 1

    def choose_step(self, traj: Trajectory) -> int:
        m = max(1, int(self.m))
        min_step = max(1, int(self.min_step))

        last = None
        run = 0
        for i, st in enumerate(traj.steps, start=1):
            if st.ans == last:
                run += 1
            else:
                run = 1
                last = st.ans
            if i >= min_step and run >= m:
                return i
        return len(traj.steps)


@dataclass
class LearnedStopPolicy(BasePolicy):
    model: LogisticRegressionModel
    threshold: float = 0.5

    def choose_step(self, traj: Trajectory) -> int:
        thr = float(self.threshold)
        n = len(traj.steps)
        for i in range(1, n + 1):
            x = extract_prefix_features(traj, i).reshape(1, -1)
            p_stop = float(self.model.predict_proba(x)[0])
            if p_stop >= thr:
                return i
        return n


@dataclass
class ExpectedImprovementPolicy(BasePolicy):
    model_now: LogisticRegressionModel
    model_next: LogisticRegressionModel
    lambda_cost: float = 0.0
    gain_margin: float = 0.0

    def choose_step(self, traj: Trajectory) -> int:
        n = len(traj.steps)
        if n == 0:
            return 1

        lam = float(self.lambda_cost)
        margin = float(self.gain_margin)

        # sequential decision
        for i in range(1, n + 1):
            if i == n:
                return n

            x = extract_prefix_features(traj, i).reshape(1, -1)
            p_now = float(self.model_now.predict_proba(x)[0])
            p_next = float(self.model_next.predict_proba(x)[0])
            gain = p_next - p_now

            # cost of taking another step (approx = tokens of next step)
            tok_next = traj.steps[i].tokens  # i is next step index (0-based)
            cost_next = float(tok_next if tok_next is not None else 0.0)

            # stop rule
            if gain <= margin + lam * cost_next:
                return i

        return n


def load_policy_from_json(policy: str, model_path: str, **kwargs: Any) -> BasePolicy:
    """
    policy:
      - "learned": expects {"weights","bias","mean","std","feature_names"}
      - "ei": expects {"model_now":{...}, "model_next":{...}}
    """
    with open(model_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if policy == "learned":
        model = LogisticRegressionModel.from_json(obj)
        thr = float(kwargs.get("threshold", 0.5))
        return LearnedStopPolicy(model=model, threshold=thr)

    if policy == "ei":
        model_now = LogisticRegressionModel.from_json(obj["model_now"])
        model_next = LogisticRegressionModel.from_json(obj["model_next"])
        lam = float(kwargs.get("lambda_cost", 0.0))
        margin = float(kwargs.get("gain_margin", 0.0))
        return ExpectedImprovementPolicy(model_now=model_now, model_next=model_next, lambda_cost=lam, gain_margin=margin)

    raise ValueError(f"Unknown policy '{policy}' for load_policy_from_json")
