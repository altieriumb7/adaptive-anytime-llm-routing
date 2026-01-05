from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .io import Step, Trajectory


FEATURE_NAMES = [
    "conf",
    "conf_delta",
    "answer_changed",
    "tokens_step",
    "tokens_cum",
    "t",
]


def _tokens_step(st: Step) -> float:
    # if missing, treat as 0 (so mean_tokens reflects what you actually logged)
    return float(st.tokens if st.tokens is not None else 0)


def _tokens_cum(steps: List[Step], i: int) -> float:
    # i is 1-indexed
    return float(sum((_tokens_step(s) for s in steps[:i])))


def extract_prefix_features(traj: Trajectory, i: int) -> np.ndarray:
    """
    Prefix-only features at step i (1-indexed).
    """
    steps = traj.steps
    if i < 1:
        i = 1
    if i > len(steps):
        i = len(steps)

    st = steps[i - 1]
    conf = float(st.conf)

    if i == 1:
        conf_delta = 0.0
        answer_changed = 0.0
    else:
        prev = steps[i - 2]
        conf_delta = float(st.conf - prev.conf)
        answer_changed = 1.0 if st.ans != prev.ans else 0.0

    tok_step = _tokens_step(st)
    tok_cum = _tokens_cum(steps, i)
    t = float(st.t if st.t is not None else i)

    return np.array([conf, conf_delta, answer_changed, tok_step, tok_cum, t], dtype=np.float32)


def build_learned_stop_dataset(trajs: List[Trajectory], lambda_cost: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds (X,y) for learned stopping:
      choose t* = argmax_t (correct_t - lambda * tokens_cum_t)
      include prefixes i<=t* with label 1 only at i==t*
    """
    Xs = []
    ys = []

    for tr in trajs:
        steps = tr.steps
        if not steps:
            continue

        # compute best utility step t*
        best_i = 1
        best_u = -1e9
        tok_cum = 0.0
        for i, st in enumerate(steps, start=1):
            tok_cum += float(st.tokens if st.tokens is not None else 0)
            corr = 1.0 if bool(st.correct) else 0.0
            u = corr - float(lambda_cost) * tok_cum
            if u > best_u:
                best_u = u
                best_i = i

        # add prefix instances i<=best_i
        for i in range(1, best_i + 1):
            Xs.append(extract_prefix_features(tr, i))
            ys.append(1.0 if i == best_i else 0.0)

    if not Xs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.stack(Xs, axis=0), np.array(ys, dtype=np.float32)


def build_expected_improvement_dataset(trajs: List[Trajectory]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds (X, y_now, y_next) where:
      X = features at i (prefix-only)
      y_now = correct at i
      y_next = correct at i+1 (only where i+1 exists)
    """
    Xs = []
    y_now = []
    y_next = []

    for tr in trajs:
        steps = tr.steps
        if len(steps) < 2:
            continue
        for i in range(1, len(steps)):  # i in [1..n-1]
            Xs.append(extract_prefix_features(tr, i))
            y_now.append(1.0 if bool(steps[i - 1].correct) else 0.0)
            y_next.append(1.0 if bool(steps[i].correct) else 0.0)

    if not Xs:
        zX = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        zy = np.zeros((0,), dtype=np.float32)
        return zX, zy, zy

    return np.stack(Xs, axis=0), np.array(y_now, dtype=np.float32), np.array(y_next, dtype=np.float32)
