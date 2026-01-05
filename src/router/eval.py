from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.calibration.conf_calibrator import ConfidenceCalibrator

from .io import Trajectory, Step
from .policies import BasePolicy


def _p95(x: List[float]) -> float:
    if not x:
        return 0.0
    return float(np.percentile(np.array(x, dtype=np.float32), 95))


def evaluate_router(
    trajs: List[Trajectory],
    policy: BasePolicy,
    calibrator: Optional[ConfidenceCalibrator] = None,
) -> Dict[str, Any]:
    if not trajs:
        return {
            "n": 0,
            "acc": 0.0,
            "mean_steps": 0.0,
            "mean_tokens": 0.0,
            "p95_steps": 0.0,
            "p95_tokens": 0.0,
            "flip_rate": 0.0,
            "regress_at_stop_rate": 0.0,
            "stop_histogram": {},
        }

    accs = []
    steps_chosen = []
    tokens_chosen = []
    flips = []
    regress = []
    stop_hist = {}

    for tr in trajs:
        if not tr.steps:
            continue

        # optional per-step calibration (IMPORTANT: uses st.t)
        if calibrator is not None:
            new_steps = []
            for st in tr.steps:
                new_steps.append(
                    Step(
                        ans=st.ans,
                        conf=calibrator.calibrate(st.t, st.conf),
                        tokens=st.tokens,
                        t=st.t,
                        correct=st.correct,
                    )
                )
            tr = Trajectory(uid=tr.uid, gold=tr.gold, steps=new_steps)

        k = int(policy.choose_step(tr))
        k = max(1, min(k, len(tr.steps)))
        st = tr.steps[k - 1]

        is_ok = bool(st.correct)
        accs.append(1.0 if is_ok else 0.0)
        steps_chosen.append(float(k))
        tok = sum((s.tokens or 0) for s in tr.steps[:k])
        tokens_chosen.append(float(tok))

        # flip: chosen answer differs from final answer
        flips.append(1.0 if st.ans != tr.steps[-1].ans else 0.0)

        # regress at stop: there exists earlier correct but chosen is incorrect
        any_prev_correct = any(bool(s.correct) for s in tr.steps[:k])
        regress.append(1.0 if (any_prev_correct and not is_ok) else 0.0)

        stop_hist[str(k)] = stop_hist.get(str(k), 0) + 1

    n = len(accs)
    if n == 0:
        return {
            "n": 0,
            "acc": 0.0,
            "mean_steps": 0.0,
            "mean_tokens": 0.0,
            "p95_steps": 0.0,
            "p95_tokens": 0.0,
            "flip_rate": 0.0,
            "regress_at_stop_rate": 0.0,
            "stop_histogram": {},
        }

    return {
        "n": int(n),
        "acc": float(np.mean(accs)),
        "mean_steps": float(np.mean(steps_chosen)),
        "mean_tokens": float(np.mean(tokens_chosen)),
        "p95_steps": _p95(steps_chosen),
        "p95_tokens": _p95(tokens_chosen),
        "flip_rate": float(np.mean(flips)),
        "regress_at_stop_rate": float(np.mean(regress)),
        "stop_histogram": stop_hist,
    }
