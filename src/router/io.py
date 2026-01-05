from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.data.load_datasets import normalize_answer
from src.data.judging import is_correct


TOKEN_KEYS = (
    "tokens",
    "token_count",
    "completion_tokens",
    "generated_tokens",
    "gen_tokens",
    "output_tokens",
    "n_tokens",
    "num_tokens",
    "tok",
    "approx_tokens",
    "max_new_tokens",
)


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _as_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _get_any(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


@dataclass
class Step:
    ans: str
    conf: float
    tokens: Optional[int] = None
    t: Optional[int] = None
    correct: Optional[bool] = None


@dataclass
class Trajectory:
    uid: str
    gold: str
    steps: List[Step]


def _parse_step_dict(s: Dict[str, Any], fallback_t: int, gold: str) -> Step:
    ans_raw = s.get("ans") or s.get("answer") or s.get("pred") or s.get("output")
    ans = normalize_answer(ans_raw or "")

    conf = _as_float(s.get("conf") or s.get("confidence") or s.get("p_correct"), default=0.0)
    conf = max(0.0, min(1.0, conf))

    tok = _as_int(_get_any(s, TOKEN_KEYS))

    t_raw = s.get("t", s.get("step", s.get("idx", None)))
    t = _as_int(t_raw)
    if t is None:
        t = fallback_t

    corr = s.get("correct", None)
    if corr is None:
        # compute correctness if not present
        corr = is_correct(ans, gold)

    return Step(ans=ans, conf=conf, tokens=tok, t=t, correct=bool(corr))


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _is_trajectory_line(obj: Dict[str, Any]) -> bool:
    return ("steps" in obj and isinstance(obj["steps"], list)) or (
        "trajectory" in obj and isinstance(obj["trajectory"], list)
    ) or ("pred_steps" in obj and isinstance(obj["pred_steps"], list))


def read_jsonl_grouped(path: str) -> List[Trajectory]:
    """
    Supports:
      (A) trajectory-per-line JSONL:
          {"id": "...", "gold": "...", "steps":[{...}, ...]}
      (B) flat per-step JSONL:
          each line is a step; we group by uid/id/idx and sort by t
    """
    rows = list(read_jsonl(path))
    if not rows:
        return []

    # Detect format (A)
    if _is_trajectory_line(rows[0]):
        trajs: List[Trajectory] = []
        for ex in rows:
            uid = str(ex.get("id") or ex.get("uid") or ex.get("idx") or ex.get("example_id") or "")
            gold_raw = ex.get("gold") or ex.get("answer") or ex.get("target")
            gold = normalize_answer(gold_raw or "")
            raw_steps = ex.get("steps") or ex.get("trajectory") or ex.get("pred_steps") or []
            steps: List[Step] = []
            for i, s in enumerate(raw_steps, start=1):
                steps.append(_parse_step_dict(s, fallback_t=i, gold=gold))
            trajs.append(Trajectory(uid=uid, gold=gold, steps=steps))
        return trajs

    # Format (B) flat-per-step
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        gid = r.get("uid", None)
        if gid is None:
            gid = r.get("id", None)
        if gid is None:
            gid = r.get("idx", None)
        if gid is None:
            gid = r.get("example_id", None)
        gid = str(gid)
        groups.setdefault(gid, []).append(r)

    trajs = []
    for gid, recs in groups.items():
        recs.sort(key=lambda x: _as_int(x.get("t", 0)) or 0)

        gold_raw = recs[0].get("gold", None)
        if gold_raw is None:
            gold_raw = recs[0].get("target", None)
        if gold_raw is None:
            gold_raw = recs[0].get("answer", None)
        gold = normalize_answer(gold_raw or "")

        raw_steps: List[Dict[str, Any]] = []
        for r in recs:
            raw_steps.append(
                {
                    "ans": r.get("answer") or r.get("pred") or r.get("output"),
                    "conf": r.get("conf") or r.get("confidence") or r.get("p_correct"),
                    "tokens": _get_any(r, TOKEN_KEYS),
                    "t": r.get("t", None),
                    "correct": r.get("correct", None),
                }
            )

        steps: List[Step] = []
        for i, s in enumerate(raw_steps, start=1):
            steps.append(_parse_step_dict(s, fallback_t=i, gold=gold))

        trajs.append(Trajectory(uid=gid, gold=gold, steps=steps))

    return trajs
