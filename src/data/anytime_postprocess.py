from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils.parsing import parse_answer_and_conf

# Keep this list shared everywhere.
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

def clamp01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return max(0.0, min(1.0, x))

def get_tokens(step: Dict[str, Any]) -> Optional[int]:
    for k in TOKEN_KEYS:
        if k in step and step[k] is not None:
            try:
                return int(step[k])
            except Exception:
                return None
    return None

def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    ans = str(ans).strip()
    ans = ans.replace(",", "")
    ans = ans.rstrip(".")
    return ans

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")

def is_correct(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p == g:
        return True
    if _NUM_RE.match(p) and _NUM_RE.match(g):
        try:
            return float(p) == float(g)
        except ValueError:
            return False
    return False

def add_step_deltas(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prev_ans = None
    prev_conf = None
    prev_tokens_cum = 0

    out = []
    for i, st in enumerate(steps, start=1):
        ans = st.get("ans")
        conf = clamp01(st.get("conf"))
        tok = get_tokens(st)
        tok_step = int(tok) if tok is not None else 0
        tok_cum = prev_tokens_cum + tok_step

        out_st = dict(st)
        out_st["t"] = st.get("t", i)
        out_st["conf"] = conf
        out_st["tokens_step"] = tok_step
        out_st["tokens_cum"] = tok_cum

        if prev_ans is None:
            out_st["answer_changed"] = False
            out_st["conf_delta"] = 0.0
        else:
            out_st["answer_changed"] = (normalize_answer(ans) != normalize_answer(prev_ans))
            if conf is None or prev_conf is None:
                out_st["conf_delta"] = 0.0
            else:
                out_st["conf_delta"] = float(conf - prev_conf)

        prev_ans = ans
        prev_conf = conf
        prev_tokens_cum = tok_cum
        out.append(out_st)

    return out

def postprocess_checkpoint(example: Dict[str, Any], best_so_far: bool = False) -> Dict[str, Any]:
    """
    Ensures each step has ans/conf, clamps conf, adds deltas/tokens.
    If best_so_far=True, enforce "monotonic by construction":
      once a correct answer appears, propagate it forward (and its conf) as target.
    """
    gold = example.get("gold") or example.get("answer") or example.get("target")
    gold = normalize_answer(gold or "")

    raw_steps = example.get("steps") or example.get("trajectory") or example.get("pred_steps") or []
    steps: List[Dict[str, Any]] = []
    for i, s in enumerate(raw_steps, start=1):
        ans = s.get("ans") or s.get("answer") or s.get("pred") or s.get("output")
        conf = s.get("conf")
        if conf is None:
            conf = s.get("confidence")
        if conf is None:
            conf = s.get("p_correct")

        # If the model output is a single string, try parse "ANSWER:/CONF:"
        if (ans is None or conf is None) and isinstance(s.get("text", None), str):
            parsed = parse_answer_and_conf(s["text"])
            if parsed is not None:
                ans, conf = parsed

        steps.append(
            {
                "t": s.get("t", i),
                "ans": ans,
                "conf": clamp01(conf),
                "tokens": get_tokens(s),
            }
        )

    steps = add_step_deltas(steps)

    if best_so_far:
        best_idx = None
        for i, st in enumerate(steps):
            if is_correct(st.get("ans"), gold):
                best_idx = i
                break
        if best_idx is not None:
            best_ans = steps[best_idx].get("ans")
            best_conf = steps[best_idx].get("conf")
            for j in range(best_idx + 1, len(steps)):
                steps[j]["ans"] = best_ans
                if best_conf is not None:
                    steps[j]["conf"] = best_conf

    out = dict(example)
    out["gold"] = gold
    out["steps"] = steps
    return out
