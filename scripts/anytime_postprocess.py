from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from src.data.judging import is_correct
from src.utils.parsing import parse_answer_and_conf

# Keep regexes consistent with training target canonicalization.
CONF_LINE_RE = re.compile(r"^\s*CONF\s*:\s*[01](?:\.\d+)?\s*$", re.IGNORECASE)
FINAL_LINE_RE = re.compile(r"^\s*####\s*.+\s*$", re.IGNORECASE)


def _strip_existing_final_and_conf(text: str) -> str:
    """Remove any trailing/embedded FINAL and CONF lines to rewrite them cleanly."""
    lines = text.splitlines()
    kept: List[str] = []
    for ln in lines:
        if FINAL_LINE_RE.match(ln):
            continue
        if CONF_LINE_RE.match(ln):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def clamp01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def approx_token_count(text: str) -> int:
    """Very rough token proxy without needing a tokenizer."""
    if not text:
        return 0
    return len(text.split())


def rewrite_raw_with_answer_conf(raw: str, answer: str, conf: Optional[float], *, round_to: int = 2) -> str:
    """Keep body, force standardized last lines."""
    body = _strip_existing_final_and_conf(raw)
    p = 0.5 if conf is None else clamp01(conf)
    if p is None:
        p = 0.5
    p_str = f"{float(p):.{round_to}f}"
    if body:
        return f"{body}\n#### {str(answer).strip()}\nCONF: {p_str}".strip()
    return f"#### {str(answer).strip()}\nCONF: {p_str}".strip()


def _compute_ttc(correct_by_t: Dict[int, bool]) -> Optional[int]:
    for t in sorted(correct_by_t.keys()):
        if correct_by_t[t]:
            return t
    return None


def postprocess_checkpoint(
    cp: Dict[str, Any],
    *,
    gold: Optional[str] = None,
    default_conf: float = 0.5,
    recompute_correct: bool = True,
) -> Dict[str, Any]:
    """In-place normalize a single checkpoint dict."""
    raw = cp.get("raw") or cp.get("raw_text") or ""

    # Ensure answer/conf exist by parsing raw.
    ans = cp.get("answer")
    conf = cp.get("conf")
    if ans is None or conf is None:
        a2, c2 = parse_answer_and_conf(raw)
        if ans is None:
            ans = a2
        if conf is None:
            conf = c2

    conf = clamp01(conf)

    cp["answer"] = ans
    cp["conf"] = default_conf if conf is None else float(conf)
    cp["parseable"] = ans is not None

    # Add approximate token features.
    cp["approx_tokens"] = approx_token_count(raw)

    # Correctness.
    if recompute_correct and (gold is not None):
        cp["correct"] = bool(is_correct(ans, gold))

    return cp


def add_step_deltas(checkpoints: List[Dict[str, Any]]) -> None:
    prev_ans: Optional[str] = None
    prev_conf: Optional[float] = None
    cum_tokens = 0
    for cp in sorted(checkpoints, key=lambda x: int(x.get("t", 0))):
        ans = cp.get("answer")
        conf = cp.get("conf")
        cp["answer_changed"] = (prev_ans is not None and ans is not None and ans != prev_ans)
        cp["conf_delta"] = None if (prev_conf is None or conf is None) else float(conf) - float(prev_conf)

        tok = int(cp.get("approx_tokens") or 0)
        cum_tokens += tok
        cp["approx_tokens_cum"] = cum_tokens

        prev_ans = ans if ans is not None else prev_ans
        prev_conf = conf if conf is not None else prev_conf


def apply_monotone_best_so_far(
    checkpoints: List[Dict[str, Any]],
    *,
    gold: str,
    prefer_high_conf: bool = True,
) -> None:
    """Prevent regressions: if a prefix step is correct, later steps inherit its final answer.
    Intended for TRAINING DATA creation.
    """

    best_correct_answer: Optional[str] = None
    best_correct_conf: Optional[float] = None
    best_correct_t: Optional[int] = None

    def _maybe_update_best(cp: Dict[str, Any]) -> None:
        nonlocal best_correct_answer, best_correct_conf, best_correct_t
        if not cp.get("parseable"):
            return
        if not bool(cp.get("correct")):
            return
        a = cp.get("answer")
        c = cp.get("conf")
        if a is None:
            return
        if best_correct_answer is None:
            best_correct_answer = a
            best_correct_conf = float(c) if c is not None else 0.5
            best_correct_t = int(cp.get("t") or 0)
            return
        if not prefer_high_conf:
            return
        cc = float(c) if c is not None else 0.5
        if best_correct_conf is None or cc > best_correct_conf:
            best_correct_answer = a
            best_correct_conf = cc
            best_correct_t = int(cp.get("t") or 0)

    for cp in sorted(checkpoints, key=lambda x: int(x.get("t", 0))):
        postprocess_checkpoint(cp, gold=gold, recompute_correct=True)
        _maybe_update_best(cp)

        if best_correct_answer is None:
            cp["best_so_far_correct"] = False
            cp["best_so_far_answer"] = None
            cp["best_so_far_conf"] = None
            cp["best_so_far_t"] = None
            continue

        cp["best_so_far_correct"] = True
        cp["best_so_far_answer"] = best_correct_answer
        cp["best_so_far_conf"] = best_correct_conf
        cp["best_so_far_t"] = best_correct_t

        # If current is not correct, override its final answer/conf (training-time).
        if not bool(cp.get("correct")):
            raw = cp.get("raw") or ""
            cp["answer"] = best_correct_answer
            cp["conf"] = float(best_correct_conf) if best_correct_conf is not None else float(cp.get("conf") or 0.5)
            cp["raw"] = rewrite_raw_with_answer_conf(raw, best_correct_answer, cp["conf"])
            cp["approx_tokens"] = approx_token_count(cp["raw"])
            cp["correct"] = True
            cp["parseable"] = True


def infer_verifier_summary(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    vc = record.get("verifier_code")
    if not isinstance(vc, dict):
        return None
    ex = vc.get("exec")
    if not isinstance(ex, dict):
        return None

    stdout = (ex.get("stdout") or "").strip()
    stderr = (ex.get("stderr") or "").strip()
    last = stdout.splitlines()[-1].strip() if stdout else ""

    return {
        "ok": bool(ex.get("ok")),
        "timeout": bool(ex.get("timeout")),
        "rejected": bool(ex.get("rejected")),
        "reason": ex.get("reason"),
        "stdout_last": last,
        "stderr": stderr,
        "verified_answer": vc.get("verified_answer"),
        "verified_correct": bool(vc.get("verified_correct")),
    }


def postprocess_trajectory_record(
    record: Dict[str, Any],
    *,
    expected_ts: Optional[Sequence[int]] = (1, 2, 3, 4),
    add_deltas: bool = True,
    monotone_best_so_far: bool = False,
    prefer_high_conf: bool = True,
    add_verifier_summary: bool = True,
) -> Dict[str, Any]:
    """Normalize + enrich a trajectory JSONL object (in-place) and return it."""

    gold = record.get("gold")
    checkpoints = record.get("checkpoints")
    if not isinstance(checkpoints, list):
        return record

    for cp in checkpoints:
        if isinstance(cp, dict):
            postprocess_checkpoint(cp, gold=gold)

    if monotone_best_so_far and isinstance(gold, str):
        apply_monotone_best_so_far(checkpoints, gold=gold, prefer_high_conf=prefer_high_conf)

    if add_deltas:
        add_step_deltas(checkpoints)

    if expected_ts is not None:
        have = {int(cp.get("t")) for cp in checkpoints if "t" in cp}
        missing = [int(t) for t in expected_ts if int(t) not in have]
        record["missing_ts"] = missing

    if isinstance(gold, str):
        correct_by_t: Dict[int, bool] = {}
        for cp in checkpoints:
            try:
                t = int(cp.get("t"))
            except Exception:
                continue
            correct_by_t[t] = bool(cp.get("correct"))
        ttc = _compute_ttc(correct_by_t)
        labels = record.get("labels")
        if not isinstance(labels, dict):
            labels = {}
        labels["correct_by_t"] = correct_by_t
        labels["ttc"] = ttc
        record["labels"] = labels

    if add_verifier_summary:
        vs = infer_verifier_summary(record)
        if vs is not None:
            record["verifier_summary"] = vs
            for cp in checkpoints:
                if isinstance(cp, dict):
                    cp["verifier_ok"] = vs.get("ok")
                    cp["verifier_timeout"] = vs.get("timeout")
                    cp["verifier_rejected"] = vs.get("rejected")
                    cp["verifier_stdout_last"] = vs.get("stdout_last")

    return record
