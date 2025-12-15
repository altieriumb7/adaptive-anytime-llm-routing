from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Optional, Tuple

from src.utils.parsing import parse_answer_and_conf

CONF_LINE_RE = re.compile(r"^\s*CONF\s*:\s*[01](?:\.\d+)?\s*$", re.IGNORECASE)
FINAL_LINE_RE = re.compile(r"^\s*####\s*.+\s*$", re.IGNORECASE)


def _strip_existing_final_and_conf(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for ln in lines:
        if FINAL_LINE_RE.match(ln):
            continue
        if CONF_LINE_RE.match(ln):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def canonicalize_response(raw: str, answer: Optional[str], conf: Optional[float]) -> Optional[str]:
    """
    Make the training target stable:
    - keep reasoning, but force standardized last lines:
        #### <answer>
        CONF: <p>
    Returns None if no answer.
    """
    if answer is None:
        return None
    body = _strip_existing_final_and_conf(raw).strip()
    p = 0.75 if conf is None else max(0.0, min(1.0, conf))
    # round to 2 decimals for stability
    p_str = f"{p:.2f}"
    if body:
        return f"{body}\n#### {answer}\nCONF: {p_str}"
    return f"#### {answer}\nCONF: {p_str}"


def make_messages(problem: str, budget_t: int) -> List[Dict[str, str]]:
    """
    Chat-style messages for instruct models (Qwen/Llama/Mistral).
    We'll use tokenizer.apply_chat_template downstream.
    """
    budget_desc = {
        1: "BUDGET=1 (draft, very short)",
        2: "BUDGET=2 (re-solve, medium)",
        3: "BUDGET=3 (verify-style, structured)",
        4: "BUDGET=4 (repair/final, full)",
    }.get(budget_t, f"BUDGET={budget_t}")

    system = (
        "You are a careful math solver. "
        "You must follow the required output format exactly."
    )
    user = (
        f"{budget_desc}\n\n"
        "Solve the problem.\n\n"
        "Output format MUST be:\n"
        "- brief reasoning (can be short)\n"
        "- final line: #### <final_answer>\n"
        "- confidence line: CONF: <number between 0 and 1>\n\n"
        f"PROBLEM:\n{problem}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass
class SFTExample:
    uid: str
    budget_t: int
    messages: List[Dict[str, str]]  # system+user
    response: str                  # assistant content (target)
    gold: str
    meta: Dict[str, Any]


def iter_trajectory_jsonl(path_jsonl: str) -> Iterable[Dict[str, Any]]:
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_sft_examples(
    traj_jsonl: str,
    *,
    budgets: Tuple[int, ...] = (1, 2, 4),
    keep_only_if_answer_present: bool = True,
    seed: int = 0,
    max_per_uid: Optional[int] = None,
) -> List[SFTExample]:
    """
    Convert one JSONL (trajectories) into a list of (messages, response) examples.

    Default budgets: (1,2,4) because t=3 is code generation in our teachers pipeline.
    """
    random.seed(seed)
    out: List[SFTExample] = []

    for obj in iter_trajectory_jsonl(traj_jsonl):
        uid = obj["uid"]
        problem = obj["problem"]
        gold = obj["gold"]
        meta = obj.get("meta", {})

        cps = {cp["t"]: cp for cp in obj["checkpoints"]}

        # choose which budgets to emit
        chosen = list(budgets)
        if max_per_uid is not None and len(chosen) > max_per_uid:
            random.shuffle(chosen)
            chosen = chosen[:max_per_uid]

        for t in chosen:
            cp = cps.get(t)
            if cp is None:
                continue

            raw = cp.get("raw", "")
            # parse answer/conf if not present
            answer = cp.get("answer")
            conf = cp.get("conf")
            if answer is None or conf is None:
                a2, c2 = parse_answer_and_conf(raw)
                answer = answer or a2
                conf = conf if conf is not None else c2

            response = canonicalize_response(raw, answer=answer, conf=conf)
            if keep_only_if_answer_present and response is None:
                continue

            out.append(
                SFTExample(
                    uid=uid,
                    budget_t=t,
                    messages=make_messages(problem, budget_t=t),
                    response=response,
                    gold=gold,
                    meta=meta,
                )
            )

    return out


def save_sft_jsonl(examples: List[SFTExample], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {
                        "uid": ex.uid,
                        "budget_t": ex.budget_t,
                        "messages": ex.messages,
                        "response": ex.response,
                        "gold": ex.gold,
                        "meta": ex.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

