from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set

from tqdm import tqdm

from src.utils.parsing import parse_answer_and_conf
from src.data.judging import is_correct
from src.trajectory.prompts import (
    prompt_t1_draft,
    prompt_t2_resolve,
    prompt_t3_verify_code,
    prompt_t3_verify_text,
    prompt_t4_repair,
)
from src.verifier.safe_exec import extract_python_code, run_python_code


# Teacher interface: any object with generate(prompt, max_new_tokens, temperature, top_p, stop_strings)->str


@dataclass
class CheckpointConfig:
    t: int
    mode: str
    max_new_tokens: int
    temperature: float = 0.2


DEFAULT_CHECKPOINTS: List[CheckpointConfig] = [
    CheckpointConfig(t=1, mode="draft",       max_new_tokens=64,  temperature=0.2),
    CheckpointConfig(t=2, mode="resolve",     max_new_tokens=128, temperature=0.2),
    CheckpointConfig(t=3, mode="verify_text", max_new_tokens=128, temperature=0.2),
    CheckpointConfig(t=4, mode="repair",      max_new_tokens=160, temperature=0.2),
]


def _load_existing_uids(path_jsonl: str) -> Set[str]:
    if not os.path.exists(path_jsonl):
        return set()
    uids: Set[str] = set()
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = obj.get("uid")
                if isinstance(uid, str):
                    uids.add(uid)
            except json.JSONDecodeError:
                continue
    return uids


def _write_jsonl(path_jsonl: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path_jsonl), exist_ok=True)
    with open(path_jsonl, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _compute_ttc(correct_by_t: Dict[int, bool]) -> Optional[int]:
    for t in sorted(correct_by_t.keys()):
        if correct_by_t[t]:
            return t
    return None


def run_anytime_trajectory(
    teacher,
    problem: str,
    gold: str,
    checkpoints: List[CheckpointConfig] = DEFAULT_CHECKPOINTS,
) -> Dict[str, Any]:

    # --- t1 draft ---
    p1 = prompt_t1_draft(problem)
    out1 = teacher.generate(p1, max_new_tokens=checkpoints[0].max_new_tokens, temperature=checkpoints[0].temperature)
    a1, c1 = parse_answer_and_conf(out1)
    corr1 = is_correct(a1, gold)

    # --- t2 resolve ---
    p2 = prompt_t2_resolve(problem, draft_text=out1)
    out2 = teacher.generate(p2, max_new_tokens=checkpoints[1].max_new_tokens, temperature=checkpoints[1].temperature)
    a2, c2 = parse_answer_and_conf(out2)
    corr2 = is_correct(a2, gold)

    candidate = a2 or a1

    # --- INTERNAL python verifier (not student checkpoint) ---
    # do it only if needed (saves cost/time)
    conf2 = 0.5 if c2 is None else float(c2)
    agree = (a1 is not None and a2 is not None and a1 == a2)
    need_verify = not (agree and conf2 >= 0.85)

    out3code = ""
    code = None
    c_code = None
    exec_res = None
    verifier_stdout = ""
    verifier_stderr = ""
    verified_answer = None
    verified_correct = False

    if need_verify:
        p3code = prompt_t3_verify_code(problem, candidate_answer=candidate)
        out3code = teacher.generate(p3code, max_new_tokens=200, temperature=0.2)
        code = extract_python_code(out3code)
        _, c_code = parse_answer_and_conf(out3code)

        if code is not None:
            exec_res = run_python_code(code, timeout_s=2)
            verifier_stdout = exec_res.stdout.strip()
            verifier_stderr = (exec_res.stderr or "").strip()
            if exec_res.ok and verifier_stdout:
                verified_answer = verifier_stdout.splitlines()[-1].strip()
                verified_correct = is_correct(verified_answer, gold)
        else:
            verifier_stderr = "No python code block found."

    # --- t3 verify_text (student-facing) ---
    p3 = prompt_t3_verify_text(problem, draft_text=out1, resolve_text=out2, verifier_stdout=verifier_stdout)
    out3 = teacher.generate(p3, max_new_tokens=checkpoints[2].max_new_tokens, temperature=checkpoints[2].temperature)
    a3, c3 = parse_answer_and_conf(out3)
    corr3 = is_correct(a3, gold)

    # --- t4 repair ---
    p4 = prompt_t4_repair(
        problem,
        draft_text=out1,
        resolve_text=out2,
        verify_text=out3,
        verifier_stdout=verifier_stdout,
        verifier_stderr=verifier_stderr,
    )
    out4 = teacher.generate(p4, max_new_tokens=checkpoints[3].max_new_tokens, temperature=checkpoints[3].temperature)
    a4, c4 = parse_answer_and_conf(out4)
    corr4 = is_correct(a4, gold)

    correct_by_t = {1: corr1, 2: corr2, 3: corr3, 4: corr4}
    ttc = _compute_ttc(correct_by_t)

    return {
        "problem": problem,
        "gold": gold,
        "checkpoints": [
            {"t": 1, "mode": "draft",       "raw": out1, "answer": a1, "conf": c1, "correct": corr1},
            {"t": 2, "mode": "resolve",     "raw": out2, "answer": a2, "conf": c2, "correct": corr2},
            {"t": 3, "mode": "verify_text", "raw": out3, "answer": a3, "conf": c3, "correct": corr3},
            {"t": 4, "mode": "repair",      "raw": out4, "answer": a4, "conf": c4, "correct": corr4},
        ],
        "verifier_code": {
            "raw": out3code,
            "conf": c_code,
            "code": code,
            "exec": None if exec_res is None else {
                "ok": exec_res.ok,
                "timeout": exec_res.timeout,
                "rejected": exec_res.rejected,
                "stdout": exec_res.stdout,
                "stderr": exec_res.stderr,
                "reason": exec_res.reason,
            },
            "verified_answer": verified_answer,
            "verified_correct": verified_correct,
        },
        "labels": {"correct_by_t": correct_by_t, "ttc": ttc},
    }


def generate_jsonl(
    teacher,
    examples: List[Any],
    out_path: str,
    *,
    resume: bool = True,
    shard_id: int = 0,
    num_shards: int = 1,
) -> None:
    existing = _load_existing_uids(out_path) if resume else set()
    sharded = [ex for i, ex in enumerate(examples) if (i % num_shards) == shard_id]

    buf: List[Dict[str, Any]] = []
    for ex in tqdm(sharded, desc=f"gen shard {shard_id}/{num_shards}"):
        if resume and ex.uid in existing:
            continue
        traj = run_anytime_trajectory(teacher=teacher, problem=ex.problem, gold=ex.gold)
        buf.append({"uid": ex.uid, "meta": ex.meta, **traj})
        if len(buf) >= 25:
            _write_jsonl(out_path, buf)
            buf = []
    if buf:
        _write_jsonl(out_path, buf)