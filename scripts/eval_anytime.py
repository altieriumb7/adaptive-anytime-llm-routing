#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from src.calibration.conf_calibrator import ConfidenceCalibrator
from src.data.judging import is_correct
from src.data.load_datasets import load_dataset_by_name
from src.train.sft_build import make_messages
from src.utils.parsing import parse_answer_and_conf


def _safe_pair(confs: List[float], corrects: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if confs is None or corrects is None:
        return None, None
    n = min(len(confs), len(corrects))
    if n <= 0:
        return None, None
    return np.asarray(confs[:n], dtype=np.float64), np.asarray(corrects[:n], dtype=np.float64)


def brier_score(confs: List[float], corrects: List[int]) -> float:
    c, y = _safe_pair(confs, corrects)
    if c is None:
        return float("nan")
    return float(np.mean((c - y) ** 2))


def ece_score(confs: List[float], corrects: List[int], n_bins: int = 10) -> float:
    c, y = _safe_pair(confs, corrects)
    if c is None:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (c >= lo) & (c < hi) if i < n_bins - 1 else (c >= lo) & (c <= hi)
        if mask.sum() == 0:
            continue
        ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(c[mask].mean()))
    return float(ece)


def auc_accuracy(acc_by_t: Dict[int, float]) -> float:
    ts = sorted(acc_by_t.keys())
    if len(ts) < 2:
        return float(acc_by_t[ts[0]]) if ts else float("nan")
    auc = 0.0
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        auc += 0.5 * (acc_by_t[t0] + acc_by_t[t1]) * (t1 - t0)
    span = ts[-1] - ts[0]
    return float(auc / span) if span > 0 else float(auc)


class StopAfterAnswerAndConf(StoppingCriteria):
    def __init__(self, tokenizer: Any, prompt_len: int, min_gen_tokens: int = 4):
        super().__init__()
        self.tok = tokenizer
        self.prompt_len = int(prompt_len)
        self.min_gen_tokens = int(min_gen_tokens)
        self.pattern = re.compile(r"####\\s*([^\\n\\r]+)\\s*[\\n\\r]+\\s*CONF\\s*:\\s*([01](?:\\.\\d+)?)", re.IGNORECASE)

    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0][self.prompt_len:]
        if gen_ids.numel() < self.min_gen_tokens:
            return False
        tail = self.tok.decode(gen_ids[-256:], skip_special_tokens=False)
        return self.pattern.search(tail) is not None


def _fallback_last_number(text: str) -> str:
    tail = (text or "")[-1200:]
    boxed = re.findall(r"\\\\boxed\{(-?\d+(?:\.\d+)?)\}", tail)
    if boxed:
        return boxed[-1]
    nums = re.findall(r"(-?\d+(?:\.\d+)?)", tail)
    return nums[-1] if nums else ""


def _clean_gold(gold: str) -> str:
    s = "" if gold is None else str(gold)
    return re.sub(r"^####\s*", "", s.strip()).strip()


def load_model(base_model: str, adapter_dir: Optional[str]) -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


def generate_with_stats(model: Any, tok: Any, prompt: str, max_new_tokens: int) -> Tuple[str, Dict[str, int]]:
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])
    stopping = StoppingCriteriaList([StopAfterAnswerAndConf(tok, prompt_len=prompt_len)])

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            stopping_criteria=stopping,
            return_dict_in_generate=True,
        )

    seq = out.sequences[0]
    gen_ids = seq[prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    stats = {
        "prompt_tokens": prompt_len,
        "gen_tokens": int(gen_ids.shape[0]),
        "total_tokens": int(seq.shape[0]),
    }
    return text, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", default=None)
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math", "svamp", "boolq", "strategyqa"])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max_examples", type=int, default=500)
    ap.add_argument("--budgets", default="1,2,3,4")
    ap.add_argument("--max_new_tokens", default="96,160,224,320")
    ap.add_argument("--save_jsonl", type=str, default=None)
    ap.add_argument("--calibrator", type=str, default=None)
    args = ap.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    max_new_tokens = [int(x.strip()) for x in args.max_new_tokens.split(",") if x.strip()]
    if len(budgets) != len(max_new_tokens):
        raise ValueError("budgets and max_new_tokens must align")

    save_jsonl_fh = None
    if args.save_jsonl:
        Path(args.save_jsonl).parent.mkdir(parents=True, exist_ok=True)
        save_jsonl_fh = open(args.save_jsonl, "w", encoding="utf-8", buffering=1)
        atexit.register(lambda: save_jsonl_fh and save_jsonl_fh.close())

    calibrator = ConfidenceCalibrator.from_json(args.calibrator) if args.calibrator else None
    tok, model = load_model(args.base_model, args.adapter_dir)

    examples = load_dataset_by_name(args.dataset, split=args.split)[: args.max_examples]
    total = len(examples)
    correct_counts = {t: 0 for t in budgets}
    confs_by_t: Dict[int, List[float]] = {t: [] for t in budgets}
    confs_cal_by_t: Dict[int, List[float]] = {t: [] for t in budgets}
    correct_by_t: Dict[int, List[int]] = {t: [] for t in budgets}
    ttc_list: List[Optional[int]] = []

    for ex in tqdm(examples, desc="eval"):
        first_correct: Optional[int] = None
        task = "math"
        if isinstance(ex.meta, dict):
            task = ex.meta.get("task", "math")

        for t, mnt in zip(budgets, max_new_tokens):
            messages = make_messages(ex.problem, budget_t=t, task=task)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            out_text, stats = generate_with_stats(model, tok, prompt, max_new_tokens=mnt)

            ans, conf = parse_answer_and_conf(out_text)
            if (ans is None or str(ans).strip() == "") and str(task).lower() not in {"yesno", "boolq", "strategyqa"}:
                ans = _fallback_last_number(out_text)

            ok = is_correct(ans, _clean_gold(ex.gold))
            correct_counts[t] += int(ok)

            conf_val = 0.5 if conf is None else float(conf)
            conf_val = max(0.0, min(1.0, conf_val))
            confs_by_t[t].append(conf_val)
            correct_by_t[t].append(int(ok))

            if calibrator is not None:
                confs_cal_by_t[t].append(float(calibrator.calibrate(t, conf_val)))

            if first_correct is None and ok:
                first_correct = t

            if save_jsonl_fh is not None:
                save_jsonl_fh.write(json.dumps({
                    "uid": ex.uid,
                    "t": int(t),
                    "max_new_tokens": int(mnt),
                    "gen_tokens": int(stats["gen_tokens"]),
                    "prompt_tokens": int(stats["prompt_tokens"]),
                    "total_tokens": int(stats["total_tokens"]),
                    "problem": ex.problem,
                    "gold": ex.gold,
                    "raw_text": out_text,
                    "answer": ans,
                    "conf": conf,
                    "correct": int(ok),
                    "meta": ex.meta,
                }, ensure_ascii=False) + "\n")

        ttc_list.append(first_correct)

    acc_by_t = {t: (correct_counts[t] / total if total else float("nan")) for t in budgets}
    print("\n=== Accuracy@t ===")
    for t in budgets:
        print(f"t={t}: {acc_by_t[t]:.4f}")

    solved = [t for t in ttc_list if t is not None]
    print("\n=== TTC ===")
    print(f"Solved: {len(solved)}/{total} ({(len(solved)/total) if total else 0.0:.3f})")
    if solved:
        print(f"Mean TTC (solved only): {np.mean(solved):.3f}")
        for t in budgets:
            pct = sum(1 for x in solved if x <= t) / len(solved)
            print(f"% solved by t={t}: {pct:.3f}")

    print("\n=== AUC (accuracy vs budget index) ===")
    print(f"AUC: {auc_accuracy(acc_by_t):.4f}")

    print("\n=== Calibration (raw) ===")
    for t in budgets:
        print(f"t={t}: Brier={brier_score(confs_by_t[t], correct_by_t[t]):.4f} | ECE={ece_score(confs_by_t[t], correct_by_t[t], n_bins=10):.4f}")

    if calibrator is not None:
        print("\n=== Calibration (calibrated) ===")
        for t in budgets:
            if confs_cal_by_t[t]:
                print(f"t={t}: Brier={brier_score(confs_cal_by_t[t], correct_by_t[t]):.4f} | ECE={ece_score(confs_cal_by_t[t], correct_by_t[t], n_bins=10):.4f}")


if __name__ == "__main__":
    main()
