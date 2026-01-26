import torch
import argparse
import json
import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.models import parse_answer_and_conf
from src.train.sft_build import make_messages
from src.data.load_datasets import load_dataset_by_name
from src.data.judging import is_correct
from src.calibration.conf_calibrator import ConfidenceCalibrator
import re
from transformers import StoppingCriteria, StoppingCriteriaList

FORMAT_SUFFIX = (
  "\n\nSTRICT OUTPUT FORMAT.\n"
  "End with EXACTLY these two lines:\n"
  "#### <final_answer>\n"
  "CONF: <p>\n"
  "Rules:\n"
  "- <final_answer> must be a single number (no words, no LaTeX, no units).\n"
  "- <p> must be a decimal between 0 and 1.\n"
  "- Do not write anything after the CONF line.\n"
)


class StopAfterAnswerAndConf(StoppingCriteria):
    """
    Stop when the generated text contains:
      #### <answer>  AND  CONF: <float>
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.tok = tokenizer
        self.pattern = re.compile(
            r"####\s*([^\n\r]+)\s*[\n\r]+\s*CONF\s*:\s*([01](?:\.\d+)?)",
            re.IGNORECASE
        )

    def __call__(self, input_ids, scores, **kwargs):
        # decode only recent tail to keep it fast
        tail = self.tok.decode(input_ids[0][-256:], skip_special_tokens=False)
        return self.pattern.search(tail) is not None


def brier_score(confs: List[float], corrects: List[int]) -> float:
    c = np.array(confs, dtype=np.float64)
    y = np.array(corrects, dtype=np.float64)
    return float(np.mean((c - y) ** 2))


def ece_score(confs: List[float], corrects: List[int], n_bins: int = 10) -> float:
    c = np.array(confs, dtype=np.float64)
    y = np.array(corrects, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (c >= lo) & (c < hi) if i < n_bins - 1 else (c >= lo) & (c <= hi)
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = c[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def auc_accuracy(acc_by_t: Dict[int, float]) -> float:
    # trapezoid over t=1..4 (or whatever keys)
    ts = sorted(acc_by_t.keys())
    auc = 0.0
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        y0, y1 = acc_by_t[t0], acc_by_t[t1]
        auc += 0.5 * (y0 + y1) * (t1 - t0)
    # normalize by max span so it’s in [0, max_acc*span]
    span = ts[-1] - ts[0]
    return float(auc / span) if span > 0 else float(auc)


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


def load_boolq(split: str, max_examples: int = None):
    from datasets import load_dataset

    # BoolQ has train/validation; if user asks test, map to validation
    sp = split
    if sp == "test":
        sp = "validation"

    ds = load_dataset("super_glue", "boolq", split=sp)
    examples = []

    for i, row in enumerate(ds):
        passage = str(row["passage"]).strip()
        question = str(row["question"]).strip()
        gold = "yes" if int(row["label"]) == 1 else "no"

        problem = (
            f"PASSAGE:\n{passage}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Answer yes or no."
        )
        examples.append({"uid": f"boolq_{sp}_{i}", "problem": problem, "gold": gold})

        if max_examples and len(examples) >= max_examples:
            break

    return examples


def generate(model, tok, prompt: str, max_new_tokens: int, *, compute_nll: bool = True):
    """
    Returns:
      text (str),
      stats (dict): prompt_tokens, gen_tokens, total_tokens, avg_nll (optional)
    """
    import torch

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            output_scores=bool(compute_nll),
        )

    seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
    prefix_len = int(inputs["input_ids"].shape[1])
    gen_ids = seq[prefix_len:]
    gen_tokens = int(gen_ids.shape[0])

    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    stats = {
        "prompt_tokens": prefix_len,
        "gen_tokens": gen_tokens,
        "total_tokens": int(seq.shape[0]),
    }

    if compute_nll and gen_tokens > 0:
        # avg negative log-likelihood of chosen tokens (greedy)
        nll_sum = 0.0
        # out.scores is a tuple length == gen_tokens; each is [batch, vocab]
        for j, logits in enumerate(out.scores):
            # logits: (1, vocab)
            logp = torch.log_softmax(logits[0], dim=-1)
            tid = int(gen_ids[j].item())
            nll_sum += float(-logp[tid].item())
        stats["avg_nll"] = nll_sum / max(1, gen_tokens)

    return text, stats


def generate_with_stats(model, tok, prompt: str, max_new_tokens: int):
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        stopping = StoppingCriteriaList([StopAfterAnswerAndConf(tok)])

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            stopping_criteria=stopping,
            return_dict_in_generate=True,  # FIX: ensure .sequences exists
        )

    seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = seq[prompt_len:]
    gen_tokens = int(gen_ids.shape[0])

    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    stats = {
        "prompt_tokens": prompt_len,
        "gen_tokens": gen_tokens,
        "total_tokens": int(seq.shape[0]),
    }
    return text, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", default=None, help="LoRA adapter dir (or none for baseline)")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k","math", "svamp", "boolq","strategyqa"])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max_examples", type=int, default=500)
    ap.add_argument("--budgets", default="1,2,4")
    ap.add_argument("--max_new_tokens", default="96,192,256", help="comma list aligned with budgets")
    ap.add_argument("--save_jsonl", type=str, default=None, help="Write per-example predictions as JSONL (one row per budget).")
    ap.add_argument("--calibrator", type=str, default=None,
                    help="Path to per-budget confidence calibrator JSON")

    args = ap.parse_args()

    save_jsonl_fh = None
    if getattr(args, "save_jsonl", None):
        from pathlib import Path
        import atexit
        Path(args.save_jsonl).parent.mkdir(parents=True, exist_ok=True)
        save_jsonl_fh = open(args.save_jsonl, "w", encoding="utf-8", buffering=1)
        atexit.register(lambda: save_jsonl_fh and save_jsonl_fh.close())

    budgets = [int(x) for x in args.budgets.split(",")]
    max_new_tokens = [int(x) for x in args.max_new_tokens.split(",")]
    assert len(budgets) == len(max_new_tokens), "budgets and max_new_tokens must align"
    calibrator = ConfidenceCalibrator.from_json(args.calibrator) if args.calibrator else None

    tok, model = load_model(args.base_model, args.adapter_dir)

    examples = load_dataset_by_name(args.dataset, split=args.split)
    examples = examples[: args.max_examples]

    correct_counts = {t: 0 for t in budgets}
    total = len(examples)
    # store conf/correct per t for calibration
    confs_by_t: Dict[int, List[float]] = {t: [] for t in budgets}
    correct_by_t: Dict[int, List[int]] = {t: [] for t in budgets}

    # calibrated confidences (only meaningful if calibrator is provided)
    confs_cal_by_t: Dict[int, List[float]] = {t: [] for t in budgets}

    ttc_list: List[Optional[int]] = []

    for ex in tqdm(examples, desc="eval"):
        first_correct = None
        for t, mnt in zip(budgets, max_new_tokens):
            task = "math"
            meta = getattr(ex, "meta", None)
            if isinstance(meta, dict):
                task = meta.get("task", "math")

            messages = make_messages(ex.problem, budget_t=t, task=task)

            if str(task).lower() in {"yesno", "boolq", "strategyqa"}:
                messages[-1]["content"] += (
                    "\n\nSTRICT OUTPUT FORMAT. PRINT ONLY THESE TWO LINES AND NOTHING ELSE:\n"
                    "#### yes\n"
                    "CONF: 0.73\n\n"
                    "Rules:\n"
                    "- Output must be EXACTLY two lines.\n"
                    "- First line must be: #### yes OR #### no (lowercase).\n"
                    "- Second line must be: CONF: <number between 0 and 1>.\n"
                    "- Do not include explanations, extra text, angle brackets <> or LaTeX.\n"
                )
            else:
                # FIX: use strict suffix to force #### + CONF presence
                messages[-1]["content"] += FORMAT_SUFFIX

            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            out_text, stats = generate_with_stats(model, tok, prompt, max_new_tokens=mnt)
            ans, conf = parse_answer_and_conf(out_text)

            ok = is_correct(ans, ex.gold)

            if save_jsonl_fh is not None:
                row = {
                    "uid": ex.uid,
                    "t": int(t),

                    # budget proxy
                    "max_new_tokens": int(mnt),

                    # real token counts
                    "gen_tokens": int(stats.get("gen_tokens", 0)),
                    "prompt_tokens": int(stats.get("prompt_tokens", 0)),
                    "total_tokens": int(stats.get("total_tokens", 0)),

                    "problem": ex.problem,
                    "gold": ex.gold,
                    "raw_text": out_text,
                    "answer": ans,
                    "conf": conf,
                    "correct": int(ok),
                    "meta": ex.meta,
                }
                save_jsonl_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            correct_counts[t] += int(ok)

            conf_val = 0.5 if conf is None else float(conf)
            conf_val = max(0.0, min(1.0, conf_val))

            confs_by_t[t].append(conf_val)
            correct_by_t[t].append(int(ok))

            if calibrator is not None:
                conf_cal = calibrator.calibrate(t, conf_val)
                confs_cal_by_t[t].append(float(conf_cal))

            if first_correct is None and ok:
                first_correct = t

        ttc_list.append(first_correct)

    acc_by_t = {t: correct_counts[t] / total for t in budgets}

    print("\n=== Accuracy@t ===")
    for t in budgets:
        print(f"t={t}: {acc_by_t[t]:.4f}")

    # TTC stats
    solved = [t for t in ttc_list if t is not None]
    print("\n=== TTC ===")
    print(f"Solved: {len(solved)}/{total} ({len(solved)/total:.3f})")
    if solved:
        print(f"Mean TTC (solved only): {np.mean(solved):.3f}")
        for t in budgets:
            pct = sum(1 for x in solved if x <= t) / len(solved)
            print(f"% solved by t={t}: {pct:.3f}")

    # AUC over budget
    auc = auc_accuracy(acc_by_t)
    print("\n=== AUC (accuracy vs budget index) ===")
    print(f"AUC: {auc:.4f}")

    print("\n=== Calibration (raw) ===")
    for t in budgets:
        bs = brier_score(confs_by_t[t], correct_by_t[t])
        ece = ece_score(confs_by_t[t], correct_by_t[t], n_bins=10)
        print(f"t={t}: Brier={bs:.4f} | ECE={ece:.4f}")

    if calibrator is not None:
        print("\n=== Calibration (calibrated) ===")
        for t in budgets:
            if len(confs_cal_by_t[t]) == 0:
                continue
            bs = brier_score(confs_cal_by_t[t], correct_by_t[t])
            ece = ece_score(confs_cal_by_t[t], correct_by_t[t], n_bins=10)
            print(f"t={t}: Brier={bs:.4f} | ECE={ece:.4f}")


if __name__ == "__main__":
    main()
