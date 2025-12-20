import argparse, math, re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.data.load_datasets import load_gsm8k
from src.train.sft_build import make_messages
from src.models import parse_answer_and_conf

def extract_number(text: str):
    """
    Best-effort: extract a single "final" number from text.
    Priority: \\boxed{...} -> #### ... -> last fraction -> last decimal/int.
    Returns a string or None.
    """
    if text is None:
        return None
    t = str(text)
    t = t.replace(",", "")  # 70,000 -> 70000

    # \boxed{...}
    m = re.search(r"\\boxed\{([^}]*)\}", t)
    if m:
        t = m.group(1)

    # "#### 18"
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", t)
    if m:
        return m.group(1)

    # fractions (e.g. 3/4)
    fracs = re.findall(r"[-+]?\d+\s*/\s*\d+", t)
    if fracs:
        return fracs[-1].replace(" ", "")

    # last number
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", t)
    if nums:
        return nums[-1]

    return None

def numeric_equal(pred_text, gold_text, tol=1e-4):
    p = extract_number(pred_text)
    g = extract_number(gold_text)
    if p is None or g is None:
        return False

    # Try exact rational match for fractions
    if "/" in p or "/" in g:
        try:
            from fractions import Fraction
            return Fraction(p) == Fraction(g)
        except Exception:
            pass

    try:
        pf = float(p)
        gf = float(g)
        return math.isclose(pf, gf, rel_tol=tol, abs_tol=tol)
    except Exception:
        return p.strip() == g.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", default=None)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--budgets", default="1,2,3,4")
    ap.add_argument("--max_new_tokens", default="96,160,224,320")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    max_new = [int(x) for x in args.max_new_tokens.split(",") if x.strip()]
    assert len(max_new) == len(budgets), "max_new_tokens must match budgets length"

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    ds = load_gsm8k(args.split)
    n = len(ds) if args.max_examples is None else min(len(ds), args.max_examples)

    correct_by_t = {t: 0 for t in budgets}

    for i in tqdm(range(n), desc="numeric-eval"):
        ex = ds[i]
        problem = getattr(ex, "problem", None)
        gold = getattr(ex, "gold", None)

        for t, mx in zip(budgets, max_new):
            msgs = make_messages(problem, budget_t=t)
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=mx,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id,
                )

            gen = out[0][inputs["input_ids"].shape[1]:]
            text = tok.decode(gen, skip_special_tokens=True)

            ans, _conf = parse_answer_and_conf(text)
            pred_for_judge = ans if ans is not None else text

            if numeric_equal(pred_for_judge, gold, tol=args.tol):
                correct_by_t[t] += 1

    print("\n=== Numeric-only Accuracy@t ===")
    for t in budgets:
        print(f"t={t}: {correct_by_t[t]/n:.4f}")

if __name__ == "__main__":
    main()
