#!/usr/bin/env python3
import argparse, json, re, sys
from typing import Any, List

sys.path.insert(0, "scripts")
from eval_depth_router import read_jsonl_grouped, extract_steps

NUM_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

def extract_last_number(x: Any):
    if x is None:
        return None
    s = str(x)
    m = NUM_RE.findall(s)
    if not m:
        return None
    return m[-1].replace(",", "")

def get_any(step: Any, keys: List[str]):
    for k in keys:
        v = getattr(step, k, None)
        if v is not None and v != "":
            return v
    d = getattr(step, "__dict__", None)
    if isinstance(d, dict):
        for k in keys:
            v = d.get(k, None)
            if v is not None and v != "":
                return v
    return None

def step_tokens(step: Any) -> int:
    v = get_any(step, ["tokens","max_new_tokens","n_tokens","token_count"])
    try:
        return int(v) if v is not None else 0
    except Exception:
        return 0

def p95(arr):
    if not arr: return 0.0
    arr = sorted(arr)
    idx = int(0.95*(len(arr)-1))
    return float(arr[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = read_jsonl_grouped(args.data)

    n=0
    correct=0
    stop_steps=[]
    token_sums=[]

    maxK = 0

    for ex in data:
        gold, steps = extract_steps(ex)
        K = len(steps)
        maxK = max(maxK, K)
        gold_num = extract_last_number(gold)

        # earliest step whose numeric answer matches gold
        stop = K
        for i, st in enumerate(steps, start=1):
            pred = get_any(st, ["ans","answer","pred","prediction","raw_text","text","output","final"])
            pred_num = extract_last_number(pred)
            if gold_num is not None and pred_num is not None and pred_num == gold_num:
                stop = i
                break

        # evaluate at stop
        pred = get_any(steps[stop-1], ["ans","answer","pred","prediction","raw_text","text","output","final"])
        pred_num = extract_last_number(pred)
        if gold_num is not None and pred_num is not None and pred_num == gold_num:
            correct += 1

        n += 1
        stop_steps.append(stop)
        token_sums.append(sum(step_tokens(s) for s in steps[:stop]))

    # histogram length = maxK
    hist = [0]*maxK
    for s in stop_steps:
        hist[s-1]+=1
    hist = [h/n for h in hist]

    out = {
        "policy":"oracle",
        "n": n,
        "acc": correct/n if n else 0.0,
        "mean_steps": sum(stop_steps)/n if n else 0.0,
        "p95_steps": p95(stop_steps),
        "mean_tokens": sum(token_sums)/n if n else 0.0,
        "p95_tokens": p95(token_sums),
        "stop_histogram": hist
    }

    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out))
if __name__ == "__main__":
    main()
