import json, math
from collections import defaultdict

PATH = "results_abl/preds_abl_full.jsonl"

def get(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def load_uid_t(path):
    idx = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)

            uid = get(o, "uid","id","example_id","qid","custom_id")
            t   = get(o, "t","budget","step","k")
            if uid is None or t is None:
                continue
            try:
                t = int(t)
            except:
                continue

            correct = get(o, "correct","is_correct","strict_correct")
            if isinstance(correct, (int,float)):
                correct = bool(correct)

            conf = get(o, "conf","confidence","p_correct")
            try:
                conf = float(conf) if conf is not None else None
            except:
                conf = None

            idx[str(uid)][t] = {"correct": correct, "conf": conf}
    return idx

idx = load_uid_t(PATH)
uids = sorted(idx.keys())

def eval_pick(pick_fn):
    chosen_t = []
    corrects = []
    regress = 0
    total = 0
    for uid in uids:
        steps = idx[uid]
        if 4 not in steps:
            continue
        total += 1
        t = pick_fn(steps)
        chosen_t.append(t)
        c = steps.get(t, {}).get("correct")
        corrects.append(bool(c))

        # regression under policy: was any earlier step correct but chosen outcome wrong
        early_correct = any(steps.get(tt, {}).get("correct") for tt in (1,2,3) if tt in steps)
        if early_correct and (not bool(c)):
            regress += 1

    acc = sum(corrects)/len(corrects)
    mean_t = sum(chosen_t)/len(chosen_t)
    return {"acc": acc, "mean_t": mean_t, "regression": regress/total, "n": total}

# Baseline: always t=4
res_t4 = eval_pick(lambda steps: 4)

# Policy 1: argmax confidence (ties -> smallest t)
def pick_argmax(steps):
    best_t = 4
    best_c = -1.0
    for t in (1,2,3,4):
        c = steps.get(t, {}).get("conf")
        if c is None:
            continue
        if c > best_c or (c == best_c and t < best_t):
            best_c = c
            best_t = t
    return best_t
res_argmax = eval_pick(pick_argmax)

# Policy 2: threshold early stop (grid search)
def pick_thresh(tau):
    def fn(steps):
        for t in (1,2,3,4):
            c = steps.get(t, {}).get("conf")
            if c is not None and c >= tau:
                return t
        return 4
    return fn

best = None
for tau in [i/100 for i in range(50, 100)]:  # 0.50..0.99
    r = eval_pick(pick_thresh(tau))
    r["tau"] = tau
    if best is None or r["acc"] > best["acc"]:
        best = r

print("== abl_full policies ==")
print("always_t4:", res_t4)
print("argmax_conf:", res_argmax)
print("best_threshold:", best)
