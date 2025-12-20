import json, random
from collections import defaultdict

PATH = "results_abl/preds_abl_full.jsonl"
SEED = 0
DEV_FRAC = 0.5  # 50/50 dev/test split for threshold selection

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
uids = sorted([u for u in idx.keys() if 4 in idx[u]])

random.seed(SEED)
random.shuffle(uids)
cut = int(len(uids) * DEV_FRAC)
dev_uids = set(uids[:cut])
test_uids = set(uids[cut:])

def eval_pick(uids_subset, pick_fn):
    chosen_t, corrects = [], []
    regress, total = 0, 0
    for uid in uids_subset:
        steps = idx[uid]
        total += 1
        t = pick_fn(steps)
        chosen_t.append(t)
        c = bool(steps.get(t, {}).get("correct"))
        corrects.append(c)

        early_correct = any(steps.get(tt, {}).get("correct") for tt in (1,2,3) if tt in steps)
        if early_correct and (not c):
            regress += 1

    acc = sum(corrects)/len(corrects)
    mean_t = sum(chosen_t)/len(chosen_t)
    return {"acc": acc, "mean_t": mean_t, "regression": regress/total, "n": total}

def pick_thresh(tau):
    def fn(steps):
        for t in (1,2,3,4):
            c = steps.get(t, {}).get("conf")
            if c is not None and c >= tau:
                return t
        return 4
    return fn

# choose tau on DEV
best = None
for tau in [i/100 for i in range(50, 100)]:
    r = eval_pick(dev_uids, pick_thresh(tau))
    r["tau"] = tau
    if best is None or r["acc"] > best["acc"]:
        best = r

# evaluate on TEST with selected tau
test_res = eval_pick(test_uids, pick_thresh(best["tau"]))
always_t4_test = eval_pick(test_uids, lambda steps: 4)

print("== DEV selection ==")
print("best_tau_on_dev:", best)
print("== TEST evaluation ==")
print("always_t4_test:", always_t4_test)
print("threshold_test:", {**test_res, "tau": best["tau"]})
