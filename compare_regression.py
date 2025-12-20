import json, re
from collections import defaultdict

def get(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def parse_answer(text):
    if text is None:
        return None
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None

def iter_rows(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o=json.loads(line)
            if "checkpoints" in o and isinstance(o["checkpoints"], list):
                for cp in o["checkpoints"]:
                    r=dict(o); r.update(cp); yield r
            else:
                yield o

def index_uid_t(path):
    idx=defaultdict(dict)
    for r in iter_rows(path):
        uid=get(r,"uid","id","example_id","qid","custom_id")
        t=get(r,"t","budget","step","k")
        if uid is None or t is None:
            continue
        try: t=int(t)
        except: continue
        pred_text=get(r,"pred","prediction","output","text","raw","completion")
        pred_ans=get(r,"pred_answer","answer","final_answer") or parse_answer(pred_text)
        correct=get(r,"correct","is_correct","strict_correct")
        if isinstance(correct,(int,float)): correct=bool(correct)
        idx[str(uid)][t]={"pred_ans":pred_ans,"correct":correct}
    return idx

def regression_stats(idx):
    total=sum(1 for uid in idx if 4 in idx[uid])
    reg=0
    for uid,steps in idx.items():
        if 4 not in steps:
            continue
        c4=steps[4].get("correct")
        if c4 is None:
            continue
        early=any(steps.get(t,{}).get("correct") for t in (1,2,3) if t in steps)
        if early and (not c4):
            reg+=1
    return total, reg, (reg/total if total else 0.0)

def flip_stats(idx):
    total=sum(1 for uid in idx if 3 in idx[uid] and 4 in idx[uid])
    flips=0
    for uid,steps in idx.items():
        if 3 in steps and 4 in steps:
            flips += int(str(steps[3].get("pred_ans")).strip() != str(steps[4].get("pred_ans")).strip())
    return total, flips, (flips/total if total else 0.0)

for name in ["abl_full","abl_b1","base"]:
    path=f"results_abl/preds_{name}.jsonl"
    idx=index_uid_t(path)
    tot, reg, regr=regression_stats(idx)
    ftot, flips, flipr=flip_stats(idx)
    print(f"{name}: t4_total={tot}, regression={reg} ({regr:.3%}), flips34={flips} ({flipr:.3%})")
