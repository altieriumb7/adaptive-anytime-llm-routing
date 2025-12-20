import json, re
from collections import defaultdict

FULL = "results_abl/preds_abl_full.jsonl"
B1   = "results_abl/preds_abl_b1.jsonl"
OUT  = "results_abl/regression_cases_full.tsv"

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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            # Case A: file is already per-(uid,t) prediction rows
            if any(k in obj for k in ["t","budget","step"]):
                yield obj
                continue

            # Case B: file stores a trajectory with checkpoints list
            if "checkpoints" in obj and isinstance(obj["checkpoints"], list):
                for cp in obj["checkpoints"]:
                    row = dict(obj)
                    row.update(cp)
                    yield row
                continue

            # Fallback: just yield as-is
            yield obj

def index_uid_t(path):
    idx = defaultdict(dict)
    for r in iter_rows(path):
        uid = get(r, "uid","id","example_id","qid","custom_id","problem_id")
        t   = get(r, "t","budget","step","k")
        if uid is None or t is None:
            continue
        try:
            t = int(t)
        except:
            continue

        pred_text = get(r, "pred","prediction","output","text","raw","completion")
        pred_ans  = get(r, "pred_answer","answer","final_answer")
        if pred_ans is None:
            pred_ans = parse_answer(pred_text)

        correct = get(r, "correct","is_correct","strict_correct")
        if isinstance(correct, (int,float)):
            correct = bool(correct)

        conf = get(r, "conf","confidence","p_correct")
        gold = get(r, "gold","label","target","gt")
        prob = get(r, "problem","question","prompt")

        idx[str(uid)][t] = {
            "pred_ans": pred_ans,
            "pred_text": pred_text,
            "correct": correct,
            "conf": conf,
            "gold": gold,
            "problem": prob,
        }
    return idx

full = index_uid_t(FULL)
b1   = index_uid_t(B1)

total_with_t4 = sum(1 for uid in full if 4 in full[uid])

# regression definition: correct at any of t=1..3 AND wrong at t=4
cases = []
for uid, steps in full.items():
    if 4 not in steps:
        continue
    c4 = steps[4].get("correct")
    if c4 is None:
        continue
    early_correct = any(steps.get(t, {}).get("correct") for t in (1,2,3) if t in steps)
    if early_correct and (not c4):
        b1_t4 = b1.get(uid, {}).get(4, {})
        cases.append({
            "uid": uid,
            "b1_correct_t4": b1_t4.get("correct"),
            "b1_pred_t4": b1_t4.get("pred_ans"),
            "full_pred_t1": steps.get(1, {}).get("pred_ans"),
            "full_pred_t2": steps.get(2, {}).get("pred_ans"),
            "full_pred_t3": steps.get(3, {}).get("pred_ans"),
            "full_pred_t4": steps.get(4, {}).get("pred_ans"),
            "full_conf_t4": steps.get(4, {}).get("conf"),
            "gold": steps.get(4, {}).get("gold"),
            "problem": (steps.get(4, {}).get("problem") or "").replace("\t"," ").replace("\n"," "),
        })

print("Total examples with t=4 in abl_full:", total_with_t4)
print("Regression cases (abl_full):", len(cases))
if total_with_t4:
    print("Regression rate:", len(cases)/total_with_t4)

# write TSV (keep first 300 for convenience)
with open(OUT, "w", encoding="utf-8") as f:
    cols = ["uid","b1_correct_t4","b1_pred_t4","full_pred_t1","full_pred_t2","full_pred_t3","full_pred_t4","full_conf_t4","gold","problem"]
    f.write("\t".join(cols) + "\n")
    for c in cases[:300]:
        f.write("\t".join("" if c[k] is None else str(c[k]) for k in cols) + "\n")

print("Wrote:", OUT, "(first 300 cases)")