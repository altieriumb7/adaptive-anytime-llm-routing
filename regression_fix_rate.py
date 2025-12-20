import csv, json
from collections import defaultdict

REG_TSV = "results_abl/regression_cases_full.tsv"
P_FULL  = "results_abl/preds_abl_full.jsonl"
P_B1    = "results_abl/preds_abl_b1.jsonl"

def load_reg_uids():
    u=set()
    with open(REG_TSV,"r",encoding="utf-8") as f:
        r=csv.DictReader(f, delimiter="\t")
        for row in r:
            u.add(row["uid"])
    return u

def load_idx(path, uids):
    idx=defaultdict(dict)
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o=json.loads(line)
            uid=str(o.get("uid"))
            if uid not in uids: continue
            t=int(o["t"])
            idx[uid][t]=o
    return idx

uids = load_reg_uids()
full = load_idx(P_FULL, uids)
b1   = load_idx(P_B1, uids)

def corr(o):
    # keep False!
    return o["correct"]

fixed = 0
still_wrong = 0
total = 0

for uid in uids:
    if uid not in full or uid not in b1:
        continue
    if 4 not in full[uid] or 4 not in b1[uid]:
        continue
    total += 1
    # regression means: some early step correct, but t=4 wrong (in FULL)
    early_ok = any(full[uid].get(t,{}).get("correct") for t in (1,2,3) if t in full[uid])
    if not (early_ok and (not corr(full[uid][4]))):
        continue
    # does b1 get t=4 correct?
    if corr(b1[uid][4]):
        fixed += 1
    else:
        still_wrong += 1

print("Regression cases (FULL):", total)
print("Fixed by B1@t4:", fixed, f"({fixed/total:.3f})")
print("Still wrong in B1@t4:", still_wrong, f"({still_wrong/total:.3f})")
