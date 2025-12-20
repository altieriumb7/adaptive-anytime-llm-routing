import json
from collections import defaultdict

P="results_abl/preds_abl_full.jsonl"

# build uid->t->pred_ans, correct
idx=defaultdict(dict)
with open(P,"r",encoding="utf-8") as f:
    for line in f:
        o=json.loads(line)
        uid=o.get("uid") or o.get("id") or o.get("custom_id")
        t=o.get("t") or o.get("budget") or o.get("step")
        if uid is None or t is None:
            continue
        try: t=int(t)
        except: continue
        pred=o.get("pred_answer") or o.get("answer") or o.get("final_answer")
        if pred is None and "raw" in o:
            pred=o["raw"]
        idx[str(uid)][t]=pred

flip_34=0
total=0
for uid,ts in idx.items():
    if 3 in ts and 4 in ts:
        total += 1
        flip_34 += int(str(ts[3]).strip()!=str(ts[4]).strip())
print("Has both t3,t4:", total)
print("Answer changed t3->t4:", flip_34, "rate=", (flip_34/total if total else 0))
