import csv, json
from collections import defaultdict

REG_TSV = "results_abl/regression_cases_full.tsv"
PREDS   = "results_abl/preds_abl_full.jsonl"
K = 10  # how many to print

# load regression uids (first K)
uids = []
with open(REG_TSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        uids.append(row["uid"])
        if len(uids) >= K:
            break
uids = set(uids)

# index preds by uid -> t -> record
idx = defaultdict(dict)
with open(PREDS, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        o = json.loads(line)
        uid = str(o.get("uid") or o.get("example_id") or o.get("custom_id") or o.get("id"))
        if uid not in uids:
            continue
        t = o.get("t") or o.get("budget") or o.get("step")
        if t is None:
            continue
        try:
            t = int(t)
        except:
            continue
        idx[uid][t] = o

# pretty print
for uid in sorted(idx.keys()):
    print("="*100)
    print("UID:", uid)
    base = idx[uid].get(1, idx[uid].get(4, {}))
    print("PROBLEM:\n", base.get("problem","<missing problem>"))
    print("GOLD:", base.get("gold","<missing gold>"))
    for t in (1,2,3,4):
        r = idx[uid].get(t, {})
        if not r:
            continue
        conf = r.get("conf")
        ans  = r.get("answer") or r.get("pred_answer")
        corr = r.get("correct")
        raw  = r.get("raw") or r.get("pred") or r.get("text") or r.get("prediction") or ""
        print("\n--- t=%d --- conf=%s correct=%s ans=%s" % (t, str(conf), str(corr), str(ans)))
        print(raw.strip())
