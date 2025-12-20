import csv, json
from collections import defaultdict

REG_TSV = "results_abl/regression_cases_full.tsv"
PREDS   = "results_abl/preds_abl_full.jsonl"
K = 10
MAX_CHARS = 1200  # truncate long generations

def pick_text(o):
    # prefer raw_text; fall back to other names if any
    return (o.get("raw_text")
            or o.get("raw")
            or o.get("text")
            or o.get("prediction")
            or o.get("pred")
            or "")

# load first K regression uids
uids = []
with open(REG_TSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        uids.append(row["uid"])
        if len(uids) >= K:
            break
uids = set(uids)

# index preds by uid -> t
idx = defaultdict(dict)
with open(PREDS, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        o = json.loads(line)
        uid = str(o.get("uid") or o.get("example_id") or o.get("custom_id") or o.get("id"))
        if uid not in uids:
            continue
        t = o.get("t") or o.get("budget") or o.get("step") or o.get("k")
        if t is None:
            continue
        try:
            t = int(t)
        except:
            continue
        idx[uid][t] = o

def fmt(x):
    return "None" if x is None else str(x)

for uid in sorted(idx.keys()):
    print("=" * 100)
    print("UID:", uid)
    base = idx[uid].get(1, idx[uid].get(4, {}))
    print("GOLD:", base.get("gold"))
    print("PROBLEM:\n", base.get("problem","").strip())

    for t in (1,2,3,4):
        o = idx[uid].get(t)
        if not o:
            continue
        ans = o.get("answer")
        conf = o.get("conf")
        corr = o.get("correct")
        txt = pick_text(o).strip().replace("\r\n","\n")
        if len(txt) > MAX_CHARS:
            txt = txt[:MAX_CHARS] + "\n...[TRUNCATED]..."
        print(f"\n--- t={t} --- ans={fmt(ans)} conf={fmt(conf)} correct={fmt(corr)}")
        print(txt)
