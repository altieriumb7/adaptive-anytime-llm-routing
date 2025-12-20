import csv, json
from collections import defaultdict

REG_TSV = "results_abl/regression_cases_full.tsv"

P_FULL = "results_abl/preds_abl_full.jsonl"
P_B1   = "results_abl/preds_abl_b1.jsonl"

K = 20  # how many regression uids to inspect

def load_uids():
    uids = []
    with open(REG_TSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            uids.append(row["uid"])
            if len(uids) >= K:
                break
    return set(uids)

def get_uid(o):
    return str(o.get("uid") or o.get("example_id") or o.get("custom_id") or o.get("id"))

def get_t(o):
    t = o.get("t") or o.get("budget") or o.get("step") or o.get("k")
    try:
        return int(t)
    except:
        return None

def load_preds(path, want_uids):
    idx = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            uid = get_uid(o)
            if uid not in want_uids:
                continue
            t = get_t(o)
            if t is None:
                continue
            idx[uid][t] = o
    return idx

uids = load_uids()
full = load_preds(P_FULL, uids)
b1   = load_preds(P_B1, uids)

def short(o):
    if not o:
        return ("-", "-", "-")
    ans = o.get("answer") or o.get("pred_answer") or o.get("final_answer")
    conf = o.get("conf") or o.get("confidence")
    corr = o.get("correct") or o.get("is_correct")
    return (str(ans), str(conf), str(corr))

print("uid\tgold\tFULL_t1\tFULL_t2\tFULL_t3\tFULL_t4\tB1_t4")
for uid in sorted(uids):
    gold = (full.get(uid, {}).get(1, {}) or full.get(uid, {}).get(4, {})).get("gold", "")
    f1 = short(full.get(uid, {}).get(1))
    f2 = short(full.get(uid, {}).get(2))
    f3 = short(full.get(uid, {}).get(3))
    f4 = short(full.get(uid, {}).get(4))
    b4 = short(b1.get(uid, {}).get(4))
    print(f"{uid}\t{gold}\t{f1}\t{f2}\t{f3}\t{f4}\t{b4}")
