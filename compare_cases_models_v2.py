import csv, json
from collections import defaultdict

REG_TSV = "results_abl/regression_cases_full.tsv"
P_FULL  = "results_abl/preds_abl_full.jsonl"
P_B1    = "results_abl/preds_abl_b1.jsonl"
K = 20

def load_uids():
    uids=[]
    with open(REG_TSV,"r",encoding="utf-8") as f:
        r=csv.DictReader(f, delimiter="\t")
        for row in r:
            uids.append(row["uid"])
            if len(uids)>=K: break
    return set(uids)

def get_uid(o):
    return str(o.get("uid") or o.get("example_id") or o.get("custom_id") or o.get("id"))

def get_t(o):
    for k in ("t","budget","step","k"):
        if k in o:
            try: return int(o[k])
            except: return None
    return None

def get_field(o, key, fallback_keys=()):
    if key in o:
        return o[key]
    for fk in fallback_keys:
        if fk in o:
            return o[fk]
    return None

def load_preds(path, want_uids):
    idx = defaultdict(dict)
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o=json.loads(line)
            uid=get_uid(o)
            if uid not in want_uids: continue
            t=get_t(o)
            if t is None: continue
            idx[uid][t]=o
    return idx

def short(o):
    if not o: return ("-", "-", "-")
    ans = get_field(o, "answer", ("pred_answer","final_answer"))
    conf = get_field(o, "conf", ("confidence","p_correct"))
    corr = get_field(o, "correct", ("is_correct","strict_correct"))
    return (ans, conf, corr)

uids = load_uids()
full = load_preds(P_FULL, uids)
b1   = load_preds(P_B1, uids)

print("uid\tgold\tFULL_t1\tFULL_t2\tFULL_t3\tFULL_t4\tB1_t4")
for uid in sorted(uids):
    gold = (full.get(uid, {}).get(1, {}) or full.get(uid, {}).get(4, {})).get("gold", "")
    f1=short(full.get(uid, {}).get(1))
    f2=short(full.get(uid, {}).get(2))
    f3=short(full.get(uid, {}).get(3))
    f4=short(full.get(uid, {}).get(4))
    b4=short(b1.get(uid, {}).get(4))
    print(f"{uid}\t{gold}\t{f1}\t{f2}\t{f3}\t{f4}\t{b4}")
