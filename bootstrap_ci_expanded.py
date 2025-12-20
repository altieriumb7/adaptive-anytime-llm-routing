import json, random
from collections import defaultdict

SEED = 0
N_BOOT = 20000
TAU = 0.71

FULL = "results_abl/preds_abl_full.jsonl"

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

            idx[str(uid)][t] = {"correct": bool(correct), "conf": conf}
    return idx

def pick_thresh(steps, tau):
    for t in (1,2,3,4):
        c = steps.get(t, {}).get("conf")
        if c is not None and c >= tau:
            return t
    return 4

full = load_uid_t(FULL)
uids = sorted([u for u in full.keys() if 4 in full[u]])
n = len(uids)
print("N examples:", n)

# Precompute per-example quantities for two systems:
# A) always t=4
# B) threshold policy (tau)

acc_t4 = []
acc_tau = []
t_chosen = []

reg_t4 = []   # 1 if regression under that selection else 0
reg_tau = []

for uid in uids:
    steps = full[uid]
    # always t=4
    c4 = int(steps[4]["correct"])
    acc_t4.append(c4)

    early_correct = any(steps.get(tt, {}).get("correct") for tt in (1,2,3) if tt in steps)
    reg_t4.append(int(early_correct and (not bool(steps[4]["correct"]))))

    # threshold
    tt = pick_thresh(steps, TAU)
    t_chosen.append(tt)
    ct = int(steps[tt]["correct"])
    acc_tau.append(ct)
    reg_tau.append(int(early_correct and (not bool(steps[tt]["correct"]))))

def mean(arr):
    return sum(arr)/len(arr)

def bootstrap_ci(delta_fn, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        deltas.append(delta_fn(idxs))
    deltas.sort()
    lo = deltas[int(0.025*n_boot)]
    hi = deltas[int(0.975*n_boot)]
    prob_pos = sum(1 for d in deltas if d > 0)/n_boot
    return lo, hi, prob_pos

print("\n== Point estimates ==")
print("Acc(tau)-Acc(t4):", mean(acc_tau)-mean(acc_t4))
print("MeanT(tau)-MeanT(t4):", mean(t_chosen)-4.0)
print("Reg(tau)-Reg(t4):", mean(reg_tau)-mean(reg_t4))

# Accuracy delta
lo, hi, ppos = bootstrap_ci(lambda I: (sum(acc_tau[i] for i in I)/n) - (sum(acc_t4[i] for i in I)/n))
print("\nAcc Δ 95% CI:", (lo, hi), "P(Δ>0)=", ppos)

# Mean budget delta (negative is good)
lo, hi, pneg = bootstrap_ci(lambda I: (sum(t_chosen[i] for i in I)/n) - 4.0)
print("MeanT Δ 95% CI:", (lo, hi), "P(Δ<0)=", sum(1 for _ in range(1) if True) and None)

# Regression delta (negative is good)
lo, hi, pneg = bootstrap_ci(lambda I: (sum(reg_tau[i] for i in I)/n) - (sum(reg_t4[i] for i in I)/n))
print("Reg Δ 95% CI:", (lo, hi), "Note: negative means fewer regressions.")
