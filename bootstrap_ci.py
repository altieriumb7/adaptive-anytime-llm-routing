import json, random
from collections import defaultdict

SEED = 0
N_BOOT = 20000  # good standard; reduce to 5000 if slow
TAU = 0.71

FULL = "results_abl/preds_abl_full.jsonl"
B1   = "results_abl/preds_abl_b1.jsonl"

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
b1   = load_uid_t(B1)

uids = sorted([u for u in full.keys() if 4 in full[u]])
# keep only uids that exist in both (safe)
uids = [u for u in uids if 4 in b1.get(u, {})]

# Build paired 0/1 arrays for metrics we care about
y_full_t4 = []
y_full_tau = []
y_b1_t4 = []

for uid in uids:
    steps_full = full[uid]
    steps_b1   = b1[uid]

    y_full_t4.append(int(steps_full[4]["correct"]))
    t_tau = pick_thresh(steps_full, TAU)
    y_full_tau.append(int(steps_full[t_tau]["correct"]))
    y_b1_t4.append(int(steps_b1[4]["correct"]))

n = len(uids)
print("N paired examples:", n)

def mean(arr):
    return sum(arr)/len(arr)

def bootstrap_ci(delta_fn, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        # sample indices with replacement
        idxs = [rng.randrange(n) for _ in range(n)]
        deltas.append(delta_fn(idxs))
    deltas.sort()
    lo = deltas[int(0.025*n_boot)]
    hi = deltas[int(0.975*n_boot)]
    prob_pos = sum(1 for d in deltas if d > 0)/n_boot
    return lo, hi, prob_pos

# Observed deltas
obs_full_tau_vs_t4 = mean(y_full_tau) - mean(y_full_t4)
obs_b1_vs_full = mean(y_b1_t4) - mean(y_full_t4)

print("\n== Observed ==")
print(f"full_tau - full_t4 = {obs_full_tau_vs_t4:.6f}")
print(f"b1_t4   - full_t4  = {obs_b1_vs_full:.6f}")

# Bootstrap paired deltas
def d_full_tau_vs_t4(idxs):
    return (sum(y_full_tau[i] for i in idxs)/len(idxs)) - (sum(y_full_t4[i] for i in idxs)/len(idxs))

def d_b1_vs_full(idxs):
    return (sum(y_b1_t4[i] for i in idxs)/len(idxs)) - (sum(y_full_t4[i] for i in idxs)/len(idxs))

lo, hi, ppos = bootstrap_ci(d_full_tau_vs_t4)
print("\n== 95% CI (paired bootstrap) ==")
print(f"full_tau - full_t4: [{lo:.6f}, {hi:.6f}]  P(Δ>0)={ppos:.4f}")

lo, hi, ppos = bootstrap_ci(d_b1_vs_full)
print(f"b1_t4 - full_t4:   [{lo:.6f}, {hi:.6f}]  P(Δ>0)={ppos:.4f}")
