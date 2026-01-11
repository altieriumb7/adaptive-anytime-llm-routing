import os, json, csv, statistics
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("eval_depth_router", "scripts/eval_depth_router.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SEEDS=[0,1,2]
TARGETS=[1,2,3,4]
THS=[0.50 + i*(0.99-0.50)/79 for i in range(80)]
OUTROOT="artifacts/router_optionB"
Path(OUTROOT).mkdir(parents=True, exist_ok=True)

def mean_std(vals):
    if not vals: return float("nan"), float("nan")
    if len(vals)==1: return vals[0], 0.0
    return statistics.mean(vals), statistics.pstdev(vals)

def choose_conf_threshold(dev_examples, target):
    best=None
    for th in THS:
        r=mod.evaluate(dev_examples, policy="conf", threshold=th, seed=0, calibrator=None)
        score=(abs(r["mean_steps"]-target), -r["acc"])
        if best is None or score < best[0]:
            best=(score, th, r)
    return best[1], best[2]

def choose_stability(dev_examples, target):
    grid=[]
    for m in [2,3,4]:
        for min_step in [1,2]:
            r=mod.evaluate(dev_examples, policy="stability", m=m, min_step=min_step, seed=0, calibrator=None)
            grid.append((abs(r["mean_steps"]-target), -r["acc"], m, min_step, r))
    grid.sort()
    _,_,m,min_step,r=grid[0]
    return m, min_step, r

def save_summary(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)

all_seed_rows=[]

for seed in SEEDS:
    dev_path=f"data/router_splits_seeds/seed{seed}/dev.jsonl"
    test_path=f"data/router_splits_seeds/seed{seed}/test.jsonl"
    out_seed=f"{OUTROOT}_seed{seed}"
    Path(out_seed).mkdir(parents=True, exist_ok=True)

    dev_ex=mod.read_jsonl_grouped(dev_path)
    test_ex=mod.read_jsonl_grouped(test_path)

    rows=[]
    for B in TARGETS:
        budget_tag=f"B{B}"

        # fixed
        rdev=mod.evaluate(dev_ex, policy="fixed", k=B, seed=0, calibrator=None)
        rtest=mod.evaluate(test_ex, policy="fixed", k=B, seed=0, calibrator=None)
        for split,r in [("dev",rdev),("test",rtest)]:
            rows.append(dict(split=split, policy="fixed", budget_tag=budget_tag, params=f"k={B}", acc=r["acc"], mean_tokens=r["mean_tokens"], mean_steps=r["mean_steps"]))

        # conf tuned to match target mean steps
        th, rdev_conf = choose_conf_threshold(dev_ex, B)
        rtest_conf=mod.evaluate(test_ex, policy="conf", threshold=th, seed=0, calibrator=None)
        hist_path=os.path.join(out_seed,f"hist_{budget_tag}.json")
        with open(hist_path,"w",encoding="utf-8") as f: json.dump(rdev_conf["stop_histogram"], f)

        for split,r in [("dev",rdev_conf),("test",rtest_conf)]:
            rows.append(dict(split=split, policy="conf", budget_tag=budget_tag, params=f"thr={th:.4f}", acc=r["acc"], mean_tokens=r["mean_tokens"], mean_steps=r["mean_steps"]))

        # random matched
        rdev_rand=mod.evaluate(dev_ex, policy="random", random_hist_path=hist_path, seed=123, calibrator=None)
        rtest_rand=mod.evaluate(test_ex, policy="random", random_hist_path=hist_path, seed=123, calibrator=None)
        for split,r in [("dev",rdev_rand),("test",rtest_rand)]:
            rows.append(dict(split=split, policy="random", budget_tag=budget_tag, params="matched", acc=r["acc"], mean_tokens=r["mean_tokens"], mean_steps=r["mean_steps"]))

        # stability tuned
        m,min_step,rdev_stab=choose_stability(dev_ex, B)
        rtest_stab=mod.evaluate(test_ex, policy="stability", m=m, min_step=min_step, seed=0, calibrator=None)
        for split,r in [("dev",rdev_stab),("test",rtest_stab)]:
            rows.append(dict(split=split, policy="stability", budget_tag=budget_tag, params=f"m={m};min={min_step}", acc=r["acc"], mean_tokens=r["mean_tokens"], mean_steps=r["mean_steps"]))

    summary_path=os.path.join(out_seed,"summary.csv")
    save_summary(summary_path, rows)
    print("Wrote", summary_path)
    all_seed_rows.append((seed, rows))

# aggregate over seeds (test split only)
policies=["fixed","conf","random","stability"]
out_table=os.path.join(OUTROOT,"paper_table_test_acc_tokens.csv")
Path(OUTROOT).mkdir(parents=True, exist_ok=True)

with open(out_table,"w",encoding="utf-8",newline="") as f:
    w=csv.writer(f)
    w.writerow(["budget_tag"]+policies)
    for B in TARGETS:
        tag=f"B{B}"
        row=[tag]
        for p in policies:
            accs=[]; toks=[]
            for seed,rows in all_seed_rows:
                for r in rows:
                    if r["split"]=="test" and r["budget_tag"]==tag and r["policy"]==p:
                        accs.append(float(r["acc"]))
                        toks.append(float(r["mean_tokens"]))
            ma,_=mean_std(accs)
            mt,_=mean_std(toks)
            row.append(f"{ma:.4f} ({mt:.0f})")
        w.writerow(row)

print("Wrote", out_table)
