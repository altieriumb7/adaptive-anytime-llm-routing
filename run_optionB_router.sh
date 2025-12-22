#!/usr/bin/env bash
set -euo pipefail

VENV_ACTIVATE="${VENV_ACTIVATE:-/venv/main/bin/activate}"

DEV_JSONL="${DEV_JSONL:-./results_abl/preds_main_adapter.jsonl}"
TEST_JSONL="${TEST_JSONL:-./results_main_full/preds_main_adapter_full.jsonl}"

SCRIPTS_DIR="${SCRIPTS_DIR:-scripts}"
OUTDIR="${OUTDIR:-artifacts/router_optionB}"

TARGET_BUDGETS="${TARGET_BUDGETS:-1,2,3,4}"
TH_MIN="${TH_MIN:-0.50}"
TH_MAX="${TH_MAX:-0.99}"
TH_N="${TH_N:-80}"

STABILITY_M_LIST="${STABILITY_M_LIST:-2 3 4}"
STABILITY_MIN_STEP="${STABILITY_MIN_STEP:-2}"

SEED_MAIN="${SEED_MAIN:-0}"
SEED_RANDOM="${SEED_RANDOM:-123}"

die() { echo "ERROR: $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

echo "=== Option B: Depth-router pipeline ==="

if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE" || true
  echo "Activated venv: $VENV_ACTIVATE"
fi

need_cmd python
need_file "${SCRIPTS_DIR}/eval_depth_router.py"
need_file "${SCRIPTS_DIR}/sweep_conf_thresholds.py"
need_file "$DEV_JSONL"
need_file "$TEST_JSONL"

echo "Using DEV_JSONL:  $DEV_JSONL"
echo "Using TEST_JSONL: $TEST_JSONL"

mkdir -p "$OUTDIR"
SUMMARY_CSV="${OUTDIR}/summary.csv"
echo "split,policy,budget_tag,params,acc,mean_steps,p95_steps,mean_tokens,p95_tokens,flip_rate,regress_at_stop_rate" > "$SUMMARY_CSV"

HELPER_PY="${OUTDIR}/_router_helper.py"
cat > "$HELPER_PY" <<'EOF'
import json, sys

def loadj(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_hist(metrics_json, out_hist):
    j = loadj(metrics_json)
    hist = j.get("stop_histogram", [])
    with open(out_hist, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

def append_csv(csv_path, split, budget_tag, metrics_json):
    j = loadj(metrics_json)
    row = [
        split,
        j.get("policy",""),
        budget_tag,
        f"\"k={j.get('k','')};thr={j.get('threshold','')};m={j.get('m','')};min_step={j.get('min_step','')}\"",
        str(j.get("acc","")),
        str(j.get("mean_steps","")),
        str(j.get("p95_steps","")),
        str(j.get("mean_tokens","")),
        str(j.get("p95_tokens","")),
        str(j.get("flip_rate","")),
        str(j.get("regress_at_stop_rate","")),
    ]
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(",".join(row) + "\n")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "write_hist":
        write_hist(sys.argv[2], sys.argv[3])
    elif cmd == "append_csv":
        append_csv(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        raise SystemExit("unknown cmd")
EOF

echo "=== 1) Sweep confidence thresholds on DEV ==="
SWEEP_GRID_JSON="${OUTDIR}/conf_sweep_grid_dev.json"
SWEEP_BEST_JSONL="${OUTDIR}/best_thresholds_dev.jsonl"

python "${SCRIPTS_DIR}/sweep_conf_thresholds.py" \
  --dev "$DEV_JSONL" \
  --threshold_min "$TH_MIN" \
  --threshold_max "$TH_MAX" \
  --n "$TH_N" \
  --target_mean_steps "$TARGET_BUDGETS" \
  --seed "$SEED_MAIN" \
  --out_grid "$SWEEP_GRID_JSON" | tee "$SWEEP_BEST_JSONL"

echo "=== 2) fixed/conf/random-matched on DEV+TEST ==="
while IFS= read -r line; do
  [[ -z "$line" ]] && continue

  B="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["target_mean_steps"])' "$line")"
  TH="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["best_threshold"])' "$line")"
  budget_tag="B${B}"
  echo "----- Budget ${B} | threshold=${TH} -----"

  K_INT="$(python -c 'import sys; B=float(sys.argv[1]); print(max(1,int(round(B))))' "$B")"

  DEV_FIXED_JSON="${OUTDIR}/dev_fixed_${budget_tag}_k${K_INT}.json"
  TEST_FIXED_JSON="${OUTDIR}/test_fixed_${budget_tag}_k${K_INT}.json"
  python "${SCRIPTS_DIR}/eval_depth_router.py" --data "$DEV_JSONL" --policy fixed --k "$K_INT" --seed "$SEED_MAIN" --out "$DEV_FIXED_JSON"
  python "${SCRIPTS_DIR}/eval_depth_router.py" --data "$TEST_JSONL" --policy fixed --k "$K_INT" --seed "$SEED_MAIN" --out "$TEST_FIXED_JSON"
  python "$HELPER_PY" append_csv "$SUMMARY_CSV" dev "$budget_tag" "$DEV_FIXED_JSON"
  python "$HELPER_PY" append_csv "$SUMMARY_CSV" test "$budget_tag" "$TEST_FIXED_JSON"

  DEV_CONF_JSON="${OUTDIR}/dev_conf_${budget_tag}_th${TH}.json"
  TEST_CONF_JSON="${OUTDIR}/test_conf_${budget_tag}_th${TH}.json"
  python "${SCRIPTS_DIR}/eval_depth_router.py" --data "$DEV_JSONL" --policy conf --threshold "$TH" --seed "$SEED_MAIN" --out "$DEV_CONF_JSON"
  python "${SCRIPTS_DIR}/eval_depth_router.py" --data "$TEST_JSONL" --policy conf --threshold "$TH" --seed "$SEED_MAIN" --out "$TEST_CONF_JSON"
  python "$HELPER_PY" append_csv "$SUMMARY_CSV" dev "$budget_tag" "$DEV_CONF_JSON"
  python "$HELPER_PY" append_csv "$SUMMARY_CSV" test "$budget_tag" "$TEST_CONF_JSON"

  DEV_HIST_JSON="${OUTDIR}/hist_${budget_tag}_from_dev_conf.json"
  python "$HELPER_PY" write_hist "$DEV_CONF_JSON" "$DEV_HIST_JSON"

  DEV_RAND_JSON="${OUTDIR}/dev_random_${budget_tag}_matched.json"
  TEST_RAND_JSON="${OUTDIR}/test_random_${budget_tag}_matched.json"
  python "${SCRIPTS_DIR}/eval_depth_router.py" --data "$DEV_JSONL" --policy random --random_hist_path "$DEV_HIST_JSON" --seed "$SEED_RANDOM" --out "$DEV_RAND_JSON"
  python "${SCRIPTS_DIR}/eval_depth_router.py" --data "$TEST_JSONL" --policy random --random_hist_path "$DEV_HIST_JSON" --seed "$SEED_RANDOM" --out "$TEST_RAND_JSON"
  python "$HELPER_PY" append_csv "$SUMMARY_CSV" dev "$budget_tag" "$DEV_RAND_JSON"
  python "$HELPER_PY" append_csv "$SUMMARY_CSV" test "$budget_tag" "$TEST_RAND_JSON"

done < "$SWEEP_BEST_JSONL"

echo "=== 3) Stability tune on DEV, run on TEST ==="
python -c '
import json, os, subprocess

SCRIPTS_DIR=os.environ.get("SCRIPTS_DIR","scripts")
OUTDIR=os.environ.get("OUTDIR","artifacts/router_optionB")
DEV_JSONL=os.environ.get("DEV_JSONL","./results_abl/preds_main_adapter.jsonl")
TEST_JSONL=os.environ.get("TEST_JSONL","./results_main_full/preds_main_adapter_full.jsonl")
SUMMARY=os.path.join(OUTDIR,"summary.csv")
HELPER=os.path.join(OUTDIR,"_router_helper.py")

targets=[x.strip() for x in os.environ.get("TARGET_BUDGETS","1,2,3,4").split(",") if x.strip()]
m_list=[int(x) for x in os.environ.get("STABILITY_M_LIST","2 3 4").split()]
min_step=int(os.environ.get("STABILITY_MIN_STEP","2"))
seed=int(os.environ.get("SEED_MAIN","0"))

def run(split, data, tag, m):
    out=os.path.join(OUTDIR, f"{split}_stability_{tag}_m{m}_min{min_step}.json")
    subprocess.check_call(["python", os.path.join(SCRIPTS_DIR,"eval_depth_router.py"),
                           "--data", data, "--policy","stability",
                           "--m", str(m), "--min_step", str(min_step),
                           "--seed", str(seed), "--out", out])
    with open(out,"r",encoding="utf-8") as f: j=json.load(f)
    return out, j

def append(split, tag, out):
    subprocess.check_call(["python", HELPER, "append_csv", SUMMARY, split, tag, out])

best_cfg = {}  # key: budget string -> (m, min_step)

for Bstr in targets:
    B=float(Bstr)
    tag=f"B{Bstr}"
    bestdist=None; bestout=None; bestj=None
    for m in m_list:
        out,j=run("dev", DEV_JSONL, tag, m)
        dist=abs(float(j["mean_steps"]) - B)
        if bestdist is None or dist < bestdist:
            bestdist=dist; bestout=out; bestj=j
    best_cfg[Bstr]=(int(bestj["m"]), int(bestj["min_step"]))
    print(f"[DEV] Best stability {tag}: m={bestj['m']} mean_steps={bestj['mean_steps']:.3f} acc={bestj['acc']:.4f}")
    append("dev", tag, bestout)

for Bstr,(m,ms) in best_cfg.items():
    tag=f"B{Bstr}"
    out,j=run("test", TEST_JSONL, tag, m)
    print(f"[TEST] Stability {tag}: m={m} mean_steps={j['mean_steps']:.3f} acc={j['acc']:.4f}")
    append("test", tag, out)
'

echo "=== DONE ==="
echo "Outputs: $OUTDIR"
echo "Summary CSV: $SUMMARY_CSV"
