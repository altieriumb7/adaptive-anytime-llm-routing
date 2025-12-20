#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Option B: Depth-router evaluation pipeline (sequential, one command)
#
# What this script does:
#  - Sweeps confidence thresholds on DEV to budget-match target mean steps
#  - Runs policy evaluations on DEV and TEST:
#      * fixed-k
#      * confidence threshold (calibrated router)
#      * stability
#      * random matched to confidence stop histogram
#  - Produces:
#      artifacts/router_optionB/*.json  (per-run metrics)
#      artifacts/router_optionB/summary.csv
#
# Usage:
#   chmod +x run_optionB_router.sh
#   ./run_optionB_router.sh
#
# Edit the CONFIG section to match your paths.
###############################################################################

#######################################
# CONFIG (EDIT THESE)
#######################################

# Activate env (edit if needed). If already activated, this will be ignored.
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"

# Trajectory files (JSONL): dev and test
DEV_JSONL="${DEV_JSONL:-data/dev_traj.jsonl}"
TEST_JSONL="${TEST_JSONL:-data/test_traj.jsonl}"

# Where the router scripts live
SCRIPTS_DIR="${SCRIPTS_DIR:-scripts}"

# Output directory
OUTDIR="${OUTDIR:-artifacts/router_optionB}"

# Desired mean-step budgets to match (as a comma list)
TARGET_BUDGETS="${TARGET_BUDGETS:-1,2,3,4}"

# Threshold sweep grid
TH_MIN="${TH_MIN:-0.50}"
TH_MAX="${TH_MAX:-0.99}"
TH_N="${TH_N:-80}"

# Stability policy settings to try (we'll pick best budget match)
# You can extend this list.
STABILITY_M_LIST="${STABILITY_M_LIST:-2 3 4}"

# Minimum step before stability can stop (often helps avoid silly early stops)
STABILITY_MIN_STEP="${STABILITY_MIN_STEP:-2}"

# Seeds
SEED_MAIN="${SEED_MAIN:-0}"
SEED_RANDOM="${SEED_RANDOM:-123}"

#######################################
# Helpers
#######################################

die() { echo "ERROR: $*" >&2; exit 1; }

need_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

json_get() {
  # Usage: json_get FILE KEY
  python - <<PY
import json, sys
p=sys.argv[1]; k=sys.argv[2]
j=json.load(open(p))
v=j
for part in k.split("."):
    if isinstance(v, dict) and part in v: v=v[part]
    else:
        # allow list indexes
        try:
            idx=int(part)
            v=v[idx]
        except Exception:
            print("")
            sys.exit(0)
print(v)
PY "$1" "$2"
}

write_hist() {
  # Usage: write_hist METRICS_JSON OUT_HIST_JSON
  python - <<'PY'
import json, sys
src=sys.argv[1]; dst=sys.argv[2]
j=json.load(open(src))
hist=j.get("stop_histogram", [])
json.dump(hist, open(dst,"w"), indent=2)
print(f"Wrote histogram: {dst} (len={len(hist)})")
PY "$1" "$2"
}

append_csv_row() {
  # Append a single row to summary.csv from a metrics json
  # Columns: split,policy,budget_tag,params,acc,mean_steps,p95_steps,mean_tokens,p95_tokens,flip_rate,regress_at_stop_rate
  local split="$1"
  local budget_tag="$2"
  local metrics_json="$3"

  local policy acc mean_steps p95_steps mean_tokens p95_tokens flip_rate reg_rate
  policy="$(json_get "$metrics_json" "policy")"
  acc="$(json_get "$metrics_json" "acc")"
  mean_steps="$(json_get "$metrics_json" "mean_steps")"
  p95_steps="$(json_get "$metrics_json" "p95_steps")"
  mean_tokens="$(json_get "$metrics_json" "mean_tokens")"
  p95_tokens="$(json_get "$metrics_json" "p95_tokens")"
  flip_rate="$(json_get "$metrics_json" "flip_rate")"
  reg_rate="$(json_get "$metrics_json" "regress_at_stop_rate")"

  # policy params string (best-effort)
  local k thr m min_step
  k="$(json_get "$metrics_json" "k")"
  thr="$(json_get "$metrics_json" "threshold")"
  m="$(json_get "$metrics_json" "m")"
  min_step="$(json_get "$metrics_json" "min_step")"
  local params="k=${k};thr=${thr};m=${m};min_step=${min_step}"

  echo "${split},${policy},${budget_tag},\"${params}\",${acc},${mean_steps},${p95_steps},${mean_tokens},${p95_tokens},${flip_rate},${reg_rate}" \
    >> "${OUTDIR}/summary.csv"
}

#######################################
# Start
#######################################

echo "=== Option B: Depth-router pipeline ==="

# Activate venv if present
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
  echo "Activated venv: $VENV_ACTIVATE"
else
  echo "No venv activation script found at $VENV_ACTIVATE (continuing)."
fi

need_cmd python
need_file "${SCRIPTS_DIR}/eval_depth_router.py"
need_file "${SCRIPTS_DIR}/sweep_conf_thresholds.py"
need_file "$DEV_JSONL"
need_file "$TEST_JSONL"

mkdir -p "$OUTDIR"

echo "Writing CSV header..."
cat > "${OUTDIR}/summary.csv" <<'CSV'
split,policy,budget_tag,params,acc,mean_steps,p95_steps,mean_tokens,p95_tokens,flip_rate,regress_at_stop_rate
CSV

echo "=== 1) Sweep confidence thresholds on DEV to match target budgets: ${TARGET_BUDGETS} ==="
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

echo "Saved sweep grid: $SWEEP_GRID_JSON"
echo "Saved best thresholds: $SWEEP_BEST_JSONL"

echo "=== 2) For each budget, run FIXED and CONF on DEV+TEST; then RANDOM matched ==="

# Iterate over each line in best_thresholds_dev.jsonl
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  B="$(python - <<PY
import json,sys
j=json.loads(sys.argv[1])
print(j["target_mean_steps"])
PY "$line")"
  TH="$(python - <<PY
import json,sys
j=json.loads(sys.argv[1])
print(j["best_threshold"])
PY "$line")"

  budget_tag="B${B}"

  echo ""
  echo "----- Budget target mean_steps ~ ${B}  | chosen threshold=${TH} -----"

  # 2.1 FIXED-k
  K_INT="$(python - <<PY
import math,sys
B=float(sys.argv[1])
print(max(1,int(round(B))))
PY "$B")"

  DEV_FIXED_JSON="${OUTDIR}/dev_fixed_${budget_tag}_k${K_INT}.json"
  TEST_FIXED_JSON="${OUTDIR}/test_fixed_${budget_tag}_k${K_INT}.json"

  python "${SCRIPTS_DIR}/eval_depth_router.py" \
    --data "$DEV_JSONL" --policy fixed --k "$K_INT" --seed "$SEED_MAIN" --out "$DEV_FIXED_JSON"
  python "${SCRIPTS_DIR}/eval_depth_router.py" \
    --data "$TEST_JSONL" --policy fixed --k "$K_INT" --seed "$SEED_MAIN" --out "$TEST_FIXED_JSON"

  append_csv_row "dev" "$budget_tag" "$DEV_FIXED_JSON"
  append_csv_row "test" "$budget_tag" "$TEST_FIXED_JSON"

  # 2.2 CONF router
  DEV_CONF_JSON="${OUTDIR}/dev_conf_${budget_tag}_th${TH}.json"
  TEST_CONF_JSON="${OUTDIR}/test_conf_${budget_tag}_th${TH}.json"

  python "${SCRIPTS_DIR}/eval_depth_router.py" \
    --data "$DEV_JSONL" --policy conf --threshold "$TH" --seed "$SEED_MAIN" --out "$DEV_CONF_JSON"
  python "${SCRIPTS_DIR}/eval_depth_router.py" \
    --data "$TEST_JSONL" --policy conf --threshold "$TH" --seed "$SEED_MAIN" --out "$TEST_CONF_JSON"

  append_csv_row "dev" "$budget_tag" "$DEV_CONF_JSON"
  append_csv_row "test" "$budget_tag" "$TEST_CONF_JSON"

  # 2.3 RANDOM matched to CONF stop histogram (from DEV)
  DEV_HIST_JSON="${OUTDIR}/hist_${budget_tag}_from_dev_conf.json"
  write_hist "$DEV_CONF_JSON" "$DEV_HIST_JSON"

  DEV_RAND_JSON="${OUTDIR}/dev_random_${budget_tag}_matched.json"
  TEST_RAND_JSON="${OUTDIR}/test_random_${budget_tag}_matched.json"

  python "${SCRIPTS_DIR}/eval_depth_router.py" \
    --data "$DEV_JSONL" --policy random --random_hist_path "$DEV_HIST_JSON" --seed "$SEED_RANDOM" --out "$DEV_RAND_JSON"
  python "${SCRIPTS_DIR}/eval_depth_router.py" \
    --data "$TEST_JSONL" --policy random --random_hist_path "$DEV_HIST_JSON" --seed "$SEED_RANDOM" --out "$TEST_RAND_JSON"

  append_csv_row "dev" "$budget_tag" "$DEV_RAND_JSON"
  append_csv_row "test" "$budget_tag" "$TEST_RAND_JSON"

done < "$SWEEP_BEST_JSONL"

echo ""
echo "=== 3) Stability policy sweep (DEV), then run best-matching configs on TEST ==="
# Strategy:
# - For each budget B, we try multiple m values (and fixed min_step) on DEV,
#   pick the one whose mean_steps is closest to B.
# - Then we run that picked stability config on TEST.

# Read budgets list into python-friendly list
python - <<PY
import json, math, os, subprocess, sys

OUTDIR = os.environ["OUTDIR"]
SCRIPTS_DIR = os.environ["SCRIPTS_DIR"]
DEV_JSONL = os.environ["DEV_JSONL"]
TEST_JSONL = os.environ["TEST_JSONL"]
TARGET_BUDGETS = [float(x.strip()) for x in os.environ["TARGET_BUDGETS"].split(",") if x.strip()]
M_LIST = [int(x) for x in os.environ["STABILITY_M_LIST"].split()]
MIN_STEP = int(os.environ["STABILITY_MIN_STEP"])
SEED_MAIN = int(os.environ["SEED_MAIN"])

def run_eval(split, data_path, m, min_step, budget_tag):
    out = f"{OUTDIR}/{split}_stability_{budget_tag}_m{m}_min{min_step}.json"
    cmd = [
        "python", f"{SCRIPTS_DIR}/eval_depth_router.py",
        "--data", data_path,
        "--policy", "stability",
        "--m", str(m),
        "--min_step", str(min_step),
        "--seed", str(SEED_MAIN),
        "--out", out
    ]
    subprocess.check_call(cmd)
    with open(out, "r", encoding="utf-8") as f:
        j = json.load(f)
    return out, j

def append_csv(split, budget_tag, metrics_json):
    # call bash helper via environment
    subprocess.check_call(["bash","-lc", f'append_csv_row "{split}" "{budget_tag}" "{metrics_json}"'])

# Expose bash functions to this python process by re-invoking bash with them defined.
# We'll instead just append CSV directly here to avoid shell funk.
def append_csv_direct(split, budget_tag, j, path):
    csv_path = f"{OUTDIR}/summary.csv"
    # mirror columns used by bash:
    row = [
        split,
        j.get("policy",""),
        budget_tag,
        f'"k={j.get("k")};thr={j.get("threshold")};m={j.get("m")};min_step={j.get("min_step")}"',
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

best_by_budget = {}

for B in TARGET_BUDGETS:
    budget_tag = f"B{B}"
    best = None
    best_out = None
    for m in M_LIST:
        out_path, j = run_eval("dev", DEV_JSONL, m=m, min_step=MIN_STEP, budget_tag=budget_tag)
        # choose closest mean_steps to budget
        dist = abs(float(j["mean_steps"]) - B)
        if best is None or dist < best:
            best = dist
            best_out = (out_path, j)
    best_by_budget[B] = best_out
    out_path, j = best_out
    print(f"[DEV] Best stability for budget {B}: m={j['m']} min_step={j['min_step']} mean_steps={j['mean_steps']:.3f} acc={j['acc']:.4f} ({out_path})")
    append_csv_direct("dev", budget_tag, j, out_path)

# Now run the selected stability configs on TEST
for B, (dev_path, dev_j) in best_by_budget.items():
    budget_tag = f"B{B}"
    m = int(dev_j["m"])
    min_step = int(dev_j["min_step"])
    out_path, j = run_eval("test", TEST_JSONL, m=m, min_step=min_step, budget_tag=budget_tag)
    print(f"[TEST] Stability for budget {B}: m={m} min_step={min_step} mean_steps={j['mean_steps']:.3f} acc={j['acc']:.4f} ({out_path})")
    append_csv_direct("test", budget_tag, j, out_path)

PY

echo ""
echo "=== DONE ==="
echo "All outputs are in: $OUTDIR"
echo "Summary CSV: $OUTDIR/summary.csv"
echo ""
echo "Tip: open the CSV and pivot by budget_tag + split to build the paper table."
