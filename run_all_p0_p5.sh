#!/usr/bin/env bash
set -euo pipefail

#############################################
# USER SETTINGS (EDIT THESE)
#############################################
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"   # <-- EDIT if needed
DATASET="${DATASET:-gsm8k}"
SPLIT="${SPLIT:-test}"

TRAIN_JSONL="${TRAIN_JSONL:-data/sft_gsm8k_anytime_v2.jsonl}"
STUDENT_DIR="${STUDENT_DIR:-artifacts/student_anytime_qlora}"

# Anytime budgets and chunk token sizes (cumulative tokens become 96,256,480,800)
BUDGETS="${BUDGETS:-1,2,3,4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96,160,224,320}"

# Paper-scale eval (gsm8k test has 1319; 5000 just means “all”)
MAX_EXAMPLES="${MAX_EXAMPLES:-5000}"

# Router split settings
ROUTER_TEST_FRAC="${ROUTER_TEST_FRAC:-0.2}"
ROUTER_SEEDS="${ROUTER_SEEDS:-0 1 2}"
export ROUTER_TEST_FRAC ROUTER_SEEDS

#############################################
# LOGGING
#############################################
mkdir -p logs results artifacts
exec > >(tee -a logs/run_p0_p5_$(date +%Y%m%d_%H%M%S).log) 2>&1

echo "== Running from repo root: $(pwd)"
echo "== BASE_MODEL: ${BASE_MODEL}"
echo "== DATASET/SPLIT: ${DATASET}/${SPLIT}"
echo "== BUDGETS: ${BUDGETS}  MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "== MAX_EXAMPLES: ${MAX_EXAMPLES}"
echo "== ROUTER_TEST_FRAC: ${ROUTER_TEST_FRAC}  ROUTER_SEEDS: ${ROUTER_SEEDS}"

#############################################
# 0) ENV + DEPS
#############################################
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

# Optional venv (uncomment if you want)
# python -m venv .venv
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.train.txt
pip install -r requirements.paper.txt

#############################################
# 0.1 Repo sanity: source files already contain the reproducibility fixes
#############################################

#############################################
# P0) Sanity checks
#############################################
echo "== P0: sanity checks"
python scripts/sanity_check_dataset.py --path "${TRAIN_JSONL}" --expected_ts 1,2,3,4 || true
if [ -f data/anytime_gsm8k_train_v2.jsonl ]; then
  python scripts/sanity_check_dataset.py --path data/anytime_gsm8k_train_v2.jsonl --expected_ts 1,2,3,4 || true
fi

#############################################
# P1) (Optional) Clean trajectories if file exists
#############################################
if [ -f data/anytime_gsm8k_train_v2.jsonl ]; then
  echo "== P1: cleaning trajectories"
  python scripts/clean_trajectories.py \
    --in  data/anytime_gsm8k_train_v2.jsonl \
    --out data/anytime_gsm8k_train_v2_clean.jsonl \
    --expected_ts 1,2,3,4 || true
fi

#############################################
# P3) Train student (QLoRA)
#############################################
if [ -d "${STUDENT_DIR}" ] && [ -f "${STUDENT_DIR}/adapter_model.safetensors" -o -f "${STUDENT_DIR}/adapter_model.bin" ]; then
  echo "== P3: student already exists at ${STUDENT_DIR}, skipping training"
else
  echo "== P3: training student -> ${STUDENT_DIR}"
  python scripts/train_student_qlora.py \
    --base_model "${BASE_MODEL}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --output_dir "${STUDENT_DIR}" \
    --epochs 1 \
    --lr 2e-4 \
    --batch_size 1 \
    --grad_accum 16
fi

#############################################
# P4) Anytime eval (BASE + STUDENT) + JSONL dumps (for paper)
#############################################
BASE_PREDS="results/preds_base_full.jsonl"
STUDENT_PREDS="results/preds_student_full.jsonl"

if [ -f "${BASE_PREDS}" ]; then
  echo "== P4: base preds already exist: ${BASE_PREDS} (skipping)"
else
  echo "== P4: eval base -> ${BASE_PREDS}"
  python scripts/eval_anytime.py \
    --base_model "${BASE_MODEL}" \
    --dataset "${DATASET}" --split "${SPLIT}" --max_examples "${MAX_EXAMPLES}" \
    --budgets "${BUDGETS}" --max_new_tokens "${MAX_NEW_TOKENS}" \
    --save_jsonl "${BASE_PREDS}" | tee results/eval_base.log
fi

if [ -f "${STUDENT_PREDS}" ]; then
  echo "== P4: student preds already exist: ${STUDENT_PREDS} (skipping)"
else
  echo "== P4: eval student -> ${STUDENT_PREDS}"
  python scripts/eval_anytime.py \
    --base_model "${BASE_MODEL}" \
    --adapter_dir "${STUDENT_DIR}" \
    --dataset "${DATASET}" --split "${SPLIT}" --max_examples "${MAX_EXAMPLES}" \
    --budgets "${BUDGETS}" --max_new_tokens "${MAX_NEW_TOKENS}" \
    --save_jsonl "${STUDENT_PREDS}" | tee results/eval_student.log
fi

#############################################
# P2) Router splits (3 seeds) + fit confidence calibrator on seed0 dev
#############################################
echo "== P2: making router splits from student preds"
python - <<PY
import json, os, random
IN="${STUDENT_PREDS}"
OUTROOT="data/router_splits_seeds"
TEST_FRAC=float("${ROUTER_TEST_FRAC}")
SEEDS = [int(s) for s in "${ROUTER_SEEDS}".replace(",", " ").split()]
router_seeds = SEEDS

rows=[]
uids=set()
with open(IN,"r",encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        o=json.loads(line)
        uid=str(o.get("uid",""))
        if not uid:
            continue
        rows.append(o)
        uids.add(uid)

uids=sorted(uids)
print("Total uids:", len(uids), " total rows:", len(rows))

for seed in SEEDS:
    rng=random.Random(int(seed))
    n_test=max(1, int(len(uids)*TEST_FRAC))
    test=set(rng.sample(uids,n_test))
    outdir=os.path.join(OUTROOT,f"seed{seed}")
    os.makedirs(outdir, exist_ok=True)
    dev_path=os.path.join(outdir,"dev.jsonl")
    test_path=os.path.join(outdir,"test.jsonl")
    with open(dev_path,"w",encoding="utf-8") as fdev, open(test_path,"w",encoding="utf-8") as ftest:
        for o in rows:
            uid=str(o.get("uid",""))
            (ftest if uid in test else fdev).write(json.dumps(o,ensure_ascii=False)+"\n")
    print("seed",seed,"dev:",dev_path,"test:",test_path)
PY

echo "== P2: fitting Platt calibrators (one per seed)"
mkdir -p artifacts/calibration
for seed in ${ROUTER_SEEDS}; do
  python scripts/fit_conf_calibrator.py \
    --in  data/router_splits_seeds/seed${seed}/dev.jsonl \
    --out artifacts/calibration/platt_seed${seed}.json \
    --method platt \
    --min_points 200
done

#############################################
# P4 Router Option-B (3 seeds) -> paper_table_test_acc_tokens.csv
#############################################
echo "== P4 Router: running Option-B router evaluation (3 seeds) -> artifacts/router_optionB"

echo "== P4 Router: running Option-B router evaluation (Strada A, compute-matched)"
python scripts/run_router_optionB_repro.py

#############################################
# P5) Paper artifacts (figures + tables)
#############################################
echo "== P5: updating configs/paper.yaml to point to our preds files"
python - <<'PY'
import yaml, pathlib
p = pathlib.Path("configs/paper.yaml")
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
cfg["router_csv"] = "artifacts/router_optionB/paper_table_test_acc_tokens.csv"
cfg["models"] = [
  {"name":"student", "preds_jsonl":"results/preds_student_full.jsonl"},
  {"name":"base",    "preds_jsonl":"results/preds_base_full.jsonl"},
]
p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("updated configs/paper.yaml")
PY

echo "== P5: generating paper figures/tables -> artifacts/paper/"
bash run_paper.sh

# Generate a LaTeX router table for inclusion in the paper
python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table.tex || true

echo ""
echo "========================================"
echo "DONE."
echo "Key outputs:"
echo " - Student adapter: ${STUDENT_DIR}"
echo " - Base preds:      ${BASE_PREDS}"
echo " - Student preds:   ${STUDENT_PREDS}"
echo " - Calibrators:     artifacts/calibration/platt_seed{0,1,2}.json"
echo " - Router table:    artifacts/router_optionB/paper_table_test_acc_tokens.csv"
echo " - Paper artifacts: artifacts/paper/figures  and  artifacts/paper/tables"
echo "========================================"
