#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SOURCE_PREDS="results/preds_student_full.jsonl"
SEEDS="0,1,2"
SPLIT_MANIFEST="data/router_splits_seeds/manifest.json"
ROUTER_OUTDIR="artifacts/router_optionB"

echo "[INFO] Building seeded router splits from canonical source: ${SOURCE_PREDS}"
python scripts/make_router_splits.py \
  --source "${SOURCE_PREDS}" \
  --out_root data/router_splits_seeds \
  --seeds "${SEEDS}" \
  --manifest "${SPLIT_MANIFEST}"

echo "[INFO] Recomputing canonical GSM8K router CSV outputs"
python scripts/run_router_optionB_repro.py --out_dir "${ROUTER_OUTDIR}" --seeds "${SEEDS}"

echo "[INFO] Regenerating GSM8K router LaTeX table"
python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table.tex \
  --label tab:main_results_allbudgets \
  --split_label test \
  --split_filter test \
  --oracle_everywhere

echo "[INFO] Refreshing paper-wide tables/figures and validating assets"
python scripts/make_paper_artifacts.py --config configs/paper.yaml
python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex

echo "[INFO] Canonical GSM8K router pipeline complete."
