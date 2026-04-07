#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [ -f requirements.paper.txt ]; then
  echo "[INFO] Installing paper requirements from requirements.paper.txt"
  if ! pip install -r requirements.paper.txt; then
    echo "[WARN] Unable to install requirements.paper.txt in this environment."
    echo "       Continuing with currently installed packages."
  fi
fi

echo "[INFO] Regenerating paper figures/tables from canonical artifacts"
if ! python scripts/make_paper_artifacts.py --config configs/paper.yaml; then
  echo "[ERROR] Failed to generate paper figures/tables."
  echo "        Ensure Python dependencies (e.g., matplotlib, pyyaml, numpy) are installed."
  exit 1
fi

GSM8K_CSV="artifacts/router_optionB/paper_table_test_full_per_seed.csv"
BOOLQ_CSV="artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv"

if [ ! -f "${GSM8K_CSV}" ]; then
  echo "[ERROR] Missing required GSM8K router CSV: ${GSM8K_CSV}"
  echo "        Regenerate it with scripts/run_router_optionB_repro.py (or restore canonical artifact)."
  exit 1
fi

if [ ! -f "${BOOLQ_CSV}" ]; then
  echo "[ERROR] Missing required BoolQ router CSV: ${BOOLQ_CSV}"
  echo "        Regenerate it with scripts/run_router_optionB_boolq.py (or restore canonical artifact)."
  exit 1
fi

echo "[INFO] Regenerating GSM8K router LaTeX table"
python scripts/make_router_latex_table.py \
  --in_csv "${GSM8K_CSV}" \
  --out_tex artifacts/paper/tables/router_table.tex \
  --label tab:main_results_allbudgets \
  --split_label test \
  --split_filter test \
  --oracle_everywhere

echo "[INFO] Regenerating BoolQ router LaTeX table (canonical validation semantics)"
python scripts/make_router_latex_table.py \
  --in_csv "${BOOLQ_CSV}" \
  --out_tex artifacts/paper/tables/router_table_boolq.tex \
  --label tab:router_boolq \
  --split_label validation \
  --split_filter validation \
  --legacy_split_aliases test \
  --oracle-single-reference

echo "[INFO] Validating LaTeX dependencies"
python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex

if command -v pdflatex >/dev/null 2>&1; then
  echo "[INFO] pdflatex found; compiling paper"
  pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex
  pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex
  echo "[INFO] Paper compile complete"
else
  echo "[WARN] pdflatex not found; skipped TeX compilation."
  echo "       Artifact generation and asset checks completed successfully."
fi

echo "[INFO] Paper artifact workflow complete. See artifacts/paper/"
