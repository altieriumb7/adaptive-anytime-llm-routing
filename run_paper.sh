#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PAPER_CONFIG="configs/paper.yaml"
VARIANT="10page"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      PAPER_CONFIG="${2:-}"
      if [ -z "${PAPER_CONFIG}" ]; then
        echo "[ERROR] --config requires a path argument."
        exit 1
      fi
      shift 2
      ;;
    --variant)
      VARIANT="${2:-}"
      if [ -z "${VARIANT}" ]; then
        echo "[ERROR] --variant requires an argument: 10page|long"
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo "Usage: bash run_paper.sh [--config <path>] [--variant 10page|long]"
      exit 1
      ;;
  esac
done

case "${VARIANT}" in
  10page)
    TEX_SOURCE="main_distilling_revised_v0_10page.tex"
    JOB_NAME="paper_10page"
    OUTPUT_PDF="build/paper_10page.pdf"
    PAGE_LIMIT=10
    ;;
  long)
    TEX_SOURCE="main_distilling_revised_v0.tex"
    JOB_NAME="paper_long"
    OUTPUT_PDF="build/paper_long.pdf"
    PAGE_LIMIT=9999
    ;;
  *)
    echo "[ERROR] Unsupported variant '${VARIANT}'. Use 10page or long."
    exit 1
    ;;
esac

if [ ! -f "${TEX_SOURCE}" ]; then
  echo "[ERROR] TeX source for variant '${VARIANT}' not found: ${TEX_SOURCE}"
  echo "        Refusing to fall back to another entrypoint."
  exit 1
fi

mkdir -p build

if [ -f requirements.paper.txt ]; then
  echo "[INFO] Installing paper requirements from requirements.paper.txt"
  if ! pip install -r requirements.paper.txt; then
    echo "[WARN] Unable to install requirements.paper.txt in this environment."
    echo "       Continuing with currently installed packages."
  fi
fi

echo "[INFO] Regenerating paper figures/tables from canonical artifacts"
python scripts/make_paper_artifacts.py --config "${PAPER_CONFIG}"

GSM8K_CSV="artifacts/router_optionB/paper_table_test_full_per_seed.csv"
BOOLQ_CSV="artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv"

[ -f "${GSM8K_CSV}" ] || { echo "[ERROR] Missing required GSM8K router CSV: ${GSM8K_CSV}"; exit 1; }
[ -f "${BOOLQ_CSV}" ] || { echo "[ERROR] Missing required BoolQ router CSV: ${BOOLQ_CSV}"; exit 1; }

echo "[INFO] Regenerating GSM8K router LaTeX table"
python scripts/make_router_latex_table.py --in_csv "${GSM8K_CSV}" --out_tex artifacts/paper/tables/router_table.tex --label tab:main_results_allbudgets --split_label test --split_filter test --oracle_everywhere

echo "[INFO] Regenerating BoolQ router LaTeX table"
python scripts/make_router_latex_table.py --in_csv "${BOOLQ_CSV}" --out_tex artifacts/paper/tables/router_table_boolq.tex --label tab:router_boolq --split_label validation --split_filter validation --oracle-single-reference

echo "[INFO] Validating LaTeX dependencies for ${TEX_SOURCE}"
python scripts/check_paper_assets.py --tex "${TEX_SOURCE}"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "[ERROR] pdflatex not found; cannot compile ${TEX_SOURCE}."
  exit 1
fi

echo "[INFO] Compiling ${TEX_SOURCE}"
pdflatex -interaction=nonstopmode -halt-on-error -jobname="${JOB_NAME}" "${TEX_SOURCE}"
pdflatex -interaction=nonstopmode -halt-on-error -jobname="${JOB_NAME}" "${TEX_SOURCE}"

if [ ! -f "${JOB_NAME}.pdf" ]; then
  echo "[ERROR] Expected output PDF missing: ${JOB_NAME}.pdf"
  exit 1
fi
mv -f "${JOB_NAME}.pdf" "${OUTPUT_PDF}"

PAGE_COUNT=$(python - <<'PY' "${OUTPUT_PDF}"
import sys
from pypdf import PdfReader
pdf = sys.argv[1]
print(len(PdfReader(pdf).pages))
PY
)

echo "[INFO] Variant: ${VARIANT} | Source: ${TEX_SOURCE} | Output: ${OUTPUT_PDF} | Pages: ${PAGE_COUNT}"

if [ "${VARIANT}" = "10page" ] && [ "${PAGE_COUNT}" -gt "${PAGE_LIMIT}" ]; then
  echo "[ERROR] Page limit exceeded for 10page submission variant."
  echo "        Source: ${TEX_SOURCE}"
  echo "        Output: ${OUTPUT_PDF}"
  echo "        Pages: ${PAGE_COUNT} (limit ${PAGE_LIMIT})"
  exit 1
fi

echo "[INFO] Paper workflow complete. Submission command: bash run_paper.sh --variant 10page"
