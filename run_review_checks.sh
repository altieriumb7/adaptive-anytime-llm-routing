#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "[1/5] Static compilation for active Python modules"
python -m compileall scripts src

echo "[2/5] LFS placeholder detection on critical canonical paths"
python scripts/check_lfs_placeholders.py

echo "[3/5] Regenerate paper-facing figures/tables from canonical config"
python scripts/make_paper_artifacts.py --config configs/paper.yaml

echo "[4/5] Rebuild canonical LaTeX tables for GSM8K and BoolQ"
python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table.tex \
  --label tab:main_results_allbudgets \
  --split_label test \
  --split_filter test \
  --oracle_everywhere
python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table_boolq.tex \
  --label tab:router_boolq \
  --split_label validation \
  --split_filter validation \
  --oracle-single-reference

echo "[5/5] Validate paper asset references"
python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex

cat <<'EOF'

[SUMMARY] Reviewer-check contract for this snapshot:
- Reproducible now: paper asset generation from bundled canonical artifacts.
- Conditionally reproducible: split-level BoolQ transfer reruns (requires non-pointer LFS payloads).
- Not guaranteed from this checkout alone: full raw-to-paper retraining/regeneration.
EOF
