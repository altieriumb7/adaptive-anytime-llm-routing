#!/usr/bin/env bash
set -euo pipefail

if [ -f requirements.paper.txt ]; then
  pip install -r requirements.paper.txt
fi

python scripts/make_paper_artifacts.py --config configs/paper.yaml
echo "Done. See artifacts/paper/"

# Generate LaTeX router table if per-seed CSV is present
if [ -f artifacts/router_optionB/paper_table_test_full_per_seed.csv ]; then
  python scripts/make_router_latex_table.py \
    --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv \
    --out_tex artifacts/paper/tables/router_table.tex
fi
