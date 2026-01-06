#!/usr/bin/env bash
set -euo pipefail

if [ -f requirements.paper.txt ]; then
  pip install -r requirements.paper.txt
fi

python scripts/make_paper_artifacts.py --config configs/paper.yaml
echo "Done. See artifacts/paper/"
