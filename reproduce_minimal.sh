#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PASS=0
FAIL=0

step() { echo; echo "[STEP] $*"; }
mark_pass() { PASS=$((PASS+1)); echo "[PASS] $*"; }
mark_fail() { FAIL=$((FAIL+1)); echo "[FAIL] $*"; }

step "Check required Python packages for paper artifact regeneration"
if python - <<'PY'
import importlib.util
mods = ["matplotlib", "yaml", "numpy"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING:", ",".join(missing))
    raise SystemExit(1)
print("OK: required packages present")
PY
then
  mark_pass "python package preflight"
else
  echo "[INFO] Installing paper dependencies from requirements.paper.txt"
  if pip install -r requirements.paper.txt; then
    if python - <<'PY'
import importlib.util
mods = ["matplotlib", "yaml", "numpy"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING_AFTER_INSTALL:", ",".join(missing))
    raise SystemExit(1)
print("OK_AFTER_INSTALL: required packages present")
PY
    then
      mark_pass "python package preflight (after install)"
    else
      mark_fail "python package preflight"
    fi
  else
    mark_fail "install requirements.paper.txt"
  fi
fi

step "Run canonical reviewer checks"
if bash run_review_checks.sh; then
  mark_pass "run_review_checks.sh"
else
  mark_fail "run_review_checks.sh"
fi

step "Verify required canonical artifacts exist"
REQ=(
  "main_distilling_revised_v0.tex"
  "artifacts/router_optionB/paper_table_test_full_per_seed.csv"
  "artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv"
  "artifacts/paper/tables/router_table.tex"
  "artifacts/paper/tables/router_table_boolq.tex"
  "artifacts/paper/figures/router_pareto_all.pdf"
  "results/preds_student_full.jsonl"
  "results_abl/preds_main_adapter.jsonl"
  "results_abl/preds_base.jsonl"
)
missing=0
for f in "${REQ[@]}"; do
  if [[ -f "$f" ]]; then
    echo "[OK] $f"
  else
    echo "[MISSING] $f"
    missing=1
  fi
done
if [[ $missing -eq 0 ]]; then mark_pass "required artifacts"; else mark_fail "required artifacts"; fi

step "Verify frozen checksums (if checksum file present)"
if [[ -f artifacts/CHECKSUMS.sha256 ]]; then
  if sha256sum -c artifacts/CHECKSUMS.sha256; then
    mark_pass "checksums"
  else
    mark_fail "checksums"
  fi
else
  echo "[WARN] artifacts/CHECKSUMS.sha256 not found"
  mark_fail "checksums"
fi

step "Regenerate paper artifacts from bundled canonical inputs"
if python scripts/make_paper_artifacts.py --config configs/paper.yaml; then
  mark_pass "make_paper_artifacts"
else
  mark_fail "make_paper_artifacts"
fi

echo
echo "========== MINIMAL REPRODUCTION SUMMARY =========="
echo "PASS: ${PASS}"
echo "FAIL: ${FAIL}"
if [[ ${FAIL} -eq 0 ]]; then
  echo "OVERALL: PASS (artifact-level reproducibility checks)"
  exit 0
else
  echo "OVERALL: FAIL (see steps above)"
  exit 1
fi
