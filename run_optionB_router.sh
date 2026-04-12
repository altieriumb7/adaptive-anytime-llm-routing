#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
[LEGACY ENTRYPOINT BLOCKED]
run_optionB_router.sh is a legacy transfer-era script and is not a canonical reviewer path.
It defaults to non-canonical/placeholder-prone inputs (e.g., results_main_full/*), which can
mislead submission-time reproduction.

Use one of the canonical scripts instead:
  - bash run_canonical_router_pipeline.sh      # regenerate GSM8K Option-B router artifacts
  - bash run_review_checks.sh                  # reviewer-facing snapshot checks
  - bash run_paper.sh                          # regenerate paper-facing tables/figures

If you intentionally need the legacy behavior, copy this script from git history and run it
manually outside the submission reproduction workflow.
EOF
exit 1
