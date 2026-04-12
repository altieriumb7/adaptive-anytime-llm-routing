# Codex Repository Repair Summary

## Scope and intent
This pass focused on reviewer entrypoint clarity, LFS/placeholder fail-fast behavior, canonical-path discipline, and conservative reproducibility wording consistent with the shipped snapshot.

## What changed and why

### 1) Misleading entrypoints were guarded
- `run_optionB_router.sh` is now a **blocked legacy entrypoint** that exits immediately and points reviewers to canonical scripts.
  - Why: it previously defaulted to transfer-era/non-canonical inputs (`results_main_full/*`) and could mislead reviewers.
- `run_all_p0_p5.sh` now requires `ALLOW_LEGACY_RUN_ALL=1` to run.
  - Why: retained for archival/end-to-end experiments, but no longer ambiguous as a submission reviewer path.

### 2) LFS/placeholder detection was added and wired in
- Added `scripts/lfs_guard.py` with shared file materialization checks and Git LFS pointer detection.
- Added `scripts/check_lfs_placeholders.py` as reviewer-facing preflight:
  - hard-fails on unresolved placeholders in canonical paper-asset-critical inputs,
  - warning-only (by default) for conditional transfer split files.
- `scripts/make_router_splits.py` now validates its canonical prediction source is materialized (not missing/LFS pointer).
- `scripts/run_router_optionB_boolq.py` now validates per-seed transfer split files before evaluation.

### 3) Single reviewer check command added
- Added `run_review_checks.sh` as the canonical one-command reviewer workflow.
- It runs:
  1. static Python compilation (`scripts`, `src`),
  2. LFS placeholder preflight,
  3. paper artifact regeneration from `configs/paper.yaml`,
  4. canonical GSM8K/BoolQ LaTeX table regeneration,
  5. LaTeX asset reference validation.
- It prints a compact reproducibility contract summary at the end.

### 4) Documentation tightened for canonical-vs-legacy clarity
- Updated `README.md` to surface `run_review_checks.sh`, and to explicitly label legacy guarded scripts.
- Updated `REPRODUCIBILITY.md` to include the reviewer preflight flow and canonical LFS check command.
- Updated `REPRODUCIBILITY_STATUS.md` with reviewer-check command and legacy entrypoint policy.

### 5) Minimal paper wording cleanup (no numbers changed)
- Updated one reproducibility sentence in `main_distilling_revised_v0.tex` to explicitly state:
  - split-level BoolQ reruns are conditional on materialized transfer payloads,
  - bundled BoolQ claims are artifact-level otherwise.
- No headline results, table values, or references were fabricated or manually altered.

## Canonical files/scripts after this repair
- Paper source: `main_distilling_revised_v0.tex`
- Paper build: `run_paper.sh`
- Reviewer preflight: `run_review_checks.sh`
- Canonical GSM8K router regeneration: `run_canonical_router_pipeline.sh`
- Canonical artifact config: `configs/paper.yaml`
- Canonical paper-facing outputs:
  - `artifacts/paper/tables/router_table.tex`
  - `artifacts/paper/tables/router_table_boolq.tex`
  - `artifacts/paper/figures/router_pareto_all.pdf`

## Legacy/archival scripts explicitly labeled
- `run_optionB_router.sh` (blocked legacy entrypoint)
- `run_all_p0_p5.sh` (legacy guarded; opt-in env var required)

## What remains not fully reproducible (by design, explicitly documented)
- Full raw-to-paper end-to-end reruns requiring external APIs/checkpoints/LFS payloads.
- Split-level transfer regeneration when BoolQ split files are unresolved pointers/missing.
- Model-seed retraining variance characterization (snapshot reports split-seed variability only).

## Issues intentionally not "fixed" to avoid fabrication
- No new experiments were invented.
- No unshipped transfer payloads were synthesized.
- No headline metrics were altered outside canonical artifact regeneration pathways.
