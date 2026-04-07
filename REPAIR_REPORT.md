# Repair Report

## Scope
This repair pass treated the repository as a single research artifact (code + paper), with fixes focused on correctness, consistency, and honest reproducibility.

## What was broken and what was changed

### 1) Paper figure mismatch
- **Issue:** `main_distilling_revised_v0.tex` referenced `router_pareto_all_fixed.pdf`, but bundled artifact is `router_pareto_all.pdf`.
- **Fix:** Updated the figure include to `router_pareto_all.pdf`.

### 2) Broken evaluation logic (`scripts/eval_anytime.py`)
- **Issues found:**
  - broken loop/indentation (`for ... else` misuse), causing budget loop logic to be incorrect;
  - duplicate/contradictory helper definitions;
  - undefined `stopping` reference in one generation path;
  - fragile parsing/metric handling.
- **Fix:** Rebuilt the script into a single clean evaluation flow that:
  - iterates all budgets deterministically per example;
  - generates with explicit stopping criteria for `####` + `CONF:` format;
  - records per-budget correctness/calibration stats;
  - writes JSONL rows with token accounting;
  - supports optional calibration application.

### 3) Broken training-data build path (`src/train/sft_build.py`, `scripts/build_sft_from_trajectories.py`)
- **Issues found:**
  - use-before-assign and undefined variables (`answer`, `t`, `task`) in confidence-target logic;
  - CLI arguments accepted by wrapper script but not forwarded to core builder;
  - type/plumbing mismatches for confidence options.
- **Fixes:**
  - rewrote `build_sft_examples` sequencing so answer/conf are parsed first, then confidence target policy is applied per budget;
  - added robust task derivation from `meta`;
  - moved calibrator load outside inner loops;
  - fixed CLI plumbing in `scripts/build_sft_from_trajectories.py` (all key args now forwarded).

### 4) Artifact-generation mismatch (`configs/paper.yaml`, `scripts/make_paper_artifacts.py`)
- **Issue:** `make_paper_artifacts.py` expected an old wide router CSV cell format (`"0.65 (96)"`), but canonical router artifacts are in long numeric format (`policy,budget_tag,acc_mean,tokens_mean,...`).
- **Fix:** Updated loader to support both formats and treat `artifacts/router_optionB/paper_table_test_acc_tokens.csv` as canonical long-form input.

### 5) Conflicting result families
- **Issue:** multiple overlapping families (`results`, `results_abl`, `results_main_full`, seed folders, etc.) without one explicit canonical manuscript source.
- **Fix:** documented and enforced canonical families in README:
  - GSM8K router: `artifacts/router_optionB/`
  - BoolQ transfer router: `artifacts/router_optionB_boolq/`
  - paper outputs: `artifacts/paper/*`
  - plot prediction sources: `results_abl/preds_main_adapter.jsonl`, `results_abl/preds_base.jsonl`

### 6) Missing artifact reality check (LFS placeholders)
- **Issue:** several files are Git LFS pointers; full end-to-end reproduction cannot be claimed from repo snapshot alone.
- **Fix:** README and paper reproducibility wording were softened to explicitly state partial reproducibility and external dependencies.

## Paper synchronization actions
- Replaced manually inlined main/BoolQ router tables in LaTeX with `\input{artifacts/paper/tables/router_table.tex}` and `\input{artifacts/paper/tables/router_table_boolq.tex}` to keep paper and generated artifacts synchronized.
- Updated reproducibility section wording to avoid overstating full reproducibility.

## What remains impossible from shipped repo alone
- Full regeneration of all experiments requiring missing LFS blobs (notably certain router splits and `results_main_full` predictions).
- End-to-end regeneration requiring unavailable external checkpoints/data/API outputs.

## Numbers status
- Reported manuscript numbers were **reconciled to existing canonical artifacts** and table generation pipeline.
- This pass did **not** regenerate all experimental predictions from raw model runs.

## Canonical result family (final)
- `artifacts/router_optionB/` and `artifacts/router_optionB_boolq/` for paper router metrics.
