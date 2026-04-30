# Distilling-Anytime-Trajectories-with-Calibrated-Confidence

This repository is a **paper+code artifact** for budget-conditioned anytime inference with depth routing.

> Audit-critical note: the GSM8K anytime-quality tables/plots and the GSM8K Option-B routing tables are maintained as **two canonical artifact families**, not one row-identical prediction bundle.

## Canonical artifact map (submission-facing)

To avoid ambiguity across historical folders, the manuscript is synchronized to the following canonical paths:

- **Paper entry point:** `main_distilling_revised_v0.tex`
- **Canonical build entry point:** `run_paper.sh`
- **Main GSM8K router family (paper-facing only):** `artifacts/router_optionB/`
- **Archived GSM8K router intermediates/non-canonical summaries:** `artifacts/router_optionB_legacy/`
- **BoolQ transfer router family:** `artifacts/router_optionB_boolq/`
- **Generated paper tables/figures:** `artifacts/paper/tables/`, `artifacts/paper/figures/`
- **Canonical GSM8K router prediction source:** `results/preds_student_full.jsonl`
- **Anytime/calibration prediction JSONLs used for paper plots and Table 1:**
  - `results_abl/preds_main_adapter.jsonl`
  - `results_abl/preds_base.jsonl`

Legacy/auxiliary families (`results_main_full/`, `artifacts/router_optionB_seed*`, `artifacts/router_optionB_boolq_seed*`) are retained for traceability but are non-canonical for paper claims.

Note: `results/` is mixed-scope in this snapshot: `results/preds_student_full.jsonl` is canonical for GSM8K Option-B routing, while other files under `results/` may be legacy or auxiliary.

## Quick artifact map (reviewer-friendly)

- **GSM8K anytime table (`tab:anytime_gsm8k`) and reliability/coverage plots**
  - Canonical JSONLs: `results_abl/preds_main_adapter.jsonl`, `results_abl/preds_base.jsonl`
- **GSM8K row-match audit between anytime-vs-router families**
  - Canonical audit CSV: `artifacts/paper/tables/provenance_rowmatch_audit.csv`
- **GSM8K lightweight paired uncertainty support (split-seed bootstrap)**
  - Canonical CSV: `artifacts/paper/tables/router_paired_bootstrap_gsm8k.csv`
- **Main GSM8K table (`tab:main_results_allbudgets`)**
  - Canonical CSV: `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
  - Canonical TeX table: `artifacts/paper/tables/router_table.tex`
- **Transfer table (`tab:router_boolq`)**
  - Canonical CSV: `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
  - Canonical TeX table: `artifacts/paper/tables/router_table_boolq.tex`
- **Main figure (`fig:pareto`)**
  - Canonical figure: `artifacts/paper/figures/router_pareto_all.pdf`
- **Canonical reproducibility guide**
  - `REPRODUCIBILITY.md`
  - BoolQ-specific status: `REPRODUCIBILITY.md#boolq-transfer-reproducibility-status-single-source-of-truth`

## ATC 2026 fit (honest scope)

This artifact is positioned as a **budget-aware / uncertainty-aware inference systems** study: lightweight stopping/routing policies for compute allocation under constrained budgets.

It is **not** presented as evidence for security, privacy, governance, safety certification, or cyber-physical deployment claims; those are outside the released experiments.

## Reproducibility levels (reviewer quick read)

- **Paper-build reproducible:** Yes, from bundled canonical assets when local deps are installed (`bash run_paper.sh`).
- **Single reviewer check command:** `bash run_review_checks.sh` (compile checks + LFS placeholder detection + artifact regeneration + asset validation).
- **Paper-results reproducible (GSM8K router):** Yes, from `results/preds_student_full.jsonl` via seeded splits/manifests and Option-B pipeline.
- **Paper-results reproducible (BoolQ transfer):** Artifact-level from bundled canonical BoolQ CSV; split-level reruns require non-placeholder BoolQ split payloads (see `REPRODUCIBILITY.md` BoolQ section).
- **Not fully reproducible from shipped files alone:** full end-to-end regeneration that depends on missing LFS payloads, external models/checkpoints, or API-backed upstream data generation.

Template-only configs are explicitly named with `.template.yaml` and are not reviewer defaults:
- `configs/train_student_qlora.template.yaml`
- `configs/eval_anytime.template.yaml`

## Reviewer manifest (canonical vs legacy)

### Canonical paper outputs
- `artifacts/paper/tables/router_table.tex` (GSM8K router table, paper `tab:main_results_allbudgets`)
- `artifacts/paper/tables/router_table_boolq.tex` (BoolQ transfer table, paper `tab:router_boolq`)
- `artifacts/paper/figures/router_pareto_all.pdf` (paper `fig:pareto`)

### Canonical source inputs for those outputs
- GSM8K router per-seed CSV: `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
- GSM8K router compact CSV: `artifacts/router_optionB/paper_table_test_acc_tokens.csv`
- GSM8K router provenance manifests: `data/router_splits_seeds/manifest.json`, `artifacts/router_optionB/gsm8k_router_manifest.json`
- BoolQ router per-seed CSV: `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
- Anytime prediction JSONLs for reliability/coverage plots:  
  `results_abl/preds_main_adapter.jsonl`, `results_abl/preds_base.jsonl`
- Paper artifact config: `configs/paper.yaml`

### Known non-canonical / legacy paths
- `results_main_full/`, `artifacts/router_optionB_seed*`, `artifacts/router_optionB_boolq_seed*`
  - See `results_main_full/README_LEGACY.md` for transfer-era placeholder semantics.
- `artifacts/router_optionB_legacy/` (archived non-canonical GSM8K router summaries/intermediates)
- `artifacts/legacy_results/` (outputs produced by `run_all_p0_p5.sh`; not paper source-of-truth)
- Archived dev backups under `archive/legacy_dev_snapshots/`

### External dependencies not bundled
- Python plotting stack from `requirements.paper.txt` (e.g., `matplotlib`) for regenerating figures/tables.
- Optional LaTeX toolchain (`pdflatex`) for PDF compilation.
- Git LFS payloads for some data/result files needed by full end-to-end reruns.
- Optional remote APIs/models for full trajectory generation/training workflows.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional (only if running training / OpenAI batch workflows):
pip install -r requirements.train.txt
pip install -r requirements.openai.txt
```

## Canonical paper reproduction workflow

Use the single canonical script:

```bash
bash run_paper.sh
```

To rebuild the GSM8K router artifacts from the canonical prediction source before paper generation:

```bash
bash run_canonical_router_pipeline.sh
```

Optional explicit config (equivalent canonical behavior):

```bash
bash run_paper.sh --config configs/paper.yaml
```

`run_paper.sh` performs:
1. Best-effort install of paper dependencies from `requirements.paper.txt`.
2. Paper artifact regeneration via `python scripts/make_paper_artifacts.py --config configs/paper.yaml`.
3. Router table regeneration:
   - GSM8K: `artifacts/router_optionB/paper_table_test_full_per_seed.csv` -> `artifacts/paper/tables/router_table.tex`
   - BoolQ: `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv` -> `artifacts/paper/tables/router_table_boolq.tex`

For full GSM8K router repro from canonical predictions, run first:
```bash
python scripts/make_router_splits.py --source results/preds_student_full.jsonl --out_root data/router_splits_seeds --seeds 0,1,2 --manifest data/router_splits_seeds/manifest.json
python scripts/run_router_optionB_repro.py --out_dir artifacts/router_optionB --seeds 0,1,2
```
4. LaTeX asset validation (`\input` + `\includegraphics`) via `scripts/check_paper_assets.py`.
5. LaTeX compile only if `pdflatex` is available.


## Reviewer one-command check

Run this first during review:

```bash
bash run_review_checks.sh
```

Submission-freeze minimal reproduction (artifact-level):

```bash
bash reproduce_minimal.sh
```

Frozen submission artifacts and digests:

```bash
sha256sum -c artifacts/CHECKSUMS.sha256
```

This is the canonical lightweight check in the shipped snapshot. It deliberately fails fast when critical inputs are unresolved Git LFS pointers and prints a compact reproducibility contract summary.

### Equivalent manual commands

```bash
python scripts/make_paper_artifacts.py --config configs/paper.yaml
python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table.tex \
  --label tab:main_results_allbudgets --split_label test --split_filter test --oracle_everywhere
python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table_boolq.tex \
  --label tab:router_boolq --split_label validation --split_filter validation \
  --oracle-single-reference
python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex
python scripts/make_router_paired_bootstrap.py --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv --out_csv artifacts/paper/tables/router_paired_bootstrap_gsm8k.csv
pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex
pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex
```

### Reviewer-safe minimal checks

```bash
python -m compileall scripts src
python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex
python scripts/make_paper_artifacts.py --config configs/paper.yaml
bash run_paper.sh
```

## Legacy/non-canonical pipeline outputs

- `run_all_p0_p5.sh` is now explicitly guard-railed as legacy (requires `ALLOW_LEGACY_RUN_ALL=1`) and writes newly generated end-to-end eval JSONLs to `artifacts/legacy_results/`.
- `run_optionB_router.sh` is blocked as a legacy entrypoint and points reviewers to canonical scripts.
- These files are intentionally **non-canonical for manuscript claims** unless manually promoted after independent verification.
- The script no longer rewrites `configs/paper.yaml`; canonical paper generation always reads the checked-in config.

## Main-result provenance (paper claims -> files)

- GSM8K compute-matched routing table in paper (`tab:main_results_allbudgets`) is loaded from:
  - `artifacts/paper/tables/router_table.tex` (generated)
  - source CSV: `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
  - canonical prediction source: `results/preds_student_full.jsonl`
  - split/artifact manifests: `data/router_splits_seeds/manifest.json`, `artifacts/router_optionB/gsm8k_router_manifest.json`
- BoolQ transfer table (`tab:router_boolq`) is loaded from:
  - `artifacts/paper/tables/router_table_boolq.tex` (generated)
  - source CSV: `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
- Pareto plot (`fig:pareto`) is loaded from:
  - `artifacts/paper/figures/router_pareto_all.pdf`
  - produced by `scripts/make_paper_artifacts.py` from `configs/paper.yaml`.
- Bundle provenance note: anytime calibration/coverage figures are built from `results_abl/*.jsonl`, while compute-matched routing tables are built from `artifacts/router_optionB/*.csv`; these are both canonical but not represented as one row-identical prediction bundle (see `artifacts/paper/tables/provenance_rowmatch_audit.csv`).
- Lightweight paired uncertainty support for key GSM8K gains is exported to `artifacts/paper/tables/router_paired_bootstrap_gsm8k.csv` from `artifacts/router_optionB/paper_table_test_full_per_seed.csv` via seed-paired bootstrap over split seeds only (`scripts/make_router_paired_bootstrap.py`).
- Table-level provenance note: GSM8K Table 1 (`tab:anytime_gsm8k`) is sourced from `results_abl/preds_main_adapter.jsonl` and `results_abl/preds_base.jsonl`, while GSM8K router Table 2 (`tab:main_results_allbudgets`) is sourced from the Option-B family rooted at `results/preds_student_full.jsonl`; `artifacts/paper/tables/provenance_rowmatch_audit.csv` is the explicit non-row-identity evidence.
- Canonical metadata refresh (no heavy regeneration): `python scripts/refresh_canonical_provenance.py` updates `data/router_splits_seeds/manifest.json` and `artifacts/router_optionB/gsm8k_router_manifest.json`.

## Reproducibility caveats (truthful limits)

Some required files are Git LFS pointers in this snapshot, so full from-scratch regeneration is not guaranteed.

Known LFS placeholders include:
- `data/router_svamp/dev.jsonl`, `data/router_svamp/test.jsonl`
- `data/router_splits_boolq_seeds/seed{0,1,2}/{dev,test}.jsonl`
- `results_main_full/preds_main_adapter_full.jsonl`, `results_main_full/preds_base_full.jsonl`
- `results/preds_base_boolq.jsonl`, `results/preds_base_boolq_reparsed.jsonl`, `results/preds_base_boolq_reparsed2.jsonl`

Therefore, this artifact should be interpreted as:
- **paper-asset reproducible from bundled canonical artifacts**,
- **split-seed reproducible** for router evaluation (seeds 0/1/2), and
- **partially reproducible end-to-end** when missing LFS/data/checkpoints are unavailable.

### Reproducibility levels (submission contract)
- **Level A (manuscript artifact regeneration):** supported from bundled canonical artifacts (`bash reproduce_minimal.sh` / `bash run_paper.sh`).
- **Level B (split/router rerun from bundled predictions):** supported for GSM8K Option-B split-seed reruns from `results/preds_student_full.jsonl`.
- **Level C (full trajectory generation/training rerun):** not guaranteed from this checkout alone due to LFS/external model/API dependencies.

Important variance scope: reported uncertainty in router tables is across split seeds only; model-seed retraining variance is not claimed in this snapshot.

## BoolQ split semantics (canonical)

- Canonical source is named `paper_table_validation_full_per_seed.csv` and is treated as **validation** in the manuscript.
- Canonical exports now use `split=validation` directly in the CSV rows, so reviewer-facing generation no longer relies on legacy split aliases.

## Teacher trajectory default (documented)

- OpenAI teacher backend defaults to `gpt-4o-mini` in `src/teachers/openai_teacher.py` and in `scripts/generate_gsm8k_trajectories.py --openai_model`.
- Re-running trajectory generation with a different teacher model is expected to change downstream distilled behavior.
