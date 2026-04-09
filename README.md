# Distilling-Anytime-Trajectories-with-Calibrated-Confidence

This repository is a **unified paper+code artifact** for budget-conditioned anytime inference with depth routing.

## Canonical artifact map (submission-facing)

To avoid ambiguity across historical folders, the manuscript is synchronized to the following canonical paths:

- **Paper entry point:** `main_distilling_revised_v0.tex`
- **Canonical build entry point:** `run_paper.sh`
- **Main GSM8K router family (paper-facing only):** `artifacts/router_optionB/`
- **Archived GSM8K router intermediates/non-canonical summaries:** `artifacts/router_optionB_legacy/`
- **BoolQ transfer router family:** `artifacts/router_optionB_boolq/`
- **Generated paper tables/figures:** `artifacts/paper/tables/`, `artifacts/paper/figures/`
- **Canonical GSM8K router prediction source:** `results/preds_student_full.jsonl`
- **Prediction JSONLs used for paper plots:**
  - `results_abl/preds_main_adapter.jsonl`
  - `results_abl/preds_base.jsonl`

Legacy/auxiliary families (`results/`, `results_main_full/`, `artifacts/router_optionB_seed*`, `artifacts/router_optionB_boolq_seed*`) are retained for traceability but are non-canonical for paper claims.

## Quick artifact map (reviewer-friendly)

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

- `run_all_p0_p5.sh` now writes newly generated end-to-end eval JSONLs to `artifacts/legacy_results/`.
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

Important variance scope: reported uncertainty in router tables is across split seeds only; model-seed retraining variance is not claimed in this snapshot.

## BoolQ split semantics (canonical)

- Canonical source is named `paper_table_validation_full_per_seed.csv` and is treated as **validation** in the manuscript.
- Canonical exports now use `split=validation` directly in the CSV rows, so reviewer-facing generation no longer relies on legacy split aliases.

## Teacher trajectory default (documented)

- OpenAI teacher backend defaults to `gpt-4o-mini` in `src/teachers/openai_teacher.py` and in `scripts/generate_gsm8k_trajectories.py --openai_model`.
- Re-running trajectory generation with a different teacher model is expected to change downstream distilled behavior.
