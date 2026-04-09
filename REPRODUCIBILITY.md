# Reproducibility Guide

This repository supports multiple reproducibility levels with different requirements.

## Level A — Paper build (LaTeX + bundled assets)

**Supported:** Yes (when TeX toolchain is available).

### Command
```bash
bash run_paper.sh
```

### Expected outputs
- `artifacts/paper/tables/router_table.tex`
- `artifacts/paper/tables/router_table_boolq.tex`
- `artifacts/paper/figures/*`
- (optional) `main_distilling_revised_v0.pdf` when `pdflatex` is installed.

## Level B — Paper results (canonical GSM8K/BoolQ artifacts)

**Supported:** Partially-to-fully, depending on available local dependencies and non-placeholder data. Reported uncertainty here is split-seed variability only.

### Canonical GSM8K source-of-truth
- Prediction source: `results/preds_student_full.jsonl`
- Split manifest: `data/router_splits_seeds/manifest.json`
- Router manifest: `artifacts/router_optionB/gsm8k_router_manifest.json`

### Commands (authoritative pipeline)
```bash
bash run_canonical_router_pipeline.sh
```

Equivalent explicit commands:
```bash
python scripts/make_router_splits.py \
  --source results/preds_student_full.jsonl \
  --out_root data/router_splits_seeds \
  --seeds 0,1,2 \
  --manifest data/router_splits_seeds/manifest.json
python scripts/run_router_optionB_repro.py --out_dir artifacts/router_optionB --seeds 0,1,2
python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table.tex \
  --label tab:main_results_allbudgets --split_label test --split_filter test --oracle_everywhere
python scripts/make_paper_artifacts.py --config configs/paper.yaml
python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex
```

### Expected canonical outputs
- `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
- `artifacts/router_optionB/paper_table_test_acc_tokens.csv`
- `artifacts/paper/tables/router_table.tex`
- `artifacts/paper/tables/router_table_boolq.tex`

## Level C — Full training/regeneration (student training + transfer data refresh)

**Supported:** Partial in this bundled snapshot.

### Why partial
Some required files are not fully bundled (e.g., Git LFS placeholders and external/model dependencies). See README reproducibility caveats.

Typical missing/blocking categories:
- transfer split files that may be LFS placeholders in some environments,
- large checkpoints/models,
- optional external APIs used by upstream trajectory generation workflows.

## Known limitations (explicit)

1. This artifact is **paper-asset reproducible** from bundled canonical outputs.
2. Router tables and anytime calibration/coverage figures come from canonical but distinct artifact families; no claim is made that they originate from one row-identical prediction bundle.
3. End-to-end training/data regeneration is **not guaranteed** in every environment.
4. BoolQ/transfer regeneration may require additional non-bundled artifacts depending on LFS availability.
5. Model-seed retraining variance is not characterized in the bundled evaluation outputs.

## BoolQ transfer reproducibility status (single source of truth)

- **Paper-facing BoolQ table is artifact-level reproducible** from bundled canonical CSV:
  - `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
- **BoolQ split-level regeneration is conditional** on resolving LFS placeholders under
  `data/router_splits_boolq_seeds/seed{0,1,2}/{dev,test}.jsonl`.

Quick check for LFS placeholders:
```bash
sed -n '1,3p' data/router_splits_boolq_seeds/seed0/dev.jsonl
```
If the file begins with `version https://git-lfs.github.com/spec/v1`, fetch LFS payloads before attempting split-level BoolQ reruns.

## Canonical vs archived directories

- Canonical GSM8K router outputs: `artifacts/router_optionB/`
- Archived non-canonical GSM8K router summaries/intermediates: `artifacts/router_optionB_legacy/`
