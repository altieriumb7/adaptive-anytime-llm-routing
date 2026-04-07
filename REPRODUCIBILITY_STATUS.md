# Reproducibility Status

## What is reproducible from this shipped repository

From the current checkout, you can reproduce:
- paper-side router LaTeX tables from canonical per-seed CSVs:
  - `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
  - `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
- paper-side artifact plots/tables that depend on available JSONL + router CSV inputs (subject to Python deps like matplotlib).
- consistency checks of LaTeX `\includegraphics`/`\input` paths via `scripts/check_paper_assets.py`.
- the canonical workflow via `bash run_paper.sh` (artifact generation always; LaTeX compile only when `pdflatex` is installed).

## What is NOT fully reproducible from this shipped repository alone

Full end-to-end regeneration is incomplete because some required artifacts are Git LFS pointers (placeholders), not materialized data.

Detected LFS placeholders include:
- `data/router_svamp/dev.jsonl`
- `data/router_svamp/test.jsonl`
- `data/router_splits_boolq_seeds/seed0/dev.jsonl`
- `data/router_splits_boolq_seeds/seed0/test.jsonl`
- `data/router_splits_boolq_seeds/seed1/dev.jsonl`
- `data/router_splits_boolq_seeds/seed1/test.jsonl`
- `data/router_splits_boolq_seeds/seed2/dev.jsonl`
- `data/router_splits_boolq_seeds/seed2/test.jsonl`
- `results_main_full/preds_main_adapter_full.jsonl`
- `results_main_full/preds_base_full.jsonl`
- `results/preds_base_boolq.jsonl`
- `results/preds_base_boolq_reparsed.jsonl`
- `results/preds_base_boolq_reparsed2.jsonl`

## External dependencies

Even with all LFS objects present, full regeneration still depends on:
- model checkpoints (base + adapters),
- Hugging Face datasets access,
- optional OpenAI batch workflow outputs for trajectory creation,
- training/evaluation compute environment.

## Canonical truthful reproduction path

Use bundled canonical artifacts to regenerate paper assets and tables:
1. `bash run_paper.sh`

Equivalent manual steps:
1. `python scripts/make_paper_artifacts.py --config configs/paper.yaml`
2. GSM8K table: `python scripts/make_router_latex_table.py --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv --out_tex artifacts/paper/tables/router_table.tex --label tab:main_results_allbudgets --split_label test --split_filter test --oracle_everywhere`
3. BoolQ table: `python scripts/make_router_latex_table.py --in_csv artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv --out_tex artifacts/paper/tables/router_table_boolq.tex --label tab:router_boolq --split_label validation --split_filter validation --legacy_split_aliases test --oracle-single-reference`
4. `python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex`
5. Compile `main_distilling_revised_v0.tex` with `pdflatex` if installed.

BoolQ semantic note:
- In this shipped artifact, BoolQ paper numbers are validation results.
- Some upstream CSV rows are historically labeled `split=test`; table generation now treats that as a documented legacy alias for validation in the BoolQ pipeline.
