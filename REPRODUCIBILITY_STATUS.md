# Reproducibility Status

## What is reproducible from this shipped repository

From the current checkout, you can reproduce:
- paper-side router LaTeX tables from canonical per-seed CSVs:
  - `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
  - `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
- paper-side artifact plots/tables that depend on available JSONL + router CSV inputs (subject to Python deps like matplotlib).
- consistency checks of LaTeX `\includegraphics`/`\input` paths.

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
1. `python scripts/make_router_latex_table.py ...` for GSM8K and BoolQ tables.
2. `python scripts/make_paper_artifacts.py --config configs/paper.yaml` for figures/metrics tables.
3. Compile `main_distilling_revised_v0.tex` once TeX dependencies are installed.
