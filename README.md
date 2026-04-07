# Distilling-Anytime-Trajectories-with-Calibrated-Confidence

This repository is a **unified paper+code artifact** for anytime depth routing with calibrated confidence.

## Canonical artifact families

To remove ambiguity across overlapping folders, use the following as canonical for the paper:

- **Main GSM8K router results:** `artifacts/router_optionB/`.
- **BoolQ transfer router results:** `artifacts/router_optionB_boolq/`.
- **Paper-ready tables/figures:** `artifacts/paper/tables/` and `artifacts/paper/figures/`.
- **Prediction JSONLs used for artifact plots:** `results_abl/preds_main_adapter.jsonl` and `results_abl/preds_base.jsonl`.

Legacy/auxiliary folders (`results/`, `results_main_full/`, `artifacts/router_optionB_seed*`, `artifacts/router_optionB_boolq_seed*`) are retained for traceability, but are not the canonical manuscript source unless explicitly noted.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional, only for dataset generation / training:
pip install -r requirements.openai.txt
pip install -r requirements.train.txt
```

## Minimal truthful reproduction (from shipped repo)

This path reproduces the paper-facing tables/figures from bundled artifacts.

```bash
# 1) Regenerate paper figures/tables from canonical artifacts
python scripts/make_paper_artifacts.py --config configs/paper.yaml

# 2) Regenerate router LaTeX tables from canonical per-seed summaries
python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table.tex \
  --label tab:main_results_allbudgets --split_label test --oracle_everywhere

python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table_boolq.tex \
  --label tab:router_boolq --split_label validation --oracle-single-reference

# 3) Compile the paper
pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex
pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex
```

## Main evaluation command (when model/checkpoint access is available)

```bash
python scripts/eval_anytime.py \
  --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
  --adapter_dir <path_to_lora_adapter_or_empty> \
  --dataset gsm8k --split test --max_examples 500 \
  --budgets 1,2,3,4 --max_new_tokens 96,160,224,320 \
  --save_jsonl results/preds_main_adapter.jsonl
```

## SFT dataset build

```bash
python scripts/build_sft_from_trajectories.py \
  --traj_jsonl data/anytime_trajectories_train.jsonl \
  --out_jsonl data/sft_train.jsonl \
  --budgets 1,2,3,4 \
  --conf_target teacher
```

## Reproducibility caveats

Some files in this repository are Git LFS pointers (not materialized blobs), including:

- `data/router_svamp/dev.jsonl`, `data/router_svamp/test.jsonl`
- `data/router_splits_boolq_seeds/seed{0,1,2}/{dev,test}.jsonl`
- `results_main_full/preds_main_adapter_full.jsonl`, `results_main_full/preds_base_full.jsonl`
- several BoolQ files under `results/`

Because of that, **full end-to-end regeneration from raw data is not guaranteed from this snapshot alone**. The provided minimal path above is the truthful “works-from-shipped-files” route.
