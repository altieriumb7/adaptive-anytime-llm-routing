# Distilling-Anytime-Trajectories-with-Calibrated-Confidence

This repository is a **unified paper+code artifact** for budget-conditioned anytime inference with depth routing.

## Canonical artifact map (submission-facing)

To avoid ambiguity across historical folders, the manuscript is synchronized to the following canonical paths:

- **Paper entry point:** `main_distilling_revised_v0.tex`
- **Canonical build entry point:** `run_paper.sh`
- **Main GSM8K router family:** `artifacts/router_optionB/`
- **BoolQ transfer router family:** `artifacts/router_optionB_boolq/`
- **Generated paper tables/figures:** `artifacts/paper/tables/`, `artifacts/paper/figures/`
- **Prediction JSONLs used for paper plots:**
  - `results_abl/preds_main_adapter.jsonl`
  - `results_abl/preds_base.jsonl`

Legacy/auxiliary families (`results/`, `results_main_full/`, `artifacts/router_optionB_seed*`, `artifacts/router_optionB_boolq_seed*`) are retained for traceability but are non-canonical for paper claims.

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

`run_paper.sh` performs:
1. Best-effort install of paper dependencies from `requirements.paper.txt`.
2. Paper artifact regeneration via `python scripts/make_paper_artifacts.py --config configs/paper.yaml`.
3. Router table regeneration:
   - GSM8K: `artifacts/router_optionB/paper_table_test_full_per_seed.csv` -> `artifacts/paper/tables/router_table.tex`
   - BoolQ: `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv` -> `artifacts/paper/tables/router_table_boolq.tex`
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
  --legacy_split_aliases test --oracle-single-reference
python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex
pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex
pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex
```

## Main-result provenance (paper claims -> files)

- GSM8K compute-matched routing table in paper (`tab:main_results_allbudgets`) is loaded from:
  - `artifacts/paper/tables/router_table.tex` (generated)
  - source CSV: `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
- BoolQ transfer table (`tab:router_boolq`) is loaded from:
  - `artifacts/paper/tables/router_table_boolq.tex` (generated)
  - source CSV: `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
- Pareto plot (`fig:pareto`) is loaded from:
  - `artifacts/paper/figures/router_pareto_all.pdf`
  - produced by `scripts/make_paper_artifacts.py` from `configs/paper.yaml`.

## Reproducibility caveats (truthful limits)

Some required files are Git LFS pointers in this snapshot, so full from-scratch regeneration is not guaranteed.

Known LFS placeholders include:
- `data/router_svamp/dev.jsonl`, `data/router_svamp/test.jsonl`
- `data/router_splits_boolq_seeds/seed{0,1,2}/{dev,test}.jsonl`
- `results_main_full/preds_main_adapter_full.jsonl`, `results_main_full/preds_base_full.jsonl`
- `results/preds_base_boolq.jsonl`, `results/preds_base_boolq_reparsed.jsonl`, `results/preds_base_boolq_reparsed2.jsonl`

Therefore, this artifact should be interpreted as:
- **paper-asset reproducible from bundled canonical artifacts**, and
- **partially reproducible end-to-end** when missing LFS/data/checkpoints are unavailable.

## BoolQ split semantics (canonical)

- Canonical source is named `paper_table_validation_full_per_seed.csv` and is treated as **validation** in the manuscript.
- Historical rows may carry `split=test` labels from older exports; canonical generation handles this with:
  - `--split_filter validation --legacy_split_aliases test`.
