# Distilling Anytime Trajectories with Calibrated Confidence

This repository is the **code + artifact bundle** for a paper on budget-conditioned anytime inference with confidence-calibrated stopping/routing.

> **Important provenance note:** GSM8K anytime artifacts and GSM8K Option-B router artifacts are both canonical for paper claims, but they are **not one row-identical prediction bundle**.

## What is canonical in this repository

### Paper and build entry points
- Paper source: `main_distilling_revised_v0.tex`
- Canonical paper pipeline: `run_paper.sh`
- Canonical config: `configs/paper.yaml`

### Canonical artifact families
- GSM8K Option-B router artifacts: `artifacts/router_optionB/`
- BoolQ transfer router artifacts: `artifacts/router_optionB_boolq/`
- Paper outputs (tables/figures): `artifacts/paper/tables/`, `artifacts/paper/figures/`
- GSM8K router prediction source: `results/preds_student_full.jsonl`
- GSM8K anytime prediction sources (Table 1 + reliability/coverage):
  - `results_abl/preds_main_adapter.jsonl`
  - `results_abl/preds_base.jsonl`

### Non-canonical / legacy (kept only for traceability)
- `results_main_full/`
- `artifacts/router_optionB_seed*`
- `artifacts/router_optionB_boolq_seed*`
- `artifacts/router_optionB_legacy/`
- `archive/`

## Reviewer quick map: paper claims to files

- **GSM8K anytime table (`tab:anytime_gsm8k`)**
  - `results_abl/preds_main_adapter.jsonl`
  - `results_abl/preds_base.jsonl`

- **GSM8K router table (`tab:main_results_allbudgets`)**
  - Source CSV: `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
  - Generated TeX: `artifacts/paper/tables/router_table.tex`

- **BoolQ transfer table (`tab:router_boolq`)**
  - Source CSV: `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
  - Generated TeX: `artifacts/paper/tables/router_table_boolq.tex`

- **Main Pareto figure (`fig:pareto`)**
  - `artifacts/paper/figures/router_pareto_all.pdf`

- **Row-match audit between artifact families**
  - `artifacts/paper/tables/provenance_rowmatch_audit.csv`

- **Paired uncertainty support (split-seed bootstrap)**
  - `artifacts/paper/tables/router_paired_bootstrap_gsm8k.csv`

## Reproducibility levels (honest scope)

- **Level A — manuscript artifact regeneration:** supported from bundled canonical assets.
- **Level B — GSM8K split/router rerun:** supported from `results/preds_student_full.jsonl` with seeded split manifests.
- **Level C — full trajectory/training regeneration:** not guaranteed from this checkout alone (depends on LFS payloads, external checkpoints/models, and optional APIs).

Variance scope for router tables is **split-seed variance only** in this artifact snapshot.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional for additional workflows:
pip install -r requirements.train.txt
pip install -r requirements.openai.txt
```

## Canonical paper reproduction

Run the canonical pipeline:

```bash
bash run_paper.sh
```

Optional explicit config (same canonical behavior):

```bash
bash run_paper.sh --config configs/paper.yaml
```

To rebuild GSM8K router artifacts first from canonical predictions:

```bash
bash run_canonical_router_pipeline.sh
```

Equivalent explicit router commands:

```bash
python scripts/make_router_splits.py \
  --source results/preds_student_full.jsonl \
  --out_root data/router_splits_seeds \
  --seeds 0,1,2 \
  --manifest data/router_splits_seeds/manifest.json

python scripts/run_router_optionB_repro.py \
  --out_dir artifacts/router_optionB \
  --seeds 0,1,2
```

## Reviewer one-command checks

Primary review command:

```bash
bash run_review_checks.sh
```

Submission-freeze lightweight check:

```bash
bash reproduce_minimal.sh
```

Verify frozen checksums:

```bash
sha256sum -c artifacts/CHECKSUMS.sha256
```

If `reproduce_minimal.sh` cannot install dependencies in an offline/no-index environment, manually install `requirements.paper.txt` and rerun.

## Equivalent manual regeneration commands

```bash
python scripts/make_paper_artifacts.py --config configs/paper.yaml

python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table.tex \
  --label tab:main_results_allbudgets --split_label test --split_filter test --oracle_everywhere

python scripts/make_router_latex_table.py \
  --in_csv artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv \
  --out_tex artifacts/paper/tables/router_table_boolq.tex \
  --label tab:router_boolq --split_label validation --split_filter validation --oracle-single-reference

python scripts/make_router_paired_bootstrap.py \
  --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv \
  --out_csv artifacts/paper/tables/router_paired_bootstrap_gsm8k.csv

python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex

pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex
pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex
```

## Legacy entry points and guardrails

- `run_all_p0_p5.sh` is legacy-guarded (`ALLOW_LEGACY_RUN_ALL=1`) and writes to `artifacts/legacy_results/`.
- `run_optionB_router.sh` is legacy-blocked and points to canonical scripts.
- Legacy outputs are not source-of-truth for manuscript claims unless explicitly re-promoted after independent validation.

## Reproducibility caveats

Some required files are Git LFS pointers in this snapshot, so full from-scratch reruns may fail without pulling LFS payloads.

Known placeholder examples include:
- `data/router_svamp/dev.jsonl`, `data/router_svamp/test.jsonl`
- `data/router_splits_boolq_seeds/seed{0,1,2}/{dev,test}.jsonl`
- `results_main_full/preds_main_adapter_full.jsonl`, `results_main_full/preds_base_full.jsonl`
- `results/preds_base_boolq.jsonl`, `results/preds_base_boolq_reparsed.jsonl`, `results/preds_base_boolq_reparsed2.jsonl`

## Additional canonical semantics

- BoolQ canonical split semantics are **validation** (`paper_table_validation_full_per_seed.csv`, exported rows use `split=validation`).
- Default teacher model for trajectory generation is `gpt-4o-mini` in:
  - `src/teachers/openai_teacher.py`
  - `scripts/generate_gsm8k_trajectories.py` (`--openai_model` default)
