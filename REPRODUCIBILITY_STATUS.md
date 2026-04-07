# REPRODUCIBILITY_STATUS

## As-a-shipped reproducibility classification

### Reproducible from this checkout (artifact-level)

- Regenerating router LaTeX tables from canonical CSVs:
  - `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
  - `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
- Validating all `\input` and `\includegraphics` references in `main_distilling_revised_v0.tex`.
- Rebuilding paper-side tables/figures **if plotting dependencies are installed**.

### Not fully reproducible from this checkout alone (end-to-end)

Full regeneration from raw data/training is incomplete because required files are LFS pointers and external dependencies are not bundled.

Detected LFS pointer files (13):
- `data/router_splits_boolq_seeds/seed0/dev.jsonl`
- `data/router_splits_boolq_seeds/seed0/test.jsonl`
- `data/router_splits_boolq_seeds/seed1/dev.jsonl`
- `data/router_splits_boolq_seeds/seed1/test.jsonl`
- `data/router_splits_boolq_seeds/seed2/dev.jsonl`
- `data/router_splits_boolq_seeds/seed2/test.jsonl`
- `data/router_svamp/dev.jsonl`
- `data/router_svamp/test.jsonl`
- `results/preds_base_boolq.jsonl`
- `results/preds_base_boolq_reparsed.jsonl`
- `results/preds_base_boolq_reparsed2.jsonl`
- `results_main_full/preds_base_full.jsonl`
- `results_main_full/preds_main_adapter_full.jsonl`

## Canonical reproducibility workflow

```bash
bash run_paper.sh
```

This is the only authoritative paper-build path and is designed to fail clearly when required inputs/dependencies are missing.

## Validation results from this runtime

- `python -m compileall scripts src` passed.
- Router table regeneration commands passed.
- `python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex` passed.
- `bash run_paper.sh` failed because `matplotlib` is unavailable and could not be installed in this environment.
- `pdflatex` command not found, so TeX compilation could not be executed in this runtime.

## Honest interpretation for submission

- The artifact is synchronized at the paper/table/figure reference level.
- The shipped repo supports conservative paper-asset regeneration from canonical bundled artifacts.
- Complete from-scratch experiment regeneration requires external assets and remains explicitly out of scope for this snapshot.
