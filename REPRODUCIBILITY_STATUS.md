# REPRODUCIBILITY_STATUS

## Release-facing reproducibility scope

This repository is a **paper artifact with canonical bundled outputs**, not a claim of full from-scratch experimental reproducibility.

### What is reproducible from bundled canonical artifacts
- Router LaTeX tables can be regenerated from canonical per-seed CSVs:
  - `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
  - `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
- Paper-side table/figure assets can be regenerated with:
  - `python scripts/make_paper_artifacts.py --config configs/paper.yaml`
  - `bash run_paper.sh`
- LaTeX asset references are checkable via:
  - `python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex`

### What is not fully reproducible from this checkout alone
End-to-end regeneration from raw trajectory generation + training depends on assets not bundled directly in this snapshot:
- Git LFS payloads (some split/result files are pointers only)
- Remote model availability/checkpoints
- Optional API-backed generation workflows

## Canonical paper path
- Canonical entrypoint: `main_distilling_revised_v0.tex`
- Canonical build flow: `run_paper.sh` (or `run_paper.sh --config configs/paper.yaml`)
- Canonical paper inputs are documented in `README.md` (Reviewer manifest section)

## Environment note (this runtime)
In this sandbox (2026-04-08 UTC), package installation for `matplotlib` is blocked by proxy restrictions, so figure/table regeneration commands that import matplotlib fail here even though the repo workflow is correct.

## Honest interpretation
- **Supported:** paper-asset regeneration from canonical bundled artifacts when required local dependencies are available.
- **Not claimed:** complete end-to-end reproduction from raw data in an offline/no-LFS/no-remote-assets environment.
