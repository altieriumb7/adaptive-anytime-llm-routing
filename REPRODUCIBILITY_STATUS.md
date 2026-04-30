# REPRODUCIBILITY_STATUS

## Release-facing reproducibility scope

This repository is a **paper artifact with canonical bundled outputs**, not a claim of full from-scratch experimental reproducibility or full model-seed variance characterization.

### Three-level reproducibility contract
- **Level A (manuscript artifact regeneration):** supported from bundled canonical artifacts (tables/figures and asset checks).
- **Level B (split/router reruns from bundled predictions):** supported for seeded router evaluation workflows (GSM8K canonical Option-B path).
- **Level C (full trajectory generation + training rerun):** not guaranteed from this checkout alone.

### What is reproducible from bundled canonical artifacts
- Router LaTeX tables can be regenerated from canonical per-seed CSVs:
  - `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
  - `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
- Paper-side table/figure assets can be regenerated with:
  - `python scripts/make_paper_artifacts.py --config configs/paper.yaml`
  - `bash run_paper.sh`
  - `bash run_review_checks.sh` (reviewer preflight: compile checks + LFS pointer checks + asset validation)
- LaTeX asset references are checkable via:
  - `python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex`

### What is not fully reproducible from this checkout alone
End-to-end regeneration from raw trajectory generation + training depends on assets not bundled directly in this snapshot:
- Git LFS payloads (some split/result files are pointers only)
- Remote model availability/checkpoints
- Optional API-backed generation workflows
- BoolQ transfer inputs that may be incomplete without LFS payload resolution in some environments

## Canonical paper path
- Canonical entrypoint: `main_distilling_revised_v0.tex`
- Canonical build flow: `run_paper.sh` (or `run_paper.sh --config configs/paper.yaml`)
- Canonical paper inputs are documented in `README.md` (Reviewer manifest section)

## Environment note
Artifact regeneration requires local availability of dependencies from `requirements.paper.txt` (notably `matplotlib`) and, for PDF build, a LaTeX toolchain (`pdflatex`).

## Frozen artifact integrity
- Canonical submission artifact digests: `artifacts/CHECKSUMS.sha256`
- Verification command: `sha256sum -c artifacts/CHECKSUMS.sha256`

## Honest interpretation
- **Supported:** paper-asset regeneration from canonical bundled artifacts when required local dependencies are available.
- **Supported:** split-seed router reproducibility (seeded dev/test partitions and compute-matched evaluation).
- **Not claimed:** model-seed retraining reproducibility/variance estimates for the distilled student.
- **Not claimed:** complete end-to-end reproduction from raw data in an offline/no-LFS/no-remote-assets environment.

## Legacy entrypoint policy
- `run_optionB_router.sh` is intentionally blocked as a legacy script to prevent non-canonical reviewer runs.
- `run_all_p0_p5.sh` remains available only behind `ALLOW_LEGACY_RUN_ALL=1` for archival end-to-end experimentation.
