# REPAIR_REPORT

## Final submission-readiness cleanup

This pass focused on release hardening after canonicalization/router fixes:
- synchronized docs/manuscript wording with current canonical artifact state,
- removed stale runtime-failure claims that were framed as repository facts,
- quarantined backup/dev snapshot files away from active source areas,
- clarified remaining reproducibility limits without overclaiming.

## Canonical contract (release)
- Paper source: `main_distilling_revised_v0.tex`
- Canonical build flow: `run_paper.sh`
- Canonical GSM8K router family: `artifacts/router_optionB/`
- Canonical BoolQ router family: `artifacts/router_optionB_boolq/`
- Canonical anytime prediction family for paper plots: `results_abl/`

## Cleanup actions
1. **Docs/manuscript synchronization**
   - Updated `README.md` with a compact reviewer manifest (canonical outputs, source inputs, legacy paths, external deps).
   - Updated `REPRODUCIBILITY_STATUS.md` to be release-oriented and environment-qualified.
   - Updated manuscript wording in `main_distilling_revised_v0.tex` to remove obsolete BoolQ `split=test` caveat prose.
   - Updated consistency documentation to match current BoolQ semantics and canonical table build behavior.

2. **Legacy/backups quarantine**
   - Moved archived backup scripts under:
     - `archive/legacy_dev_snapshots/dev_old/`
   - This keeps provenance while removing dev snapshot naming from active top-level paths.

## Remaining reproducibility limitations (explicit)
- Reported table uncertainty is split-seed variability (router partition seeds), not model-seed retraining variability.
- Full from-scratch end-to-end reruns still depend on unbundled assets (Git LFS payloads, remote/base models, optional API workflows).
- BoolQ transfer reproducibility is partial in this checkout when LFS-backed transfer files are unavailable.
- Some manuscript claims are intentionally artifact-level claims tied to canonical generated tables/figures rather than raw-data-to-paper reruns.
- Local environment must provide plotting deps (`matplotlib` via `requirements.paper.txt`) for artifact regeneration.
- Local LaTeX toolchain (`pdflatex`) is optional but required to build final PDF from `.tex`.

## Validation executed in this pass (2026-04-08 UTC)
- `python -m compileall scripts src` ✅
- `python scripts/check_paper_assets.py --tex main_distilling_revised_v0.tex` ✅
- `python scripts/make_paper_artifacts.py --config configs/paper.yaml` ⚠️ failed in this sandbox due to missing `matplotlib` (proxy-blocked install)
- `bash run_paper.sh` ⚠️ same dependency limitation in this sandbox

These warnings reflect runtime environment restrictions, not silent repo/script inconsistency.
