# Submission Freeze Report (Final Pass)

Date: 2026-04-30
Scope: final submission-safe pass for manuscript + repository metadata. No scientific results changed.

## Checks run
1. Placeholder/TODO scan on reviewer-visible docs and manuscript.
2. TeX asset resolution check (`\input` and `\includegraphics`) via `scripts/check_paper_assets.py`.
3. Venue naming consistency scan (ATC/FLLM mentions) in reviewer-visible files.
4. LaTeX log warning scan for undefined citations/references and major warnings.
5. README path-reference sanity script for concrete file references.
6. Minimal reviewer reproduction command rerun (`bash reproduce_minimal.sh`).

## Files inspected
- `main_distilling_revised_v0.tex`
- `README.md`
- `REPRODUCIBILITY.md`
- `REPRODUCIBILITY_STATUS.md`
- `CLAIMS_TO_ARTIFACTS.md`
- `reproduce_minimal.sh`
- `requirements.paper.txt`
- `main_distilling_revised_v0.log` (when available)

## Issues fixed in this pass
- Added explicit compatible matplotlib requirement in `requirements.paper.txt` (`matplotlib>=3.8,<3.11`).
- Improved `reproduce_minimal.sh` with explicit Python dependency preflight (`matplotlib`, `yaml`, `numpy`) and clear install/recheck behavior before artifact regeneration.

## Findings (current state)
- **Broken TeX asset links:** none detected (`check_paper_assets.py` passed).
- **Unresolved TODOs/placeholders:** none reviewer-facing in scanned files (excluding intentional mention of Git LFS placeholders as reproducibility caveats).
- **README/TeX referenced missing files:** no concrete missing canonical artifact reference found; README contains some wildcard/template/path-pattern strings by design.
- **Page-limit status:** not fully verifiable in this environment (`pdfinfo` unavailable).
- **Citation/reference warnings:** no undefined citation/reference warnings were found in available log scan; standard overfull/underfull box warnings remain.
- **Venue-name consistency:** ATC framing is present and consistent with repository positioning.
- **Overclaiming check (abstract/conclusion):** wording remains conservative and explicitly scoped (artifact-level reproducibility, split-seed variability only, limited backbone/task scope).

## Remaining known limitations
1. End-to-end raw-to-paper rerun is still partial due to LFS/external model/API dependencies.
2. Reported variance is split-seed only; no retraining/model-seed variance characterization.
3. Experimental breadth remains limited (one backbone, four budgets, GSM8K main + BoolQ transfer).
4. `bash reproduce_minimal.sh` may fail in network-restricted environments when pip cannot fetch missing packages.

## Final readiness judgment
- **Repository/manuscript are submission-freeze ready for ATC-facing artifact transparency and reviewer safety**, with known reproducibility limits explicitly documented.
- **Acceptance-risk caveat remains scientific-scope related** (limited rerun completeness/variance breadth), not hidden process/documentation defects.
