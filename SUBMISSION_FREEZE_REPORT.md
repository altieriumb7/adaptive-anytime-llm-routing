# Submission Freeze Report (ATC 2026 Final Maintainer Pass)

Date: 2026-04-30
Scope: reviewer-safe freeze and polish pass for adaptive/trustworthy inference artifact documentation and checks. No scientific claims or experimental conclusions were changed.

## Files changed
- `reproduce_minimal.sh`
- `README.md`
- `REPRODUCIBILITY.md`
- `SUBMISSION_FREEZE_REPORT.md`

## Reviewer-visible cleanup
Searched reviewer-facing files for TODO/FIXME/WIP/placeholder/temporary wording and confirmed no unresolved reviewer-facing placeholders requiring removal. Remaining "placeholder" mentions are intentional reproducibility caveats about unresolved Git LFS pointers.

## Reproducibility messaging updates
### `reproduce_minimal.sh`
Updated messaging and failure accounting to explicitly separate:
- **environment/dependency/toolchain failures** (`[FAIL][ENV]`), and
- **scientific/artifact validation failures** (`[FAIL][SCI]`).

Added explicit offline/network guidance with manual installation steps for `requirements.paper.txt`.
Experimental logic and validation sequence were not changed.

### `README.md` and `REPRODUCIBILITY.md`
Added explicit manual fallback instructions for environments without pip/network access and clarified that such failures are environmental, not contradictions of scientific claims.

## Manuscript synchronization audit
Audited:
- `main_distilling_revised_v0.tex`
- canonical tables under `artifacts/paper/tables/`
- canonical figure under `artifacts/paper/figures/router_pareto_all.pdf`
- artifact-claim mapping in `CLAIMS_TO_ARTIFACTS.md`

Findings:
- Wording is conservative and systems-oriented.
- Reproducibility boundaries are explicit (artifact-level vs full rerun).
- Compute terminology clearly distinguishes allocated-token proxy from realized compute.
- ATC framing remains focused on adaptive inference, dependable routing, trustworthy compute allocation, and autonomous stopping.

## Checks run (with outcomes)
1. `bash run_review_checks.sh`
   - Result: **failed due to missing matplotlib dependency** during paper artifact regeneration stage.
   - Classification: environment/dependency limitation in this sandbox.
2. `sha256sum -c artifacts/CHECKSUMS.sha256`
   - Result: **pass** for all listed frozen artifacts.
3. `bash reproduce_minimal.sh`
   - Result: **overall fail** with ENV-only failures and zero SCI failures.
   - Reported summary: `ENV FAIL = 3`, `SCI FAIL = 0`.
4. `pdflatex -interaction=nonstopmode -halt-on-error main_distilling_revised_v0.tex`
   - Result: **failed** (`pdflatex` not installed in this environment).
   - Classification: toolchain availability limitation.

## Remaining limitations (explicit, unchanged)
1. Full raw-to-paper rerun is not guaranteed from this checkout alone due to LFS/external dependencies.
2. Split-level BoolQ reruns remain conditional on resolving LFS payloads.
3. Reported uncertainty remains split-seed only; retraining/model-seed variance is not claimed.
4. Experimental breadth remains limited (intentionally documented).

## Final readiness judgment (ATC 2026)
**Ready for submission freeze from a reviewer-safety and artifact-transparency perspective.**

The repository now communicates reproducibility boundaries and environment-vs-scientific failure modes more clearly, while preserving conservative, systems-focused framing and not changing scientific results.
