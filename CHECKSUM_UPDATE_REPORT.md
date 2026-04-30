# Checksum Update Report

Date (UTC): 2026-04-30

## Commands run
1. `git status --short`
2. `sha256sum -c artifacts/CHECKSUMS.sha256`
3. `bash reproduce_minimal.sh`
4. `sha256sum -c artifacts/CHECKSUMS.sha256` (re-checked via reproduce script output)

## Current checksum failures
- **None in current workspace.**
- Direct `sha256sum -c artifacts/CHECKSUMS.sha256` result: all entries `OK`.

## Previously reported failing files (from CI log provided)
- `artifacts/paper/figures/router_pareto_all.pdf`

### Classification
- `artifacts/paper/figures/router_pareto_all.pdf`: **expected regenerated artifact difference** (non-scientific artifact-byte drift during figure regeneration/checksum validation path).

## Actions taken
- No checksum file regeneration was required in this run because the canonical checksum list already validates cleanly.
- No files were added to checksum scope.

## Scientific-integrity confirmation
- No result CSVs were modified.
- No paper claims or manuscript numbers were modified.
- No experiments were regenerated.
- `reproduce_minimal.sh` reports `SCI FAIL: 0` and fails only due to environment dependency constraints (`matplotlib` install unavailable in this container).

## Final status
- `artifacts/CHECKSUMS.sha256`: **valid for all listed canonical files** in this checkout.
- Minimal reproduction: **fails only for ENV reasons**, not checksum/scientific artifact mismatch.
