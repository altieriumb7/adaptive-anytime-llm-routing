# PAPER_CODE_CONSISTENCY

## Canonical synchronization contract

This artifact now uses one consistent contract:

- Paper source: `main_distilling_revised_v0.tex`
- Generated table inputs: `artifacts/paper/tables/router_table.tex`, `artifacts/paper/tables/router_table_boolq.tex`
- Generated figure input: `artifacts/paper/figures/router_pareto_all.pdf`
- Source router CSVs:
  - GSM8K: `artifacts/router_optionB/paper_table_test_full_per_seed.csv`
  - BoolQ: `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv`
- Canonical build script: `run_paper.sh`

## Verified method/code alignment points

1. **Compute-matched Option-B routing in paper vs scripts**
   - Paper describes compute-matched routing and deterministic mixtures.
   - Router artifacts consumed by table generation are from `artifacts/router_optionB/` (canonical Option-B outputs).

2. **BoolQ split semantics**
   - Paper states BoolQ transfer as validation.
   - Table build enforces `--split_filter validation` with `--legacy_split_aliases test` to handle historical labels conservatively.

3. **Paper references are generated artifacts, not hand-copied tables**
   - Main and BoolQ results are injected via `\input{...}` files under `artifacts/paper/tables/`.

4. **Figure path semantics**
   - Paper points to `router_pareto_all.pdf` and resolves through `\graphicspath` to `artifacts/paper/figures/`.

## Audited limitations (explicitly documented, not hidden)

- Full end-to-end re-run from raw data is blocked by missing LFS objects and external dependencies.
- In this runtime, `pdflatex` is not available, so final TeX compilation cannot be executed here.
- `run_paper.sh` fails clearly when Python plotting dependencies (e.g., `matplotlib`) are absent.

## Result claim checks against canonical CSVs

The headline GSM8K claims in abstract/results were checked against means over seeds in
`artifacts/router_optionB/paper_table_test_full_per_seed.csv`:
- Fixed B2 vs Confidence B2 accuracy delta matches paper wording.
- Fixed B3 vs Stability B3 accuracy delta matches paper wording.
- Oracle reference magnitude matches table-derived values.

No overstated claim requiring unshipped artifacts was introduced.
