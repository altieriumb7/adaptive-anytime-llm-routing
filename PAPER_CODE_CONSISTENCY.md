# Paper–Code Consistency Report

## Confirmed matches
- Main paper figure now references an existing file: `artifacts/paper/figures/router_pareto_all.pdf`.
- Main and BoolQ router tables in the manuscript are now sourced directly from generated artifacts via `\input`.
- Artifact generator config and script now agree on canonical router CSV format (`policy,budget_tag,acc_mean,tokens_mean,...`).
- SFT build CLI arguments now match core builder function parameters.
- Evaluation script now executes a consistent budget loop and produces per-budget rows for downstream router/calibration scripts.

## Fixed mismatches
1. **LaTeX figure filename mismatch**
   - from `router_pareto_all_fixed.pdf` -> `router_pareto_all.pdf`.

2. **Evaluation flow mismatch**
   - fixed broken loop structure and undefined generation/stopping state in `scripts/eval_anytime.py`.

3. **SFT confidence plumbing mismatch**
   - resolved undefined variables and reordered per-step confidence assignment logic.

4. **Router artifact parser mismatch**
   - updated `scripts/make_paper_artifacts.py` to parse current canonical long-form router CSV.

5. **Paper source vs generated tables mismatch risk**
   - replaced hard-coded table bodies with generated table inputs.

## Remaining limitations
- Some datasets/results are present only as Git LFS pointers in this checkout.
- Full from-scratch reproduction (raw trajectory generation through final tables) is not guaranteed without external assets.
- `pdflatex` is not installed in this runtime, so full compile validation is documented but not executed here.

## Claims softened in the paper
- Reproducibility wording now states **partial reproducibility from shipped artifacts** instead of implying complete end-to-end reproduction from this snapshot alone.
