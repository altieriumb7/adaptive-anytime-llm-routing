# Claims to Artifacts Traceability (Submission Freeze)

Date: 2026-04-30
Paper: `main_distilling_revised_v0.tex`

Legend:
- **Status scope**: `artifact-level` (supported from bundled artifacts) vs `full-rerun` (raw-to-paper rerun).
- **Support**: supported / partially supported / unsupported.

| Claim | Manuscript location | Artifact(s) | Row/field mapping | Regeneration command | Status scope | Support | Notes |
|---|---|---|---|---|---|---|---|
| GSM8K trajectories: 7,473 problems | Abstract + Method | `data/anytime_gsm8k_train_v2.jsonl` | line count | `python scripts/peek_jsonl.py data/anytime_gsm8k_train_v2.jsonl` | artifact-level | supported | Training corpus is bundled. |
| 29,892 budget-labeled SFT examples | Abstract + Method | `data/sft_gsm8k_anytime_v2.jsonl` | line count (4x 7,473) | `python scripts/peek_jsonl.py data/sft_gsm8k_anytime_v2.jsonl` | artifact-level | partially supported | Requires verifying count in shipped file. |
| Backbone: Qwen2.5-Math-1.5B-Instruct + QLoRA | Method | `configs/train_student_qlora.template.yaml`, `scripts/train_student_qlora.py` | model/config fields | `python scripts/train_student_qlora.py --help` | artifact-level | supported | Config/script encode training recipe; no new retraining claim. |
| GSM8K anytime Acc@1..Acc@4 (base/adapter table) | Table `tab:anytime_gsm8k` | `results_abl/preds_base.jsonl`, `results_abl/preds_main_adapter.jsonl` | grouped by `t` and `correct` | `python scripts/eval_anytime.py --preds results_abl/preds_main_adapter.jsonl` (and base) | artifact-level | supported | Explicitly separate family from router table family. |
| B2 confidence routing improvement at matched compute | Abstract + Results | `artifacts/router_optionB/paper_table_test_full_per_seed.csv` | `budget_tag=B2`, compare `policy=fixed` vs `policy=conf`; mean over seeds | `python scripts/make_router_latex_table.py --in_csv artifacts/router_optionB/paper_table_test_full_per_seed.csv --out_tex artifacts/paper/tables/router_table.tex --label tab:main_results_allbudgets --split_label test --split_filter test --oracle_everywhere` | artifact-level | supported | Paper values must come from this canonical CSV. |
| B3 stability routing improvement at matched compute | Abstract + Results | `artifacts/router_optionB/paper_table_test_full_per_seed.csv` | `budget_tag=B3`, compare `policy=fixed` vs `policy=stability` | same as above | artifact-level | supported | Split-seed variability only (n=3). |
| Oracle stopping reference result | Abstract + Results | `artifacts/router_optionB/paper_table_test_full_per_seed.csv` | `policy=oracle` rows aggregated over seeds | same as above | artifact-level | supported | Not retraining variance. |
| BoolQ transfer: confidence weaker than fixed at B2/B3 | Results + Discussion | `artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv` | `budget_tag in {B2,B3}`, compare `conf` vs `fixed` | `python scripts/make_router_latex_table.py --in_csv artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv --out_tex artifacts/paper/tables/router_table_boolq.tex --label tab:router_boolq --split_label validation --split_filter validation --oracle-single-reference` | artifact-level | supported | Artifact-level transfer claim. |
| Router split seeds are {0,1,2} | Experimental setup | `data/router_splits_seeds/manifest.json`, per-seed split files | manifest `seeds` | `python scripts/make_router_splits.py --source results/preds_student_full.jsonl --out_root data/router_splits_seeds --seeds 0,1,2 --manifest data/router_splits_seeds/manifest.json` | full-rerun (split-level) | supported | Split regeneration supported from bundled predictions. |
| Per-step budgets {96,160,224,320}; cumulative {96,256,480,800} | Method | `scripts/run_router_optionB_repro.py`, router outputs CSV | `mean_tokens` tiers and policy construction | `python scripts/run_router_optionB_repro.py --out_dir artifacts/router_optionB --seeds 0,1,2` | artifact-level | supported | Compute proxy is allocated tokens. |
| Family separation (anytime vs router not row-identical) | Results + Reproducibility | `artifacts/paper/tables/provenance_rowmatch_audit.csv` | `exact_match_rate` 0.0 for `results_abl` vs split files | `python scripts/refresh_canonical_provenance.py` | artifact-level | supported | Central provenance audit. |

## Submission freeze note
All claims above are bound to canonical artifacts and checksums in `artifacts/CHECKSUMS.sha256`. Full raw-to-paper rerun (trajectory generation + retraining) is **not guaranteed** from this checkout alone.
