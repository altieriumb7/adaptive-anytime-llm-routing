# Distilling-Anytime-Trajectories-with-Calibrated-Confidence

This repository is the **artifact companion** for the paper *Distilling Anytime Trajectories with Calibrated Confidence*. The final LaTeX manuscript source is **not** included here, so this release should be treated as **artifact-only**, not as a paper+artifact source bundle.

The artifact studies how to distill the full improvement trajectory of a slow teacher into a student that can stream intermediate answers—draft → verified answer → corrected answer → final answer—together with calibrated confidence and a learnable stop/continue policy.

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements.openai.txt   # only if you are creating the dataset
pip install -r requirements.train.txt    # only if you are training models
```

## Dataset creation

To speed up dataset generation on Windows PowerShell, run the following commands in separate shells:

```powershell
python scripts\make_batch_anytime_requests.py --split train --model gpt-4o-mini --out data\batch_anytime_train.jsonl
python scripts\submit_batch.py --infile data\batch_anytime_train.jsonl
```

## Terminology note

In manuscript-facing documentation, the source-side ordering signal should be described as **source-side surface-complexity**. Some internal artifact field names still use legacy `difficulty` terminology for compatibility with existing outputs and scripts; those names are retained intentionally.
