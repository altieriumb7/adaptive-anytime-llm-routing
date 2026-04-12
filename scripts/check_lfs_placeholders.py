#!/usr/bin/env python3
"""Reviewer-facing LFS placeholder checks for critical reproducibility paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lfs_guard import is_lfs_pointer

# required_now: must be materialized for canonical paper-asset regeneration.
REQUIRED_NOW: Sequence[Tuple[str, str]] = (
    ("results/preds_student_full.jsonl", "canonical GSM8K router prediction source"),
    ("artifacts/router_optionB/paper_table_test_full_per_seed.csv", "canonical GSM8K router per-seed table"),
    ("artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv", "canonical BoolQ transfer per-seed table"),
    ("results_abl/preds_main_adapter.jsonl", "canonical anytime prediction source (student)"),
    ("results_abl/preds_base.jsonl", "canonical anytime prediction source (base)"),
)

# conditional: only needed for transfer split-level reruns.
CONDITIONAL_PATHS: Sequence[Tuple[str, str]] = (
    ("data/router_splits_boolq_seeds/seed0/dev.jsonl", "BoolQ split-level reruns: seed0/dev"),
    ("data/router_splits_boolq_seeds/seed0/test.jsonl", "BoolQ split-level reruns: seed0/test"),
)


def _scan(paths: Iterable[Tuple[str, str]], *, fail_on_pointer: bool, label: str) -> List[str]:
    failures: List[str] = []
    print(f"[{label}] checking {len(list(paths))} path(s)")
    for raw_path, role in paths:
        path = Path(raw_path)
        if not path.exists():
            msg = f"[WARN] missing: {path} ({role})"
            if fail_on_pointer:
                failures.append(msg)
            else:
                print(msg)
            continue
        if is_lfs_pointer(path):
            msg = f"[FAIL] unresolved Git LFS pointer: {path} ({role})"
            if fail_on_pointer:
                failures.append(msg)
            else:
                print(msg.replace("[FAIL]", "[WARN]"))
            continue
        print(f"[OK] materialized: {path} ({role})")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect unresolved Git LFS pointer placeholders.")
    ap.add_argument("--strict-conditional", action="store_true", help="Fail if conditional transfer split files are placeholders/missing.")
    args = ap.parse_args()

    hard_failures = _scan(REQUIRED_NOW, fail_on_pointer=True, label="required-now")
    conditional_failures = _scan(CONDITIONAL_PATHS, fail_on_pointer=args.strict_conditional, label="conditional-transfer")

    if hard_failures or conditional_failures:
        if hard_failures:
            print("\n" + "\n".join(hard_failures))
        if conditional_failures:
            print("\n" + "\n".join(conditional_failures))
        raise SystemExit(
            "LFS placeholder check failed for required canonical inputs. "
            "Artifact-level paper regeneration requires fully materialized required-now files."
        )

    print("[OK] Required canonical inputs are materialized.")
    if not args.strict_conditional:
        print("[INFO] Conditional transfer split checks are warning-only in default mode.")


if __name__ == "__main__":
    main()
