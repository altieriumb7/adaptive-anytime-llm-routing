#!/usr/bin/env python3
"""Utilities to detect unresolved Git LFS pointer placeholders."""

from __future__ import annotations

from pathlib import Path

LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    """Return True when file content starts with the Git LFS pointer header."""
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
    except UnicodeDecodeError:
        # Binary files are not expected in JSONL/CSV checks.
        return False
    return first.startswith(LFS_POINTER_PREFIX)


def assert_materialized(path: Path, role: str) -> None:
    """Fail fast if path is missing or unresolved as an LFS pointer."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required {role}: {path}. "
            "This repository snapshot is partial for some transfer assets; "
            "see REPRODUCIBILITY.md before attempting end-to-end reruns."
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Required {role} is empty: {path}")
    if is_lfs_pointer(path):
        raise ValueError(
            f"Required {role} is an unresolved Git LFS pointer: {path}. "
            "Fetch LFS payloads before running this step. "
            "For reviewer-safe checks, run: bash run_review_checks.sh"
        )
