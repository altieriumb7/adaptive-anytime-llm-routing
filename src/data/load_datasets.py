from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from datasets import load_dataset


# -------------------------
# Common normalization
# -------------------------

# GSM8K answers are often like: "... #### 42"
GSM8K_FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)

def normalize_answer(ans: str) -> str:
    """Lightweight, shared normalization.

    - strip
    - remove commas in numbers
    - remove trailing period
    """
    if ans is None:
        return ""
    ans = str(ans).strip()
    ans = ans.replace(",", "")
    ans = ans.rstrip(".")
    return ans


# -------------------------
# Dataset examples
# -------------------------

@dataclass
class Example:
    uid: str
    problem: str
    gold: str
    meta: Dict[str, Any]


# -------------------------
# GSM8K
# -------------------------

def load_gsm8k(split: str = "train") -> List[Example]:
    """
    Loads GSM8K from HF: dataset 'gsm8k', config 'main'
    Each row has: question, answer
    """
    ds = load_dataset("gsm8k", "main", split=split)
    out: List[Example] = []

    for i, row in enumerate(ds):
        question = row["question"]
        answer = row["answer"]
        m = GSM8K_FINAL_RE.search(answer)
        if not m:
            gold = normalize_answer(answer)
        else:
            gold = normalize_answer(m.group(1))

        out.append(
            Example(
                uid=f"gsm8k_{split}_{i}",
                problem=question,
                gold=gold,
                meta={"dataset": "gsm8k", "config": "main", "split": split},
            )
        )

    return out


# -------------------------
# MATH (pinned, reproducible)
# -------------------------

# Paper-reproducible default:
# https://huggingface.co/datasets/EleutherAI/hendrycks_math
DEFAULT_MATH_DATASET_ID = os.getenv("MATH_DATASET_ID", "EleutherAI/hendrycks_math")


def _extract_balanced_braces(text: str, open_brace_idx: int) -> Optional[Tuple[str, int]]:
    """Given text and index of '{', return (content, close_idx) for balanced braces."""
    if open_brace_idx < 0 or open_brace_idx >= len(text) or text[open_brace_idx] != "{":
        return None
    depth = 0
    for j in range(open_brace_idx, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_idx + 1 : j], j
    return None


_BOXED_CMD_RE = re.compile(r"\\boxed\s*\{")

def extract_math_final(solution_text: str) -> str:
    """Best-effort final answer extraction for MATH-style solutions.

    Priority:
      1) last \boxed{...} (balanced braces)
      2) last non-empty line
    """
    if solution_text is None:
        return ""

    text = str(solution_text)

    # Find last \boxed{ ... } with balanced braces
    matches = list(_BOXED_CMD_RE.finditer(text))
    for m in reversed(matches):
        open_idx = m.end() - 1  # points at '{'
        got = _extract_balanced_braces(text, open_idx)
        if got is not None:
            inside, _ = got
            return normalize_answer(_strip_latex_wrappers(inside))

    # fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return normalize_answer(_strip_latex_wrappers(lines[-1]))

    return normalize_answer(_strip_latex_wrappers(text))


_LATEX_WRAP_RE = [
    re.compile(r"^\$+(.*)\$+$", re.DOTALL),
    re.compile(r"^\\\((.*)\\\)$", re.DOTALL),
    re.compile(r"^\\\[(.*)\\\]$", re.DOTALL),
]

def _strip_latex_wrappers(s: str) -> str:
    """Remove common LaTeX wrappers that often surround final answers."""
    if s is None:
        return ""
    out = str(s).strip()

    # unwrap $...$, \(...\), \[...\]
    for rx in _LATEX_WRAP_RE:
        m = rx.match(out)
        if m:
            out = m.group(1).strip()

    # drop \left/\right and spacing commands
    out = out.replace("\\left", "").replace("\\right", "")
    out = re.sub(r"\\,", "", out)

    # remove text wrappers that don't affect numeric value
    out = re.sub(r"\\text\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", out)

    return out.strip()


def load_math(split: str = "train") -> List[Example]:
    """Loads MATH from a pinned HF dataset ID for reproducibility."""
    ds = load_dataset(DEFAULT_MATH_DATASET_ID, split=split)

    out: List[Example] = []
    for i, row in enumerate(ds):
        # EleutherAI/hendrycks_math schema: problem, solution, level, type
        problem = row.get("problem") or row.get("question") or row.get("prompt")
        solution = row.get("solution") or row.get("answer")

        if problem is None or solution is None:
            continue

        gold = extract_math_final(solution)
        out.append(
            Example(
                uid=f"math_{split}_{i}",
                problem=problem,
                gold=gold,
                meta={"dataset": DEFAULT_MATH_DATASET_ID, "split": split},
            )
        )
    return out


# -------------------------
# SVAMP (extra dataset for generalization)
# -------------------------

# Common IDs:
# - Dahoas/svamp (small)
# - ChilleD/SVAMP (popular)
DEFAULT_SVAMP_DATASET_ID = os.getenv("SVAMP_DATASET_ID", "Dahoas/svamp")


def load_svamp(split: str = "train") -> List[Example]:
    """Load SVAMP-style arithmetic word problems."""
    ds = load_dataset(DEFAULT_SVAMP_DATASET_ID, split=split)

    out: List[Example] = []
    for i, row in enumerate(ds):
        # Try common column names
        body = row.get("Body") or row.get("body") or row.get("context") or ""
        q = row.get("Question") or row.get("question") or row.get("query") or row.get("Problem") or row.get("problem") or ""
        problem = (str(body).strip() + " " + str(q).strip()).strip()
        if not problem:
            # some variants just have "text"
            problem = str(row.get("text") or row.get("prompt") or "").strip()

        gold = row.get("Answer") or row.get("answer") or row.get("label") or row.get("target")
        if gold is None:
            gold = row.get("result")  # some variants
        gold = normalize_answer(_strip_latex_wrappers(str(gold) if gold is not None else ""))

        out.append(
            Example(
                uid=f"svamp_{split}_{i}",
                problem=problem,
                gold=gold,
                meta={"dataset": DEFAULT_SVAMP_DATASET_ID, "split": split},
            )
        )
    return out


# -------------------------
# Convenience
# -------------------------

def load_dataset_by_name(name: str, split: str = "train") -> List[Example]:
    name = (name or "").lower().strip()
    if name in ("gsm8k", "gsm"):
        return load_gsm8k(split=split)
    if name in ("math", "hendrycks_math", "competition_math"):
        return load_math(split=split)
    if name in ("svamp",):
        return load_svamp(split=split)
    raise ValueError(f"Unknown dataset name: {name}")
