from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Optional, Union

from .load_datasets import normalize_answer


# Basic numeric patterns
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_FRAC_RE = re.compile(r"^\s*(-?\d+)\s*/\s*(-?\d+)\s*$")
_LATEX_FRAC_RE = re.compile(r"^\s*\\(?:d)?frac\{\s*(-?\d+)\s*\}\{\s*(-?\d+)\s*\}\s*$")


def _strip_math_wrappers(s: str) -> str:
    if s is None:
        return ""
    out = str(s).strip()

    # unwrap $...$ and \(...\) / \[...\]
    out = re.sub(r"^\$+(.*)\$+$", r"\1", out, flags=re.DOTALL).strip()
    out = re.sub(r"^\\\((.*)\\\)$", r"\1", out, flags=re.DOTALL).strip()
    out = re.sub(r"^\\\[(.*)\\\]$", r"\1", out, flags=re.DOTALL).strip()

    out = out.replace("\\left", "").replace("\\right", "")
    out = re.sub(r"\\,", "", out)
    out = re.sub(r"\\text\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", out)

    return out.strip()


def _to_number(s: str) -> Optional[Union[Fraction, float]]:
    """Parse ints/decimals/fractions (including LaTeX \frac) into a comparable numeric type."""
    if s is None:
        return None
    x = _strip_math_wrappers(normalize_answer(str(s)))
    if not x:
        return None

    m = _LATEX_FRAC_RE.match(x)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            return None
        return Fraction(a, b)

    m = _FRAC_RE.match(x)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            return None
        return Fraction(a, b)

    if _NUM_RE.match(x):
        # prefer int-like floats as Fractions? keep float for simplicity
        try:
            if "." in x:
                return float(x)
            return Fraction(int(x), 1)
        except Exception:
            return None

    return None


def is_correct(pred: Optional[str], gold: str, tol: float = 1e-9) -> bool:
    """Correctness check.

    - exact match after normalize_answer + stripping common LaTeX wrappers
    - else numeric equivalence for:
        * ints
        * decimals
        * fractions: a/b and \frac{a}{b}
    """
    if pred is None:
        return False

    p = _strip_math_wrappers(normalize_answer(str(pred)))
    g = _strip_math_wrappers(normalize_answer(str(gold)))

    if p == g:
        return True

    pn = _to_number(p)
    gn = _to_number(g)
    if pn is None or gn is None:
        return False

    # Compare Fractions exactly when possible
    if isinstance(pn, Fraction) and isinstance(gn, Fraction):
        return pn == gn

    # Mix Fraction/float -> compare as floats with tolerance
    try:
        pf = float(pn) if isinstance(pn, Fraction) else float(pn)
        gf = float(gn) if isinstance(gn, Fraction) else float(gn)
        return math.isclose(pf, gf, rel_tol=tol, abs_tol=tol)
    except Exception:
        return False
