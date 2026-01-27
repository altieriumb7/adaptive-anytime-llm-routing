import re

_RE_HASH = re.compile(r'^\s*####\s*(.*?)\s*$', re.MULTILINE)
_RE_CONF = re.compile(r'CONF\s*:\s*([01](?:\.\d+)?)', re.IGNORECASE)

def _clean_text(s: str) -> str:
    s = (s or "").strip()
    # remove common wrappers
    s = s.replace("<yes>", "yes").replace("<no>", "no")
    s = re.sub(r"</?final_answer>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[{}]", " ", s)
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)  # \text{yes} -> yes
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _canon_yesno(s: str) -> str:
    s = _clean_text(s)
    # Keep only letters for yes/no decision
    letters = re.sub(r"[^a-z]", "", s)
    if letters in {"yes", "y", "true", "t"}:
        return "yes"
    if letters in {"no", "n", "false", "f"}:
        return "no"
    # fallback: contains yes/no anywhere
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    return s.strip()

import re

def parse_answer_and_conf(text: str):
    text = text or ""
    ans = ""
    conf = None

    # Take the LAST match for both "####" and "CONF:"
    # Find the last occurrence of the final answer
    matches = list(re.finditer(r"####\s*([^\n\r]+)", text))
    if matches:
        ans = matches[-1].group(1).strip()  # Take the last match for the final answer

    # Find the last occurrence of CONF:
    cmatches = list(re.finditer(r"CONF\s*:\s*([01](?:\.\d+)?)", text, re.IGNORECASE))
    if cmatches:
        try:
            conf = float(cmatches[-1].group(1))  # Extract confidence
            conf = max(0.0, min(1.0, conf))  # Ensure confidence is between 0 and 1
        except Exception:
            conf = None

    return ans, conf
