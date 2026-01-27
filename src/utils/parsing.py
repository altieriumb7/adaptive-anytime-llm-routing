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

def parse_answer_and_conf(raw_text: str):
    # Strip out LaTeX and unnecessary symbols
    raw_text = re.sub(r'\\[a-zA-Z]+{[^}]+}', '', raw_text)  # Remove LaTeX
    raw_text = re.sub(r'[^\w\s\.]', '', raw_text)  # Remove punctuation

    match = re.search(r"####\s*([^\n\r]+)\s*[\n\r]+\s*CONF\s*:\s*([01](?:\.\d+)?)", raw_text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        confidence = match.group(2).strip()
        return answer, confidence
    return None, None
