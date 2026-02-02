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

    # take the LAST #### match (important for multi-line outputs)
    matches = list(_RE_HASH.finditer(text))
    if matches:
        ans = matches[-1].group(1).strip()

    # take the LAST CONF match
    cmatches = list(_RE_CONF.finditer(text))
    if cmatches:
        try:
            conf = float(cmatches[-1].group(1))
            conf = max(0.0, min(1.0, conf))
        except Exception:
            conf = None

    # --- NEW: boxed fallback (SVAMP often uses \\boxed{...}) ---
    if ans == "" or "<final_answer>" in ans.lower() or "final_answer" in ans.lower():
        try:
            _RE_BOXED = re.compile(r"\\\\boxed\\{([^}]*)\\}")
            bmatches = list(_RE_BOXED.finditer(text))
            if bmatches:
                ans = bmatches[-1].group(1).strip()
        except Exception:
            pass

    # --- NEW: last number in tail fallback (SVAMP/GSM8K messy formatting) ---
    if ans == "" or "<final_answer>" in ans.lower() or "final_answer" in ans.lower():
        tail = text[-1000:]
        nums = re.findall(r"(-?\\d+(?:\\.\\d+)?)", tail)
        if nums:
            ans = nums[-1].strip()

    # robust yes/no cleanup if needed
    if ans == "" or "<final_answer>" in ans.lower() or "final_answer" in ans.lower():
        ans = _canon_yesno(text[-800:])  # look at tail
    else:
        ans = _canon_yesno(ans)

    # --- NEW: canonicalize numeric strings like "17.0" -> "17" ---
    try:
        af = float(ans)
        if af.is_integer():
            ans = str(int(af))
        else:
            ans = str(af)
    except Exception:
        pass

    return ans, conf
