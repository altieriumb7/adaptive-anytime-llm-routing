import re
from typing import Optional, Tuple

CONF_RE = re.compile(r"(?:^|\n)\s*CONF\s*:\s*([01](?:\.\d+)?)\s*(?:\n|$)", re.IGNORECASE)
FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)

def parse_answer_and_conf(text: str) -> Tuple[Optional[str], Optional[float]]:
    ans = None
    conf = None

    m_ans = FINAL_RE.search(text)
    if m_ans:
        ans = m_ans.group(1).strip()

    m_conf = CONF_RE.search(text)
    if m_conf:
        try:
            v = float(m_conf.group(1))
            conf = max(0.0, min(1.0, v))
        except ValueError:
            conf = None

    return ans, conf