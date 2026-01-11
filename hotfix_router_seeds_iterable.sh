#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path
import re

# Search .sh/.bash/.py for the python loop "for seed in router_seeds:" or "for seed in ROUTER_SEEDS:"
candidates = []
for p in Path(".").rglob("*"):
    if not p.is_file():
        continue
    if p.suffix not in (".sh", ".bash", ".py"):
        continue
    try:
        txt = p.read_text(encoding="utf-8")
    except Exception:
        continue
    if ("router_seeds" in txt) or ("ROUTER_SEEDS" in txt):
        candidates.append(p)

patched = []

loop_patterns = [
    re.compile(r"^(?P<indent>[ \t]*)for[ \t]+seed[ \t]+in[ \t]+router_seeds[ \t]*:[ \t]*(#.*)?$", re.M),
    re.compile(r"^(?P<indent>[ \t]*)for[ \t]+seed[ \t]+in[ \t]+ROUTER_SEEDS[ \t]*:[ \t]*(#.*)?$", re.M),
]

for p in candidates:
    txt = p.read_text(encoding="utf-8")
    orig = txt

    for pat in loop_patterns:
        m = pat.search(txt)
        if not m:
            continue

        indent = m.group("indent") or ""
        # Avoid double-inserting
        if (f"{indent}# ---- hotfix: normalize router seed list" in txt) and ("_parse_seed_list" in txt):
            break

        # Insert a tiny helper right before the loop; keep indentation consistent
        insertion_lines = [
            "",
            f"{indent}# ---- hotfix: normalize router seed list (string/int -> list[int]) ----",
            f"{indent}def _parse_seed_list(v):",
            f"{indent}    if v is None:",
            f"{indent}        return [0]",
            f"{indent}    if isinstance(v, int):",
            f"{indent}        return [v]",
            f"{indent}    if isinstance(v, (list, tuple)):",
            f"{indent}        return [int(x) for x in v]",
            f"{indent}    s = str(v).strip()",
            f"{indent}    if not s:",
            f"{indent}        return [0]",
            f"{indent}    return [int(x) for x in s.replace(',', ' ').split()]",
        ]

        # Figure out which variable name is being iterated in this file at this match
        loop_line = m.group(0)
        var_name = "router_seeds" if "router_seeds" in loop_line else "ROUTER_SEEDS"

        insertion_lines += [
            f"{indent}{var_name} = _parse_seed_list({var_name})",
            f"{indent}# --------------------------------------------------------------------",
            "",
        ]
        insertion = "\n".join(insertion_lines)

        txt = txt[:m.start()] + insertion + txt[m.start():]
        break  # patch at most one loop per file

    if txt != orig:
        # backup then write
        bak = p.with_suffix(p.suffix + ".bak")
        bak.write_text(orig, encoding="utf-8")
        p.write_text(txt, encoding="utf-8")
        patched.append(str(p))

print("=== hotfix_router_seeds_iterable ===")
if patched:
    print("Patched:")
    for f in patched:
        print(" -", f)
    print("Backups saved as: <file>.<ext>.bak")
else:
    print("No files patched. (Could not find a python loop: 'for seed in router_seeds/ROUTER_SEEDS:')")
PY
