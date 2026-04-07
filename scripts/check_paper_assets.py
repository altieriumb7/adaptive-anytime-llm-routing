#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List


def parse_graphicspath(tex: str) -> List[str]:
    line = None
    for raw in tex.splitlines():
        if "\\graphicspath" in raw:
            line = raw
            break
    if not line:
        return [""]
    paths = re.findall(r"\{([^{}]+)\}", line)
    out = []
    for p in paths:
        p = p.strip()
        if p.startswith("./"):
            p = p[2:]
        if p == "graphicspath":
            continue
        out.append(p)
    return out or [""]


def find_commands(tex: str, cmd: str) -> List[str]:
    # supports optional args: \cmd[...]{target}
    pat = rf"\\{cmd}(?:\[[^\]]*\])?\{{([^}}]+)\}}"
    return [x.strip() for x in re.findall(pat, tex)]


def resolve_input(root: Path, tex_file: Path, target: str) -> Path | None:
    base = tex_file.parent
    cand = (base / target).resolve()
    if cand.exists():
        return cand
    if cand.suffix == "":
        cand_tex = cand.with_suffix(".tex")
        if cand_tex.exists():
            return cand_tex
    return None


def resolve_graphic(root: Path, tex_file: Path, target: str, graphicspaths: List[str]) -> Path | None:
    exts = ["", ".pdf", ".png", ".jpg", ".jpeg", ".eps"]

    def iter_base_paths():
        yield tex_file.parent
        for gp in graphicspaths:
            yield root / gp

    for base in iter_base_paths():
        for ext in exts:
            p = (base / f"{target}{ext}").resolve()
            if p.exists():
                return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate LaTeX paper asset dependencies.")
    ap.add_argument("--tex", default="main_distilling_revised_v0.tex", help="Main LaTeX file to validate.")
    args = ap.parse_args()

    tex_file = Path(args.tex)
    if not tex_file.exists():
        print(f"[ERROR] TeX file not found: {tex_file}")
        return 2

    root = Path.cwd().resolve()
    text = tex_file.read_text(encoding="utf-8")
    graphicspaths = parse_graphicspath(text)

    missing = []

    for target in find_commands(text, "input"):
        resolved = resolve_input(root, tex_file, target)
        if resolved is None:
            missing.append(("input", target))

    for target in find_commands(text, "includegraphics"):
        resolved = resolve_graphic(root, tex_file, target, graphicspaths)
        if resolved is None:
            missing.append(("includegraphics", target))

    if missing:
        print("[ERROR] Missing paper dependencies:")
        for kind, target in missing:
            print(f"  - {kind}: {target}")
        return 1

    print(f"[OK] All \\input and \\includegraphics dependencies resolved for {tex_file}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
