#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml  # pyyaml


def to_args(args_dict: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k, v in (args_dict or {}).items():
        if v is None:
            continue
        key = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                out.append(key)
            continue
        if isinstance(v, (list, tuple)):
            out.append(key)
            out.append(",".join(str(x) for x in v))
            continue
        out.append(key)
        out.append(str(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a python script from a YAML config.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--dry", action="store_true", help="Print command without executing.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    script = cfg.get("script")
    if not script:
        raise SystemExit("Config must contain: script: <path>")
    py = cfg.get("python", sys.executable)
    cmd = [py, script] + to_args(cfg.get("args", {}))

    env = os.environ.copy()
    for k, v in (cfg.get("env", {}) or {}).items():
        env[str(k)] = str(v)

    print("CMD:", " ".join(cmd))
    if args.dry:
        return
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
