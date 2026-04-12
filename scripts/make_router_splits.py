#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from scripts.lfs_guard import assert_materialized


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_source(source: Path) -> tuple[List[str], List[dict], List[str]]:
    raw_lines: List[str] = []
    rows: List[dict] = []
    seen: Dict[str, str] = {}
    for line in source.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        uid = str(obj.get('uid', '')).strip()
        if not uid:
            raise ValueError('Source row missing uid; split key must be example/problem id.')
        problem = str(obj.get('problem', '')).strip()
        if uid in seen and seen[uid] != problem:
            raise ValueError(f'uid {uid} maps to multiple problems; split key is ambiguous.')
        seen.setdefault(uid, problem)
        raw_lines.append(line)
        rows.append(obj)
    return raw_lines, rows, sorted(seen)


def write_split(path: Path, raw_lines: List[str], rows: List[dict], keep_uids: set[str]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    uids = set()
    with path.open('w', encoding='utf-8') as out:
        for line, obj in zip(raw_lines, rows):
            uid = str(obj['uid'])
            if uid in keep_uids:
                out.write(line + '\n')
                n_rows += 1
                uids.add(uid)
    return {'rows': n_rows, 'examples': len(uids), 'sha256': sha256_file(path)}


def main() -> None:
    ap = argparse.ArgumentParser(description='Build seeded router splits by uid and emit provenance manifest.')
    ap.add_argument('--source', default='results/preds_student_full.jsonl')
    ap.add_argument('--out_root', default='data/router_splits_seeds')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--dev_ratio', type=float, default=0.8)
    ap.add_argument('--manifest', default='data/router_splits_seeds/manifest.json')
    ap.add_argument('--no_write_splits', action='store_true', help='Do not rewrite split files; only record manifest from existing files.')
    args = ap.parse_args()

    source = Path(args.source)
    assert_materialized(source, role='router source predictions JSONL')
    out_root = Path(args.out_root)
    seeds = [int(x) for x in str(args.seeds).replace(',', ' ').split() if x.strip()]

    raw_lines, rows, uid_list = load_source(source)
    n_examples = len(uid_list)
    dev_n = int(math.ceil(n_examples * float(args.dev_ratio)))
    test_n = n_examples - dev_n

    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_schema_version': '1.1',
        'generator_script': 'scripts/make_router_splits.py',
        'source_file_path': str(source),
        'source_sha256': sha256_file(source),
        'seeds': seeds,
        'counts': {
            'source_rows': len(rows),
            'source_examples': n_examples,
            'dev_examples_per_seed': dev_n,
            'test_examples_per_seed': test_n,
        },
        'split_key': 'uid (example/problem id)',
        'metrics': ['rows', 'examples', 'sha256'],
        'per_seed': [],
    }

    for seed in seeds:
        seed_dir = out_root / f'seed{seed}'
        dev_path = seed_dir / 'dev.jsonl'
        test_path = seed_dir / 'test.jsonl'

        if args.no_write_splits:
            dev_stats = {'rows': sum(1 for _ in dev_path.open('r', encoding='utf-8')), 'examples': None, 'sha256': sha256_file(dev_path)}
            test_stats = {'rows': sum(1 for _ in test_path.open('r', encoding='utf-8')), 'examples': None, 'sha256': sha256_file(test_path)}
        else:
            ids = uid_list[:]
            random.Random(seed).shuffle(ids)
            dev_ids = set(ids[:dev_n])
            test_ids = set(ids[dev_n:])
            dev_stats = write_split(dev_path, raw_lines, rows, dev_ids)
            test_stats = write_split(test_path, raw_lines, rows, test_ids)

        manifest['per_seed'].append({
            'seed': seed,
            'dev_path': str(dev_path),
            'test_path': str(test_path),
            'dev': dev_stats,
            'test': test_stats,
        })

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {manifest_path}')


if __name__ == '__main__':
    main()
