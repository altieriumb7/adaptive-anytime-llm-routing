#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def count_examples(jsonl_path: Path) -> int:
    uids = set()
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get('uid')
            if uid is not None:
                uids.add(uid)
    return len(uids)


def refresh_split_manifest(split_manifest_path: Path) -> Dict[str, Any]:
    manifest = json.loads(split_manifest_path.read_text(encoding='utf-8'))

    per_seed = manifest.get('per_seed', [])
    for entry in per_seed:
        dev_path = Path(entry['dev_path'])
        test_path = Path(entry['test_path'])
        if dev_path.exists():
            entry.setdefault('dev', {})['examples'] = count_examples(dev_path)
        if test_path.exists():
            entry.setdefault('test', {})['examples'] = count_examples(test_path)

    manifest['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    split_manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def refresh_router_manifest(split_manifest: Dict[str, Any], out_manifest_path: Path) -> None:
    canonical_outputs = [
        ("artifacts/router_optionB/paper_table_test_full_per_seed.csv", "router_long_csv"),
        ("artifacts/router_optionB/paper_table_test_acc_tokens.csv", "router_compact_csv"),
        ("artifacts/paper/tables/router_table.tex", "paper_table_tex"),
        ("data/router_splits_seeds/manifest.json", "split_manifest"),
    ]

    canonical_entries = []
    for path_str, kind in canonical_outputs:
        p = Path(path_str)
        canonical_entries.append({
            'path': path_str,
            'kind': kind,
            'sha256': sha256_file(p) if p.exists() else None,
        })

    payload = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_schema_version': '1.2',
        'generator_script': 'scripts/refresh_canonical_provenance.py',
        'source_file_path': split_manifest.get('source_file_path', 'results/preds_student_full.jsonl'),
        'source_sha256': split_manifest.get('source_sha256'),
        'seeds': split_manifest.get('seeds', [0, 1, 2]),
        'split_counts': split_manifest.get('counts', {}),
        'per_seed_counts': [
            {
                'seed': e.get('seed'),
                'dev_rows': e.get('dev', {}).get('rows'),
                'dev_examples': e.get('dev', {}).get('examples'),
                'test_rows': e.get('test', {}).get('rows'),
                'test_examples': e.get('test', {}).get('examples'),
            }
            for e in split_manifest.get('per_seed', [])
        ],
        'metric_names': ['acc', 'mean_tokens', 'mean_steps'],
        'canonical_outputs': canonical_entries,
        'canonical_sources': {
            'gsm8k_router_source': 'results/preds_student_full.jsonl',
            'anytime_plot_sources': [
                'results_abl/preds_main_adapter.jsonl',
                'results_abl/preds_base.jsonl',
            ],
            'boolq_transfer_source': 'artifacts/router_optionB_boolq/paper_table_validation_full_per_seed.csv',
        },
        'reproducibility_scope': {
            'artifact_level_reproducible': [
                'router tables from canonical CSVs',
                'paper figures/tables from bundled artifacts',
            ],
            'raw_data_end_to_end': 'partial (depends on external models/APIs/LFS payloads)',
        },
    }

    out_manifest_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='Refresh canonical provenance manifests without regenerating heavy outputs.')
    ap.add_argument('--split-manifest', default='data/router_splits_seeds/manifest.json')
    ap.add_argument('--router-manifest', default='artifacts/router_optionB/gsm8k_router_manifest.json')
    args = ap.parse_args()

    split_path = Path(args.split_manifest)
    router_path = Path(args.router_manifest)

    split_manifest = refresh_split_manifest(split_path)
    refresh_router_manifest(split_manifest, router_path)


if __name__ == '__main__':
    main()
