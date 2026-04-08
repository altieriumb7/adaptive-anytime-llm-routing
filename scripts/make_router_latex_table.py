#!/usr/bin/env python3
r"""Generate a LaTeX router results table from per-seed CSV.

Input:
  artifacts/router_optionB/paper_table_test_full_per_seed.csv

Output:
  artifacts/paper/tables/router_table.tex

This writes a complete table* (tabularx) ready to be included via \input{}.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def mean_std(xs: List[float]) -> Tuple[float, float]:
    """Return mean and sample std-dev (ddof=1) for cross-seed reporting."""
    if not xs:
        return float('nan'), float('nan')
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def fmt_pm(mu: float, sd: float, decimals: int) -> str:
    if math.isnan(mu) or math.isnan(sd):
        return '--'
    return f"{mu:.{decimals}f}$\\pm${sd:.{decimals}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_csv', required=True)
    ap.add_argument('--out_tex', required=True)
    ap.add_argument('--caption', default=None, help='If omitted, an automatic caption is generated from --split_label.')
    ap.add_argument('--label', default='tab:main_results_allbudgets')
    ap.add_argument('--split_label', default='test', help='Label used in the header row, e.g. test or validation.')
    ap.add_argument('--split_filter', default=None, help='CSV split value to keep. Defaults to --split_label.')
    ap.add_argument(
        '--legacy_split_aliases',
        default='',
        help='Comma-separated additional split values accepted for filtering (e.g. test for historical BoolQ validation CSVs).',
    )
    ap.add_argument('--oracle_everywhere', action='store_true', help='Repeat Oracle across all budget columns.')
    ap.add_argument('--oracle-single-reference', dest='oracle_single_reference', action='store_true', help='Render Oracle once as a single offline reference point and show -- for later budgets.')
    ap.add_argument('--no-oracle-single-reference', dest='oracle_single_reference', action='store_false', help='Disable single-reference oracle rendering.')
    ap.set_defaults(oracle_single_reference=None)
    args = ap.parse_args()

    split_filter = args.split_filter or args.split_label
    keep_splits = {split_filter}
    keep_splits.update({x.strip() for x in str(args.legacy_split_aliases).split(',') if x.strip()})

    rows: List[Dict[str, str]] = []
    seen_splits = set()
    with open(args.in_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            seen_splits.add(row.get('split'))
            if row.get('split') in keep_splits:
                rows.append(row)

    if not rows:
        raise SystemExit(
            f'No rows found for split filter {sorted(keep_splits)} in {args.in_csv}. '
            f'Available split values: {sorted(s for s in seen_splits if s is not None)}'
        )

    budgets = sorted({row['budget_tag'] for row in rows}, key=lambda x: int(x.lstrip('B')))
    policies = ['fixed', 'conf', 'random', 'stability', 'oracle']
    pretty = {
        'fixed': 'Fixed',
        'conf': 'Confidence',
        'random': 'Random (matched)',
        'stability': 'Stability (tuned)',
        'oracle': 'Oracle',
    }

    cell: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: {'acc': [], 'tok': [], 'steps': []})
    for row in rows:
        p = row.get('policy', '')
        b = row.get('budget_tag', '')
        if p not in policies or b not in budgets:
            continue
        key = (p, b)
        cell[key]['acc'].append(float(row['acc']))
        cell[key]['tok'].append(float(row['mean_tokens']))
        cell[key]['steps'].append(float(row['mean_steps']))

    lines: List[str] = []
    lines.append('\\begin{table*}[t]')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{3pt}')
    lines.append('\\renewcommand{\\arraystretch}{1.1}')
    lines.append('\\begin{tabularx}{\\textwidth}{l' + ('Y' * len(budgets)) + '}')
    lines.append('\\toprule')
    split_label = args.split_label
    oracle_single_reference = (not args.oracle_everywhere) if args.oracle_single_reference is None else args.oracle_single_reference
    lines.append(f'Policy ({split_label}, mean$\\pm$std over seeds, sample std) & ' + ' & '.join(budgets) + ' \\\\')
    lines.append('\\midrule')

    for pol in policies:
        parts = [pretty.get(pol, pol)]
        for b in budgets:
            # Oracle does not depend on budget; show once for readability unless requested.
            if oracle_single_reference and (pol == 'oracle') and (b != budgets[0]):
                parts.append('--')
                continue
            mu_a, sd_a = mean_std(cell[(pol, b)]['acc'])
            mu_t, sd_t = mean_std(cell[(pol, b)]['tok'])
            mu_s, sd_s = mean_std(cell[(pol, b)]['steps'])
            parts.append(f"{fmt_pm(mu_a, sd_a, 4)} ({fmt_pm(mu_t, sd_t, 0)}, {fmt_pm(mu_s, sd_s, 2)})")
        lines.append(' & '.join(parts) + ' \\\\')
        if pol == 'fixed':
            lines.append('\\addlinespace')

    lines.append('\\bottomrule')
    lines.append('\\end{tabularx}')
    caption = args.caption
    if caption is None:
        caption = (
            f'{split_label.capitalize()} performance across compute tiers '
            '(accuracy; mean tokens and mean steps in parentheses). '
            'Values are mean$\\pm$std over three split seeds (sample std).'
        )
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{args.label}}}")
    lines.append('\\end{table*}')

    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
