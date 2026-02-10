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
    ap.add_argument('--caption', default='Test performance across compute tiers (accuracy; mean tokens and mean steps in parentheses). Values are mean$\\pm$std over three split seeds.')
    ap.add_argument('--label', default='tab:main_results_allbudgets')
    ap.add_argument('--oracle_everywhere', action='store_true', help='Repeat Oracle across all budget columns (default: show only in the first budget column).')
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    with open(args.in_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get('split') == 'test':
                rows.append(row)

    if not rows:
        raise SystemExit('No test rows found. Run scripts/run_router_optionB_repro.py first.')

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
    lines.append('Policy (test, mean$\\pm$std over seeds) & ' + ' & '.join(budgets) + ' \\\\')
    lines.append('\\midrule')

    for pol in policies:
        parts = [pretty.get(pol, pol)]
        for b in budgets:
            # Oracle does not depend on budget; show once for readability unless requested.
            if (not args.oracle_everywhere) and (pol == 'oracle') and (b != budgets[0]):
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
    lines.append(f"\\caption{{{args.caption}}}")
    lines.append(f"\\label{{{args.label}}}")
    lines.append('\\end{table*}')

    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
