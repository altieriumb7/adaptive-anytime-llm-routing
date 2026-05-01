# Paper build variants

This repository now supports two explicit LaTeX build variants:

- **Submission (canonical):** `main_distilling_revised_v0_10page.tex` → `build/paper_10page.pdf`
- **Long draft:** `main_distilling_revised_v0.tex` → `build/paper_long.pdf`

## Canonical submission command

```bash
bash run_paper.sh --variant 10page
```

Behavior:
1. Regenerates metadata-driven paper artifacts from `configs/paper.yaml`.
2. Rebuilds manuscript tables from canonical CSVs.
3. Validates LaTeX assets against `main_distilling_revised_v0_10page.tex`.
4. Compiles with `pdflatex`.
5. Writes `build/paper_10page.pdf`.
6. Checks PDF page count using `pypdf` and fails if pages > 10.

## Long draft command

```bash
bash run_paper.sh --variant long
```

This writes `build/paper_long.pdf` and does not apply the 10-page submission limit.

## Notebook workflow note

There is currently no checked-in notebook-based PDF builder in this repository snapshot.
Use the shell command above as the single canonical build path to avoid accidental long-draft compilation.
