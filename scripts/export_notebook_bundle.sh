#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK="sst_helper_notebook.ipynb"
OUT_DIR="deliverables"

mkdir -p "$OUT_DIR"

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "Notebook not found: $NOTEBOOK" >&2
  exit 1
fi

echo "Exporting HTML..."
python -m jupyter nbconvert --to html "$NOTEBOOK" --output-dir "$OUT_DIR"

echo "Exporting PDF..."
if python -m jupyter nbconvert --to pdf "$NOTEBOOK" --output-dir "$OUT_DIR"; then
  echo "PDF export completed."
else
  echo "PDF export failed (likely missing TeX/WebPDF dependencies)."
  echo "Tip: install TeX or use 'python -m jupyter nbconvert --to webpdf' with Playwright." >&2
fi

echo "Done. Files available in: $OUT_DIR"
