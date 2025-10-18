#!/bin/bash
set -ex

# Run from project root
# Non-zero exit code on fail

flake8 --version

# Quick syntax errors
flake8 \
    graphistry \
    --count \
    --select=E9,F63,F7,F82 \
    --show-source \
    --statistics

# Deeper check
flake8 \
  graphistry \
  --exclude graphistry/graph_vector_pb2.py,graphistry/_version.py \
  --count \
  --ignore=C901,E121,E122,E123,E124,E125,E128,E131,E144,E201,E202,E203,E231,E251,E265,E301,E302,E303,E401,E501,E722,F401,W291,W293,W503 \
  --max-complexity=10 \
  --max-line-length=127 \
  --statistics

# Check for relative imports with '..' using flake8-quotes or custom regex
# This will fail if any relative imports with .. are found
echo "Checking for relative imports with '..' ..."
if grep -r "from \.\." graphistry --include="*.py" --exclude-dir="__pycache__"; then
    echo "ERROR: Found relative imports with '..'. Use absolute imports instead."
    exit 1
fi

# Note: DataFrame.style property access requires optional Jinja2 dependency
# Documented in CONTRIBUTING.md - avoid `df.style` (property), use `g.style()` (method) instead
# Automated check disabled due to false positives (cfg.style, Plottable.style references)
# Manual review: Check isinstance() before accessing result attributes that might be DataFrames
