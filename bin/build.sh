#!/bin/bash
set -ex

# Run from project root
# Non-zero exit code on fail

python -m build --version

OUTPUT_DIR=${OUTPUT_DIR:-dist/}

python -m build \
    --sdist \
    --wheel \
    --outdir "${OUTPUT_DIR}"
