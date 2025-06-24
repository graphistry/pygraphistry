#!/bin/bash
set -ex

# Minimal test to reproduce the ModuleNotFoundError
echo "=== Testing PyGraphistry pip install (minimal) ==="

PYTHON_VERSION=${PYTHON_VERSION:-3.11}

# Test with non-editable install to reproduce the issue
docker run --rm \
  -v "$(pwd)":/workspace \
  python:${PYTHON_VERSION}-slim \
  bash -c "
    cd /tmp && \
    pip install /workspace && \
    python -c 'from graphistry.models.gfql.types.temporal import DateTimeWire'
  "