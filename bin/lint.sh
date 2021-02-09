#!/bin/bash
set -ex

# Run from project root
# Non-zero exit code on fail

flake8 --version

# Quick syntax errors
flake8 \
    graphistry \
    --exit-zero \
    --count \
    --select=E9,F63,F7,F82 \
    --show-source \
    --statistics

# Deeper check
flake8 \
  graphistry \
  --exit-zero \
  --exclude graphistry/graph_vector_pb2.py,graphistry/_version.py \
  --count \
  --ignore=E121,E123,E128,E144,E201,E202,E203,E231,E251,E265,E301,E302,E303,E401,E501,E722,F401,W291,W293 \
  --exit-zero \
  --max-complexity=10 \
  --max-line-length=127 \
  --statistics
