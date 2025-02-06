#!/bin/bash
set -ex

# Run from project root
# Non-zero exit code on fail

mypy --version

if [ -n "$PYTHON_VERSION" ]; then
  SHORT_VERSION=$(echo "$PYTHON_VERSION" | cut -d. -f1,2)
  mypy --python-version "$SHORT_VERSION" --config-file mypy.ini graphistry
else
  mypy --config-file mypy.ini graphistry
fi
