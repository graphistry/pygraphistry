#!/bin/bash
set -ex

# Run from project root
# Non-zero exit code on fail

mypy --version

if [ -n "$PYTHON_VERSION" ]; then
  mypy --python-version "$PYTHON_VERSION" --config-file mypy.ini graphistry
else
  mypy --config-file mypy.ini graphistry
fi
