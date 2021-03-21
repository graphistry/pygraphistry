#!/bin/bash
set -ex

# Run from project root
# Non-zero exit code on fail

mypy --version

# Check core
mypy --config-file mypy.ini graphistry
