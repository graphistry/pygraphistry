#!/bin/bash
# Validate RST documentation files
set -e

# Check rstcheck is installed
if ! command -v rstcheck &> /dev/null; then
    echo "Error: rstcheck not installed. Install with: pip install 'rstcheck[sphinx]'"
    exit 1
fi

# Validate RST files
if [ "$1" = "--changed" ]; then
    # Check only changed files
    git diff --name-only HEAD -- '*.rst' | xargs -r rstcheck --config docs/.rstcheck.cfg
else
    # Check all source files  
    rstcheck --config docs/.rstcheck.cfg docs/source/**/*.rst
fi