#!/bin/bash
# Validate RST documentation files
set -e

# Check rstcheck is installed
if ! command -v rstcheck &> /dev/null; then
    echo "Error: rstcheck not installed. Install with: pip install 'rstcheck[sphinx]'"
    exit 1
fi

# Determine config path based on current directory
if [ -f ".rstcheck.cfg" ]; then
    CONFIG_PATH=".rstcheck.cfg"
    SOURCE_PATH="source/**/*.rst"
elif [ -f "docs/.rstcheck.cfg" ]; then
    CONFIG_PATH="docs/.rstcheck.cfg"
    SOURCE_PATH="docs/source/**/*.rst"
else
    echo "Error: Could not find .rstcheck.cfg"
    exit 1
fi

# Validate RST files
if [ "$1" = "--changed" ]; then
    # Check only changed files
    git diff --name-only HEAD -- '*.rst' | xargs -r rstcheck --config "$CONFIG_PATH"
else
    # Check all source files  
    rstcheck --config "$CONFIG_PATH" $SOURCE_PATH
fi