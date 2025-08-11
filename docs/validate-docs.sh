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
    DEFAULT_SOURCE="source/**/*.rst"
elif [ -f "docs/.rstcheck.cfg" ]; then
    CONFIG_PATH="docs/.rstcheck.cfg"
    DEFAULT_SOURCE="docs/source/**/*.rst"
else
    echo "Error: Could not find .rstcheck.cfg"
    exit 1
fi

# If no args provided, check all source files
if [ $# -eq 0 ]; then
    # Use eval to properly expand the glob pattern
    eval "exec rstcheck --config \"$CONFIG_PATH\" $DEFAULT_SOURCE"
else
    # Pass through all arguments to rstcheck, adding our config
    exec rstcheck --config "$CONFIG_PATH" "$@"
fi