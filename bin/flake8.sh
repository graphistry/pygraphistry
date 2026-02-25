#!/bin/bash
# DEPRECATED: flake8 has been replaced by ruff (see issue #466).
# This wrapper delegates to ruff.sh for backwards compatibility.
echo "WARNING: flake8.sh is deprecated. Use ruff.sh instead." >&2
exec "$(dirname "$0")/ruff.sh" "$@"
