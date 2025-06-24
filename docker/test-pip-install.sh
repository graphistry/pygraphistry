#!/bin/bash
set -ex

# Test PyGraphistry pip install in a clean environment
# This verifies that all imports work correctly after pip install

echo "=== Testing PyGraphistry pip install ==="

PYTHON_VERSION=${PYTHON_VERSION:-3.11}

# Run in Docker container with inline Python test
docker run --rm \
  -v "$(pwd)":/workspace \
  python:${PYTHON_VERSION}-slim \
  bash -c "cd /workspace && pip install . && python -c '
import sys
import traceback

try:
    print(\"Testing: import graphistry\")
    import graphistry
    print(\"✓ Success: import graphistry\")
    
    print(\"\\nTesting: from graphistry.compute.predicates.types import NormalizedIsInElement\")
    from graphistry.compute.predicates.types import NormalizedIsInElement
    print(\"✓ Success: predicates.types import\")
    
    print(\"\\nTesting: from graphistry.models.gfql.types.temporal import DateTimeWire\")
    from graphistry.models.gfql.types.temporal import DateTimeWire
    print(\"✓ Success: gfql.types.temporal import\")
    
    print(\"\\nTesting: from graphistry.compute.ast_temporal import TemporalValue\")
    from graphistry.compute.ast_temporal import TemporalValue
    print(\"✓ Success: ast_temporal import\")
    
    print(\"\\nAll imports successful!\")
    sys.exit(0)
except Exception as e:
    print(f\"\\n✗ Error: {type(e).__name__}: {e}\")
    traceback.print_exc()
    sys.exit(1)
'"