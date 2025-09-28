#!/usr/bin/env python3
"""Script to update policy tests for Accept/Deny only pattern."""

import os
import re

# Test files to update
test_files = [
    'graphistry/tests/test_policy_exceptions.py',
    'graphistry/tests/test_policy_hooks.py',
    'graphistry/tests/test_policy_validation.py',
    'graphistry/tests/test_policy_behavior_modification.py',
    'graphistry/tests/test_policy_integration.py',
    'graphistry/tests/test_policy_recursion.py',
    'graphistry/tests/test_policy_closure_state.py',
]

replacements = [
    # Remove PolicyModification import
    (r'from graphistry\.compute\.gfql\.policy import \(\s*PolicyContext,\s*PolicyException,\s*PolicyModification.*?\)',
     'from graphistry.compute.gfql.policy import (\n    PolicyContext,\n    PolicyException\n)'),

    # Remove validate_modification import
    (r',?\s*validate_modification', ''),

    # Fix function signatures
    (r'\) -> Optional\[PolicyModification\]:', ') -> None:'),

    # Remove Optional import if it's only for PolicyModification
    (r'^from typing import Optional$', ''),
    (r'^from typing import Optional, ', 'from typing import '),

    # Fix return statements that return modifications
    (r"return \{'engine': .*?\}", "raise PolicyException('preload', 'Engine override not allowed')"),
    (r"return \{'query': .*?\}", "raise PolicyException('preload', 'Query modification not allowed')"),
    (r"return \{'params': .*?\}", "raise PolicyException('call', 'Parameter modification not allowed')"),

    # Fix bare return None to pass or nothing
    (r'^\s+return None\s*$', ''),
]

def fix_file(filepath):
    """Fix a single test file."""
    print(f"Fixing {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Updated {filepath}")
    else:
        print(f"  No changes needed for {filepath}")

if __name__ == '__main__':
    for test_file in test_files:
        if os.path.exists(test_file):
            fix_file(test_file)
        else:
            print(f"Skipping {test_file} - not found")