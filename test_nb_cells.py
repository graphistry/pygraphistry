#!/usr/bin/env python
"""Execute each cell from the temporal predicates notebook"""

import nbformat
import sys

# Read the notebook
nb_path = 'demos/gfql/temporal_predicates.ipynb'
with open(nb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Count cells
code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
print(f"Found {len(code_cells)} code cells in {nb_path}")

# Execute each cell
for i, cell in enumerate(code_cells):
    print(f"\nExecuting cell {i+1}/{len(code_cells)}...")
    try:
        exec(cell.source)
        print(f"✓ Cell {i+1} executed successfully")
    except Exception as e:
        print(f"✗ Cell {i+1} failed: {e}")
        print(f"Cell content:\n{cell.source[:200]}...")
        sys.exit(1)

print(f"\n✅ All {len(code_cells)} cells executed successfully!")