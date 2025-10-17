#!/bin/bash
# Find Dynamic Imports for HOISTIMPORTS Protocol
#
# Usage: ./ai/assets/find_dynamic_imports.sh [base_branch] [output_file]
#
# This script automates Phase 1 of the HOISTIMPORTS protocol by extracting all
# dynamic imports (imports inside functions/methods/conditionals) added in a PR
# and formatting them with context for categorization.
#
# Example:
#   ./ai/assets/find_dynamic_imports.sh master plans/my-feature/dynamic_imports.md

set -euo pipefail

# Parse arguments
BASE_BRANCH=${1:-master}
OUTPUT_FILE=${2:-}

# Detect current branch
CURRENT_BRANCH=$(git branch --show-current)

# Try to get PR number if available
PR_NUMBER=$(gh pr view --json number -q .number 2>/dev/null || echo "N/A")

# Generate output
{
cat <<EOF
# Dynamic Import Inventory - PR #${PR_NUMBER}

Generated: $(date +%Y-%m-%d)
Base branch: $BASE_BRANCH
PR branch: $CURRENT_BRANCH

---

**Instructions**:
1. Review each dynamic import below
2. Mark category as HOIST or KEEP
3. Document reasoning for each decision
4. For HOIST imports, identify target section (stdlib/third-party/internal-absolute/internal-relative)
5. Proceed to Phase 3 (hoisting) for all HOIST imports

**HOIST Criteria:**
- ❌ Stdlib imports (json, warnings, uuid, etc.)
- ❌ Already loaded elsewhere in same file
- ❌ No documented reason for being dynamic
- ❌ Common third-party (pandas, numpy, requests)
- ❌ Internal project imports (unless circular)
- ❌ Inside try/except without reason

**KEEP Criteria:**
- ✅ Circular dependency (documented with comment)
- ✅ Heavy optional dependency (cudf, cuml, cugraph, torch, dgl, tensorflow)
- ✅ Conditional feature dependency (behind feature flags)
- ✅ TYPE_CHECKING pattern (already correct)
- ✅ Lazy loading optimization (documented performance reason)
- ✅ Platform-specific imports (behind sys.platform checks)

---

EOF

# Extract dynamic imports with context using git diff
# Strategy: Find added lines with 'import' that have indentation (not top-level)
import_num=0
current_file=""
context_lines=()
in_hunk=false

git diff "$BASE_BRANCH...HEAD" --unified=10 | while IFS= read -r line; do
    # Track current file
    if [[ "$line" =~ ^\+\+\+\ b/(.*) ]]; then
        current_file="${BASH_REMATCH[1]}"
        continue
    fi

    # Track hunk headers to get line numbers
    if [[ "$line" =~ ^@@\ -[0-9]+,[0-9]+\ \+([0-9]+),[0-9]+\ @@ ]]; then
        current_line="${BASH_REMATCH[1]}"
        in_hunk=true
        continue
    fi

    # Skip non-Python files
    if [[ ! "$current_file" =~ \.py$ ]]; then
        continue
    fi

    # Detect added import lines that are indented (dynamic)
    if [[ "$line" =~ ^\+[[:space:]]+(import\ |from\ .*\ import\ ) ]]; then
        # Skip docstring lines
        if [[ "$line" =~ ^\+[[:space:]]*\"\"\" ]] || [[ "$line" =~ ^\+[[:space:]]*\'\'\' ]]; then
            continue
        fi

        # Skip TYPE_CHECKING blocks (these are intentionally dynamic)
        if [[ "$line" =~ TYPE_CHECKING ]]; then
            continue
        fi

        import_num=$((import_num + 1))

        # Extract the import statement (remove the + prefix)
        import_line=$(echo "$line" | sed 's/^+//')

        # Try to determine context (this is approximate - actual context would require parsing)
        # For now, just flag it as "Inside indented block"

        cat <<IMPORT_BLOCK

## Import $import_num

**File:** \`$current_file\`
**Line:** ~$current_line (approximate from diff)
**Import:** \`\`\`python
$import_line
\`\`\`

**Context:** Inside indented block (function/method/conditional)

**Surrounding Code:**
\`\`\`python
[View in diff for ±10 lines of context]
$import_line
\`\`\`

**Category:** [TODO: HOIST or KEEP]
**Reason:** [TODO: Explain categorization]
**Target Section (if HOIST):** [TODO: stdlib / third-party / internal-absolute / internal-relative]

---

IMPORT_BLOCK
    fi
done

cat <<EOF

## Summary Statistics

**Total dynamic imports found:** $import_num
**HOIST:** [COUNT after review]
**KEEP:** [COUNT after review]

**HOIST target sections:**
- Standard library: [COUNT]
- Third-party: [COUNT]
- Internal absolute: [COUNT]
- Internal relative: [COUNT]

**KEEP reasons:**
- Circular dependencies: [COUNT]
- Heavy optional dependencies: [COUNT]
- TYPE_CHECKING pattern: [COUNT]
- Lazy loading optimization: [COUNT]
- Platform-specific: [COUNT]
- Other: [COUNT]

**Next Steps:**
1. Review each import above and mark as HOIST or KEEP
2. For HOIST imports, determine target section
3. Proceed to Phase 3 (hoisting) following import ordering rules
4. Run tests after hoisting to catch circular imports

EOF
} > "${OUTPUT_FILE:-/dev/stdout}"

# If output to file, show confirmation message
if [ -n "$OUTPUT_FILE" ]; then
    # Count imports by reading what we just generated
    IMPORT_COUNT=$(grep -c "^## Import [0-9]" "$OUTPUT_FILE" || echo "0")

    echo "✅ Dynamic import inventory generated: $OUTPUT_FILE" >&2
    echo "" >&2
    echo "Next steps:" >&2
    echo "  1. Review and categorize each import (HOIST vs KEEP)" >&2
    echo "  2. For HOIST imports, identify target section" >&2
    echo "  3. Update counts in Summary Statistics" >&2
    echo "  4. Proceed to Phase 3 (hoisting for all HOIST imports)" >&2
    echo "" >&2
    echo "Found $IMPORT_COUNT dynamic imports to review" >&2
fi
