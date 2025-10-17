#!/bin/bash
# Generate Comment Inventory for DECOMMENT Protocol
#
# Usage: ./ai/bin/generate_comment_inventory.sh [base_branch] [output_file]
#
# This script automates Phase 1 of the DECOMMENT protocol by extracting all
# comments added in a PR and formatting them with context for categorization.
#
# Example:
#   ./ai/bin/generate_comment_inventory.sh master plans/my-feature/comment_inventory.md

set -euo pipefail

# Parse arguments
BASE_BRANCH=${1:-master}
OUTPUT_FILE=${2:-}

# Detect current branch
CURRENT_BRANCH=$(git branch --show-current)

# Try to get PR number if available
PR_NUMBER=$(gh pr view --json number -q .number 2>/dev/null || echo "N/A")

# Output to file or stdout
if [ -n "$OUTPUT_FILE" ]; then
    exec > "$OUTPUT_FILE"
fi

# Header
cat <<EOF
# Comment Inventory - PR #${PR_NUMBER}

Generated: $(date +%Y-%m-%d)
Base branch: $BASE_BRANCH
PR branch: $CURRENT_BRANCH
Total comments found: [PLACEHOLDER - count manually]

EOF

# Extract all added comment lines with file context
# This captures Python (#), JavaScript (//), and other comment styles
git diff "$BASE_BRANCH...HEAD" --unified=10 | \
awk '
BEGIN {
    file = ""
    in_hunk = 0
    comment_num = 0
}

# Track current file
/^\+\+\+ b\// {
    file = substr($0, 7)  # Remove "+++ b/" prefix
    next
}

# Track hunk headers to get line numbers
/^@@ / {
    # Extract starting line number from hunk header
    # Format: @@ -old_start,old_count +new_start,new_count @@
    match($0, /\+([0-9]+)/, arr)
    line_num = arr[1]
    in_hunk = 1

    # Store context lines for this hunk
    context_before = ""
    context_after = ""
    context_count = 0
    next
}

# Track line numbers in hunks
in_hunk && /^\+/ && !/^\+\+\+/ {
    line_num++
}
in_hunk && /^[^+\-]/ {
    line_num++
}

# Detect added comment lines (various styles)
/^\+.*#/ || /^\+.*\/\// || /^\+.*\/\*/ || /^\+.*\*\// {
    # Skip lines that are just adding to existing comments
    line = substr($0, 2)  # Remove leading "+"

    # Extract the comment text
    comment = line
    gsub(/^[[:space:]]+/, "", comment)  # Trim leading whitespace

    # Skip if this looks like a docstring marker or code with # in string
    if (match(comment, /^"""/)) next

    comment_num++

    print "## " file
    print ""
    print "### Comment " comment_num
    print "**Line:** ~" line_num
    print "**Comment:** `" comment "`"
    print "**Context:**"
    print "```python"

    # Get surrounding context (would need to re-parse diff for full context)
    # For now, just show the line itself as minimal context
    print line

    print "```"
    print "**Category:** [TODO]"
    print "**Reason:** [TODO]"
    print ""
    print "---"
    print ""
}
'

# Check if awk command succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to parse git diff. Make sure you're in a git repository." >&2
    echo "Usage: $0 [base_branch] [output_file]" >&2
    exit 1
fi

# Footer
cat <<EOF

## Summary Statistics

**Total comments reviewed:** [COUNT]
**KEEP:** [COUNT]
**REMOVE:** [COUNT]

**KEEP reasons:**
- Section markers in large functions: [COUNT]
- Non-obvious behavior/design decisions: [COUNT]
- Type ignore overrides (need explanation added): [COUNT]
- Backwards compatibility notes: [COUNT]
- Other: [COUNT]

**REMOVE reasons:**
- Redundant with code: [COUNT]
- Redundant with variable/function names: [COUNT]
- Obvious from immediate context: [COUNT]
- Other: [COUNT]

EOF

# If output to file, confirm
if [ -n "$OUTPUT_FILE" ]; then
    echo "Comment inventory generated: $OUTPUT_FILE" >&2
    echo "Next steps:" >&2
    echo "  1. Review and categorize each comment (KEEP vs REMOVE)" >&2
    echo "  2. Update counts in Summary Statistics" >&2
    echo "  3. Proceed to Phase 3 (removal)" >&2
fi
