#!/bin/bash
# Generate Comment Inventory for DECOMMENT Protocol
#
# Usage: ./ai/assets/generate_comment_inventory.sh [base_branch] [output_file]
#
# This script automates Phase 1 of the DECOMMENT protocol by extracting all
# comments added in a PR and formatting them with context for categorization.
#
# Example:
#   ./ai/assets/generate_comment_inventory.sh master plans/my-feature/comment_inventory.md

set -euo pipefail

# Parse arguments
BASE_BRANCH=${1:-master}
OUTPUT_FILE=${2:-}

# Detect current branch
CURRENT_BRANCH=$(git branch --show-current)

# Try to get PR number if available
PR_NUMBER=$(gh pr view --json number -q .number 2>/dev/null || echo "N/A")

# Count comments added in this PR
COMMENT_COUNT=$(git diff "$BASE_BRANCH...HEAD" | grep -c "^+.*#" || echo "0")

# Generate output
{
cat <<EOF
# Comment Inventory - PR #${PR_NUMBER}

Generated: $(date +%Y-%m-%d)
Base branch: $BASE_BRANCH
PR branch: $CURRENT_BRANCH
Total comments found: $COMMENT_COUNT

---

**Instructions**:
1. Review each comment below
2. Mark category as KEEP or REMOVE
3. Document reasoning for each decision
4. Update Summary Statistics at bottom
5. Proceed to Phase 3 (removal) for all REMOVE comments

---

EOF

# Extract comments with context using git diff
comment_num=0
current_file=""

git diff "$BASE_BRANCH...HEAD" --unified=10 | while IFS= read -r line; do
    # Track current file
    if [[ "$line" =~ ^\+\+\+\ b/(.*) ]]; then
        current_file="${BASH_REMATCH[1]}"
        continue
    fi

    # Detect added comment lines (Python style)
    if [[ "$line" =~ ^\+.*# ]]; then
        # Skip docstring lines
        if [[ "$line" =~ ^\+[[:space:]]*\"\"\" ]] || [[ "$line" =~ ^\+[[:space:]]*\'\'\' ]]; then
            continue
        fi

        comment_num=$((comment_num + 1))

        # Extract just the comment text
        comment_text=$(echo "$line" | sed 's/^+//')

        cat <<COMMENT_BLOCK

## $current_file

### Comment $comment_num
**Line:** [Approximate - check manually]
**Comment:** \`$comment_text\`
**Context:**
\`\`\`python
$comment_text
\`\`\`
**Category:** [TODO: KEEP or REMOVE]
**Reason:** [TODO: Explain why]

---

COMMENT_BLOCK
    fi
done

cat <<EOF

## Summary Statistics

**Total comments reviewed:** $COMMENT_COUNT
**KEEP:** [COUNT after review]
**REMOVE:** [COUNT after review]

**KEEP reasons:**
- Section markers in large functions: [COUNT]
- Non-obvious behavior/design decisions: [COUNT]
- Type ignore overrides (need explanation added): [COUNT]
- Backwards compatibility notes: [COUNT]
- GitHub issue references: [COUNT]
- TODOs: [COUNT]
- Other: [COUNT]

**REMOVE reasons:**
- Redundant with code: [COUNT]
- Redundant with variable/function names: [COUNT]
- Obvious from immediate context: [COUNT]
- Redundant with docstrings: [COUNT]
- Ephemeral dev notes: [COUNT]
- Other: [COUNT]

EOF
} > "${OUTPUT_FILE:-/dev/stdout}"

# If output to file, show confirmation message
if [ -n "$OUTPUT_FILE" ]; then
    echo "✅ Comment inventory generated: $OUTPUT_FILE" >&2
    echo "" >&2
    echo "Next steps:" >&2
    echo "  1. Review and categorize each comment (KEEP vs REMOVE)" >&2
    echo "  2. Update counts in Summary Statistics" >&2
    echo "  3. Proceed to Phase 3 (removal for all REMOVE comments)" >&2
    echo "" >&2
    echo "Found $COMMENT_COUNT comments to review" >&2
fi
