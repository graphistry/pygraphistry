# Decomment Protocol

**Purpose:** Remove unnecessary comments from PRs while preserving valuable documentation.

## Overview

This protocol guides the systematic removal of redundant comments from code changes, keeping only those that provide non-obvious value. It follows a phased TDD-like approach: identify → categorize → remove → verify.

## When to Use

- After completing a PR's implementation
- Before requesting code review
- When PR feedback mentions excessive comments
- As part of code cleanup before merge

## Prerequisites

1. **Plan File**: Ensure you have a current plan file (see `ai/prompts/PLAN.md`)
   - If no plan exists for this work, create one following PLAN.md template
   - If plan exists but is stale, refresh it with current state
2. **Clean Git State**: All changes committed to feature branch
3. **PR Context**: Know which branch the PR will land into (usually `master` or `main`)

## Protocol Phases

### Phase 1: Identify All Comments Added in PR

**Goal:** Create a comprehensive inventory of all comments added in this PR.

**Actions:**
1. Determine base branch (branch PR will merge into)
2. Generate diff of PR branch against base branch
3. Extract all added comments (+) with context:
   - File path
   - Line number
   - Comment text
   - ±10 lines of surrounding code for context

**Commands:**
```bash
# Automated inventory generation (RECOMMENDED)
./ai/assets/generate_comment_inventory.sh master plans/[task]/comment_inventory.md

# Manual alternatives:
# Get base branch
BASE_BRANCH=$(gh pr view --json baseRefName -q .baseRefName)

# Generate diff showing only additions
git diff $BASE_BRANCH...HEAD

# Alternative: Get PR diff directly
gh pr diff

# Extract comments with context (example for Python)
git diff $BASE_BRANCH...HEAD | grep -E "^\+.*#" -B 10 -A 10
```

**Output Format:**
Create a structured inventory:
```
File: graphistry/io/metadata.py
Line: 42
Comment: "# Extract node bindings"
Context:
    def serialize_node_bindings(g: 'Plottable') -> Dict[str, str]:
        """Extract node bindings from Plottable.

        Maps internal Plottable attributes (_node, _point_color, etc.)
        to server format (node, node_color, etc.).
        """
        # Extract node bindings  <-- THIS COMMENT
        return serialize_bindings(g, [
            ['_node', 'node'],
```

**Save To:** `plans/[task]/comment_inventory.md`

### Phase 2: Categorize Comments

**Goal:** Classify each comment as KEEP or REMOVE based on value criteria.

**Keep Criteria (preserve these):**
- ✅ **Non-obvious behavior**: Explains surprising/subtle behavior not clear from code
- ✅ **GitHub issues**: References specific issues (`# See #123`, `# Fixes #456`)
- ✅ **TODOs**: Action items for future work (`# TODO: Add support for...`)
- ✅ **Bug workarounds**: Explains why code works around a bug (`# Workaround for pandas bug #789`)
- ✅ **Performance notes**: Explains optimization choices (`# O(n) instead of O(n²) because...`)
- ✅ **API constraints**: Documents external API limitations (`# API only supports 100 items max`)
- ✅ **Type ignore explanations**: Why type checker is overridden (`# type: ignore[...] - safe because...`)
- ✅ **Security notes**: Security-relevant documentation
- ✅ **Complex algorithms**: High-level algorithm explanation when not obvious

**Remove Criteria (redundant with code):**
- ❌ **Obvious from code**: Repeats what code clearly states
  - `# Set x to 5` before `x = 5`
  - `# Loop through items` before `for item in items:`
  - `# Return result` before `return result`
- ❌ **Redundant with names**: Comment just repeats variable/function name
  - `# Serialize node bindings` before `serialize_node_bindings()`
  - `# Node count` before `node_count = len(nodes)`
- ❌ **Ephemeral notes**: Temporary dev notes
  - `# Testing this`
  - `# Debug line`
  - `# WIP`
- ❌ **Redundant with docstrings**: Already documented in function docstring
- ❌ **Section markers for small blocks**: `# Node encodings` when only 3 lines follow
- ❌ **Commented-out code**: Unless there's a good reason (document why if keeping)

**Special Cases:**
- **Type hints**: `# type: ignore` without explanation → REMOVE (add explanation or fix type)
- **Section markers in large functions**: KEEP if function >50 lines and sections are distinct
- **Deprecation notices**: KEEP if documenting deprecated behavior

**Output Format:**
Update `plans/[task]/comment_inventory.md` with decisions:
```
File: graphistry/io/metadata.py
Line: 42
Comment: "# Extract node bindings"
Category: REMOVE
Reason: Redundant with function name serialize_node_bindings() directly below
---
File: graphistry/compute/chain_remote.py
Line: 156
Comment: "# Note: PyGraphistry #793 fixed in PR #798 - bindings now update correctly"
Category: KEEP
Reason: References specific issue and explains why workaround was removed
```

### Phase 3: Remove Redundant Comments

**Goal:** Systematically remove all REMOVE-categorized comments.

**Actions:**
1. Process each REMOVE comment in inventory
2. Use Edit tool to remove comment lines
3. Verify surrounding code still makes sense
4. Check if removing comment reveals need for better naming/structure

**Best Practices:**
- Remove entire comment line (including leading whitespace)
- If comment removal makes code unclear → improve code instead
  - Rename variables/functions
  - Extract method
  - Add type hints
- Batch similar removals (e.g., all "obvious from code" comments)
- Commit frequently: `git commit -m "chore: Remove redundant comments in [module]"`

**Commands:**
```bash
# After each batch of removals
git add [files]
git commit -m "chore: Remove redundant comments in [module]"

# Run tests after each commit
python3.12 -m pytest [relevant_tests] -v
bin/lint.sh
```

### Phase 4: Extra Pass - Verify Additions vs Removals

**Goal:** Double-check that removals were appropriate and no valuable comments were lost.

**Actions:**
1. Review the diff of changes made:
   ```bash
   git diff HEAD~[N]  # N = number of decomment commits
   ```

2. For each removed comment, verify:
   - ✅ Code is still clear without it
   - ✅ No loss of important context
   - ✅ Variable/function names are descriptive enough
   - ✅ Complex logic has sufficient explanation (docstring or inline comment)

3. Check for patterns:
   - Did we remove too many comments from one area? (might indicate unclear code)
   - Did we keep inconsistent comment styles? (standardize if so)
   - Are there new comments we should reconsider? (review KEEP decisions)

4. Run full validation:
   ```bash
   # All tests
   python3.12 -m pytest graphistry/tests/[your_module]/ -v

   # Linting
   bin/lint.sh

   # Type checking (if applicable)
   bin/mypy.sh
   ```

5. If any issues found:
   - Restore valuable comments that were mistakenly removed
   - Improve code clarity where comment removal exposed unclear code
   - Document findings in plan file

**Output:**
Update plan with Phase 4 results:
```
Phase 4: Verification Results
- Comments removed: 47
- Comments kept: 12
- Code improvements made: 3 (renamed variables for clarity)
- Tests: All passing ✅
- Linting: Clean ✅
- Final review: Ready for PR ✅
```

## Plan Integration

### Initial Plan Check

**Before starting decomment work:**
```bash
# Check if plan exists
ls plans/[current-task]/plan.md

# If no plan exists:
# 1. Create plan following ai/prompts/PLAN.md template
# 2. Document decomment work as Phase N.A: "Remove redundant comments"

# If plan exists but stale:
# 1. Read current plan
# 2. Add new phase for decomment work
# 3. Update "Completed Phase Summary" when done
```

### Plan Phase Template

Add this phase to your plan:

```markdown
### Phase N.A: Remove Redundant Comments
**Status:** 📝 TODO → 🔄 IN_PROGRESS → ✅ DONE
**Branch:** [your-branch]
**PR:** [#number]
**Started:** [date]
**Completed:** [date]
**Description:** Systematic removal of redundant comments from PR changes

**Actions:**
1. ✅ Generate comment inventory from PR diff
2. ✅ Categorize comments (KEEP vs REMOVE)
3. ✅ Remove redundant comments
4. ✅ Verification pass

**Results:**
- Total comments in PR: [N]
- Removed (redundant): [N]
- Kept (valuable): [N]
- Code improvements: [N]
- Tests: All passing ✅
- Linting: Clean ✅

**Tool Calls:**
```bash
git diff master...HEAD > plans/[task]/pr_diff.txt
# [other commands used]
git commit -m "chore: Remove redundant comments"
git push
```
```

## Examples

### Example 1: Obvious from Code (REMOVE)

**Before:**
```python
# Set the source column
g = g.bind(source='src')

# Set the destination column
g = g.bind(destination='dst')

# Return the result
return g
```

**After:**
```python
g = g.bind(source='src')
g = g.bind(destination='dst')
return g
```

**Reasoning:** Comments just repeat what code obviously does.

### Example 2: Non-obvious Behavior (KEEP)

**Before:**
```python
# Type ignore safe here: metadata dict guaranteed to have 'bindings' key
# by schema validation in deserialize_plottable_metadata
encodings[encoding_key] = node_bindings[server_key]  # type: ignore[literal-required]
```

**After:** (Keep as-is)

**Reasoning:** Explains why type checker is overridden, not obvious from context.

### Example 3: GitHub Issue Reference (KEEP)

**Before:**
```python
# Note: PyGraphistry #793 fixed in PR #798 - bindings now update correctly after UMAP
# The defensive validation workaround is no longer needed
if hasattr(g, '_source') and g._source:
    # ...
```

**After:** (Keep as-is)

**Reasoning:** Historical context linking to specific issue, valuable for future maintainers.

### Example 4: Redundant with Docstring (REMOVE)

**Before:**
```python
def serialize_node_bindings(g: 'Plottable') -> Dict[str, str]:
    """Extract node bindings from Plottable.

    Maps internal Plottable attributes (_node, _point_color, etc.)
    to server format (node, node_color, etc.).
    """
    # Extract node bindings
    return serialize_bindings(g, [
        ['_node', 'node'],
        # ...
    ])
```

**After:**
```python
def serialize_node_bindings(g: 'Plottable') -> Dict[str, str]:
    """Extract node bindings from Plottable.

    Maps internal Plottable attributes (_node, _point_color, etc.)
    to server format (node, node_color, etc.).
    """
    return serialize_bindings(g, [
        ['_node', 'node'],
        # ...
    ])
```

**Reasoning:** Docstring already explains what function does, inline comment is redundant.

### Example 5: Section Marker in Large Function (KEEP)

**Before:**
```python
def serialize_plottable_metadata(g: 'Plottable') -> PlottableMetadata:
    """Serialize complete Plottable metadata to JSON format."""

    # Collect all bindings (both node and edge)
    bindings: Dict[str, str] = {}
    bindings.update(serialize_node_bindings(g))
    bindings.update(serialize_edge_bindings(g))

    # Collect all simple encodings
    encodings: EncodingsDict = {}
    # ... 20 more lines ...

    # Build metadata
    metadata_obj: MetadataDict = {}
    # ... 10 more lines ...

    # Build style
    style: Dict[str, Any] = {}
    # ... 5 more lines ...

    return result
```

**After:** (Keep as-is)

**Reasoning:** Function is >50 lines with distinct sections, section markers improve readability.

## Success Criteria

**Phase 1 Complete:**
- ✅ Full inventory of added comments created
- ✅ Each comment has file, line, text, and context

**Phase 2 Complete:**
- ✅ Every comment categorized as KEEP or REMOVE
- ✅ Category reasoning documented
- ✅ Edge cases identified and decided

**Phase 3 Complete:**
- ✅ All REMOVE comments deleted from code
- ✅ Tests still passing
- ✅ Linting clean
- ✅ Changes committed

**Phase 4 Complete:**
- ✅ Verification pass done (reviewed all removals)
- ✅ No valuable comments lost
- ✅ Code clarity maintained or improved
- ✅ Final tests and lint passing
- ✅ Plan updated with results

## Common Pitfalls

1. **Over-removal**: Removing comments that explain non-obvious behavior
   - *Fix:* Be conservative, when in doubt KEEP

2. **Inconsistent standards**: Different removal criteria per file
   - *Fix:* Apply same KEEP/REMOVE criteria consistently

3. **Ignoring code quality**: Not improving code when comment removal exposes unclear code
   - *Fix:* Treat unclear code as an opportunity to improve naming/structure

4. **Skipping verification**: Not reviewing the full diff of removals
   - *Fix:* Always do Phase 4 verification pass

5. **Batch commits too large**: Hard to review what was removed
   - *Fix:* Commit per module or logical grouping

## Integration with Existing Workflows

### With TDD Workflow
- Run decomment after implementation complete and tests passing (green phase)
- Decomment is a "refactor" step in red-green-refactor cycle

### With PR Review
- Run decomment before requesting review
- Reduces reviewer cognitive load
- Cleaner diffs for reviewers to read

### With Git Workflow
- Create dedicated decomment commits
- Use conventional commit format: `chore: Remove redundant comments in [module]`
- Keep decomment commits separate from feature commits for easier review/revert

## Checklist

```markdown
- [ ] Plan file exists and is current (or created new one)
- [ ] Phase 1: Comment inventory generated and saved
- [ ] Phase 2: All comments categorized with reasoning
- [ ] Phase 3: REMOVE comments deleted, tests passing
- [ ] Phase 4: Verification pass complete
- [ ] Plan updated with decomment phase results
- [ ] Changes committed with clear messages
- [ ] Final validation (tests + lint) passing
```

---

**Remember:** Comments should explain *why*, not *what*. If a comment just repeats what the code obviously does, the code should be clear enough without it. If removing a comment makes code unclear, improve the code, don't keep the comment.
