# Hoist Imports Protocol

**Purpose:** Move dynamic imports to top-level unless there's a documented reason not to.

## Quick Reference (60-second version)

1. **Identify**: Run `./ai/assets/find_dynamic_imports.sh master plans/[task]/dynamic_imports.md`
2. **Categorize**: Review each as HOIST (move to top) or KEEP (document reason)
3. **Hoist**: Move imports to top-level following PEP 8 section ordering
4. **Verify**: Run tests, confirm no circular imports or heavy load issues

See full protocol below for detailed criteria and import ordering rules.

---

## Overview

This protocol guides the systematic refactoring of dynamic imports (imports inside functions, methods, or conditionals) to top-level imports, improving code clarity, IDE support, and static analysis. Dynamic imports should only remain when there's a documented technical reason (circular dependencies, optional heavy dependencies, etc.).

## When to Use

- After completing a PR's implementation
- Before requesting code review
- When PR feedback mentions dynamic imports
- When adding new functionality that introduces imports
- As part of code cleanup before merge

## Prerequisites

1. **Plan File**: Ensure you have a current plan file (see `ai/prompts/PLAN.md`)
2. **Clean Git State**: All changes committed to feature branch
3. **PR Context**: Know which branch the PR will land into (usually `master` or `main`)

## Protocol Phases

### Phase 1: Identify All Dynamic Imports Added in PR

**Goal:** Create a comprehensive inventory of all dynamic imports added in this PR.

**Actions:**
1. Determine base branch (branch PR will merge into)
2. Generate diff of PR branch against base branch
3. Extract all added import statements inside functions/methods/conditionals:
   - File path
   - Line number
   - Import statement
   - Context (function/method name, why it's dynamic)
   - ±10 lines of surrounding code

**Commands:**
```bash
# Automated inventory generation (RECOMMENDED)
./ai/assets/find_dynamic_imports.sh master plans/[task]/dynamic_imports.md

# Manual alternatives (if automation script unavailable):
# Get base branch
BASE_BRANCH=$(gh pr view --json baseRefName -q .baseRefName)

# Find dynamic imports (imports not at module top-level)
# Look for added lines with 'import' that have indentation
git diff $BASE_BRANCH...HEAD --unified=10 | grep -E "^\+[[:space:]]+(import |from .* import )" -B 10 -A 2

# Alternative: Get full diff and search manually
git diff $BASE_BRANCH...HEAD > /tmp/pr_diff.txt
# Then search for indented import statements in the diff
```

**Output Format:**
Create a structured inventory:
```
File: graphistry/compute/chain_remote.py
Line: 174
Import: import json
Context: Inside try block in chain_remote_generic()
Reason: [Analyze why it's dynamic]
Code:
    try:
        import json  # <-- DYNAMIC IMPORT
        metadata_content = zip_ref.read('metadata.json')
        metadata = json.loads(metadata_content.decode('utf-8'))
```

**Save To:** `plans/[task]/dynamic_imports.md`

### Phase 2: Categorize Dynamic Imports

**Goal:** Classify each dynamic import as HOIST or KEEP based on technical criteria.

**KEEP Criteria (preserve dynamic import):**
- ✅ **Circular dependency**: Documented with comment `# Avoid circular import` or similar
- ✅ **Heavy optional dependency**: `cudf`, `cuml`, `cugraph`, `torch`, `dgl`, `tensorflow` - imports that load GBs of libraries
- ✅ **Conditional feature dependency**: Behind feature flags or optional extras
- ✅ **TYPE_CHECKING pattern**: Already using `if TYPE_CHECKING:` pattern for type hints
- ✅ **Lazy loading optimization**: Documented performance reason (e.g., CLI tools that need fast startup)
- ✅ **Platform-specific imports**: Behind `sys.platform` checks or similar

**HOIST Criteria (move to top-level):**
- ❌ **Stdlib imports**: `import json`, `import warnings`, `import uuid` - always hoist
- ❌ **Already loaded elsewhere**: If module is already imported at top in same file
- ❌ **No documented reason**: No comment explaining why it's dynamic
- ❌ **Common third-party**: `pandas`, `numpy`, `requests` - already in core deps
- ❌ **Internal project imports**: `from graphistry.Plottable import Plottable` unless circular
- ❌ **Inside try/except without reason**: Just error handling, not a real dependency issue

**Special Cases:**
- **`from graphistry.X import Y`**: Check if circular - if not, HOIST
- **`import pandas as pd`**: Common dependency, HOIST unless in optional feature
- **Multiple imports in same function**: If one should HOIST, likely all should
- **Imports in `__init__` methods**: Usually safe to HOIST unless late binding needed

**Output Format:**
Update `plans/[task]/dynamic_imports.md` with decisions:
```
File: graphistry/compute/chain_remote.py
Line: 174
Import: import json
Category: HOIST
Reason: Standard library, no circular dependency risk, no performance concern
Target Location: Top of file, stdlib section

---

File: graphistry/plugins/gpu_utils.py
Line: 42
Import: import cudf
Category: KEEP
Reason: Heavy optional dependency (RAPIDS), only needed when GPU available
Comment to add: # Lazy load cudf - heavy RAPIDS dependency only needed for GPU operations
```

### Phase 3: Hoist Imports to Top-Level

**Goal:** Systematically move all HOIST-categorized imports to module top-level following PEP 8 ordering.

**Import Ordering Rules (PEP 8 + Project Conventions):**

```python
# 1. Standard library imports (alphabetical)
import json
import sys
import warnings
from typing import Any, Dict, List

# 2. Third-party imports (alphabetical)
import numpy as np
import pandas as pd
import requests
from typing_extensions import Literal

# 3. Internal absolute imports from project root (alphabetical)
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject
from graphistry.io.metadata import deserialize_plottable_metadata

# 4. Internal relative imports (alphabetical)
from .client_session import ClientSession
from .utils import setup_logger
```

**Section Ordering Rules:**
1. **Standard library** - Python built-ins (json, sys, typing, etc.)
2. **Blank line**
3. **Third-party dependencies** - PyPI packages (pandas, numpy, requests, etc.)
4. **Blank line**
5. **Internal absolute imports** - `from graphistry.X import Y` (rooted at project)
6. **Blank line** (if relative imports present)
7. **Internal relative imports** - `from .X import Y` (relative to current module)

**Within each section**: Alphabetical by module name

**Actions:**
1. For each HOIST import in inventory:
   - Determine which section it belongs to
   - Find the correct alphabetical position in that section
   - **NINJA MODE**: Insert the import WITHOUT resorting existing imports
   - Remove the dynamic import from function/method body
   - Add comment to dynamic location if helpful: `# json imported at top`

2. Verify surrounding code still makes sense
3. Check if removing import reveals any issues (unused except for one function, etc.)

**Best Practices:**
- **Be a ninja**: Don't resort existing imports unless they're clearly wrong
- **Insert alphabetically**: Find correct spot, insert new line
- **Preserve existing style**: Match spacing, line breaks, `as` aliases
- **Don't fix unrelated imports**: Only touch the imports you're hoisting
- **Group related imports**: If hoisting multiple from same module, combine them
- **Use `# type: ignore` sparingly**: Only if unavoidable circular import for typing

**Example - Before:**
```python
# graphistry/compute/chain_remote.py
from typing import Any, Dict
import pandas as pd

from graphistry.Plottable import Plottable

def chain_remote_generic(self, ...):
    # ... 150 lines ...

    if 'metadata.json' in zip_ref.namelist():
        try:
            import json  # <-- DYNAMIC IMPORT
            import uuid  # <-- DYNAMIC IMPORT
            metadata = json.loads(metadata_content)
            viztoken = str(uuid.uuid4())
```

**Example - After (NINJA MODE):**
```python
# graphistry/compute/chain_remote.py
from typing import Any, Dict
import json  # <-- HOISTED (inserted alphabetically in stdlib section)
import pandas as pd
import uuid  # <-- HOISTED (inserted alphabetically in stdlib section)

from graphistry.Plottable import Plottable

def chain_remote_generic(self, ...):
    # ... 150 lines ...

    if 'metadata.json' in zip_ref.namelist():
        try:
            # json and uuid imported at top
            metadata = json.loads(metadata_content)
            viztoken = str(uuid.uuid4())
```

**Commands:**
```bash
# After hoisting imports in a file
git add [file]
python3.12 -m pytest [relevant_tests] -xvs  # Test for circular imports
bin/lint.sh  # Check import ordering
```

### Phase 4: Verify and Test

**Goal:** Ensure hoisted imports don't introduce circular dependencies or performance issues.

**Actions:**
1. Run tests for each file modified:
   ```bash
   # Test the specific module
   python3.12 -m pytest graphistry/tests/test_[module].py -xvs

   # Watch for ImportError or circular import errors
   ```

2. Check for circular import issues:
   - ✅ Tests pass without `ImportError`
   - ✅ No "circular import" errors in test output
   - ✅ Module can be imported standalone: `python -c "import graphistry.module"`

3. Verify no performance regression:
   - ✅ Import time acceptable (use `time python -c "import graphistry"` if concerned)
   - ✅ No unexpected heavy dependencies loaded

4. Check linting:
   ```bash
   bin/lint.sh
   # Should show no import ordering issues
   ```

5. Run full test suite:
   ```bash
   python3.12 -m pytest graphistry/tests/test_*.py -v
   ```

**If circular import detected:**
- Revert the specific import causing the issue
- Mark it as KEEP in inventory
- Add comment explaining circular dependency
- Consider if refactoring could break the cycle

**If heavy dependency issue:**
- Revert the import
- Mark as KEEP in inventory
- Add comment explaining lazy load reason

**Output:**
Update plan with Phase 4 results:
```
Phase 4: Verification Results
- Imports hoisted: 8
- Imports kept dynamic: 2 (cudf, circular dep in Plottable)
- Circular imports found: 1 (fixed by keeping dynamic)
- Tests: All passing ✅
- Linting: Clean ✅
- Import time: No regression ✅
```

## Plan Integration

### Plan Phase Template

Add this phase to your plan:

```markdown
### Phase N.A: Hoist Dynamic Imports
**Status:** 📝 TODO → 🔄 IN_PROGRESS → ✅ DONE
**Branch:** [your-branch]
**PR:** [#number]
**Started:** [date]
**Completed:** [date]
**Description:** Systematic hoisting of dynamic imports to module top-level

**Actions:**
1. ✅ Generate dynamic import inventory from PR diff
2. ✅ Categorize imports (HOIST vs KEEP)
3. ✅ Hoist imports to top-level following PEP 8 ordering
4. ✅ Verification pass (tests, circular import check)

**Results:**
- Total dynamic imports in PR: [N]
- Hoisted to top-level: [N]
- Kept dynamic (documented): [N]
- Reasons for keeping:
  - Circular dependencies: [N]
  - Heavy optional deps: [N]
  - TYPE_CHECKING: [N]
- Tests: All passing ✅
- Linting: Clean ✅

**Tool Calls:**
```bash
./ai/assets/find_dynamic_imports.sh master plans/[task]/dynamic_imports.md
# [edit commands]
git commit -m "refactor: Hoist dynamic imports to top-level"
git push
```
```

## Examples

### Example 1: Stdlib Import (HOIST)

**Before:**
```python
# graphistry/compute/chain_remote.py
from graphistry.Plottable import Plottable

def chain_remote_generic(self, ...):
    if 'metadata.json' in zip_ref.namelist():
        try:
            import json
            metadata = json.loads(metadata_content)
```

**After:**
```python
# graphistry/compute/chain_remote.py
import json

from graphistry.Plottable import Plottable

def chain_remote_generic(self, ...):
    if 'metadata.json' in zip_ref.namelist():
        try:
            metadata = json.loads(metadata_content)
```

**Reasoning:** Standard library, no circular dependency risk, no performance concern.

### Example 2: Heavy Dependency (KEEP)

**Before:**
```python
# graphistry/plugins/igraph.py
def from_igraph(self, ig, ...):
    import igraph
    # process igraph...
```

**After (Keep + Document):**
```python
# graphistry/plugins/igraph.py
def from_igraph(self, ig, ...):
    # Lazy load igraph - optional heavy dependency
    import igraph
    # process igraph...
```

**Reasoning:** Optional dependency, only needed when user calls this specific method.

### Example 3: Circular Import (KEEP)

**Before:**
```python
# graphistry/compute/chain.py
def to_plottable(self):
    from graphistry.Plottable import Plottable  # avoiding circular
    return Plottable()._chain(self)
```

**After (Keep as-is):**
```python
# graphistry/compute/chain.py
def to_plottable(self):
    # Avoid circular import: Plottable imports Chain
    from graphistry.Plottable import Plottable
    return Plottable()._chain(self)
```

**Reasoning:** Circular dependency, documented with comment.

### Example 4: TYPE_CHECKING Pattern (KEEP)

**Before:**
```python
# graphistry/io/metadata.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable

def serialize_plottable_metadata(g: 'Plottable') -> dict:
    ...
```

**After:** (Keep as-is - this is correct pattern)

**Reasoning:** TYPE_CHECKING pattern avoids circular imports for type hints only.

### Example 5: Multiple Imports in Same Function (HOIST ALL)

**Before:**
```python
# graphistry/compute/chain_remote.py
def generate_url(self, dataset_id):
    import uuid
    from graphistry.client_session import DatasetInfo

    info: DatasetInfo = {
        'name': dataset_id,
        'viztoken': str(uuid.uuid4())
    }
```

**After:**
```python
# graphistry/compute/chain_remote.py
import uuid

from graphistry.client_session import DatasetInfo

def generate_url(self, dataset_id):
    info: DatasetInfo = {
        'name': dataset_id,
        'viztoken': str(uuid.uuid4())
    }
```

**Reasoning:** Both imports can safely move to top - no circular deps, stdlib + internal.

## Success Criteria

**Phase 1 Complete:**
- ✅ Full inventory of dynamic imports created
- ✅ Each import has file, line, context, and reason

**Phase 2 Complete:**
- ✅ Every import categorized as HOIST or KEEP
- ✅ Category reasoning documented
- ✅ Target section identified for HOIST imports

**Phase 3 Complete:**
- ✅ All HOIST imports moved to top-level
- ✅ Imports inserted alphabetically in correct sections
- ✅ No existing imports resorted (ninja mode)
- ✅ Dynamic locations documented with comments if helpful

**Phase 4 Complete:**
- ✅ All tests passing (no circular imports)
- ✅ Linting clean (import order correct)
- ✅ No performance regression
- ✅ Plan updated with results

## Common Pitfalls

1. **Circular import not detected**: Moving an import creates circular dependency
   - *Fix:* Revert, mark as KEEP, add comment explaining circular dep

2. **Heavy dependency loaded**: Hoisting `cudf` causes slow imports for non-GPU users
   - *Fix:* Revert, mark as KEEP, add comment explaining lazy load reason

3. **Resorting all imports**: Changing unrelated import ordering
   - *Fix:* Be a ninja - only insert new imports, don't touch existing ones

4. **Breaking TYPE_CHECKING pattern**: Moving TYPE_CHECKING imports to top
   - *Fix:* Keep TYPE_CHECKING imports where they are - this is correct pattern

5. **Assuming all function imports are bad**: Some are legitimately dynamic
   - *Fix:* Apply KEEP criteria carefully, document reasons

## Integration with Existing Workflows

### With TDD Workflow
- Run HOISTIMPORTS after tests are passing (green phase)
- HOISTIMPORTS is a "refactor" step in red-green-refactor cycle

### With PR Review
- Run HOISTIMPORTS before requesting review
- Cleaner imports improve code readability
- Better IDE support for reviewers

### With Git Workflow
- Create dedicated HOISTIMPORTS commit
- Use conventional commit format: `refactor: Hoist dynamic imports to top-level`
- Keep HOISTIMPORTS commits separate from feature commits for easier review/revert

## Checklist

```markdown
- [ ] Plan file exists and is current
- [ ] Phase 1: Dynamic import inventory generated and saved
- [ ] Phase 2: All imports categorized with reasoning
- [ ] Phase 3: HOIST imports moved to top-level (ninja mode)
- [ ] Phase 4: Tests passing, no circular imports
- [ ] Plan updated with HOISTIMPORTS phase results
- [ ] Changes committed with clear message
- [ ] Final validation (tests + lint) passing
```

---

**Remember:** Dynamic imports make code harder to understand, break IDE support, and hide dependencies. Always hoist unless there's a documented technical reason not to. When in doubt, try hoisting - tests will tell you if there's a problem.
