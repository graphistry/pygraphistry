# Lint and Type Check Template for PyGraphistry

<!-- FILE TRACKING HEADER - FILL IN WHEN FILE CREATED -->
```
Run Name: lint_check_[YYYY_MM_DD_HHMMSS]
Main File: plans/lint_check_[YYYY_MM_DD_HHMMSS]/progress.md
Created: [YYYY-MM-DD HH:MM:SS]
Status: [IN_PROGRESS/COMPLETE/BLOCKED]
```

## Instructions for AI Assistant

This template guides systematic code quality checks using flake8 (linting) and mypy (type checking) for PyGraphistry.

### Quick Start - Docker Commands
```bash
# Run lint and typecheck only (recommended approach)
cd docker && WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh

# Skip lint check
cd docker && WITH_LINT=0 WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh

# Skip typecheck
cd docker && WITH_TYPECHECK=0 WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh
```

1. **Start without creating files** - Run Steps 1-2 first
2. **MANDATORY: FIX ALL FIXABLE ISSUES** - You MUST attempt to fix every error found
3. **Create plans file only if needed**:
   - When manual fixes are required
   - When Step 3 indicates iteration needed
   - When complex issues found
4. **Follow the iterative process** - Steps 1-3 repeat until clean
5. **Always complete with Step 4** - Print report and handle cleanup
6. **Ask about file deletion** in interactive mode (after showing report)

### File Structure
- **Main tracking file**: `plans/lint_check_[YYYY_MM_DD_HHMMSS]/progress.md`
  - This is the ONLY file that tracks overall progress across all iterations
  - Always update this file when iterating back with progress updates
  - Include the run name at the top of each update for easy tracking
- **Additional files allowed**: Task-specific scratchpads for complex issues
  - `issues_catalog.md` - Detailed issue breakdown if needed
  - `implementation_plan.md` - For complex refactoring plans

### ⚠️ CRITICAL: You MUST Fix Issues, Not Just Report Them ⚠️
**NEVER** skip to "BLOCKED" status without attempting fixes. The process is:
1. Find issues → 2. FIX THEM → 3. Verify → 4. Repeat until clean

**BLOCKED means**: "I tried to fix these but cannot due to [specific technical reason]"
**BLOCKED does NOT mean**: "I found many issues" or "Issues look complex"

### Iterative Process Overview
The lint/type check process is **iterative by design**:
- **Steps 1-2**: Identify issues AND FIX THEM IMMEDIATELY
- **Step 3**: Verify and decide to iterate or finish
- **Repeat**: Continue until clean (0 flake8, 0 mypy)
- **Step 4**: Generate final report and cleanup
- **Exit**: When clean OR blocked AFTER ATTEMPTING FIXES

**When returning with progress updates**:
- Always start with: "Continuing run: lint_check_[YYYY_MM_DD_HHMMSS]"
- Update the main progress.md file with current iteration status
- Include iteration number and timestamp for each update

### When to Create progress.md File
- **NO FILE NEEDED**: Clean in first pass (no issues found)
- **CREATE progress.md**: Any of these conditions:
  - Multiple iterations needed (Step 3 → Iterate)
  - Manual fixes required in Steps 1-2
  - Complex issues found requiring documentation
  - More than 5 total issues across all tools

**Remember**: progress.md is the main tracking file that persists across all iterations

## Execution Protocol

### Step 1: Check Lint Issues with flake8
**Started**: [YYYY-MM-DD HH:MM:SS]
**Command (containerized)**: `cd docker && WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh`
**Command (direct)**: `./bin/lint.sh` (requires local environment)
**Purpose**: Identify all linting issues
**Action**: Run and categorize issues by type

<!-- FILL IN: Lint check results -->
**Quick syntax check (Critical errors only)**:
```bash
# First run - critical errors only
./bin/lint.sh 2>&1 | grep -E "E9|F63|F7|F82" | head -20
```

**Full lint check results**:
- **Total issues found**: [count]
- **Critical errors** (E9, F63, F7, F82): [count]
- **Import issues** (F401): [count]
- **Other issues**: [count]

**Issues by Priority**:
- **P0 - Critical** (breaking functionality):
  - [ ] Issue: [file:line] - [error code] - [description]
  
- **P1 - High** (type safety violations, imports):
  - [ ] Issue: [file:line] - [error code] - [description]
  
- **P2 - Medium** (code style consistency):
  - [ ] Issue: [file:line] - [error code] - [description]
  
- **P3 - Low** (minor improvements):
  - [ ] Issue: [file:line] - [error code] - [description]
  
- **P4 - Nice to Have** (cosmetic, minimal impact, already suppressed):
  - [ ] Issue: [file:line] - [error code] - [description]
  
- **P5 - Unimportant** (won't fix, intentional, or too minor to matter):
  - [ ] Issue: [file:line] - [error code] - [description]

**Quick fixes** (< 5 min each): [count]
**Complex fixes** (require analysis): [count]
**Can ignore** (P4/P5): [count]

### Step 2: Type Check with MyPy
**Started**: [YYYY-MM-DD HH:MM:SS]
**Command (containerized)**: `cd docker && WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh`
**Command (direct)**: `./bin/typecheck.sh` (requires local environment)
**Purpose**: Verify type safety across the codebase
**Action**: Run and filter results

<!-- FILL IN: Type check results -->
**Type check results**:
- **Total errors**: [count]
- **Excluded files** (from mypy.ini): tests/, _version.py, graph_vector_pb2.py
- **Ignored imports**: [count from ignore_missing_imports]

**Issues by Priority**:
- **P0 - Critical** (breaking functionality):
  - [ ] Issue: [file:line] - [description]
  
- **P1 - High** (type safety violations):
  - [ ] Issue: [file:line] - [description]
  
- **P2 - Medium** (missing annotations):
  - [ ] Issue: [file:line] - [description]
  
- **P3 - Low** (optional improvements):
  - [ ] Issue: [file:line] - [description]
  
- **P4 - Nice to Have** (third-party stubs, test files, minimal benefit):
  - [ ] Issue: [file:line] - [description]
  
- **P5 - Unimportant** (dynamic typing required, legacy patterns, not worth fixing):
  - [ ] Issue: [file:line] - [description]

**Quick fixes** (obvious annotations): [count]
**Complex fixes** (require refactoring): [count]
**Can ignore** (P4/P5): [count]

**⚠️ MANDATORY FIXING PHASE ⚠️**
If ANY errors found above, you MUST:
1. Start fixing immediately (begin with quick fixes)
2. Use Edit/MultiEdit tools to fix the issues
3. Document each fix in the iteration tracking
4. Continue to Step 3 ONLY after attempting fixes

### Step 3: Repeat-or-Finish
**Started**: [YYYY-MM-DD HH:MM:SS]
**Purpose**: Verify all fixes and determine if another iteration is needed
**Action**: Re-run both tools to check for clean state

<!-- FILL IN: Verification results -->
**Verification Commands**:
```bash
# Containerized approach (recommended)
cd docker && WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh

# Or run directly if you have local environment
# ./bin/lint.sh
# ./bin/typecheck.sh
```

**Results**:
- **Flake8 issues remaining**: [count]
- **MyPy issues remaining**: [count]

**Decision**:
- [ ] ✅ **CLEAN** - 0 flake8 issues AND 0 mypy issues (excluding P4/P5) → Go to Step 4
- [ ] 🔄 **ITERATE** - P0-P3 issues remain → REPEAT Steps 1-3
- [ ] 🛑 **BLOCKED** - Cannot fix remaining P0-P3 issues → Document blockers → Go to Step 4

**Note**: P4/P5 issues don't require iteration - document them but consider the check "clean" if only these remain.

**⚠️ BLOCKED CRITERIA ⚠️**
You can ONLY choose BLOCKED if ALL of these are true:
1. You have attempted to fix at least 5-10 issues
2. You encountered a specific technical blocker (document it)
3. The blocker prevents further progress

Examples of valid blockers:
- "Circular import that breaks when fixed"
- "Third-party library missing type stubs configured in mypy.ini"
- "Type system limitation requiring major refactor"
- "Flake8 rule conflicts with project style guide"

NOT valid reasons for BLOCKED:
- "Too many errors" 
- "Errors look complex"
- "Would take too long"
- "Not sure how to fix"

### Step 4: Final Report and Cleanup
**Started**: [YYYY-MM-DD HH:MM:SS]
**Purpose**: Generate comprehensive report and handle file cleanup
**Action**: Print final summary and manage plans file

<!-- FILL IN: Final report -->
```
=== LINT & TYPE CHECK FINAL REPORT ===
Started: [YYYY-MM-DD HH:MM:SS]
Completed: [YYYY-MM-DD HH:MM:SS]
Total Iterations: [number]
Starting Issues: [count]
Issues Fixed: [count] ([percentage]%)
Time Elapsed: [duration]

Summary by Tool:
- Flake8 issues fixed: [count]
- MyPy issues fixed: [count]
- Remaining issues: [count]

Docker testing command:
cd docker && WITH_BUILD=0 ./test-cpu-local.sh

File Status:
- progress.md created: Yes/No
- Location: plans/lint_check_[date]/progress.md

Result: ✅ CLEAN / 🔧 IMPROVED / 🛑 BLOCKED
```

**Cleanup Decision**:
- [ ] **Interactive Mode** - Ask user: "Delete progress.md file? (y/n)"
- [ ] **Non-interactive Mode** - Keep file, note location in output
- [ ] **No File Created** - Nothing to clean up

## Issue Summary

<!-- FILL IN: After running all tools -->
**Iteration Tracking**:
- **Current Iteration**: [number]
- **Starting Issues**: [initial count]
- **Current Status**: [count remaining]

**Per-Iteration Progress**:
| Iteration | Started | Flake8 Fixed | Flake8 Remaining | MyPy Fixed | MyPy Remaining | P4/P5 | Status |
|-----------|---------|--------------|------------------|------------|----------------|-------|--------|
| 1         | [time]  | [count]      | [count]          | [count]    | [count]        | [count]| 🔄     |
| 2         | [time]  | [count]      | [count]          | [count]    | [count]        | [count]| ✅     |

**Total Issues Fixed**:
- Flake8 issues fixed: [count]
- MyPy errors fixed: [count]
- **P0-P3 remaining**: [count]
- **P4/P5 ignored**: [count]

## PyGraphistry-Specific Patterns

### Common Flake8 Fixes
```python
# E501: Line too long (max 127)
# Break long lines
result = very_long_function_call(
    parameter1, parameter2, 
    parameter3, parameter4
)

# F401: Unused import
# Remove the import or add to __all__

# E722: Bare except
# Replace: except:
# With: except Exception:

# W291/W293: Trailing whitespace
# Remove whitespace at end of lines
```

### Common MyPy Fixes
```python
# Missing return type
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    return df

# Optional types
from typing import Optional, Union
def get_value(key: str) -> Optional[str]:
    return None

# DataFrame types
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import cudf
    
def process(df: Union[pd.DataFrame, 'cudf.DataFrame']) -> Union[pd.DataFrame, 'cudf.DataFrame']:
    return df

# Functional style (PyGraphistry pattern)
# Good: Return new objects
df = df.assign(new_col=values)
# Bad: Mutate in place
df['new_col'] = values  # Avoid
```

### Configuration Reference

**Flake8 Ignored Rules (from bin/lint.sh)**:
- C901: Function complexity
- E121-E128: Indentation rules
- E201-E203: Whitespace around brackets
- E501: Line length (but still check at 127 chars)
- F401: Unused imports (but still fix obvious ones)
- W291, W293: Trailing whitespace

**MyPy Configuration (from mypy.ini)**:
- Python version: 3.8
- Excluded: tests/, _version.py, graph_vector_pb2.py
- Many imports ignored due to missing stubs

### P4/P5 Issue Examples (Following Product Management Priority Conventions)

**P4 - Nice to Have (low impact)**:
- Minor style inconsistencies that don't affect readability
- Flake8 rules explicitly ignored in bin/lint.sh (E121-E128, etc.)
- MyPy errors in excluded files (tests/, _version.py)
- Import errors for packages with `ignore_missing_imports = True`
- Optional type annotations that would add minimal safety benefit

**P5 - Unimportant (not worth fixing)**:
- Style violations in generated code (graph_vector_pb2.py)
- Complex type annotations that would hurt readability
- Dynamic attribute access required by the functional API
- Legacy patterns that work but don't follow modern conventions
- Third-party library limitations requiring # type: ignore
- Cosmetic issues with no functional impact

## Docker Commands Reference

### Prerequisites
The Docker test scripts require a pre-built test-cpu image. If not available:
```bash
cd docker
# Build the test image first (one-time setup)
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 \
  docker compose build \
  --build-arg PYTHON_VERSION=3.10 \
  --build-arg PIP_DEPS="-e .[test,build]" \
  test-cpu
```

### Running Lint/Type Checks (Containerized)
```bash
# RECOMMENDED: Run both lint and typecheck only
cd docker && WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh

# Full test suite including lint/typecheck/tests/build
cd docker && ./test-cpu-local.sh

# Fast iteration (skip build)
WITH_BUILD=0 ./test-cpu-local.sh

# Only run lint check
WITH_TYPECHECK=0 WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh

# Only run type check  
WITH_LINT=0 WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh

# Skip lint/typecheck for focused testing
WITH_LINT=0 WITH_TYPECHECK=0 WITH_BUILD=0 ./test-cpu-local.sh

# Specific test file
WITH_BUILD=0 ./test-cpu-local.sh graphistry/tests/test_compute.py
```

### Direct Script Execution (requires local environment)
If you have the dependencies installed locally:
```bash
# From project root
./bin/lint.sh
./bin/typecheck.sh

# For specific files
flake8 graphistry/embed_utils.py --max-line-length=127
mypy graphistry/embed_utils.py
```

### Understanding Output
- The docker scripts mount the code as read-only and run the bin scripts inside the container
- Lint errors show file:line:column followed by error code and description
- Type errors show file:line followed by error description
- Exit code 0 means all checks passed

## Decision Point

<!-- FILL IN: Based on issue count and complexity -->
**File Creation Decision**:
- [ ] **No File Mode** - Clean or simple fixes in first pass
- [ ] **Create File** - Multiple iterations or complex fixes needed

**Execution Mode**: 
- [ ] **Direct Fix** - Simple issues, fix immediately
- [ ] **Tracked Fix** - Create plans file to track iterations
- [ ] **Task Mode** - Complex issues need full implementation plan

## Task Mode Actions

<!-- Use this section if progress.md needed -->
### ⚠️ REMINDER: You MUST Actually Fix Issues Here ⚠️
This section is for DOCUMENTING fixes you've already made, not planning future fixes.
If you haven't fixed anything yet, GO BACK and start fixing!

### Task Created:
**Task Name**: `plans/lint_check_[YYYY_MM_DD_HHMMSS]/`

**Files Created**:
- [ ] `progress.md` - Main tracking file (ALWAYS created, tracks all iterations)
- [ ] `implementation_plan.md` - Systematic fix plan (optional, for complex issues)
- [ ] `issues_catalog.md` - Detailed issue list (optional, for many issues)

**Priority Order for Fixes**:
1. **P0**: Fix immediately - breaking functionality
2. **P1**: Fix next - type safety and imports
3. **P2**: Fix if time - style consistency
4. **P3**: Fix if easy - minor improvements
5. **P4**: Skip - nice to have but not important
6. **P5**: Skip - unimportant or not worth the effort

### Progress Update Format (for progress.md):
```
=== ITERATION [number] UPDATE ===
Run Name: lint_check_[YYYY_MM_DD_HHMMSS]
Started: [YYYY-MM-DD HH:MM:SS]

Fixes Applied:
1. File: [path]
   - Issue: [flake8/mypy code] - [description]
   - Fix: [what was changed]
   - Status: ✅ Fixed

2. File: [path]
   - Issue: [flake8/mypy code] - [description]
   - Fix: [what was changed]
   - Status: ❌ Failed - [reason]

Verification:
- Flake8: [count] remaining
- MyPy: [count] remaining
- Next: [ITERATE/COMPLETE/BLOCKED]
```

## Example Step 4 Output

```
=== LINT & TYPE CHECK FINAL REPORT ===
Started: 2024-01-15 14:30:00
Completed: 2024-01-15 14:35:15
Total Iterations: 2
Starting Issues: 23
Issues Fixed: 23 (100%)
Time Elapsed: 5 minutes 15 seconds

Summary by Tool:
- Flake8 issues fixed: 21
- MyPy issues fixed: 2
- Remaining issues: 0

Docker testing command:
cd docker && WITH_BUILD=0 ./test-cpu-local.sh

File Status:
- progress.md created: Yes
- Location: plans/lint_check_2024_01_15_143000/progress.md

Result: ✅ CLEAN

Would you like to delete the progress.md tracking file? (y/n): _
```

## ⚠️ FINAL REMINDER: The Goal is FIXING, Not Just Finding ⚠️

When you run this template:
1. **DO**: Find issues → Fix them → Verify → Repeat
2. **DON'T**: Find issues → Report them → Give up

Remember: Users want clean code, not detailed reports about dirty code!