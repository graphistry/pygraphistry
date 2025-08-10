# Conventional Commits Template for PyGraphistry

<!-- FILE TRACKING HEADER - FILL IN WHEN FILE CREATED -->
```
File: plans/commits_[YYYY_MM_DD_HHMMSS]/progress.md
Run Name: commits_[YYYY_MM_DD_HHMMSS]
Created: [YYYY-MM-DD HH:MM:SS]
Status: [IN_PROGRESS/COMPLETE/BLOCKED]
Total Commits: [number]
```

## Instructions for AI Assistant

This template provides a systematic approach for creating well-organized conventional commits in PyGraphistry. When asked to create commits:

1. **Analyze changes first** - Run Steps 1-2 to understand the full scope
2. **Get user feedback** - Present analysis and proposed batching strategy
3. **Create topical commits** - Group related changes by feature/scope
4. **Use line-level precision** - Don't commit entire files if only parts changed
5. **Follow conventional format** - type(scope): clear, concise subject
6. **Update CHANGELOG after each commit** - Add entry once hash is available

### File Structure
- **Main tracking file**: `progress.md` - persists across all commit operations
- **Optional scratchpads**: Create additional files only for complex analysis
- **CHANGELOG section**: At end of progress.md, updated after each commit

### Progress Update Format
When returning with updates, ALWAYS start with:
```
Continuing run: commits_[YYYY_MM_DD_HHMMSS]
Iteration: [number]
Status: Creating commit [X] of [Y]
```

### Conventional Commit Format
```
type(scope): subject

[optional body]

[optional footer(s)]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc
- `refactor`: Code restructuring without feature/fix
- `perf`: Performance improvements
- `test`/`tests`: Adding/updating tests
- `chore`: Build process, auxiliary tools
- `ci`: CI configuration files and scripts
- `build`/`infra`: Build system or dependencies

**PyGraphistry-Specific Scopes**:
- `plottable`: Plotting functionality
- `umap`: UMAP embeddings
- `gfql`: Graph Frame Query Language
- `gpu`/`gpu ci`: GPU-related functionality
- `cugraph`/`cudf`: RAPIDS integration
- `dbscan`: Clustering functionality
- `featurize`: Feature engineering
- `transform`: Data transformation
- `dgl`: Deep Graph Library integration
- `layout`: Layout algorithms (ring, tree, fa2, gib)
- `compute`: Compute chain/hop operations
- `api`: API changes
- `hyper`: Hypergraph functionality

## Execution Protocol

### Step 1: Initial Status Analysis
**Started**: [YYYY-MM-DD HH:MM:SS]
**Commands**: 
- `git status`
- `git diff --stat`
**Purpose**: Get overview of all changes
**Action**: Identify modified, new, and deleted files

<!-- FILL IN: Status analysis -->
**Repository State**:
- **Current branch**: [branch name]
- **Files modified**: [count]
- **Files added**: [count]
- **Files deleted**: [count]
- **Total changes**: [lines added/deleted]

**Change Categories**:
- [ ] Infrastructure/config files
- [ ] Documentation updates
- [ ] Source code changes
- [ ] Test modifications
- [ ] Dependencies/build files

### Step 2: Detailed Change Analysis
**Started**: [YYYY-MM-DD HH:MM:SS]
**Purpose**: Analyze changes by logical grouping
**Action**: Review diffs to understand change context

<!-- FILL IN: For each significant file/area -->
**Change Analysis**:

#### Group 1: [Descriptive Name]
**Files**: 
- `path/to/file1.py` - [brief description of changes]
- `path/to/file2.py` - [brief description of changes]

**Change Summary**: [What feature/fix/improvement this represents]
**Diff Preview**:
```bash
git diff --cached path/to/file1.py | head -20
```
**Key Changes**:
- [ ] Added functionality: [describe]
- [ ] Modified behavior: [describe]
- [ ] Removed code: [describe]

#### Group 2: [Descriptive Name]
[Repeat structure for each logical group]

### Step 3: Commit Strategy Proposal
**Started**: [YYYY-MM-DD HH:MM:SS]
**Purpose**: Present batching strategy for user approval
**Action**: Propose logical commit groupings

<!-- FILL IN: Proposed commit strategy -->
**Proposed Commit Order**:

1. **Infrastructure/Supporting Changes** (if any)
   - Commit 1: `chore(build): [description]`
     - Files: [list files]
   - Commit 2: `chore(config): [description]`
     - Files: [list files]

2. **Core Feature Changes**
   - Commit 3: `feat(scope): [description]`
     - Files: [list files]
   - Commit 4: `fix(scope): [description]`
     - Files: [list files]

3. **Documentation/Tests**
   - Commit 5: `docs(scope): [description]`
     - Files: [list files]
   - Commit 6: `test(scope): [description]`
     - Files: [list files]

**User Decision Point**:
```
Please review the proposed commit strategy:
- Approve as-is? (y)
- Modify grouping? (m)
- Different order? (o)
- Skip some commits? (s)
```

### Step 4: Pre-commit Preparation
**Started**: [YYYY-MM-DD HH:MM:SS]
**Purpose**: Ensure clean working state
**Action**: Verify no unintended changes

<!-- FILL IN: Pre-commit checks -->
**Pre-commit Checklist**:
- [ ] Run linting: `./bin/lint.sh`
- [ ] Run type checking: `./bin/typecheck.sh`
- [ ] Verify tests pass: `python -m pytest -xvs` (if test changes)
- [ ] Check for secrets: Review diffs for sensitive data
- [ ] Unstage unrelated changes: `git reset HEAD <file>`
- [ ] Remove Claude comments: No explanatory comments in code
- [ ] Verify functional style: No `df[col] = val`, use `df.assign()`
- [ ] Check for unnecessary `copy()`: DataFrames already return new objects

### Step 5: Execute Commits
**Started**: [YYYY-MM-DD HH:MM:SS]
**Purpose**: Create commits according to approved strategy
**Action**: Stage and commit changes in batches

<!-- FILL IN: For each commit -->
#### Commit [number]: [type(scope): subject]
**Started**: [YYYY-MM-DD HH:MM:SS]
**CHANGELOG Entry** (to add after commit):
```
### [Section based on type]
- scope: concise description (#hash_here)
```

**Staging Commands**:
```bash
# Check current staging status
git status

# Stage specific changes (from unstaged state)
git add -p path/to/file1.py  # Interactive staging for partial changes
git add path/to/file2.py      # Full file staging

# Or if files are already staged and you need to unstage specific ones:
git reset HEAD path/to/unwanted.py  # Unstage specific file
```

**Staged Files**:
- [ ] `path/to/file1.py` - [portions staged]
- [ ] `path/to/file2.py` - [full file]

**Commit Message**:
```
[type]([scope]): [subject]

[Detailed explanation of what changed and why]
[Impact of the changes]
[Any breaking changes or migration notes]

[Footer if needed - e.g., Fixes #123]
```

**Execution**:
```bash
git commit -m "$(cat <<'EOF'
[Insert actual commit message here]
EOF
)"
```

**Verification**:
- [ ] Commit created successfully
- [ ] `git show HEAD` displays expected changes
- [ ] No unintended files included

**Post-commit Actions**:
1. Get commit hash: `git rev-parse --short HEAD`
2. Add entry to appropriate CHANGELOG section (Features/Fixes/etc)
3. Format: `- scope: description (hash)`

### Step 6: Post-commit Review
**Started**: [YYYY-MM-DD HH:MM:SS]
**Purpose**: Verify all commits and final state
**Action**: Review commit history and remaining changes

<!-- FILL IN: Post-commit verification -->
**Commit Summary**:
```bash
git log --oneline -[number of commits]
```

**Created Commits**:
1. [hash] [type(scope): subject]
2. [hash] [type(scope): subject]
3. [hash] [type(scope): subject]

**CHANGELOG Status**:
- [ ] All commits added to CHANGELOG section
- [ ] Commit hashes are short format (7 chars)
- [ ] Links will work on GitHub after push

**Remaining Changes**:
- [ ] All planned changes committed
- [ ] Unstaged changes: [list if any]
- [ ] Untracked files: [list if any]

### Step 7: Final Report
**Started**: [YYYY-MM-DD HH:MM:SS]
**Purpose**: Generate comprehensive summary
**Action**: Document what was accomplished

<!-- FILL IN: Final report -->
```
=== CONVENTIONAL COMMITS FINAL REPORT ===
Started: [YYYY-MM-DD HH:MM:SS]
Completed: [YYYY-MM-DD HH:MM:SS]
Run Name: commits_[YYYY_MM_DD_HHMMSS]
Total Commits Created: [number]
Files Modified: [count]
Lines Added: [count]
Lines Deleted: [count]

Commits by Type:
- feat: [count]
- fix: [count]
- docs: [count]
- chore: [count]
- test: [count]
- other: [count]

Key Accomplishments:
1. [Major feature/fix implemented]
2. [Infrastructure improvement]
3. [Documentation updated]

Next Steps:
- [ ] Push to remote: git push origin [branch]
- [ ] Create PR if needed
- [ ] Update issue tracker

Result: ✅ COMPLETE / 🛑 BLOCKED
```

## Commit Message Best Practices

### Subject Line Rules
- **Limit to 50 characters** (hard limit 72)
- **Use imperative mood**: "Add feature" not "Added feature"
- **Don't end with period**
- **Capitalize first letter**

### Body Guidelines
- **Wrap at 72 characters**
- **Explain what and why**, not how
- **Reference issues and PRs**
- **Note breaking changes**

### Good PyGraphistry Examples
```
feat(gfql): add multi-hop traversal support

Implement efficient multi-hop graph traversals with column
conflict resolution. Uses temporary column prefixing to avoid
namespace collisions during chain operations.

* Supports up to 10 hops with automatic memoization
* Handles both pandas and cudf DataFrames
* Preserves original column names in final output
```

```
fix(umap): handle timeout in GPU embedding computation

Previous implementation could hang on large graphs when GPU
memory was insufficient. Now implements fallback to CPU with
proper memory estimation and chunking.

Fixes #660
```

```
breaking fix(api): change from_cugraph return type

BREAKING CHANGE: from_cugraph now returns PlotterBase instead
of tuple when return_graph=True. This aligns with other
from_* methods.

Migration: Change `g, _ = from_cugraph(G)` to `g = from_cugraph(G)`
```

### Atomic Commits
Each commit should:
- **Focus on one logical change**
- **Be revertable independently**
- **Tell a clear story in history**
- **Include related test updates**

## Interactive Staging Tips

### Using `git add -p`
When files have multiple unrelated changes:

```bash
git add -p path/to/file.py
```

Options:
- `y` - stage this hunk
- `n` - skip this hunk
- `s` - split into smaller hunks
- `e` - manually edit hunk
- `?` - help

### Staging Strategies
1. **New features first** - Infrastructure before implementation
2. **Bug fixes separate** - Don't mix fixes with features
3. **Refactoring alone** - Keep behavior changes separate
4. **Tests with implementation** - Include in same commit
5. **Docs can be separate** - Unless integral to feature

## Common PyGraphistry Patterns

### GFQL Feature Implementation
```
1. feat(gfql): add base predicate AST structure
2. feat(gfql): implement numeric predicate operators
3. feat(compute): integrate predicates into hop operations
4. test(gfql): add predicate evaluation tests
5. docs(gfql): update query language reference
```

### GPU/CPU Fallback Pattern
```
1. feat(umap): add GPU memory estimation
2. feat(umap): implement CPU fallback mechanism
3. perf(umap): optimize chunking for large graphs
4. test(umap): add GPU OOM handling tests
```

### Breaking API Change
```
1. test(api): add tests for new return type behavior
2. breaking fix(api): change from_cugraph return signature
3. docs(api): update migration guide for v2.x
4. chore(changelog): mark breaking change with 🔥
```

### Layout Algorithm Addition
```
1. feat(layout): add tree layout algorithm base
2. feat(layout/tree): implement Sugiyama method
3. feat(plottable): expose tree layout in plot API
4. test(layout): add tree layout test cases
5. docs(layout): add tree layout examples
```

## Troubleshooting

### When Commits Get Messy
- **Too many changes staged**: Use `git reset HEAD` and re-stage selectively
- **Wrong commit message**: `git commit --amend` (before push only)
- **Need to split commit**: `git reset HEAD~1` and recommit in parts
- **Accidentally committed file**: `git rm --cached <file>` in new commit

### Pre-commit Hooks
If commits fail due to hooks:
1. Fix the issues (formatting, linting)
2. Re-stage fixed files
3. Retry commit
4. If hooks modified files, amend commit to include changes

## Decision Points Summary

1. **After Step 2**: Review analysis, decide on grouping strategy
2. **After Step 3**: Approve/modify commit batching plan
3. **During Step 5**: Decide on exact staging for each commit
4. **After Step 6**: Verify satisfaction with commits created

## File Cleanup

**Cleanup Decision**:
- [ ] **Keep progress.md** - Contains useful commit history
- [ ] **Delete progress.md** - No longer needed
- [ ] **Archive to project notes** - Move to documentation

---

**Remember**:
- Always analyze before committing
- Group related changes logically
- Use precise staging for clean history
- Follow conventional format strictly
- Document the why, not just the what
- **Update CHANGELOG after each commit**

## CHANGELOG Updates

<!-- 
This section tracks changes to be added to CHANGELOG.md
Update AFTER each commit when hash is available.
PyGraphistry format: - description (#commit_hash)
-->

**Run**: commits_[YYYY_MM_DD_HHMMSS]  
**Date**: [YYYY-MM-DD]

### To add to CHANGELOG.md under `## [Development]`:

#### Breaking 🔥
<!-- Add breaking changes here as they're committed -->
- Change from_cugraph return type to PlotterBase (#xxxxxxx)

#### Feat
<!-- Add new features here as they're committed -->
- Add multi-hop graph traversal with memoization (#xxxxxxx)
- Support GPU fallback for UMAP embeddings (#xxxxxxx)

#### Fixed
<!-- Add bug fixes here as they're committed -->
- Handle GPU memory exhaustion in embeddings (#xxxxxxx)
- Resolve column conflicts in chain operations (#xxxxxxx)

#### Docs
<!-- Add documentation updates here as they're committed -->
- Update GFQL query examples (#xxxxxxx)
- Add GPU memory management guide (#xxxxxxx)

#### Infra
<!-- Add infrastructure/build changes here as they're committed -->
- Upgrade to cudf 23.x compatibility (#xxxxxxx)
- Add type stubs for pandas 2.0 (#xxxxxxx)

#### Tests
<!-- Add test additions/updates here as they're committed -->
- Add GPU fallback integration tests (#xxxxxxx)
- Improve GFQL chain coverage (#xxxxxxx)

### Session Summary
- Total commits: [number]
- Breaking changes: [list if any]
- Related issues: [#660, #657]
- Remember to update CHANGELOG.md in Development section!