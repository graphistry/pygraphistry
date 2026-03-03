---
name: conventional-commits
description: "Create atomic PyGraphistry commits with conventional commit messages and patch-based staging. Use when preparing commit sequences, selecting staged hunks, or writing commit messages."
---

# Conventional Commits Guide for PyGraphistry

**⚠️ BEFORE STARTING: Reread [the planning skill](../plan/SKILL.md). Reuse existing `plans/*/plan.md` if one exists, else start a new one.**

PyGraphistry-specific commit conventions only. See PLAN.md for execution protocol.

## Conventional Commit Format

```
type(scope): subject
```

### Types
`feat` `fix` `docs` `style` `refactor` `perf` `test` `chore` `ci` `build` `infra`

### PyGraphistry Scopes
`plottable` `umap` `gfql` `gpu` `cugraph` `cudf` `dbscan` `featurize` `transform` `dgl` `layout` `compute` `api` `hyper`

## Message Format

**Subject**: ≤50 chars, imperative ("add" not "added"), capitalize, no period
**Body**: ≤72 char lines, explain why not how, reference issues with `Fixes #123`
**Breaking**: Use `BREAKING CHANGE:` in body with migration notes

```
✅ feat(gfql): add multi-hop traversal support
✅ fix(umap): handle GPU OOM with CPU fallback
✅ breaking fix(api): change from_cugraph return type

❌ Added GFQL feature (past tense)
❌ fix bug (no scope, vague)
❌ refactor(gfql): add feature + fix bug (multiple changes)
```

**Atomic**: One logical change per commit, independently revertable, include related tests

## Staging Strategy

**DEFAULT: Many small commits via `git add -p`. Do NOT commit entire files or large batches.**

### Workflow
1. **Validation check**: Check context for recent lint/typecheck/test runs. Warn if unclear.
2. **Group changes** into logical commits (analyze first, see PLAN.md)
3. **Stage interactively** with `git add -p path/to/file.py` for each group
4. **Commit atomically** - one logical change per commit
5. **Repeat** for each small logical unit

```bash
# Create sequence of small commits
git add -p file1.py  # y/n/s/e to select hunks for commit 1
git commit -m "feat(gfql): add predicate AST structure"

git add -p file1.py  # Select remaining hunks for commit 2
git add -p file2.py
git commit -m "feat(gfql): implement numeric predicates"

git add -p file3.py  # Commit 3
git commit -m "test(gfql): add predicate evaluation tests"
```

**Options**: `y` stage | `n` skip | `s` split | `e` edit | `?` help

**Prefer**: 5 small commits over 1 large commit
**Order**: Infrastructure → Features → Fixes → Tests → Docs (adjust per context)

## Pre-commit Hooks

If commits fail due to hooks, amend to include hook modifications after fixing issues.

## Pre-commit Checklist
- [ ] Run linting: `./bin/lint.sh`
- [ ] Run type checking: `./bin/mypy.sh`
- [ ] Verify tests pass: `pytest -xvs` (if test changes)
- [ ] Check for secrets: Review diffs for sensitive data
- [ ] Remove Claude comments: No explanatory comments in code
- [ ] Verify functional style: No `df[col] = val`, use `df.assign()`
- [ ] Check for unnecessary `copy()`: DataFrames already return new objects

## CHANGELOG Updates

**⚠️ Add to `## [Development]` section only. Do NOT add to released tags. If `## [Development]` missing, create it.**

```
### Breaking 🔥 | Feat | Fixed | Docs | Infra | Tests
- scope: description (#hash)
```

Get hash: `git rev-parse --short HEAD`
