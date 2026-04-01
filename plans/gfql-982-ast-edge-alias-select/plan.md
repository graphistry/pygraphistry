---
name: gfql-982-ast-edge-alias-select
description: Native GFQL rows/select edge-alias property projection after traversal
---
# GFQL #982 Plan
**THIS PLAN FILE**: `plans/gfql-982-ast-edge-alias-select/plan.md`
**Issue**: #982 https://github.com/graphistry/pygraphistry/issues/982
**Priority**: p2
**Branch**: `fix/issue-982-ast-edge-alias-select`
**Base**: `origin/master`
**PR**: pending

## Status

**PLANNING SCAFFOLD CREATED**

Known current state from the issue and recent #880 work:
- Direct Cypher multi-alias projection already supports edge alias properties via bindings rows
- Native GFQL `rows() + select()` is reported to still fail on edge alias property access such as `r.creationDate`
- #880 landed the native chain `rows(binding_ops=...)` injection path, so #982 may already be partially or fully unblocked on latest `master`

## Problem statement

On the native AST/GFQL path, row projection after traversal should allow edge alias
property access:

```python
g.gfql([
    n(name="n"),
    e_undirected(name="r"),
    n(name="friend"),
    rows(),
    select([("cd", "r.creationDate")]),
])
```

The historical failure was:

```text
GFQLTypeError [invalid-node-reference] Error executing 'select': unsupported token in row expression: 'r'
```

Expected behavior:
- edge alias properties project successfully from the bindings row table
- result rows include values like `123` for `r.creationDate`

## Working hypothesis

There are two plausible states to verify on current `master`:

1. **Already fixed by #880**
- Native chain `rows()` now injects `binding_ops`
- `_gfql_connected_bindings_row_table()` already materializes edge alias columns like `r.creationDate`
- If so, #982 only needs verification, regression tests, changelog/docs issue closure

2. **Still partially broken**
- The bindings table may materialize edge alias properties, but the native `select()`
  path may still mishandle some combinations:
  - `e_forward` / `e_reverse` / `e_undirected`
  - named vs unnamed first node
  - mixed node + edge alias projections in one `select()`
  - empty match and null propagation cases

## Scope

In scope:
- Verify current `master` behavior for the native AST path from the #982 repro
- Add/fix native GFQL tests covering edge alias property projection after `rows()`
- Fix remaining runtime/lowering gaps if latest `master` still fails on any #982 shape
- Update changelog / issue handoff notes as needed

Out of scope:
- General row-expression redesign beyond what #982 needs
- New Cypher compiler features unrelated to native AST projection
- Broader OPTIONAL MATCH / branching bindings-table work

## Initial validation targets

Primary:
- `graphistry/tests/test_compute_chain.py`
- `graphistry/tests/compute/gfql/cypher/test_lowering.py`

Focused repro commands:
- native AST repro from issue body
- `bash bin/pytest.sh graphistry/tests/test_compute_chain.py -k 'edge_alias or ChainBindingsTable' -q`
- `python3.12 -B -m pytest -q graphistry/tests/compute/gfql/cypher/test_lowering.py -k 'edge_alias'`

## Implementation plan

### Step 1: Re-verify #982 on latest master
- Run the exact issue repro on current branch
- Confirm whether the bug still exists after #880 / v0.53.16

### Step 2: Map the exact failing surface
- Check forward, reverse, and undirected traversals
- Check mixed node + edge alias projections in the same `select()`
- Check empty result and null behavior

### Step 3: Fix the narrowest broken layer
- Prefer fixing native bindings-row materialization / row-expression access if needed
- Avoid special-casing Cypher if the bug is actually in shared row execution

### Step 4: Test amplification
- Add direct native AST regressions for every verified failing shape
- Keep Cypher overlap coverage only where it validates shared runtime behavior

### Step 5: Closeout
- Update `CHANGELOG.md` if behavior changes on this branch
- Push branch and open/update PR
- If latest `master` is already correct, convert the branch into a verification-only closure PR or close it with findings

## Key code locations to inspect

| What | File |
|------|------|
| Native chain rows injection | `graphistry/compute/chain.py` |
| Shared bindings serializer | `graphistry/compute/ast.py` |
| Connected bindings row materialization | `graphistry/compute/gfql/row/pipeline.py` |
| Row-expression property access | `graphistry/compute/gfql/row/pipeline.py` |
| Native chain bindings tests | `graphistry/tests/test_compute_chain.py` |
| Cypher overlap tests | `graphistry/tests/compute/gfql/cypher/test_lowering.py` |

## Notes

- New `plans/` entries are ignored by default in this repo, so this file may require
  force-add when committing if we want it included in the PR branch.
