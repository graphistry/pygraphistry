# M3 B3 Compatibility Matrix — `CompiledCypher*` Preflight

**Issue**: #1165  
**Parent**: #1160 (M3 execution DAG)  
**Date**: 2026-04-21  
**Status**: Preflight complete — no deletion/cutover in this artifact.  
**Purpose**: Establish which surfaces depend on `CompiledCypherQuery` / `CompiledCypherUnionQuery` / `CompiledCypherGraphQuery` / `compile_cypher_query` and the closure strategy for each before M3-PR3 deletion gate.

---

## Scope

Three target types and one factory function, all defined in `graphistry/compute/gfql/cypher/lowering.py`:

| Symbol | Line | Description |
|--------|------|-------------|
| `CompiledCypherQuery` | 154 | Core compiled query: `chain`, optional `post_processing`, `execution_extras` (carries `logical_plan`, `query_graph`, reentry state), `graph_bindings`, `use_ref` |
| `CompiledCypherUnionQuery` | 225 | UNION wrapper: tuple of `CompiledCypherQuery` branches |
| `CompiledCypherGraphQuery` | 250 | GRAPH-constructor result: graph bindings + chain; NOT in public `__all__` |
| `compile_cypher_query()` | 8368 | Internal factory; returns `Union[CompiledCypherQuery, CompiledCypherUnionQuery, CompiledCypherGraphQuery]` |

---

## Compatibility Matrix

### Row 1 — Compiler internals (lowering.py)

| Field | Detail |
|-------|--------|
| **Surface** | Compiler internals |
| **Owner file** | `graphistry/compute/gfql/cypher/lowering.py` |
| **Usage** | ~50 call sites constructing `CompiledCypherQuery`; `compile_cypher_query()` factory at line 8368; `CompiledCypherUnionQuery` at line 8414; `CompiledCypherGraphQuery` at line 8185+ |
| **Closure strategy** | **Retain until M3-PR3** — these are the compiler output types; PhysicalPlanner will consume `logical_plan` already embedded in `execution_extras`; type deletion requires zero internal construction sites |
| **Proof target** | All `CompiledCypherQuery(...)` constructors eliminated from `lowering.py` in M3-PR3; `compile_cypher_query()` either deleted or becomes a thin alias to the planner path |
| **Blocking on** | M3-PR2 (runtime cutover), M3-PR3 (deletion gate) |
| **Follow-up** | None; tracked under #1160 DAG steps |

---

### Row 2 — Public API exports (cypher/__init__.py + api.py)

| Field | Detail |
|-------|--------|
| **Surface** | Public API |
| **Owner files** | `graphistry/compute/gfql/cypher/__init__.py` lines 36–37, 52–54; `graphistry/compute/gfql/cypher/api.py` lines 8, 79–94 |
| **Usage** | `CompiledCypherQuery`, `CompiledCypherUnionQuery`, `compile_cypher_query` in `__all__`; `compile_cypher()` in `api.py` (public function) returns `Union[CompiledCypherQuery, CompiledCypherUnionQuery, CompiledCypherGraphQuery]` |
| **Closure strategy** | **Shim/defer to M3-PR3** — symbols must remain exported for backward compatibility until all callers can migrate; `compile_cypher()` return type changes to a new plan type in M3-PR3; add deprecation docstring note at cutover |
| **Proof target** | M3-PR3: `compile_cypher()` return type updated; old types removed from `__all__` or retained as deprecated re-exports with removal target in next minor; migration note added to `CHANGELOG.md` |
| **Blocking on** | M3-PR3 (public API cutover) |
| **Follow-up** | New issue needed for deprecation/removal tracking in the next minor version after M3 lands |

---

### Row 3 — Runtime execution engine (gfql_unified.py)

| Field | Detail |
|-------|--------|
| **Surface** | Runtime execution |
| **Owner file** | `graphistry/compute/gfql_unified.py` |
| **Usage** | 8 internal functions accepting `CompiledCypherQuery`/`CompiledCypherUnionQuery`/`CompiledCypherGraphQuery`:<br>• `_execute_graph_constructor_compiled()` line 471<br>• `_execute_graph_query()` line 518<br>• `_execute_query_with_graph_context()` line 542<br>• `_execute_compiled_query()` line 571 (main dispatch)<br>• `_execute_compiled_query_with_reentry()` line 686<br>• `_compiled_query_reentry_state()` line 834<br>• `_compiled_query_scalar_reentry_state()` line 942<br>• `_compiled_query_reentry_contract()` line 1006<br>• Top-level dispatch at line ~1431 (`isinstance` gate) |
| **Closure strategy** | **Migrate in M3-PR2** — this is the primary M3 target; `_execute_compiled_query()` is the main routing function to replace with a `PhysicalPlanner`-dispatched path; reentry functions fold into `PatternMatch(input=...)` execution model |
| **Proof target** | M3-PR2: `_execute_compiled_query()` routes through PhysicalPlanner for covered shapes; `CompiledCypherQuery` is still consumed as the lowering output but execution is planner-dispatched; differential parity tests green |
| **Blocking on** | M3-PR2 (PhysicalPlanner skeleton from #1164) |
| **Follow-up** | None; tracked under #1160 DAG step 3 |

---

### Row 4 — Remote wire path (chain_remote.py)

| Field | Detail |
|-------|--------|
| **Surface** | Remote wire serialization |
| **Owner file** | `graphistry/compute/chain_remote.py` |
| **Usage** | `_compiled_to_let_json()` line 46 — serializes `CompiledCypherQuery.chain` and `graph_bindings` to Let wire format; dispatch at lines 117/122 checks `isinstance(compiled, CompiledCypherUnionQuery)` / `isinstance(compiled, CompiledCypherQuery)` |
| **Closure strategy** | **Shim/defer** — remote wire format is independent of the local compiler IR; the `Chain` + `graph_bindings` fields needed for JSON serialization are part of `CompiledCypherQuery` today; migrating this path requires either a new wire-protocol-facing type or extracting the serialization fields; deferred until M3-PR3 stabilizes the public API surface |
| **Proof target** | Deferred; new follow-up issue to track remote wire migration explicitly |
| **Blocking on** | M3-PR3 public API shape; wire protocol decision |
| **Follow-up** | **Requires new issue**: "M3 follow-up: migrate chain_remote.py off CompiledCypherQuery for remote wire path" |

---

### Row 5 — Test suite

| Field | Detail |
|-------|--------|
| **Surface** | Tests |
| **Owner files** | `graphistry/tests/compute/gfql/cypher/test_lowering.py` (29 references); `graphistry/tests/compute/gfql/cypher/test_m1_differential_scaffold.py` (2 references) |
| **Usage** | `isinstance(result, CompiledCypherQuery)` assertions; direct attribute access (`result.chain`, `result.execution_extras`, etc.) |
| **Closure strategy** | **Migrate with implementation** — tests follow the compilation/execution changes; when `compile_cypher_query()` return type changes in M3-PR3, test assertions update to the new plan type; no independent test-surface action needed |
| **Proof target** | All `isinstance(*, CompiledCypherQuery)` assertions in tests converted to the new type in M3-PR3 |
| **Blocking on** | M3-PR2/M3-PR3 implementation |
| **Follow-up** | None |

---

### Row 6 — Public documentation (cypher.rst)

| Field | Detail |
|-------|--------|
| **Surface** | Docs |
| **Owner file** | `docs/source/api/gfql/cypher.rst` |
| **Usage** | Lines 18, 77, 89, 94 — references `compile_cypher()` return type as `CompiledCypherQuery`/`CompiledCypherUnionQuery` |
| **Closure strategy** | **Update with public API changes** in M3-PR3; when `compile_cypher()` return type changes the doc must be updated simultaneously |
| **Proof target** | Docs updated in same PR as `compile_cypher()` return type change |
| **Blocking on** | M3-PR3 public API cutover |
| **Follow-up** | None; tracked as part of M3-PR3 scope |

---

## Summary: Closure Sequencing

```
M3-PR1b (this) ─── preflight only; no code changes
     │
M3-PR2 (#1164) ─── PhysicalPlanner skeleton
     │              migrate Row 3 (runtime execution)
     │
M3-PR3 ────────── public API cutover + deletion gate
     │              migrate Row 1 (compiler internals: delete constructors)
     │              migrate Row 2 (public exports: deprecate/remove from __all__)
     │              migrate Row 5 (tests: update isinstance assertions)
     │              migrate Row 6 (docs: update return type docs)
     │
M3-PR4+ ───────── deferred
                   migrate Row 4 (remote wire path: new follow-up issue)
                   migrate Row 2 deprecation removal (next minor version)
```

## Deferred Items Requiring Follow-up Issues

| # | Item | Owner |
|---|------|-------|
| A | Remote wire migration: `chain_remote.py` off `CompiledCypherQuery` | New issue (see Row 4) |
| B | Public API deprecation/removal tracking post-M3 | New issue (see Row 2) |
