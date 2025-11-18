# GFQL Reference Enumerator / Oracle

Internal notes for the `graphistry.gfql.ref.enumerator` module introduced in issue #836.

## Purpose

- Provide a correctness oracle for fixed-length GFQL chain queries.
- Enumerates every path that satisfies local node/edge filters and same-path WHERE clauses.
- Emits node/edge subsets, alias tags, and optional explicit path bindings for debugging.
- Serves as a golden reference for pandas + cuDF engines and higher-level documentation/tests.

## Inputs

- `g: Plottable` — must have `_nodes`, `_edges`, `_node`, `_source`, `_destination` bindings. `_edge` optional; oracle auto-injects a synthetic id column when missing.
- `ops: Sequence[ASTObject]` — alternating `ASTNode`/`ASTEdge` starting and ending with nodes. Hops must be single-step today.
- `where: Sequence[WhereComparison]` — optional same-path predicates comparing alias columns (`id`, scalar columns) using `==, !=, <, <=, >, >=`.
- `include_paths: bool` — toggles explicit `{alias -> id}` path bindings in the result.
- `caps: OracleCaps` — safety limits (defaults below) to prevent runaway enumeration.

## Outputs (`OracleResult`)

- `nodes`: pandas DataFrame subset containing every vertex participating in any satisfying path.
- `edges`: pandas DataFrame subset containing every edge participating in any satisfying path.
- `tags`: `Dict[str, Set[Any]]` mapping alias name → set of ids seen on that alias across all surviving paths.
- `paths`: optional list of per-path bindings (only populated when `include_paths=True`).

## Safety Caps

- Default caps keep exhaustive enumeration tractable and dovetail with CI:
  - `max_nodes = 12`
  - `max_edges = 40`
  - `max_length = 5` (node steps)
  - `max_partial_rows = 200_000`
- Overrides should be used sparingly for local testing; CI relies on defaults.
- Caps are enforced after cloning the source DataFrames into pandas to ensure consistent behavior for pandas/cuDF inputs.

## Null/NaN Semantics

- Any comparison touching a null/NaN evaluates to False.
- No implicit coercion (e.g., strings vs ints). Users must pre-clean data.
- Nulls never leak into the alias propagation tables; predicates prune them immediately.

## DataFrame Handling

- Oracle always operates on pandas DataFrames. Inputs that implement `to_pandas()` (cuDF, proxy wrappers) are converted automatically via `_ensure_pandas_frame()`.
- If `_edge` is unbound, a stable `__enumerator_edge_id__` column is injected to keep downstream comparisons deterministic.

## Usage Pattern

```python
from graphistry.compute import n, e_forward
from graphistry.gfql.ref.enumerator import enumerate_chain, OracleCaps, compare, col

ops = [
    n({"type": "account"}, name="a"),
    e_forward({"type": "owns"}, name="r"),
    n({"type": "user"}, name="c"),
]
where = [compare(col("a", "owner_id"), "==", col("c", "id"))]

oracle = enumerate_chain(g, ops, where=where, include_paths=True, caps=OracleCaps())
print(oracle.nodes, oracle.edges, oracle.tags, oracle.paths)
```

## Test Coverage

- Deterministic table-driven cases mirror tricky scenarios (triangles, alias reuse, null semantics, cap enforcement, cuDF conversions).
- GFQL parity tests (`graphistry/tests/gfql/test_enumerator_parity.py`) ensure pandas engine outputs match the oracle on canonical chains.
- Hypothesis-based property test exercises random tiny graphs to validate tag/path consistency.
