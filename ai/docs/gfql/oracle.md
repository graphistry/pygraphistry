# GFQL Oracle Cheatsheet

Small reference for `graphistry.gfql.ref.enumerator`, the test-only oracle for fixed-length GFQL chains and same-path predicates.

## What it does
- Enumerates every path that satisfies alternating `ASTNode`/`ASTEdge` steps, node/edge filters, and WHERE comparisons across aliases.
- Returns `(nodes, edges, tags, optional paths)` capturing exactly the ids that participate in at least one surviving path.
- Enforces strict caps (default: |V|≤12, |E|≤40, length≤5, partial rows≤200k) so exhaustive enumeration stays cheap.

## Inputs
- `Plottable` with `_nodes`, `_edges`, `_node`, `_source`, `_destination`; `_edge` auto-synthesized when missing.
- Chain or `Chain[...]` object; only single-hop edges today.
- Optional WHERE clauses built with `col(alias, column)` + `compare()`; comparisons touching NULL/NaN always fail.

## Implementation notes
- Everything runs on pandas; cuDF/other backends are copied through `to_pandas()`.
- Alias values flow as lightweight columns named `alias::field`; tags/path outputs simply read those columns back.
- WHERE evaluation happens after the final merge; rejected rows are pruned before emitting outputs.

## Usage
```python
from graphistry.compute import n, e_forward
from graphistry.gfql.ref.enumerator import enumerate_chain, col, compare, OracleCaps

ops = [n({"type": "account"}, name="a"), e_forward({"type": "owns"}, name="r"), n({"type": "user"}, name="c")]
where = [compare(col("a", "owner_id"), "==", col("c", "id"))]

oracle = enumerate_chain(g, ops, where=where, include_paths=True, caps=OracleCaps())
```

## Tests
- Deterministic fixtures + Hypothesis self-checks live in `graphistry/tests/gfql/test_ref_enumerator.py`.
- Engine-vs-oracle parity lives in `graphistry/tests/gfql/test_enumerator_parity.py` (forward, reverse, multi-hop, undirected, cycles, branches, empty).
