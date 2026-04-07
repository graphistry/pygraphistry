# GFQL Oracle Cheatsheet

Small reference for `graphistry.gfql.ref.enumerator`, the test-only oracle for fixed-length GFQL chains and same-path predicates.

## What it does
- Enumerates every path that satisfies alternating `ASTNode`/`ASTEdge` steps.
- Handles (1) **local predicates** via `filter_dict`/`query` on each step, (2) **same-path multihop WHERE** clauses like `a.x == c.y`, and defers full **path predicates** (`length(p)`, `simplePath(p)`, etc.) to future path-mode work that will reuse this oracle.
- Returns `(nodes, edges, tags, optional paths)` capturing exactly the ids that participate in at least one surviving path.
- Enforces strict caps (default: |V|≤12, |E|≤40, length≤5, partial rows≤200k) so exhaustive enumeration stays cheap.

## In-scope vs out-of-scope

The oracle covers **fixed alternating chain semantics** — the chain-kernel layer.  Use it for parity testing whenever you add or change a chain-kernel feature.

| Feature | In scope | Notes |
|---|---|---|
| Forward, reverse, undirected single-hop | ✅ | All direction modes |
| Multi-hop variable-length (`min_hops`/`max_hops`) | ✅ | Including `min_hops=0` (self-hop) |
| Multi-hop **edge aliases** (`e_forward(name="e")`) | ✅ | Oracle validates node/edge membership; engine also writes a bare presence-marker column — parity holds because oracle does not check column values |
| `filter_dict` / `query` node and edge predicates | ✅ | Applied per-step |
| Same-path WHERE (`col(alias, field)` + `compare()`) | ✅ | Pruned mid-path when both aliases bound |
| Hop labels / `label_seeds` / output slicing | ✅ | |
| `to_fixed_point` edges | ❌ out of scope | Unbounded BFS; enumerator raises `ValueError` |
| Multi-hop **multi-alias** (same edge step, multiple names) | ❌ out of scope | Raises `ValueError` |
| OPTIONAL MATCH / outer-join semantics | ❌ out of scope | Control-flow layer; verified via TCK |
| WITH / UNWIND multi-stage pipelines | ❌ out of scope | Control-flow layer; verified via TCK |
| CASE expressions | ❌ out of scope | Expression evaluation layer |
| shortestPath() | ❌ out of scope | Algorithm dispatch; verified via TCK |
| rows() / binding-table materialization | ❌ out of scope | Row-materialisation layer |

**CI guard**: `tests/gfql/ref/test_enumerator_parity.py` asserts `len(CASES) >= _MIN_PARITY_CASES`.  Bump that constant when you add chain-kernel features.

## Inputs
- `Plottable` with `_nodes`, `_edges`, `_node`, `_source`, `_destination`; `_edge` auto-synthesized when missing.
- Chain or `Chain[...]` object.
- Optional WHERE clauses built with `col(alias, column)` + `compare()`; comparisons touching NULL/NaN always fail.

## Implementation notes
- Everything runs on pandas; cuDF/other backends are copied through `to_pandas()`.
- Alias values flow as lightweight columns named `alias::field`; tags/path outputs simply read those columns back.
- Null/NaN policy: any comparison touching null/NaN ⇒ `False` (`==`/`!=` included); no implicit casts.
- WHERE evaluation can prune mid-loop when both referenced aliases are bound, and is reapplied at the end for completeness.

## Usage
```python
from graphistry.compute import n, e_forward
from graphistry.gfql.ref.enumerator import enumerate_chain, col, compare, OracleCaps

ops = [n({"type": "account"}, name="a"), e_forward({"type": "owns"}, name="r"), n({"type": "user"}, name="c")]
where = [compare(col("a", "owner_id"), "==", col("c", "id"))]

oracle = enumerate_chain(g, ops, where=where, include_paths=True, caps=OracleCaps())
```

## Tests
- Deterministic fixtures + Hypothesis self-checks live in `tests/gfql/ref/test_ref_enumerator.py`.
- Engine-vs-oracle parity lives in `tests/gfql/ref/test_enumerator_parity.py` (forward, reverse, multi-hop, undirected, cycles, branches, empty).
