.. _gfql-where:

GFQL WHERE (Same-Path Constraints)
==================================

WHERE adds constraints between named steps in a chain. Use it to relate
attributes across the same path (for example, start.owner_id equals
end.owner_id).

This page documents MATCH-stage `where=[...]` constraints.
For RETURN-stage row filtering (`where_rows(...)`), see :doc:`return`.

Basic Usage
-----------

.. code-block:: python

    from graphistry import n, e_forward, col, compare

    g_filtered = g.gfql(
        [
            n({"type": "account"}, name="a"),
            e_forward(name="e"),
            n({"type": "user"}, name="c"),
        ],
        where=[
            compare(col("a", "owner_id"), "==", col("c", "owner_id")),
            compare(col("e", "org_id"), "==", col("a", "org_id")),
        ],
    )
    g_filtered.plot()

Use `g.gfql([...], where=[...])` for WHERE. `Chain(..., where=[...])` is the
equivalent explicit form. WHERE only applies to aliases in the chain
(same-path scope), not to unrelated nodes elsewhere in the graph.
All WHERE comparisons are ANDed (all must match).

Aliases come from `name=`. Column references use `alias.column`.

Boolean Semantics (`where=[...]`)
---------------------------------

`where` is a Python list of comparison clauses. Commas in that list mean
logical AND.

.. doc-test: skip

.. code-block:: python

    from graphistry import n, e_forward, col, compare

    g.gfql(
        [n(name="a"), e_forward(name="e"), n(name="b")],
        where=[
            compare(col("a", "org_id"), "==", col("b", "org_id")),  # AND
            compare(col("e", "risk"), ">=", col("a", "min_risk")),  # AND
        ],
    )

- Supported now: conjunction (AND) across entries.
- Not supported yet in same-path WHERE: `OR`, `NOT`, grouping parentheses.

Comparator Surface (Same-Path WHERE)
------------------------------------

`compare(col(...), op, col(...))` supports these operators:

- `==`, `!=`, `<`, `<=`, `>`, `>=`

JSON wire format uses these names:

- `eq`, `neq`, `lt`, `le`, `gt`, `ge`

Why predicate helpers are not used in same-path `where`
-------------------------------------------------------

Predicate helpers (for example `gt(10)`, `between(...)`, `isna()`) are
single-column filters, and belong in `n({...})` / `e_forward({...})`
`filter_dict`s or in `where_rows(filter_dict=...)`.

Same-path `where=[...]` is currently restricted to column-vs-column
comparisons across aliases so the validator can statically verify aliases and
columns before execution in both pandas and cuDF vectorized paths.

.. doc-test: skip

.. code-block:: python

    from graphistry import n, e_forward, col, compare, gt

    # Good: single-step predicate helper
    g.gfql([n({"score": gt(10)}, name="a"), e_forward(), n(name="b")])

    # Good: cross-step column-vs-column comparison
    g.gfql(
        [n(name="a"), e_forward(name="e"), n(name="b")],
        where=[compare(col("a", "score"), ">", col("b", "score"))],
    )

    # Not supported in same-path WHERE (predicate helper inside compare)
    g.gfql(
        [n(name="a"), e_forward(name="e"), n(name="b")],
        where=[compare(col("a", "score"), ">", gt(10))],
    )
    # ValueError: where[...] must use StepColumnRef for left/right ...

When to use predicates vs WHERE
-------------------------------

Predicates live inside `n(...)`/`e_forward(...)` filter dicts and apply to
one step. WHERE compares fields across steps.

.. doc-test: skip

.. code-block:: python

    from graphistry import n, e_forward, col, compare, gt

    # Single-step predicate (preferred when you only filter one entity)
    g.gfql([n({"a": gt(10)}, name="n1"), e_forward(), n(name="n2")])

    # Cross-step comparison (needs WHERE)
    g.gfql(
        [n(name="n1"), e_forward(name="e1"), n(name="n2"), e_forward(name="e2"), n()],
        where=[
            compare(col("n1", "a"), ">", col("n2", "b")),
            compare(col("e1", "x"), "==", col("e2", "y")),
        ],
    )

JSON wire format details live in :doc:`/gfql/spec/wire_protocol`.
Supported operators: `==`, `!=`, `<`, `<=`, `>`, `>=` (JSON uses `eq`, `neq`,
`lt`, `le`, `gt`, `ge`).

Current scope:

- Same-path column-vs-column comparisons across named aliases
- AND semantics across `where=[...]` entries

In progress:

- Boolean composition (`OR`, `NOT`, grouping)
- Column-vs-literal comparisons
- Predicate/function expressions in `where`
- Computed expressions
- Cross-path/global constraints

Validation Behavior
-------------------

WHERE is validated before same-path execution starts.

Validation checks include:

- **Alias bindings**: Every referenced alias must exist as a `name=` on a node
  or edge step in the chain.
- **Column visibility**: Each referenced column must exist on the visible
  schema at that step. This includes columns added by prior safelisted
  `call(...)` operations whose schema effects are known.
- **Clause shape**: In Python, each `where` entry must be a
  `compare(col(...), op, col(...))` object (or equivalent dict clause); in
  JSON, each entry must use exactly one operator key (`eq`, `neq`, `lt`, `le`,
  `gt`, `ge`) with string `left`/`right` values.

Common failures:

.. doc-test: skip

.. code-block:: python

    from graphistry import n, e_forward, col, compare

    # Missing alias binding ("missing" was never introduced via name=)
    g.gfql(
        [n(name="a"), e_forward(name="e"), n(name="c")],
        where=[compare(col("missing", "x"), "==", col("c", "owner_id"))],
    )
    # ValueError: WHERE references aliases with no node/edge bindings: missing

    # Missing column on an alias
    g.gfql(
        [n(name="a"), e_forward(name="e"), n(name="c")],
        where=[compare(col("a", "missing_col"), "==", col("c", "owner_id"))],
    )
    # ValueError: WHERE references missing column 'missing_col' on alias 'a' ...

    # Invalid where entry type
    g.gfql([n(name="a"), e_forward(name="e"), n(name="c")], where=[123])
    # ValueError: where[0] must be a WhereComparison or dict clause ...

Advanced troubleshooting (opt-in): set environment variables
`GRAPHISTRY_WHERE_VALIDATION_IGNORE_ERRORS` and
`GRAPHISTRY_WHERE_VALIDATION_IGNORE_CALLS` to selectively suppress specific
missing-column validation paths during migration/debugging.

WHERE can compare columns from node or edge steps when the types align.
Null handling follows predicate semantics; use `isna()`/`notna()` in per-step
filters when needed (for example, `n({"owner_id": notna()})`).

Use selective per-step filters in `n(...)`/`e_forward(...)` first; WHERE ties
steps together and can be more expensive on dense graphs.

WHERE works with pandas and cuDF; select an engine via
`g.gfql(..., engine='cudf')`. For full JSON schema details, see
:doc:`/gfql/spec/wire_protocol`.

Row-Table Filtering with `where_rows(...)`
------------------------------------------

Use `where_rows(...)` when filtering the active row table selected by
`rows(...)` in a `MATCH ... RETURN`-style pipeline.

.. code-block:: python

    from graphistry import n, e_forward
    from graphistry.compute import rows, where_rows, return_

    filtered = g.gfql([
        n(name="a"),
        e_forward(name="e"),
        n(name="b"),
        rows(table="nodes", source="b"),
        where_rows(expr="score >= 10 AND name CONTAINS 'alice'"),
        return_(["id", "name", "score"]),
    ])

`where` and `where_rows` solve different problems:

- `where=[...]`: same-path alias comparisons across chain steps.
- `where_rows(...)`: row-level filtering on the active table (nodes/edges).

`where_rows` accepts:

- `filter_dict={...}` predicate filters.
- `expr=\"...\"` Cypher-like scalar expressions.
- both together (AND semantics).
- In `expr=\"...\"`, comparison operators are `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`.
- For temporal/date-time row filtering, `filter_dict` uses the same predicate
  operators as MATCH filters (for example, `gt`, `ge`, `lt`, `le`, `eq`, `ne`,
  `between`).

Validation behavior:

- Expression forms outside the supported subset are rejected by validator/runtime.
- Column references are validated against the active row table.
- Execution stays vectorized on pandas/cuDF backends.
