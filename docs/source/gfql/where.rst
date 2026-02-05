.. _gfql-where:

GFQL WHERE (Same-Path Constraints)
==================================

WHERE adds constraints between named steps in a chain. Use it to relate
attributes across the same path (for example, start.owner_id equals
end.owner_id).

Basic Usage
-----------

.. code-block:: python

    from graphistry import n, e_forward, col, compare
    from graphistry.compute.chain import Chain

    chain = Chain(
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

    g_filtered = g.gfql(chain)
    g_filtered.plot()

Use `Chain(..., where=[...])` when you need WHERE; list form is for chains
without WHERE. WHERE only applies to aliases in the chain (same-path scope),
not to unrelated nodes elsewhere in the graph.
All WHERE comparisons are ANDed (all must match).

Aliases come from `name=`. Column references use `alias.column`.

When to use predicates vs WHERE
-------------------------------

Predicates live inside `n(...)`/`e_forward(...)` filter dicts and apply to
one step. WHERE compares fields across steps.

.. code-block:: python

    from graphistry import n, e_forward, col, compare, gt
    from graphistry.compute.chain import Chain

    # Single-step predicate (preferred when you only filter one entity)
    Chain([n({"a": gt(10)}, name="n1"), e_forward(), n(name="n2")])

    # Cross-step comparison (needs WHERE)
    Chain(
        [n(name="n1"), e_forward(name="e1"), n(name="n2"), e_forward(name="e2"), n()],
        where=[
            compare(col("n1", "a"), ">", col("n2", "b")),
            compare(col("e1", "x"), "==", col("e2", "y")),
        ],
    )

JSON wire format details live in :doc:`/gfql/spec/wire_protocol`.
Supported operators: `==`, `!=`, `<`, `<=`, `>`, `>=` (JSON uses `eq`, `neq`,
`lt`, `le`, `gt`, `ge`).

WHERE can compare columns from node or edge steps when the types align.
Null handling follows predicate semantics; use `isna()`/`notna()` in per-step
filters when needed (for example, `n({"owner_id": notna()})`).

Use selective per-step filters in `n(...)`/`e_forward(...)` first; WHERE ties
steps together and can be more expensive on dense graphs.

WHERE works with pandas and cuDF; select an engine via
`g.gfql(..., engine='cudf')`. For full JSON schema details, see
:doc:`/gfql/spec/wire_protocol`.
