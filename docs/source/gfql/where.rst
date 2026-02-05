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
Multiple comparisons in `where=[...]` are combined with AND (all must match).

Aliases come from `name=`. Column references use `alias.column`.

JSON Form
---------

.. code-block:: python

    g_filtered = g.gfql({
        "chain": [
            n({"type": "account"}, name="a").to_json(),
            e_forward().to_json(),
            n({"type": "user"}, name="c").to_json(),
        ],
        "where": [
            {"eq": {"left": "a.owner_id", "right": "c.owner_id"}},
            {"neq": {"left": "a.status", "right": "c.status"}}
        ],
    })

Supported operators: `==`, `!=`, `<`, `<=`, `>`, `>=`.
JSON uses `eq`, `neq`, `lt`, `le`, `gt`, `ge`.

WHERE can compare columns from node or edge steps when the types align.
Null handling follows predicate semantics; use `isna()`/`notna()` in per-step
filters when needed (for example, `n({"owner_id": notna()})`).

Use selective per-step filters in `n(...)`/`e_forward(...)` first; WHERE ties
steps together and can be more expensive on dense graphs.

WHERE works with pandas and cuDF; select an engine via
`g.gfql(..., engine='cudf')`. For full JSON schema details, see
:doc:`/gfql/spec/wire_protocol`.
