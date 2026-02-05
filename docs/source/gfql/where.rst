.. _gfql-where:

GFQL WHERE (Same-Path Constraints)
==================================

WHERE adds constraints between named steps in a chain. Use it to relate
attributes across the same path (for example, start.owner_id equals
end.owner_id).

Basic Usage
-----------

.. code-block:: python

    from graphistry import n, e_forward
    from graphistry.compute.chain import Chain
    from graphistry.compute.gfql.same_path_types import col, compare

    chain = Chain(
        [
            n({"type": "account"}, name="a"),
            e_forward(),
            n({"type": "user"}, name="c"),
        ],
        where=[
            compare(col("a", "owner_id"), "==", col("c", "owner_id")),
        ],
    )

    g_filtered = g.gfql(chain)

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
            {"eq": {"left": "a.owner_id", "right": "c.owner_id"}}
        ],
    })

Supported operators: `==`, `!=`, `<`, `<=`, `>`, `>=`.
JSON uses `eq`, `neq`, `lt`, `le`, `gt`, `ge`.

Use per-step filters in `n(...)`/`e_forward(...)`; WHERE ties steps together.
