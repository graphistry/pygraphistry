.. _gfql-return:

GFQL RETURN (Row Pipelines)
===========================

Use row-pipeline operators for Cypher-style `MATCH ... RETURN` flows after
pattern matching.

Scope
-----

- This page covers row-table operations: `rows`, `where_rows`, `with_`,
  `return_`, `select`, `order_by`, `skip`, `limit`, `distinct`, `unwind`, `group_by`.
- For same-path MATCH constraints, use :doc:`where` (`where=[...]`).

Minimal Example
---------------

.. code-block:: python

    from graphistry import n, e_forward, gt
    from graphistry.compute import rows, where_rows, return_, order_by, limit

    top_people = g.gfql([
        n({"type": "Person"}),
        e_forward({"type": "FOLLOWS"}),
        n({"type": "Person", "score": gt(0)}, name="p"),
        rows(table="nodes", source="p"),
        where_rows(expr="score >= 50"),
        return_(["id", "name", "score"]),
        order_by([("score", "desc"), ("name", "asc")]),
        limit(10),
    ])

Key Semantics
-------------

- `rows(table="nodes" or table="edges", source="alias")` selects the active row table.
- `source` must reference a prior matcher alias from `name="..."`.
- `where_rows(...)` filters the active row table (not chain aliases).
- `return_`, `with_`, and `select` use the same projection shape.

Projection Equivalence
----------------------

.. code-block:: python

    # Equivalent projections
    return_(["id", ("score2", "score * 2")])
    with_(["id", ("score2", "score * 2")])
    select([("id", "id"), ("score2", "score * 2")])

Notes
-----

- `return_(["id"])` is shorthand for `return_([("id", "id")])`.
- `order_by([("col", "asc" | "desc")])` sorts by one or more keys.
- `skip(n)` and `limit(n)` are row offsets/caps.
- In `where_rows(expr="...")`, comparison operators are
  `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`.
- For temporal/date-time row filtering, `where_rows(filter_dict=...)` uses the
  same predicate operators as MATCH filters (`gt`, `ge`, `lt`, `le`, `eq`,
  `ne`, `between`).
- Unsupported row expressions are rejected by validator/runtime.

See also: :doc:`quick`, :doc:`where`, :doc:`spec/cypher_mapping`.
