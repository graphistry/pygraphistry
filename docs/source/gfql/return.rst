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

`rows(table=..., source=...)` in practice
-----------------------------------------

.. code-block:: python

    from graphistry import n, e_forward
    from graphistry.compute import rows, return_, order_by

    # Node rows matched by alias "p"
    people_rows = g.gfql([
        n({"type": "Person"}, name="p"),
        e_forward(name="r"),
        n(name="q"),
        rows(table="nodes", source="p"),
        return_(["id", "name", "score"]),
        order_by([("id", "asc")]),
    ])

    # Edge rows matched by alias "r"
    edge_rows = g.gfql([
        n(name="a"),
        e_forward({"type": "FOLLOWS"}, name="r"),
        n(name="b"),
        rows(table="edges", source="r"),
        return_(["s", "d", "type", "weight"]),
    ])

- `table="nodes"` switches to node rows; `table="edges"` switches to edge rows.
- `source="p"` (or `"r"`) keeps only rows participating in that named matcher.
- If `source` is omitted (`rows(table="nodes")`), the full active table is used.
- For edge rows, replace `s`/`d` with your graph's configured edge endpoint column names.

`where_rows(expr="...")`: expression language
---------------------------------------------

.. doc-test: skip

.. code-block:: python

    from graphistry import n
    from graphistry.compute import rows, where_rows, return_

    filtered = g.gfql([
        n({"type": "Person"}, name="p"),
        rows(table="nodes", source="p"),
        where_rows(expr="score >= 50 AND name STARTS WITH 'A' AND manager_id IS NOT NULL"),
        return_(["id", "name", "score", "manager_id"]),
    ])

- `expr` uses the GFQL row-expression parser (Cypher-like subset).
- Columns are referenced by active row-table column name (for example, `score`, `name`).
- Common operators: `AND`, `OR`, `NOT`, `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`,
  `IS NULL`, `IS NOT NULL`, `IN`, `CONTAINS`, `STARTS WITH`, `ENDS WITH`.
- For predicate helpers like `lt(...)`/`between(...)`, use `where_rows(filter_dict=...)`.
- Unsupported row expressions are rejected by validator/runtime.

`with_`, `select`, `return_`: same projection model
---------------------------------------------------

.. code-block:: python

    from graphistry import n
    from graphistry.compute import rows, with_, where_rows, return_

    ranked = g.gfql([
        n({"type": "Person"}, name="p"),
        rows(table="nodes", source="p"),
        with_([
            "id",                       # shorthand for ("id", "id")
            ("score2", "score * 2"),   # tuple is (output_name, expression)
            ("person_name", "name"),   # rename
        ]),
        where_rows(expr="score2 >= 100"),
        return_(["id", "person_name", "score2"]),
    ])

.. code-block:: python

    from graphistry import n
    from graphistry.compute import rows, select

    projected = g.gfql([
        n({"type": "Person"}, name="p"),
        rows(table="nodes", source="p"),
        select([
            ("person_id", "id"),
            ("score2", "score * 2"),
        ]),
    ])

`with_(...)`, `select(...)`, and `return_(...)` all accept:

- Shorthand string: `"col"` means `( "col", "col" )`.
- Tuple form: `(output_name, expression_or_source_column)`.
- Mixed lists of shorthand + tuples.

Notes
-----

- `return_(["id"])` is shorthand for `return_([("id", "id")])`.
- Multiple projection steps are allowed and applied left-to-right:
  `return_(...)`, `with_(...)`, and `select(...)` each project from the current
  active row table produced by prior steps.
  Later projections can only reference columns that still exist after earlier
  projections.
- `order_by([("col", "asc" | "desc")])` sorts by one or more keys.
- `skip(n)` and `limit(n)` are row offsets/caps.
- In `where_rows(expr="...")`, comparison operators are
  `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`.
- For temporal/date-time row filtering, `where_rows(filter_dict=...)` uses the
  same predicate operators as MATCH filters (`gt`, `ge`, `lt`, `le`, `eq`,
  `ne`, `between`).
- Call-step placement rule: row-pipeline calls (`rows`, `where_rows`, `return_`,
  `with_`, `select`, `order_by`, `skip`, `limit`, `distinct`, `unwind`,
  `group_by`) are chain-list steps. Do not interleave call steps with `n()/e()`
  traversals in the chain interior; place calls in boundary prefix/suffix
  segments around traversal steps.
- Unsupported row expressions are rejected by validator/runtime.

See also: :doc:`quick`, :doc:`where`, :doc:`spec/cypher_mapping`.
