.. _gfql-cypher-api:

GFQL Local Cypher API Reference
===============================

This page documents the Python helper APIs behind PyGraphistry's local Cypher
support.

- **Cypher** is a graph query language popularized by Neo4j and related tools.
- **GFQL** is PyGraphistry's dataframe-native graph query language for querying
  a bound graph in memory.
- PyGraphistry supports a read-only local Cypher subset that can be parsed,
  validated, compiled, and executed through GFQL.

Use this page when you want to:

- run a supported local Cypher query through ``g.gfql("MATCH ...")``
- preflight a query with ``parse_cypher()`` or ``compile_cypher()``
- translate a supported query into a GFQL ``Chain`` programmatically

This page is an API reference, not the main tutorial. It does not call remote
Bolt/Neo4j-style Cypher backends; use ``g.cypher(...)`` or
``graphistry.cypher(...)`` for that remote execution path.

See also:

- :doc:`/gfql/cypher` for the user-facing local Cypher guide and support matrix
- :doc:`/gfql/index` or :doc:`/gfql/quick` if you are new to GFQL itself
- :doc:`/gfql/spec/cypher_mapping` for translation-oriented guidance

Start Here: Local Execution
---------------------------

If you only want to run a supported local Cypher query on a bound graph, start
with ``g.gfql(...)``. The method always returns a ``Plottable``, but the result
shape depends on what you ask for:

- native GFQL chains preserve graph state in ``_nodes`` and ``_edges``
- local Cypher ``RETURN`` projections surface tabular rows in the returned
  ``_nodes`` dataframe

For the broader graph-state vs row-state model, see :doc:`/gfql/quick`.

.. code-block:: python

    from graphistry.compute.ast import e_forward, n

    # Graph/subgraph result: native GFQL chains stay in graph state.
    g2 = g1.gfql([n({"type": "Person"}), e_forward(), n()])

    # Row/table result: local Cypher projections surface rows in _nodes.
    df = g1.gfql(
        "MATCH (p:Person) RETURN p.name AS name ORDER BY name DESC LIMIT $top_n",
        params={"top_n": 5},
    )._nodes

When the query argument is a string, the ``language`` selector defaults to
``"cypher"``. Top-level ``params=...`` is currently only supported for string
query compilation; regular GFQL AST / Chain inputs use normal Python values in
the AST itself.

Helper Functions
----------------

Import the helpers from ``graphistry.compute.gfql.cypher``:

.. code-block:: python

    from graphistry.compute.gfql.cypher import (
        parse_cypher,
        compile_cypher,
        cypher_to_gfql,
        gfql_from_cypher,
    )

``parse_cypher(query)``
~~~~~~~~~~~~~~~~~~~~~~~

- Parses supported local Cypher text into the typed AST used by the local
  compiler.
- Returns ``CypherQuery`` or ``CypherUnionQuery``.

``compile_cypher(query, params=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Parses and lowers a supported local Cypher query into the compiled program
  used by local execution.
- Returns ``CompiledCypherQuery`` or ``CompiledCypherUnionQuery``.
- Use this when you want to inspect the compiler output before execution.

``cypher_to_gfql(query, params=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Compiles a supported local Cypher query into a single GFQL ``Chain``.
- Use this when you want the translated GFQL chain object instead of immediate
  execution.
- Queries that require ``UNION`` or a row-returning ``CALL`` flow intentionally
  raise ``GFQLValidationError`` here; execute those directly through
  ``g.gfql("...", language="cypher")`` instead.

``gfql_from_cypher(query, params=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Alias for ``cypher_to_gfql(...)`` for callers that prefer GFQL-first naming.
