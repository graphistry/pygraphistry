.. _gfql-cypher-api:

GFQL Cypher Syntax API Reference
================================

This page documents the Python helper APIs behind PyGraphistry's Cypher-syntax
support in GFQL.

- **Cypher** is a graph query language popularized by Neo4j and related tools.
- **GFQL** is PyGraphistry's dataframe-native graph query language: the first
  fully vectorized graph query implementation with an open-source GPU runtime.
- PyGraphistry supports a read-only Cypher surface on bound graphs that can be
  parsed, validated, compiled, and executed through GFQL's columnar engine.

Use this page when you want to:

- run a supported Cypher query through ``g.gfql("MATCH ...")`` on a bound graph
- preflight a query with ``parse_cypher()`` or ``compile_cypher()``
- translate a supported query into a GFQL ``Chain`` programmatically

This page is an API reference, not the main tutorial. It covers Cypher syntax
through ``g.gfql("MATCH ...")`` on a bound graph, which is the on-ramp for
Cypher users who want familiar graph-pattern syntax without giving up GFQL's
fully vectorized dataframe/GPU execution model. For **remote GFQL** execution
on Graphistry infrastructure, use ``g.gfql_remote([...])``. For **remote
database Cypher** over Bolt/Neo4j-style backends, use ``g.cypher(...)`` or
``graphistry.cypher(...)``.

See also:

- :doc:`/gfql/cypher` for the user-facing guide, supported syntax forms,
  and current boundaries
- :doc:`/gfql/remote` for remote GFQL execution
- :doc:`/gfql/index` or :doc:`/gfql/quick` if you are new to GFQL itself
- :doc:`/gfql/spec/cypher_mapping` for translation-oriented guidance

Start Here: Cypher Syntax Through ``g.gfql()``
----------------------------------------------

If you only want to run a supported Cypher query on a bound graph, start
with ``g.gfql(...)``. The method always returns a ``Plottable``, but the result
shape depends on what you ask for:

- native GFQL chains preserve graph state in ``_nodes`` and ``_edges``
- Cypher ``RETURN`` projections surface tabular rows in the returned
  ``_nodes`` dataframe

For the broader graph-state vs row-state model, see :doc:`/gfql/quick`.

.. code-block:: python

    from graphistry.compute.ast import e_forward, n

    # Graph/subgraph result: native GFQL chains stay in graph state.
    g2 = g1.gfql([n({"type": "Person"}), e_forward(), n()])

    # Row/table result: Cypher RETURN projections surface rows in _nodes.
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

- Parses supported Cypher text into the typed AST used by the GFQL Cypher
  compiler.
- Returns ``CypherQuery`` or ``CypherUnionQuery``.

``compile_cypher(query, params=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Deprecated** — scheduled for removal in a future release (tracked in `#1169 <https://github.com/graphistry/pygraphistry/issues/1169>`_).
- Parses and lowers a supported Cypher query into the compiled program used by
  ``g.gfql("MATCH ...")`` execution.
- Returns compiler-internal shapes (``CompiledCypherQuery`` /
  ``CompiledCypherUnionQuery`` / ``CompiledCypherGraphQuery``) that are also
  deprecated and scheduled for removal.
- Prefer ``g.gfql("...", language="cypher")`` for execution and
  ``cypher_to_gfql(...)`` / ``gfql_from_cypher(...)`` for single-chain
  translation.

``cypher_to_gfql(query, params=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Compiles a supported Cypher query into a single GFQL ``Chain``.
- Use this when you want the translated GFQL chain object instead of immediate
  execution.
- Queries that require ``UNION`` or a row-returning ``CALL`` flow intentionally
  raise ``GFQLValidationError`` here; execute those directly through
  ``g.gfql("...", language="cypher")`` instead.

``gfql_from_cypher(query, params=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Alias for ``cypher_to_gfql(...)`` for callers that prefer GFQL-first naming.
