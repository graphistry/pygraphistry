.. _gfql-cypher-api:

GFQL Local Cypher API Reference
===============================

These helpers expose the local GFQL-flavored Cypher parser and compiler.

For execution-first usage, prefer ``g.gfql("MATCH ...")`` on a bound graph. Use
the helper functions below when you want to parse, compile, or translate a
supported local Cypher query programmatically. They do not call remote
Bolt/Neo4j-style Cypher backends.

See also:

- :doc:`/gfql/cypher` for the user-facing local Cypher guide
- :doc:`/gfql/spec/cypher_mapping` for translation-oriented guidance

Direct Execution Entry Point
----------------------------

Use ``g.gfql(...)`` for local string execution on a bound graph:

.. code-block:: python

    result = g.gfql("MATCH (p:Person) RETURN p.name AS name")

When the query argument is a string, the ``language`` selector defaults to
``"cypher"``. Use ``params=...`` for parameter substitution instead of manual
string interpolation:

.. code-block:: python

    result = g.gfql(
        "MATCH (p:Person) RETURN p.name AS name ORDER BY name DESC LIMIT $top_n",
        params={"top_n": 5},
    )

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
