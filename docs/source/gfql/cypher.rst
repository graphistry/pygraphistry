.. _gfql-cypher:

Local Cypher In GFQL
====================

PyGraphistry can run a supported Cypher subset directly on a bound graph in
memory. Use this page when you want to execute a Cypher string locally through
GFQL instead of translating the query by hand.

Choose The Right Cypher Entrypoint
----------------------------------

- Use ``g.gfql("MATCH ...")`` or ``g.gfql("...", language="cypher")`` for
  local in-memory Cypher execution on the current ``Plottable``.
- Use ``g.gfql_remote([...])`` for remote GFQL execution when the dataset size
  or hardware profile calls for running GFQL on remote infrastructure.
- Use ``graphistry.cypher("...")`` or ``g.cypher("...")`` for remote database
  Cypher over Bolt/Neo4j-style integrations. That is a separate execution path.

Quickstart
----------

Assume ``g`` is a bound graph with nodes and edges already attached.

.. code-block:: python

    top_people = g.gfql(
        "MATCH (p:Person) "
        "RETURN p.name AS name "
        "ORDER BY name DESC "
        "LIMIT 5"
    )

    top_people._nodes

String queries default to ``language="cypher"``, so the explicit selector is
usually optional:

.. code-block:: python

    same_result = g.gfql(
        "MATCH (p:Person) RETURN p.name AS name ORDER BY name DESC LIMIT 5",
        language="cypher",
    )

Parameters
----------

Use ``params=...`` instead of manual string interpolation:

.. code-block:: python

    limited = g.gfql(
        "MATCH (p:Person) "
        "RETURN p.name AS name "
        "ORDER BY name DESC "
        "LIMIT $top_n",
        params={"top_n": 2},
    )

    limited._nodes

Supported Local Cypher Surface
------------------------------

The shipped local compiler is intentionally bounded. High-level supported
families include:

- ``MATCH`` and a bounded ``OPTIONAL MATCH`` subset
- ``WHERE``
- ``RETURN`` and ``WITH``
- ``ORDER BY``, ``SKIP``, ``LIMIT``, and ``DISTINCT``
- ``UNWIND``
- supported local ``UNION`` and ``CALL graphistry.*`` flows when executed
  directly through ``g.gfql("...")``

For exact row-pipeline semantics after pattern matching, see :doc:`return`. For
same-path comparisons, see :doc:`where`.

Validation And Unsupported Shapes
---------------------------------

- Unsupported but syntactically valid local Cypher shapes raise
  ``GFQLValidationError``.
- Invalid local Cypher syntax raises ``GFQLSyntaxError``.
- Unsupported string languages also raise ``GFQLValidationError``.

That fail-fast behavior is intentional: the local compiler prefers explicit
validation over silently returning wrong rows.

Compiler Helper APIs
--------------------

Use the helper APIs when you want to inspect or reuse compiler output rather
than execute immediately:

.. code-block:: python

    from graphistry.compute.gfql.cypher import (
        parse_cypher,
        compile_cypher,
        cypher_to_gfql,
        gfql_from_cypher,
    )

    parsed = parse_cypher("MATCH (p:Person) RETURN p.name AS name")
    compiled = compile_cypher("MATCH (p:Person) RETURN p.name AS name")
    chain = cypher_to_gfql("MATCH (p:Person) RETURN p.name AS name")

``cypher_to_gfql()`` / ``gfql_from_cypher()`` are intentionally limited to
queries that can be represented as a single GFQL ``Chain``. If a query requires
``UNION`` or a row-returning ``CALL`` flow, execute it directly through
``g.gfql("...", language="cypher")`` instead.

See :doc:`/api/gfql/cypher` for the helper reference.

Translation Vs Direct Execution
-------------------------------

This page is about **direct local execution** of supported Cypher strings.

- If you want to run the query now, use ``g.gfql("MATCH ...")``.
- If you want to understand how Cypher maps into GFQL operators and wire
  protocol, use :doc:`spec/cypher_mapping`.
- If you want native GFQL chain syntax instead of strings, start with
  :doc:`quick`.
