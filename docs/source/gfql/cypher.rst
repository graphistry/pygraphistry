.. _gfql-cypher:

Local Cypher In GFQL
====================

PyGraphistry can run a supported Cypher subset directly on a bound graph in
memory. Start here when you want to execute a Cypher string locally through
GFQL instead of translating the query into GFQL by hand.

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

The local compiler intentionally supports a bounded subset. At a high level,
that includes:

- ``MATCH`` and a bounded ``OPTIONAL MATCH`` subset
- ``WHERE``
- ``RETURN`` and ``WITH``
- ``ORDER BY``, ``SKIP``, ``LIMIT``, and ``DISTINCT``
- ``UNWIND``
- supported local ``UNION`` and ``CALL graphistry.*`` flows when executed
  directly through ``g.gfql("...")``

For exact ``RETURN`` / ``WITH`` row semantics after pattern matching, see
:doc:`return`. For same-path ``WHERE`` comparisons, see :doc:`where`.

Support Matrix
--------------

.. list-table::
   :header-rows: 1

   * - Query shape
     - Status
     - Notes
   * - ``MATCH`` / ``WHERE`` / ``RETURN`` / ``WITH`` / ``ORDER BY`` / ``SKIP`` / ``LIMIT`` / ``DISTINCT``
     - Supported
     - Core local read-only path.
   * - ``OPTIONAL MATCH``
     - Partial
     - Supported for a bounded local subset, not the full Cypher null-extension surface.
   * - ``UNWIND``
     - Partial
     - Supported in the local subset, but not in every placement and combination.
   * - ``UNION`` and ``CALL graphistry.*``
     - Partial
     - Execute directly through ``g.gfql("...")``. Helper translation to a single ``Chain`` is stricter.
   * - Variable-length relationship patterns
     - Not yet supported
     - Rewrite in native GFQL with explicit hop bounds today.
   * - ``CREATE`` / ``DELETE`` / ``SET``
     - Not supported
     - Local GFQL-flavored Cypher is read-only.
   * - Multiple disconnected ``MATCH`` patterns and arbitrary joins
     - Not supported
     - Split the work into separate GFQL / dataframe steps.
   * - Full Cypher expression and function surface in row expressions
     - Partial
     - The local row-expression subset is intentionally smaller than full Cypher; finish advanced logic in pandas/cuDF when needed.

Validation And Unsupported Shapes
---------------------------------

- Unsupported but syntactically valid local Cypher shapes raise
  ``GFQLValidationError``, usually before execution starts.
- Invalid local Cypher syntax raises ``GFQLSyntaxError``.
- Passing a string query with an unsupported ``language=...`` selector also
  raises ``GFQLValidationError``.

That fail-fast behavior is intentional: the local compiler prefers explicit
validation over silently returning wrong rows.

Static Validation / Preflight Check
-----------------------------------

If you want to know whether a query fits the current local subset before
execution, preflight it with the helper APIs:

.. code-block:: python

    from graphistry.compute.exceptions import GFQLSyntaxError, GFQLValidationError
    from graphistry.compute.gfql.cypher import parse_cypher, compile_cypher

    query = "MATCH (p:Person) RETURN p.name AS name"

    try:
        parse_cypher(query)      # grammar + AST checks
        compile_cypher(query)    # local compiler / lowering checks
    except GFQLSyntaxError as exc:
        print("Invalid local Cypher syntax:", exc)
    except GFQLValidationError as exc:
        print("Valid Cypher, but outside the current local subset:", exc)

- Use ``parse_cypher()`` when you only want syntax and AST validation.
- Use ``compile_cypher()`` for the strongest local preflight, because it also
  catches unsupported-but-valid query shapes in lowering.
- Use ``cypher_to_gfql()`` only when you specifically need a single GFQL
  ``Chain``. It is intentionally stricter than direct execution through
  ``g.gfql("...")``.

Common Rewrites
---------------

- Need remote database Cypher instead of local in-memory execution? Use
  ``graphistry.cypher("...")`` or ``g.cypher("...")``.
- Need a pure GFQL chain object? Use ``cypher_to_gfql()`` when the query fits a
  single ``Chain``.
- Need variable-length traversal today? Rewrite in native GFQL with explicit
  hop bounds such as ``e_forward(min_hops=1, max_hops=3)``.
- Need write semantics or arbitrary joins? Keep local Cypher for the supported
  read-only part and finish the rest in a database or in pandas/cuDF.

Compiler Helper APIs
--------------------

Use the helper APIs when you want to inspect or reuse compiler output rather
than run the query immediately:

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
