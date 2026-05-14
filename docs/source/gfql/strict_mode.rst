Strict Schema Checks
====================

GFQL can check Cypher queries against the graph schema before the query runs.
For Cypher users, this means typos in labels, variables, and properties fail
early with a validation error instead of producing confusing empty results or
later execution failures.

Use :py:meth:`g.gfql_validate(...) <graphistry.compute.ComputeMixin.ComputeMixin.gfql_validate>`
when you want a report without running the query. Use
:py:meth:`g.gfql(..., validate=True) <graphistry.compute.ComputeMixin.ComputeMixin.gfql>`
when you want the same checks before execution.

Local Cypher execution uses these schema checks. Environment variables or
keyword arguments do not switch local Cypher execution back to a looser mode.

What Gets Checked
-----------------

For Cypher queries, strict schema checks verify:

* Labels used in ``MATCH`` exist in the graph schema.
* Variables referenced in ``WHERE``, ``RETURN``, ``UNWIND``, and ``CALL`` are
  in scope.
* Property names exist for the node or edge variable they are read from.

Invalid queries raise ``GFQLValidationError`` before execution. Valid queries
run the same as before.

It does **not** check every dataframe value's Python or Arrow type. This page is
about Cypher names and schema references.

Validate Without Running
------------------------

``g.gfql_validate(...)`` returns structured diagnostics and never executes query
operators:

.. doc-test: skip

.. code-block:: python

   report = g.gfql_validate(
       "MATCH (p:Person) RETURN p.name AS name",
       strict=True,
   )
   if not report["ok"]:
       for diag in report["diagnostics"]:
           print(diag["code"], diag["message"])

Validate Before Running
-----------------------

Use ``validate=True`` on ``g.gfql(...)`` to run the same checks before executing
the query:

.. doc-test: skip

.. code-block:: python

   result = g.gfql(
       "MATCH (p:Person) RETURN p.name AS name",
       validate=True,
   )

These APIs are the recommended way to make validation explicit in request
handlers, notebooks, and CI checks.

Configuration Notes
-------------------

Most users do not need to configure these checks directly. Prefer
``g.gfql_validate(...)`` or ``g.gfql(..., validate=True)``.

Code can also set a catalog metadata flag:

.. doc-test: skip

.. code-block:: python

   from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog
   catalog = GraphSchemaCatalog.from_schema_parts(
       node_columns={"id", "label__Person"},
       edge_columns={"src", "dst", "label__KNOWS"},
       metadata={"strict": True},
   )

or a process-wide environment variable:

.. code-block:: bash

   export GRAPHISTRY_GFQL_STRICT_SCHEMA=true

Truthy values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
Falsy / unset: anything else (default ``false``).

Treat these as opt-in signals, not as switches that disable validation. Setting
them to ``false`` or leaving them unset does not make local Cypher execution
looser.

The explicit validation APIs (``g.gfql_validate(strict=True)`` and
``g.gfql(validate=True)``) are unaffected by these helpers.

Error Messages
--------------

Schema-check failures raise ``GFQLValidationError`` with deterministic messages
and sorted availability hints:

.. code-block:: text

    Cypher label is missing from the graph schema.
    Use labels that exist in the node schema or extend the schema catalog.
    available labels: [Comment, Person, Post]

Use the message text to identify the gap, then either fix the query or extend
the catalog while iterating.

When To Use It
--------------

Recommended:

* Production query gates where unknown identifiers should fail closed.
* CI / pre-merge quality bars over a curated catalog.
* Multi-team environments where the graph schema is managed centrally.

Before relying on these checks:

* Exploratory / notebook usage should make sure GFQL knows the labels and
  properties in the graph being queried.
* Pipelines with intentionally partial schemas should validate only after the
  schema has enough labels and properties for the queries being checked.

Recommended usage
-----------------

Use explicit validation for the tightest path:

1. **Validate each call explicitly** — for example, in a request handler that
   should never accept unknown labels, variables, or properties:

.. doc-test: skip

.. code-block:: python

   result = g.gfql(query, validate=True)

This is the clearest option for application code that wants strict checks.

Clearing the env var or removing the catalog flag does not make local Cypher
execution looser. Explicit validation remains strict when requested.

See also
--------

* :doc:`validation/fundamentals` — preflight + execution-time validation
  primitives, including ``g.gfql_validate(...)``.
* :doc:`cypher` — Cypher syntax reference and preflight examples.
