GFQL Validation Fundamentals
============================

Learn how to use GFQL's built-in validation system to catch errors early and build robust graph applications.

.. note::
   This guide is accompanied by an interactive Jupyter notebook. To run the examples yourself, see 
   `GFQL Validation Fundamentals notebook <../../demos/gfql/gfql_validation_fundamentals.html>`_.

What You'll Learn
-----------------

* How GFQL automatically validates queries
* Understanding structured error messages with error codes
* Schema validation against your data
* Pre-execution validation for performance
* Collecting all errors vs fail-fast mode

Prerequisites
-------------

* Basic Python knowledge
* PyGraphistry installed (``pip install graphistry[ai]``)

Quick Start
-----------

.. code-block:: python

   from graphistry.compute.chain import Chain
   from graphistry.compute.ast import n, e_forward
   from graphistry.compute.exceptions import GFQLValidationError
   
   # Automatic validation during construction
   try:
       chain = Chain([
           n({'type': 'customer'}),
           e_forward(),
           n()
       ])
       print("Valid chain created!")
   except GFQLValidationError as e:
       print(f"Error: [{e.code}] {e.message}")

Key Concepts
------------

Built-in Validation
^^^^^^^^^^^^^^^^^^^

GFQL validates automatically - no separate validation calls needed:

* **Syntax validation**: Happens during chain construction
* **Schema validation**: Happens by default during ``g.chain()`` execution
* **Structured errors**: Error codes (E1xx, E2xx, E3xx) for programmatic handling

Error Types
^^^^^^^^^^^

* **GFQLSyntaxError** (E1xx): Structural issues in query
* **GFQLTypeError** (E2xx): Type mismatches and invalid values
* **GFQLSchemaError** (E3xx): Missing columns, incompatible types

Common Errors and Fixes
-----------------------

Invalid Parameters
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Wrong - negative hops
   try:
       chain = Chain([n(), e_forward(hops=-1)])
   except GFQLTypeError as e:
       print(f"Error: {e.message}")  # "hops must be a positive integer"

   # Correct
   chain = Chain([n(), e_forward(hops=2)])

Missing Columns
^^^^^^^^^^^^^^^

.. code-block:: python

   # Wrong - column doesn't exist
   try:
       result = g.chain([n({'category': 'VIP'})])
   except GFQLSchemaError as e:
       print(f"Error: {e.message}")  # Column "category" does not exist
       print(f"Suggestion: {e.context.get('suggestion')}")

   # Correct - use existing columns
   result = g.chain([n({'type': 'customer'})])

Type Mismatches
^^^^^^^^^^^^^^^

.. code-block:: python

   # Wrong - string value on numeric column
   try:
       result = g.chain([n({'score': 'high'})])
   except GFQLSchemaError as e:
       print(f"Error: {e.message}")  # Type mismatch

   # Correct - use numeric predicate
   from graphistry.compute.predicates.numeric import gt
   result = g.chain([n({'score': gt(80)})])

Temporal Comparisons
^^^^^^^^^^^^^^^^^^^^

.. doc-test: xfail

.. code-block:: python

   import pandas as pd
   from graphistry.compute.predicates.numeric import gt, lt
   
   # Compare datetime columns
   result = g.chain([
       n({'created_at': gt(pd.Timestamp('2024-01-01'))})
   ])
   
   # Find recent activity (last 7 days)
   result = g.chain([
       e_forward({
           'timestamp': gt(pd.Timestamp.now() - pd.Timedelta(days=7))
       })
   ])

How Validation Works
--------------------

Default Behavior
^^^^^^^^^^^^^^^^

GFQL validates automatically - just write your queries and run them:

.. code-block:: python

   # Validation happens automatically
   result = g.chain([n({'type': 'customer'})])

   # Errors are caught and reported clearly
   try:
       result = g.chain([n({'invalid_column': 'value'})])
   except GFQLSchemaError as e:
       print(f"Error: {e.message}")

Pre-Execution Validation Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the inline GFQL entrypoints first:

1. ``g.gfql_validate(...)`` for validate-only preflight (no execution)
2. ``g.gfql(..., validate=True)`` for preflight + execution
3. ``validate_chain_schema()`` for low-level chain-schema checks only

``g.gfql_validate(...)`` (validate-only, no execution) supports:

* **Input forms**: Cypher strings, GFQL JSON payloads, and GFQL Python objects
  (for example ``Chain(...)``, ``[n(), e(), n()]``, and ``ASTLet(...)``)
  String inputs are always validated as Cypher (no separate string-shape precheck).
* **Predicate + structural validation**: yes
* **Schema validation**:

  * GFQL JSON and GFQL Python chain-like forms: yes (default ``schema=True``)
  * GFQL Let/DAG forms: DAG structure + schema checks for direct graph-bound
    steps; reference-based steps stay structural-only
  * Cypher strings: syntax/compile + schema-aware name checks against the bound
    graph schema by default (``strict=True``); pass ``strict=False`` for
    syntax/compile-only preflight

.. code-block:: python

   # Chain / JSON-style GFQL
   report = g.gfql_validate([n({'type': 'customer'})], collect_all=True)
   if not report["ok"]:
       print(report["diagnostics"])

   # Cypher
   cypher_report = g.gfql_validate(
       "MATCH (c) RETURN c.id AS id LIMIT $n",
       params={"n": 10},
   )
   if not cypher_report["ok"]:
       print(cypher_report["diagnostics"])

``g.gfql(..., validate=True)`` accepts the same query inputs as ``g.gfql(...)``
(Cypher string, GFQL JSON, GFQL Python objects), runs local preflight first, and
executes only when preflight passes. Its preflight uses ``g.gfql_validate(...)``
defaults, so local bound-graph execution runs schema-aware checks by default.

.. code-block:: python

   # Run preflight first; execute only if preflight passes
   result = g.gfql(
       "MATCH (c) RETURN c.id AS id LIMIT $n",
       params={"n": 10},
       validate=True,
   )

Use ``validate_chain_schema()`` when you specifically want the low-level chain-schema helper:

.. code-block:: python

   from graphistry.compute.validate_schema import validate_chain_schema

   # Step 1: Validate (no execution)
   try:
       validate_chain_schema(g, chain)  # Only validates, doesn't execute
       print("Chain is valid for this graph schema")
   except GFQLSchemaError as e:
       print(f"Schema incompatibility: {e}")

   # Step 2: Execute (after validation passes)
   result = g.gfql(chain.chain)
   print(f"Query executed: {len(result._nodes)} nodes")

Execution-time Preflight Toggles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For remote execution, ``g.gfql_remote(..., validate=True)`` runs local query
prevalidation before implicit upload/network execution, so invalid queries fail
before data upload when possible. For Cypher strings, remote prevalidation uses
``strict=False`` by default because the authoritative schema is on the remote dataset.

Error Collection
^^^^^^^^^^^^^^^^

Choose between fail-fast and collect-all modes:

.. code-block:: python

   # Fail-fast (default)
   try:
       chain = Chain([problematic_operations])
   except GFQLValidationError as e:
       print(f"First error: {e}")
   
   # Collect all errors
   errors = chain.validate(collect_all=True)
   for error in errors:
       print(f"[{error.code}] {error.message}")

Next Steps
----------

* :doc:`llm` - AI integration patterns
* :doc:`production` - Production deployment patterns

See Also
--------

* :doc:`../spec/language` - Complete language specification
* :doc:`../overview` - GFQL overview
