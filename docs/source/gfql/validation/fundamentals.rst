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
* **Schema validation**: Happens by default during ``g.gfql()`` execution
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
       result = g.gfql([n({'category': 'VIP'})])
   except GFQLSchemaError as e:
       print(f"Error: {e.message}")  # Column "category" does not exist
       print(f"Suggestion: {e.context.get('suggestion')}")
   
   # Correct - use existing columns
   result = g.gfql([n({'type': 'customer'})])

Type Mismatches
^^^^^^^^^^^^^^^

.. code-block:: python

   # Wrong - string value on numeric column
   try:
       result = g.gfql([n({'score': 'high'})])
   except GFQLSchemaError as e:
       print(f"Error: {e.message}")  # Type mismatch
   
   # Correct - use numeric predicate
   from graphistry.compute.predicates.numeric import gt
   result = g.gfql([n({'score': gt(80)})])

Temporal Comparisons
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   from graphistry.compute.predicates.numeric import gt, lt
   
   # Compare datetime columns
   result = g.gfql([
       n({'created_at': gt(pd.Timestamp('2024-01-01'))})
   ])
   
   # Find recent activity (last 7 days)
   result = g.gfql([
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
   result = g.gfql([n({'type': 'customer'})])
   
   # Errors are caught and reported clearly
   try:
       result = g.gfql([n({'invalid_column': 'value'})])
   except GFQLSchemaError as e:
       print(f"Error: {e.message}")

Pre-Execution Validation Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You have two options for validating queries:

1. **Validate-only** (no execution): Use ``validate_chain_schema()`` to check compatibility without running the query
2. **Validate-and-run**: Use ``g.gfql(..., validate_schema=True)`` to validate before execution

.. code-block:: python

   # Method 1: Validate-only (no execution)
   from graphistry.compute.validate_schema import validate_chain_schema
   
   try:
       validate_chain_schema(g, chain)  # Only validates, doesn't execute
       print("Chain is valid for this graph schema")
   except GFQLSchemaError as e:
       print(f"Schema incompatibility: {e}")
   
   # Method 2: Validate-and-run
   try:
       result = g.gfql(chain.chain, validate_schema=True)  # Validates, then executes if valid
       print(f"Query executed: {len(result._nodes)} nodes")
   except GFQLSchemaError as e:
       print(f"Validation failed, query not executed: {e}")

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