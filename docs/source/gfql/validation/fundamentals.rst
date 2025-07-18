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

Advanced: Manual Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For advanced users who need to validate before execution:

.. code-block:: python

   # Validate syntax without running
   chain = Chain([n(), e_forward()])
   errors = chain.validate(collect_all=True)
   
   # Pre-validate against schema (rarely needed)
   from graphistry.compute.validate_schema import validate_chain_schema
   schema_errors = validate_chain_schema(g, chain, collect_all=True)

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