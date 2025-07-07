GFQL Validation Fundamentals
============================

Learn the basics of validating GFQL queries to catch errors early and build robust graph applications.

.. note::
   This guide is accompanied by an interactive Jupyter notebook. To run the examples yourself, see 
   `demos/gfql/gfql_validation_fundamentals.ipynb <https://github.com/graphistry/pygraphistry/blob/master/demos/gfql/gfql_validation_fundamentals.ipynb>`_.

What You'll Learn
-----------------

* How to validate GFQL query syntax
* Understanding validation error messages  
* Basic schema validation with DataFrames
* Common syntax errors and how to fix them

Prerequisites
-------------

* Basic Python knowledge
* PyGraphistry installed (``pip install graphistry[ai]``)

Quick Start
-----------

.. code-block:: python

   from graphistry.compute.validate import validate_syntax, validate_query
   
   # Validate query syntax
   query = [
       {"type": "n", "filter": {"type": {"eq": "customer"}}},
       {"type": "e_forward"},
       {"type": "n"}
   ]
   
   issues = validate_syntax(query)
   if not issues:
       print("✅ Query syntax is valid!")

Key Concepts
------------

Error Levels
^^^^^^^^^^^^

* **error**: Query will fail if executed
* **warning**: Query may work but has potential issues

Common Validation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``validate_syntax(query)``: Check query structure and syntax
* ``validate_schema(query, schema)``: Validate against data schema
* ``validate_query(query, nodes_df, edges_df)``: Combined validation

Common Errors and Fixes
-----------------------

Invalid Operation Type
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # ❌ Wrong
   [{"type": "node"}]  # Should be "n"
   
   # ✅ Correct
   [{"type": "n"}]

Missing Operator
^^^^^^^^^^^^^^^^

.. code-block:: python

   # ❌ Wrong
   {"filter": {"name": "Alice"}}  # Missing operator
   
   # ✅ Correct
   {"filter": {"name": {"eq": "Alice"}}}

Column Not Found
^^^^^^^^^^^^^^^^

Always validate against your schema to catch column name errors early.

Next Steps
----------

* :doc:`advanced` - Complex queries and multi-hop validation
* :doc:`llm` - AI integration patterns
* :doc:`production` - Production deployment patterns

See Also
--------

* :doc:`../spec/language` - Complete language specification
* :doc:`../overview` - GFQL overview