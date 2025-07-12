Advanced GFQL Validation Patterns
=================================

Deep dive into complex GFQL validation scenarios, performance considerations, and advanced patterns.

.. note::
   Run the interactive examples yourself in 
   `demos/gfql/gfql_validation_advanced.ipynb <https://github.com/graphistry/pygraphistry/blob/master/demos/gfql/gfql_validation_advanced.ipynb>`_.

Prerequisites
-------------

* Complete :doc:`fundamentals` first
* Experience writing GFQL queries
* Understanding of graph traversal concepts

Complex Multi-Hop Queries
-------------------------

Validate queries with multiple hops and complex traversal patterns.

.. code-block:: python

   # Multi-hop with bounded traversal
   query = [
       {"type": "n", "filter": {"type": {"eq": "user"}}},
       {"type": "e_forward", "hops": 2},  # 2-hop traversal
       {"type": "n", "filter": {"risk_score": {"gt": 50}}}
   ]

Named Operations
^^^^^^^^^^^^^^^^

Use named operations for complex patterns:

.. code-block:: python

   query = [
       {"type": "n", "name": "start_users", "filter": {"type": {"eq": "user"}}},
       {"type": "e_forward", "filter": {"rel_type": {"eq": "purchased"}}},
       {"type": "n", "name": "products"},
       {"type": "e_reverse", "filter": {"rel_type": {"eq": "viewed"}}},
       {"type": "n", "name": "viewers"}
   ]

Advanced Predicates
-------------------

Temporal Predicates
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   query = [
       {"type": "n", "filter": {
           "created_at": {
               "gt": {"type": "datetime", "value": "2024-01-10T00:00:00Z"}
           }
       }}
   ]

Nested Predicates
^^^^^^^^^^^^^^^^^

.. code-block:: python

   query = [
       {"type": "n", "filter": {
           "_and": [
               {"type": {"in": ["user", "payment"]}},
               {"_or": [
                   {"risk_score": {"gte": 75}},
                   {"tags": {"contains": "urgent"}}
               ]}
           ]
       }}
   ]

Performance Considerations
--------------------------

Bounded vs Unbounded Hops
^^^^^^^^^^^^^^^^^^^^^^^^^

Always specify hop limits for better performance:

.. code-block:: python

   # [OK] Good - bounded
   {"type": "e_forward", "hops": 3}
   
   # [WARNING] Warning - unbounded
   {"type": "e_forward"}  # No hop limit

Query Complexity Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Monitor query complexity to prevent performance issues in production.

Schema Evolution
----------------

Handle schema changes gracefully:

.. code-block:: python

   def create_compatible_query(query, column_mapping):
       """Update query to use new column names."""
       # Implementation to map old columns to new ones
       pass

Custom Validation
-----------------

Extend validation for domain-specific requirements:

.. code-block:: python

   def validate_business_rules(query, schema):
       """Add custom business rule validation."""
       custom_issues = []
       
       # Check for sensitive columns without filters
       # Warn about expensive patterns
       # Enforce domain-specific constraints
       
       return custom_issues

Best Practices
--------------

1. **Multi-hop queries**: Always specify hop limits
2. **Complex predicates**: Use nested AND/OR for sophisticated filtering
3. **Schema evolution**: Plan for column changes
4. **Custom validation**: Extend for business rules
5. **Performance**: Consider query complexity

Next Steps
----------

* :doc:`llm` - LLM integration patterns
* :doc:`production` - Production deployment
* :doc:`../spec/language` - Language specification