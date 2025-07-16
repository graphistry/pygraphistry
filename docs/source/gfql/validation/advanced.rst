Advanced GFQL Validation Patterns
=================================

Deep dive into complex GFQL validation scenarios, performance considerations, and advanced patterns using the built-in validation system.

.. note::
   Run the interactive examples yourself in 
   `demos/gfql/gfql_validation_fundamentals.ipynb <https://github.com/graphistry/pygraphistry/blob/master/demos/gfql/gfql_validation_fundamentals.ipynb>`_.

Prerequisites
-------------

* Complete :doc:`fundamentals` first
* Experience writing GFQL queries
* Understanding of graph traversal concepts

Complex Multi-Hop Queries
-------------------------

GFQL automatically validates complex queries during construction, catching errors early in multi-hop traversal patterns.

.. code-block:: python

   from graphistry.compute.chain import Chain
   from graphistry.compute.ast import n, e_forward
   from graphistry.compute.predicates.numeric import gt
   from graphistry.compute.predicates.str import eq
   from graphistry.compute.exceptions import GFQLValidationError

   # Multi-hop with bounded traversal - validates automatically
   try:
       chain = Chain([
           n({'type': eq('user')}),
           e_forward(hops=2),  # 2-hop traversal
           n({'risk_score': gt(50)})
       ])
       print("✅ Complex query validated successfully")
   except GFQLValidationError as e:
       print(f"❌ Validation failed: [{e.code}] {e.message}")

Named Operations
^^^^^^^^^^^^^^^^

Use named operations for complex patterns with automatic validation:

.. code-block:: python

   from graphistry.compute.predicates.str import eq

   # Named operations with automatic validation
   try:
       chain = Chain([
           n({'type': eq('user')}, name='start_users'),
           e_forward({'rel_type': eq('purchased')}),
           n(name='products'),
           e_reverse({'rel_type': eq('viewed')}),
           n(name='viewers')
       ])
       
       # Execute with schema validation
       result = g.chain(chain)  # validate_schema=True by default
       
       # Access named results
       start_users = result._nodes[result._nodes['start_users']]
       products = result._nodes[result._nodes['products']]
       
   except GFQLValidationError as e:
       print(f"Error in named operations: [{e.code}] {e.message}")

Advanced Predicates
-------------------

Temporal Predicates
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   from graphistry.compute.predicates.temporal import after
   from graphistry.compute.exceptions import GFQLTypeError

   # Temporal validation with proper datetime handling
   try:
       chain = Chain([
           n({'created_at': after(pd.Timestamp('2024-01-10T00:00:00Z'))})
       ])
   except GFQLTypeError as e:
       if e.code == 'E203':  # Invalid datetime format
           print(f"Use pd.Timestamp: {e.context.get('suggestion')}")

Nested Predicates
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from graphistry.compute.predicates.logical import and_, or_
   from graphistry.compute.predicates.str import is_in, contains
   from graphistry.compute.predicates.numeric import gte

   # Complex nested predicates with validation
   try:
       chain = Chain([
           n(and_(
               {'type': is_in(['user', 'payment'])},
               or_(
                   {'risk_score': gte(75)},
                   {'tags': contains('urgent')}
               )
           ))
       ])
   except GFQLValidationError as e:
       print(f"Nested predicate error: [{e.code}] {e.message}")

Performance Considerations
--------------------------

Bounded vs Unbounded Hops
^^^^^^^^^^^^^^^^^^^^^^^^^

GFQL validation warns about performance issues with unbounded traversals:

.. code-block:: python

   from graphistry.compute.exceptions import GFQLTypeError

   # Good - bounded hops
   try:
       chain = Chain([n(), e_forward(hops=3)])  # ✅ Explicit hop limit
   except GFQLTypeError as e:
       # Won't trigger - valid configuration
       pass

   # Warning - unbounded hops (still valid, but may be slow)
   chain = Chain([n(), e_forward()])  # ⚠️ No hop limit - validate manually

Pre-execution Validation
^^^^^^^^^^^^^^^^^^^^^^^

Use pre-execution validation to catch performance issues early:

.. code-block:: python

   from graphistry.compute.validate_schema import validate_chain_schema

   # Validate schema before expensive execution
   chain = Chain([n(), e_forward(hops=5)])  # Syntax validated
   
   # Pre-validate against actual data
   try:
       validate_chain_schema(g, chain, collect_all=False)
       print("✅ Schema validation passed")
   except GFQLSchemaError as e:
       print(f"❌ Schema issue: [{e.code}] {e.message}")
       # Handle before expensive execution

Query Complexity Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Monitor query complexity using collect-all validation:

.. code-block:: python

   # Get all validation issues at once
   errors = chain.validate(collect_all=True)
   
   # Count different error types
   syntax_errors = [e for e in errors if e.code.startswith('E1')]
   performance_warnings = [e for e in errors if 'performance' in e.message.lower()]
   
   print(f"Performance concerns: {len(performance_warnings)}")

Schema Evolution
----------------

Handle schema changes gracefully with structured error handling:

.. code-block:: python

   from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError

   def create_compatible_query(operations, g, column_mapping=None):
       """Update query to handle schema changes."""
       try:
           # Try original query first
           return g.chain(operations)
       except GFQLSchemaError as e:
           if e.code == ErrorCode.E301:  # Column not found
               missing_col = e.context.get('field')
               if column_mapping and missing_col in column_mapping:
                   # Update operations to use new column name
                   updated_ops = map_column_names(operations, column_mapping)
                   return g.chain(updated_ops)
           raise  # Re-raise if can't handle

Custom Validation
-----------------

Extend validation for domain-specific requirements:

.. code-block:: python

   from graphistry.compute.exceptions import GFQLValidationError, ErrorCode

   class BusinessRuleValidator:
       def __init__(self, sensitive_columns=None):
           self.sensitive_columns = sensitive_columns or []
       
       def validate_business_rules(self, chain, collect_all=False):
           """Add custom business rule validation."""
           errors = []
           
           # Check for sensitive columns without filters
           for op in chain.chain:
               if hasattr(op, 'filter') and op.filter:
                   for col in op.filter.keys():
                       if col in self.sensitive_columns:
                           errors.append(GFQLValidationError(
                               'B001',  # Custom business rule code
                               f'Sensitive column "{col}" requires additional approval',
                               field=col,
                               suggestion='Contact security team for approval'
                           ))
                           if not collect_all:
                               raise errors[0]
           
           return errors if collect_all else None

   # Usage
   validator = BusinessRuleValidator(sensitive_columns=['ssn', 'credit_card'])
   business_errors = validator.validate_business_rules(chain, collect_all=True)

Best Practices
--------------

1. **Built-in validation**: Let GFQL automatically validate during construction
2. **Multi-hop queries**: Always specify hop limits for performance
3. **Error handling**: Use structured error codes for programmatic responses
4. **Pre-execution validation**: Validate schema before expensive operations
5. **Collect-all mode**: Use for comprehensive error reporting in development
6. **Custom validation**: Extend with domain-specific business rules
7. **Schema evolution**: Handle column changes with graceful error recovery

Next Steps
----------

* :doc:`llm` - LLM integration patterns
* :doc:`production` - Production deployment
* :doc:`../spec/language` - Language specification