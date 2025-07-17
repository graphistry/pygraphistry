GFQL Validation in Production
=============================

Production-ready patterns for GFQL built-in validation in platform engineering and DevOps contexts.

.. note::
   See complete implementation examples in 
   `demos/gfql/gfql_validation_fundamentals.ipynb <https://github.com/graphistry/pygraphistry/blob/master/demos/gfql/gfql_validation_fundamentals.ipynb>`_.

Target Audience
---------------

* Platform Engineers
* DevOps Teams  
* Backend Developers
* System Architects

Performance & Caching
---------------------

Schema Caching
^^^^^^^^^^^^^^

.. code-block:: python

   from functools import lru_cache
   import time
   
   class CachedSchemaValidator:
       def __init__(self, cache_size=1000, ttl_seconds=3600):
           self.cache_size = cache_size
           self.ttl_seconds = ttl_seconds
           self._cache = {}
           self._cache_times = {}
           
           # Cache the actual validation function
           self._validate_uncached = lru_cache(maxsize=cache_size)(
               self._validate_impl
           )
       
       def _validate_impl(self, operations_hash, plottable_id):
           """Actual validation implementation."""
           # Implementation depends on having stable plottable references
           pass
       
       def validate(self, operations, plottable):
           """Validate with caching."""
           # Use built-in validation with caching layer
           cache_key = (hash(str(operations)), id(plottable))
           
           if cache_key in self._cache:
               cache_time = self._cache_times.get(cache_key, 0)
               if time.time() - cache_time < self.ttl_seconds:
                   return self._cache[cache_key]
           
           # Perform validation
           validator = PlottableValidator(plottable)
           result = validator.validate(operations, collect_all=True)
           
           # Cache result
           self._cache[cache_key] = result
           self._cache_times[cache_key] = time.time()
           
           return result

Batch Validation
^^^^^^^^^^^^^^^^

.. code-block:: python

   def batch_validate_queries(operation_sets, plottable):
       """Validate multiple queries efficiently with built-in validation."""
       validator = PlottableValidator(plottable)
       
       results = []
       for operations in operation_sets:
           try:
               errors = validator.validate(operations, collect_all=True)
               results.append({
                   "valid": len(errors) == 0,
                   "errors": [
                       {
                           "code": e.code,
                           "message": e.message,
                           "field": e.context.get("field"),
                           "suggestion": e.context.get("suggestion")
                       }
                       for e in errors
                   ]
               })
           except Exception as e:
               results.append({
                   "valid": False,
                   "error": str(e)
               })
       
       return results

Testing Patterns
----------------

pytest Fixtures
^^^^^^^^^^^^^^^

.. code-block:: python

   import pytest
   import pandas as pd
   import graphistry
   from graphistry.compute.chain import Chain
   from graphistry.compute.ast import n, e_forward
   from graphistry.compute.predicates.str import eq
   from graphistry.compute.exceptions import GFQLValidationError

   @pytest.fixture
   def sample_plottable():
       nodes = pd.DataFrame({
           'id': [1, 2, 3],
           'type': ['A', 'B', 'A']
       })
       edges = pd.DataFrame({
           'src': [1, 2],
           'dst': [2, 3]
       })
       g = graphistry.nodes(nodes, node='id').edges(edges, source='src', destination='dst')
       return g
   
   def test_valid_query(sample_plottable):
       operations = [n({'type': eq('A')})]
       
       # Test syntax validation
       chain = Chain(operations)  # Should not raise
       
       # Test schema validation
       result = sample_plottable.chain(operations)  # Should not raise
       assert len(result._nodes) > 0
   
   def test_invalid_query_syntax(sample_plottable):
       with pytest.raises(GFQLValidationError) as exc_info:
           chain = Chain([n({'type': eq('A')}, name=123)])  # Invalid name type
       assert exc_info.value.code.startswith('E2')  # Type error
   
   def test_invalid_query_schema(sample_plottable):
       operations = [n({'missing_column': eq('value')})]
       
       with pytest.raises(GFQLValidationError) as exc_info:
           result = sample_plottable.chain(operations)  # Schema validation fails
       assert exc_info.value.code == 'E301'  # Column not found

API Integration
---------------

Flask Example
^^^^^^^^^^^^^

.. code-block:: python

   from flask import Flask, request, jsonify
   from graphistry.compute.chain import Chain
   from graphistry.compute.exceptions import GFQLValidationError
   from graphistry.compute.ast import from_json
   
   app = Flask(__name__)
   
   @app.route('/api/v1/validate', methods=['POST'])
   def validate_gfql():
       data = request.get_json()
       operations_json = data.get('operations')
       
       try:
           # Parse operations from JSON
           operations = [from_json(op) for op in operations_json]
           
           # Validate syntax (automatic during Chain construction)
           chain = Chain(operations)
           syntax_errors = chain.validate(collect_all=True)
           
           # Prepare response
           response = {
               'valid': len(syntax_errors) == 0,
               'errors': [
                   {
                       'code': e.code,
                       'message': e.message,
                       'field': e.context.get('field'),
                       'suggestion': e.context.get('suggestion')
                   }
                   for e in syntax_errors
               ]
           }
           
           return jsonify(response)
           
       except Exception as e:
           return jsonify({
               'valid': False,
               'error': str(e)
           }), 400
   
   @app.route('/api/v1/validate-with-schema', methods=['POST'])
   def validate_gfql_with_schema():
       data = request.get_json()
       operations_json = data.get('operations')
       plottable_data = data.get('plottable')  # Serialized plottable
       
       try:
           # Parse operations from JSON
           operations = [from_json(op) for op in operations_json]
           
           # Would need to reconstruct plottable from data
           # and use validate_chain_schema
           from graphistry.compute.validate_schema import validate_chain_schema
           
           # This is a placeholder - actual implementation would need
           # to deserialize plottable_data into a plottable instance
           # errors = validate_chain_schema(plottable, operations, collect_all=True)
           
           return jsonify({
               'valid': True,
               'message': 'Schema validation endpoint placeholder'
           })
           
       except Exception as e:
           return jsonify({
               'valid': False,
               'error': str(e)
           }), 500

Security Considerations
-----------------------

.. code-block:: python

   import time
   from collections import defaultdict
   from graphistry.compute.exceptions import GFQLValidationError

   class SecureValidator:
       def __init__(self, max_operations=50, rate_limit_per_minute=100):
           self.max_operations = max_operations
           self.rate_limit_per_minute = rate_limit_per_minute
           self.request_counts = defaultdict(list)
       
       def validate_secure(self, operations, user_id, plottable=None):
           """Validate with security checks."""
           current_time = time.time()
           
           # Check rate limit
           user_requests = self.request_counts[user_id]
           # Clean old requests (older than 1 minute)
           user_requests[:] = [t for t in user_requests if current_time - t < 60]
           
           if len(user_requests) >= self.rate_limit_per_minute:
               raise GFQLValidationError(
                   'S001',
                   f'Rate limit exceeded: {self.rate_limit_per_minute} requests per minute',
                   field='rate_limit',
                   suggestion=f'Wait {60 - (current_time - user_requests[0]):.1f} seconds'
               )
           
           # Check query size
           if len(operations) > self.max_operations:
               raise GFQLValidationError(
                   'S002',
                   f'Query too large: {len(operations)} operations (max: {self.max_operations})',
                   field='operations',
                   suggestion=f'Reduce query to {self.max_operations} operations or fewer'
               )
           
           # Record request
           user_requests.append(current_time)
           
           # Perform validation
           chain = Chain(operations)
           syntax_errors = chain.validate(collect_all=True)
           
           if plottable:
               from graphistry.compute.validate_schema import validate_chain_schema
               schema_errors = validate_chain_schema(plottable, operations, collect_all=True) or []
               return syntax_errors + schema_errors
           else:
               return syntax_errors

Production Checklist
--------------------

* **Built-in Validation**: Use GFQL's automatic validation system
* **Caching**: Implement validation result caching
* **Batch Processing**: Validate multiple queries efficiently
* **Testing**: Comprehensive test coverage with pytest
* **API Design**: RESTful endpoints with structured error responses
* **Security**: Rate limiting and operation count limits
* **Error Codes**: Use structured error codes for programmatic handling

Performance Guidelines
----------------------

1. **Schema Validation**: Use validate_schema=True (default) for production safety
2. **Pre-execution Validation**: Validate before expensive operations
3. **Caching**: Cache validation results with appropriate TTL
4. **Batch Processing**: Use collect_all=True for multiple error reporting
5. **Rate Limiting**: Set reasonable per-user request limits

Next Steps
----------

* Implement production validation service
* Create runbooks for common issues
* Establish performance benchmarks

See Also
--------

* :doc:`../spec/wire_protocol` - Wire protocol specification
* `PyGraphistry API Reference <https://docs.graphistry.com/api/>`_
* `Production Deployment Guide <https://docs.graphistry.com/deployment/>`_