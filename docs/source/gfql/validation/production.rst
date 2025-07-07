GFQL Validation in Production
=============================

Production-ready patterns for GFQL validation in platform engineering and DevOps contexts.

.. note::
   See complete implementation examples in 
   `demos/gfql/gfql_validation_production.ipynb <https://github.com/graphistry/pygraphistry/blob/master/demos/gfql/gfql_validation_production.ipynb>`_.

Target Audience
---------------

* Platform Engineers
* DevOps Teams  
* Backend Developers
* System Architects

Plottable Integration
---------------------

Seamlessly validate queries against Plottable objects:

.. code-block:: python

   from graphistry.compute.validate import extract_schema_from_plottable
   
   class PlottableValidator:
       def __init__(self, plottable):
           self.plottable = plottable
           self.schema = extract_schema_from_plottable(plottable)
       
       def validate(self, query):
           return validate_query(
               query,
               nodes_df=self.plottable._nodes,
               edges_df=self.plottable._edges
           )

Performance & Caching
---------------------

Schema Caching
^^^^^^^^^^^^^^

.. code-block:: python

   from functools import lru_cache
   
   class CachedSchemaValidator:
       def __init__(self, cache_size=1000, ttl_seconds=3600):
           self._schema_cache = {}
           self._query_cache = lru_cache(maxsize=cache_size)(
               self._validate_uncached
           )

Batch Validation
^^^^^^^^^^^^^^^^

.. code-block:: python

   def batch_validate_queries(queries, plottable):
       """Validate multiple queries efficiently."""
       schema = extract_schema_from_plottable(plottable)
       
       results = []
       for query in queries:
           issues = validate_query(query, plottable._nodes, plottable._edges)
           results.append({
               "valid": len(issues) == 0,
               "issues": issues
           })
       
       return results

Testing Patterns
----------------

pytest Fixtures
^^^^^^^^^^^^^^^

.. code-block:: python

   @pytest.fixture
   def sample_data():
       nodes = pd.DataFrame({
           'id': [1, 2, 3],
           'type': ['A', 'B', 'A']
       })
       edges = pd.DataFrame({
           'src': [1, 2],
           'dst': [2, 3]
       })
       return nodes, edges
   
   def test_valid_query(sample_data):
       nodes, edges = sample_data
       query = [{"type": "n", "filter": {"type": {"eq": "A"}}}]
       issues = validate_query(query, nodes, edges)
       assert len(issues) == 0

CI/CD Integration
-----------------

GitHub Actions
^^^^^^^^^^^^^^

.. code-block:: yaml

   name: GFQL Query Validation
   
   on:
     pull_request:
       paths:
         - 'queries/**/*.json'
   
   jobs:
     validate-queries:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Validate GFQL queries
           run: python scripts/validate_queries.py queries/

Pre-commit Hooks
^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # .pre-commit-config.yaml
   repos:
     - repo: local
       hooks:
         - id: validate-gfql
           name: Validate GFQL Queries
           entry: python scripts/validate_gfql_hook.py
           language: system
           files: '\.(json|py)$'

Monitoring & Logging
--------------------

.. code-block:: python

   class ValidationMonitor:
       def log_validation(self, query, issues, elapsed_ms, context=None):
           log_data = {
               "timestamp": datetime.utcnow().isoformat(),
               "validation_time_ms": elapsed_ms,
               "errors": len([i for i in issues if i.level == "error"]),
               "warnings": len([i for i in issues if i.level == "warning"]),
               "context": context or {}
           }
           
           if errors:
               logger.error("GFQL validation failed", extra=log_data)

API Integration
---------------

Flask Example
^^^^^^^^^^^^^

.. code-block:: python

   @app.route('/api/v1/validate', methods=['POST'])
   def validate_gfql():
       data = request.get_json()
       query = data.get('query')
       
       issues = validate_syntax(query)
       
       return jsonify({
           'valid': not any(i.level == 'error' for i in issues),
           'issues': [issue_to_dict(i) for i in issues]
       })

Security Considerations
-----------------------

.. code-block:: python

   class SecureValidator:
       def __init__(self, max_query_size=1000, rate_limit_per_minute=100):
           self.max_query_size = max_query_size
           self.rate_limit_per_minute = rate_limit_per_minute
       
       def validate_secure(self, query, user_id):
           # Check rate limit
           # Check query size
           # Sanitize query
           # Validate

Production Checklist
--------------------

* ✅ **Plottable Integration**: Use ``extract_schema_from_plottable()``
* ✅ **Caching**: Implement schema and query result caching
* ✅ **Batch Processing**: Validate multiple queries efficiently
* ✅ **Testing**: Comprehensive test coverage
* ✅ **CI/CD**: Automated validation in pipelines
* ✅ **Monitoring**: Track metrics and error patterns
* ✅ **API Design**: RESTful endpoints with error handling
* ✅ **Security**: Rate limiting and sanitization

Performance Guidelines
----------------------

1. Cache schemas with appropriate TTL
2. Use batch validation for multiple queries
3. Monitor p95 validation times
4. Set reasonable query size limits

Next Steps
----------

* Implement production validation service
* Set up monitoring dashboards
* Create runbooks for common issues
* Establish SLOs for validation performance

See Also
--------

* :doc:`../spec/wire_protocol` - Wire protocol specification
* `PyGraphistry API Reference <https://docs.graphistry.com/api/>`_
* `Production Deployment Guide <https://docs.graphistry.com/deployment/>`_