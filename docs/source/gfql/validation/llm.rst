GFQL Validation for LLMs
========================

Learn how to integrate GFQL's built-in validation with Large Language Models and automation pipelines.

.. note::
   Explore the complete examples in 
   `demos/gfql/gfql_validation_fundamentals.ipynb <https://github.com/graphistry/pygraphistry/blob/master/demos/gfql/gfql_validation_fundamentals.ipynb>`_.

Target Audience
---------------

* AI/ML Engineers building GFQL generation systems
* Developers integrating LLMs with graph queries
* Teams building automated query generation pipelines

JSON Integration
----------------

GFQL queries use JSON for LLM integration:

.. code-block:: json

   {
       "type": "Chain",
       "chain": [
           {"type": "Node", "filter_dict": {"type": "user"}},
           {"type": "Edge", "direction": "forward", "hops": 2},
           {"type": "Node", "filter_dict": {"score": {"type": "GT", "val": 80}}}
       ]
   }

Validation Workflow
-------------------

Parse and Validate
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from graphistry.compute.exceptions import GFQLValidationError
   from graphistry.compute.chain import Chain

   def json_to_chain(json_data):
       """Parse JSON from LLM into query object."""
       try:
           return Chain.from_json(json_data, validate=True)
       except GFQLValidationError as e:
           # Handle parse errors
           return None, e

   def chain_to_json(chain):
       """Convert query to JSON for LLM training/examples."""
       return chain.to_json(validate=False)  # Already validated

   def validation_error_to_dict(error: GFQLValidationError) -> dict:
       """Convert validation error to LLM-friendly format."""
       return {
           "code": error.code,
           "message": error.message,
           "field": error.context.get("field"),
           "value": str(error.context.get("value")) if error.context.get("value") else None,
           "suggestion": error.context.get("suggestion"),
           "operation_index": error.context.get("operation_index"),
           "error_type": error.__class__.__name__
       }

Validate Query Syntax
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Validate syntax and structure
   syntax_errors = chain.validate(collect_all=True)
   
   if syntax_errors:
       print(f"Found {len(syntax_errors)} syntax errors")
       for error in syntax_errors:
           print(f"  [{error.code}] {error.message}")

Validate Against Schema
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from graphistry.compute.validate_schema import validate_chain_schema
   
   # Validate against actual data schema
   schema_errors = []
   if g:  # Your Plottable instance with data
       schema_errors = validate_chain_schema(g, chain, collect_all=True) or []
       
       if schema_errors:
           print(f"Found {len(schema_errors)} schema errors")
           for error in schema_errors:
               print(f"  [{error.code}] {error.message}")

Combined Validation
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Complete validation pipeline
   def validate_llm_query(json_data, graph=None):
       """Full validation with detailed feedback."""
       # Parse
       result = json_to_chain(json_data)
       if isinstance(result, tuple):
           return {"success": False, "parse_errors": [validation_error_to_dict(result[1])]}
       
       chain = result
       
       # Validate syntax
       syntax_errors = chain.validate(collect_all=True)
       
       # Validate schema if graph provided
       schema_errors = []
       if graph:
           schema_errors = validate_chain_schema(graph, chain, collect_all=True) or []
       
       # Return results
       if syntax_errors or schema_errors:
           return {
               "success": False,
               "syntax_errors": [validation_error_to_dict(e) for e in syntax_errors],
               "schema_errors": [validation_error_to_dict(e) for e in schema_errors]
           }
       
       return {"success": True, "chain": chain}

Automated Fix Suggestions
-------------------------

Generate actionable suggestions using structured error context:

.. code-block:: python

   def generate_fix_suggestions(errors):
       """Generate fix suggestions from validation errors."""
       fixes = []
       
       for error in errors:
           fix = {
               "error_code": error.code,
               "operation_index": error.context.get("operation_index"),
               "field": error.context.get("field"),
               "current_value": error.context.get("value"),
               "suggested_action": error.context.get("suggestion")
           }
           
           # Add specific fix actions based on error code
           if error.code == ErrorCode.E103:  # Invalid parameter value (e.g., negative hops)
               fix["action"] = "replace_parameter"
               # Extract valid value from suggestion if present
               if "positive integer" in error.message:
                   fix["fix_hint"] = "Use a positive integer value"
           elif error.code == ErrorCode.E301:  # Column not found
               fix["action"] = "replace_column"
               # Available columns are in the suggestion text
               if error.context.get("suggestion") and "Available columns:" in error.context.get("suggestion"):
                   fix["available_columns_hint"] = error.context.get("suggestion")
           elif error.code == ErrorCode.E302:  # Type mismatch
               fix["action"] = "fix_type_mismatch"
               fix["column_type"] = error.context.get("column_type")
           
           fixes.append(fix)
       
       return fixes

Best Practices
--------------

1. **Built-in Validation**: Use GFQL's automatic validation during construction
2. **Error Codes**: Leverage structured error codes (E1xx, E2xx, E3xx) for programmatic handling
3. **Collect-All Mode**: Use ``collect_all=True`` for comprehensive error reporting to LLMs
4. **Schema Context**: Provide available columns and types in LLM prompts
5. **Pre-execution Validation**: Validate schema before expensive operations

See Also
--------

* :doc:`production` - Production patterns
* :doc:`../spec/language` - Language specification
* :doc:`../spec/cypher_mapping` - Cypher to GFQL mapping