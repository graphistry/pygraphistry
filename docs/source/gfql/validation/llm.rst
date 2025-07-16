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

JSON Serialization
------------------

Convert validation results to structured formats for LLMs:

.. code-block:: python

   from graphistry.compute.exceptions import GFQLValidationError

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

   # Usage with collect-all mode
   from graphistry.compute.chain import Chain
   
   chain = Chain(operations)
   errors = chain.validate(collect_all=True)
   
   serialized_errors = [validation_error_to_dict(error) for error in errors]

Error Categorization
--------------------

Prioritize fixes for LLM processing using error codes:

.. code-block:: python

   from graphistry.compute.exceptions import ErrorCode

   def categorize_errors(errors):
       """Categorize errors by severity for LLM processing."""
       categories = {
           "critical": [],    # Must fix - syntax errors (E1xx)
           "important": [],   # Should fix - type errors (E2xx)
           "data_issues": []  # Schema errors (E3xx)
       }
       
       for error in errors:
           error_dict = validation_error_to_dict(error)
           
           if error.code.startswith('E1'):
               categories["critical"].append(error_dict)
           elif error.code.startswith('E2'):
               categories["important"].append(error_dict)
           elif error.code.startswith('E3'):
               categories["data_issues"].append(error_dict)
       
       return categories

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
           if error.code == ErrorCode.E103:  # Invalid parameter value
               fix["action"] = "replace_parameter"
               fix["new_value"] = error.context.get("valid_range")
           elif error.code == ErrorCode.E301:  # Column not found
               fix["action"] = "replace_column"
               fix["available_columns"] = error.context.get("available_columns")
           
           fixes.append(fix)
       
       return fixes

LLM Integration Pipeline
------------------------

.. code-block:: python

   from graphistry.compute.chain import Chain
   from graphistry.compute.exceptions import GFQLValidationError
   from graphistry.compute.validate_schema import validate_chain_schema

   class GFQLValidationPipeline:
       def __init__(self, plottable_graph=None, max_iterations=3):
           self.graph = plottable_graph  # For schema validation
           self.max_iterations = max_iterations
       
       def validate_and_report(self, operations):
           """Comprehensive validation with LLM-friendly reporting."""
           report = {
               "valid": True,
               "syntax_errors": [],
               "schema_errors": [],
               "fixes": []
           }
           
           try:
               # Syntax validation (automatic)
               chain = Chain(operations)
               syntax_errors = chain.validate(collect_all=True)
               
               if syntax_errors:
                   report["valid"] = False
                   report["syntax_errors"] = [validation_error_to_dict(e) for e in syntax_errors]
               
               # Schema validation if graph provided
               if self.graph:
                   try:
                       validate_chain_schema(self.graph, operations, collect_all=False)
                   except GFQLValidationError as e:
                       report["valid"] = False
                       report["schema_errors"] = [validation_error_to_dict(e)]
               
               # Generate fix suggestions
               all_errors = syntax_errors + report.get("schema_errors", [])
               report["fixes"] = generate_fix_suggestions(all_errors)
               
           except Exception as e:
               report["valid"] = False
               report["error"] = str(e)
           
           return report
       
       def create_llm_prompt(self, report, operations):
           """Format validation feedback for LLM consumption."""
           if report["valid"]:
               return "Query is valid."
           
           prompt_parts = ["The GFQL query has the following issues:\n"]
           
           # Add syntax errors
           for error in report["syntax_errors"]:
               prompt_parts.append(f"- SYNTAX ERROR [{error['code']}]: {error['message']}")
               if error.get("suggestion"):
                   prompt_parts.append(f"  Suggestion: {error['suggestion']}")
           
           # Add schema errors
           for error in report["schema_errors"]:
               prompt_parts.append(f"- SCHEMA ERROR [{error['code']}]: {error['message']}")
               if error.get("suggestion"):
                   prompt_parts.append(f"  Suggestion: {error['suggestion']}")
           
           prompt_parts.append("\nPlease fix these issues and return a corrected GFQL query.")
           return "\n".join(prompt_parts)

Prompt Engineering
------------------

System Prompt Template
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   You are a GFQL expert. Generate valid GFQL queries using the built-in validation system.
   
   GFQL Rules:
   1. Use Chain() constructor with list of operations
   2. Valid operations: n(), e_forward(), e_reverse(), e_undirected()
   3. Use predicate functions: eq(), gt(), contains(), is_in(), etc.
   4. Complex filters use and_(), or_() predicates
   5. Schema validation happens automatically with validate_schema=True (default)
   
   Available columns:
   Nodes: [id, name, type, score]
   Edges: [src, dst, weight]
   
   Error Codes:
   - E1xx: Syntax errors (structure, parameters)
   - E2xx: Type errors (wrong value types)
   - E3xx: Schema errors (missing columns, type mismatches)

Iterative Refinement
--------------------

.. code-block:: python

   def refine_query_with_llm(operations, pipeline, llm_client):
       """Iteratively refine GFQL query using validation feedback."""
       
       for iteration in range(pipeline.max_iterations):
           report = pipeline.validate_and_report(operations)
           
           if report["valid"]:
               return operations, report
           
           # Create LLM prompt with validation feedback
           prompt = pipeline.create_llm_prompt(report, operations)
           
           # Get LLM response
           response = llm_client.generate(prompt)
           
           # Parse new operations from LLM response
           try:
               operations = parse_operations_from_llm(response)
           except Exception as e:
               print(f"Failed to parse LLM response: {e}")
               break
       
       return operations, report

   # Usage example
   initial_operations = [n({'type': 'user'}), e_forward(hops=-1)]  # Invalid hops
   
   pipeline = GFQLValidationPipeline(plottable_graph=g)
   refined_ops, final_report = refine_query_with_llm(initial_operations, pipeline, llm_client)
   
   if final_report["valid"]:
       result = g.chain(refined_ops)
   else:
       print("Could not generate valid query after refinement")

Best Practices
--------------

1. **Built-in Validation**: Use GFQL's automatic validation during construction
2. **Error Codes**: Leverage structured error codes (E1xx, E2xx, E3xx) for programmatic handling
3. **Collect-All Mode**: Use ``collect_all=True`` for comprehensive error reporting to LLMs
4. **Schema Context**: Provide available columns and types in LLM prompts
5. **Iterative Approach**: Allow multiple refinement rounds with validation feedback
6. **Pre-execution Validation**: Validate schema before expensive operations
7. **Rate Limiting**: Implement for production APIs

Integration Checklist
---------------------

* ✅ Use structured error codes for LLM consumption
* ✅ Implement collect-all validation mode
* ✅ Create iterative validation pipeline with built-in validation
* ✅ Provide schema context in prompts
* ✅ Handle both syntax and schema validation
* ✅ Log validation metrics and fix success rates
* ✅ Implement graceful error recovery

Next Steps
----------

* Integrate with real LLM providers (OpenAI, Anthropic)
* Build production validation pipelines
* Create domain-specific templates
* Monitor generation accuracy

See Also
--------

* :doc:`production` - Production patterns
* :doc:`../spec/language` - Language specification
* :doc:`../spec/cypher_mapping` - Cypher to GFQL mapping