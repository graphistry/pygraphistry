GFQL Validation for LLMs
========================

Learn how to integrate GFQL validation with Large Language Models and automation pipelines.

.. note::
   Explore the complete examples in 
   `demos/gfql/gfql_validation_llm.ipynb <https://github.com/graphistry/pygraphistry/blob/master/demos/gfql/gfql_validation_llm.ipynb>`_.

Target Audience
---------------

* AI/ML Engineers building GFQL generation systems
* Developers integrating LLMs with graph queries
* Teams building automated query generation pipelines

JSON Serialization
------------------

Convert validation results to structured formats for LLMs:

.. code-block:: python

   def validation_issue_to_dict(issue):
       return {
           "level": issue.level,
           "message": issue.message,
           "operation_index": issue.operation_index,
           "suggestion": issue.suggestion
       }

Error Categorization
--------------------

Prioritize fixes for LLM processing:

.. code-block:: python

   categories = {
       "critical": [],    # Must fix - syntax errors
       "important": [],   # Should fix - schema errors  
       "suggested": []    # Nice to fix - warnings
   }

Automated Fix Suggestions
-------------------------

Generate actionable suggestions:

.. code-block:: python

   fixes = [
       {
           "action": "replace",
           "path": "[0].type",
           "old_value": "node",
           "new_value": "n"
       }
   ]

LLM Integration Pipeline
------------------------

.. code-block:: python

   class GFQLValidationPipeline:
       def __init__(self, schema=None, max_iterations=3):
           self.schema = schema
           self.max_iterations = max_iterations
       
       def validate_and_report(self, query):
           # Validate syntax and schema
           # Create comprehensive report
           # Generate fix suggestions
           pass
       
       def create_llm_prompt(self, report):
           # Format validation feedback for LLM
           pass

Prompt Engineering
------------------

System Prompt Template
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   You are a GFQL expert. 
   
   GFQL Rules:
   1. Queries are JSON arrays of operations
   2. Valid types: "n", "e_forward", "e_reverse", "e"
   3. Filters use operators: eq, ne, gt, gte, lt, lte
   4. Complex filters use _and, _or
   
   Available columns:
   Nodes: [id, name, type, score]
   Edges: [src, dst, weight]

Iterative Refinement
--------------------

.. code-block:: python

   for iteration in range(max_iterations):
       report = pipeline.validate_and_report(query)
       
       if report["valid"]:
           break
       
       # LLM fixes based on validation feedback
       query = llm.fix_query(query, report["fixes"])

Best Practices
--------------

1. **Structured Formats**: Always use JSON for LLM consumption
2. **Error Prioritization**: Fix critical → important → suggested
3. **Schema Context**: Provide available columns to LLMs
4. **Iterative Approach**: Allow multiple refinement rounds
5. **Rate Limiting**: Implement for production APIs

Integration Checklist
---------------------

* [✓] Serialize validation issues to JSON
* [✓] Implement fix suggestion generation
* [✓] Create iterative validation pipeline
* [✓] Provide schema context in prompts
* [✓] Handle rate limiting and retries
* [✓] Log validation metrics

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