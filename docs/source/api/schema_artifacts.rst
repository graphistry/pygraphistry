Schema Artifacts
================

PyGraphistry ships structural JSON Schema artifacts for downstream tooling and LLM contract generation:

- ``schemas/encodings.schema.json``
- ``schemas/react-settings.schema.json``
- ``schemas/url-params.schema.json``

Generate or check them with:

.. code-block:: bash

   python -m graphistry.devschemas.export
   python -m graphistry.devschemas.export --check

These schemas are structural contracts. Runtime semantic validation remains owned by the existing PyGraphistry validators.
