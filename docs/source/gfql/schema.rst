Declarative Graph Schemas
=========================

GFQL accepts public schema declarations through the stable
``graphistry.schema`` import path. Use this when application code owns a graph
contract and wants Cypher preflight checks to fail before query execution.
The API is experimental in this release: the import path and core declaration
objects are intended to be stable, while coercion, remote transport, and
planner use are still follow-on surfaces. Inference is also experimental and
must be requested explicitly.

The schema is optional. When you provide one, PyGraphistry uses it as the
declared contract for local GFQL validation. When you do not provide one,
validation falls back to the columns already visible on the bound local
``nodes`` and ``edges`` dataframes. If neither a public schema nor local
dataframes are available, Cypher validation still parses and compiles the query,
but it cannot reject unknown labels or properties because there is no schema to
check against.

.. code-block:: python

   import graphistry
   import pandas as pd
   import pyarrow as pa
   from graphistry.schema import EdgeType, GraphSchema, NodeType

   Person = NodeType(
       "Person",
       pa.schema([
           pa.field("id", pa.int64(), nullable=False),
           pa.field("name", pa.large_string()),
       ]),
   )
   Company = NodeType(
       "Company",
       pa.schema([
           pa.field("id", pa.int64(), nullable=False),
           pa.field("name", pa.large_string()),
       ]),
   )
   WorksAt = EdgeType(
       "WORKS_AT",
       source=Person,
       destination=Company,
       properties=pa.schema([pa.field("since", pa.int64(), nullable=False)]),
   )

   schema = GraphSchema(
       node_types=[Person, Company],
       edge_types=[WorksAt],
       node_id_column="id",
       edge_source_column="src",
       edge_destination_column="dst",
   )

   nodes_df = pd.DataFrame({
       "id": [1, 2],
       "name": ["Ada", "Graphistry"],
       "label__Person": [True, False],
       "label__Company": [False, True],
   })
   edges_df = pd.DataFrame({
       "src": [1],
       "dst": [2],
       "since": [2024],
       "label__WORKS_AT": [True],
   })

   g = (
       graphistry
       .edges(edges_df, "src", "dst")
       .nodes(nodes_df, "id")
       .bind(schema=schema)
   )

   g.gfql_validate("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name")

Schema Objects
--------------

``NodeType(name, properties, labels=None)``
  Declares a node contract. ``labels`` defaults to ``(name,)`` and maps to the
  existing GFQL label-column convention ``label__<Label>``. ``properties``
  accepts a ``pyarrow.Schema``, a GFQL ``RowSchema``, or a mapping shorthand
  such as ``{"id": pa.int64(), "name": pa.large_string()}`` or
  ``{"id": int, "name": str}``. Arrow schemas are the preferred declaration
  path because they preserve dtype and nullability.

``EdgeType(name, source, destination, properties=None)``
  Declares an edge contract and topology. ``source`` and ``destination`` accept
  ``NodeType`` objects, label strings, or label iterables. Edge properties use
  the same Arrow-aligned schema inputs as node properties.

``GraphSchema(node_types, edge_types, strict=True, ...)``
  Groups node/edge contracts and adapts them to the internal
  ``GraphSchemaCatalog`` used by binder/preflight validation. ``strict=False``
  makes schema-bound ``g.gfql_validate(...)`` permissive by default; callers can
  still override per call with ``g.gfql_validate(..., strict=True)``.
  ``metadata`` is descriptive provenance for callers and exports; it is not part
  of validation semantics.

``NodeType.to_arrow()`` and ``EdgeType.to_arrow()``
  Export declarations as ``pyarrow.Schema`` objects through GFQL's row-schema
  bridge. Label/type columns are included by default so exports line up with
  the table columns used by binder/preflight validation.

``NodeType.from_arrow(...)`` and ``EdgeType.from_arrow(...)``
  Import explicit Arrow declarations back into public schema objects. This is
  declaration import, not inference: edge imports still require source and
  destination labels, and graph-level imports require named node/edge entries.

``GraphSchema.to_arrow()`` and ``GraphSchema.from_arrow(...)``
  Export/import a declaration payload containing per-node/per-edge Arrow
  schemas plus merged ``nodes`` and ``edges`` table schemas. The merged schemas
  are useful for dataframe boundary validation; the per-type entries preserve
  type names and edge topology.

What Preflight Checks
---------------------

When a schema is bound to a graph, Cypher preflight checks validate:

* node labels against declared node types,
* node and edge property names against declared properties,
* relationship types against declared edge types, and
* relationship source/destination labels against declared topology when the
  query provides enough label information.

Invalid queries raise ``GFQLValidationError`` with structured context.

This is a correctness and documentation surface first: applications can state
what labels, relationship types, properties, and topology they expect, then
validate user-authored or generated Cypher before running it. The same typed
contract is also used by inference and is the foundation for later coercion,
remote transport, and planner/performance work.

Arrow Boundary Validation
-------------------------

You can also opt into declared-schema checks at Arrow conversion and upload
boundaries. This is off by default so existing ``plot()``, ``upload()``, and
``to_arrow()`` calls keep their current behavior.

``schema_validate="strict"``
  Requires every declared node/edge schema column to exist and match the
  declared Arrow type. Non-nullable declared columns must not contain nulls.

``schema_validate="autofix"``
  Performs the same presence and non-null checks, and casts compatible columns
  to the declared Arrow type after normal Arrow conversion. Existing
  ``validate="autofix"`` mixed-type coercion still runs first.

.. code-block:: python

   # Debug a bound edge table against the schema.
   edges_arrow = g.to_arrow(schema_validate="strict")

   # Coerce compatible values such as string-encoded integers to the declared
   # Arrow type before local conversion. The same option is accepted by plot()
   # and upload().
   edges_arrow_autofix = g.to_arrow(schema_validate="autofix")

   # Validate the node table explicitly.
   nodes_arrow = g.validate_arrow_schema("nodes", validate="strict")

Provided vs. Inferred Schema
----------------------------

You can provide a schema directly or infer one from bound local data.

Use a provided schema when application code owns the contract:

.. code-block:: python

   declared_g = (
       graphistry
       .edges(edges_df, "src", "dst")
       .nodes(nodes_df, "id")
       .bind(schema=schema)
   )

Use inference when the graph data should define the first draft contract:

.. code-block:: python

   inferred_base_g = graphistry.edges(edges_df, "src", "dst").nodes(nodes_df, "id")
   inferred_schema = inferred_base_g.infer_schema()
   inferred_g = inferred_base_g.bind(schema=inferred_schema)

For one-step local binding, use:

.. code-block:: python

   inferred_g = (
       graphistry
       .edges(edges_df, "src", "dst")
       .nodes(nodes_df, "id")
       .bind(infer_schema=True)
   )

Inference is opt-in. ``graphistry.bind(...)`` and ``g.bind(...)`` do not infer a
schema unless ``infer_schema=True`` is passed.

Inference Rules
---------------

``graphistry.infer_schema(g)`` and ``g.infer_schema()`` return a public
``GraphSchema``. They inspect currently bound ``nodes`` and ``edges`` dataframes:

* Node types come from boolean ``label__<Label>`` columns on the node table.
* Edge types come from boolean ``label__<TYPE>`` columns on the edge table.
* Node properties are non-label node columns observed on rows for a label.
* Edge properties are non-label edge columns, excluding the bound source,
  destination, and edge-id columns.
* Source/destination topology is inferred when edges reference bound node ids
  and those nodes have label columns. Edge-only graphs keep edge types and
  properties, but do not invent endpoint labels.
* A remote-only graph such as ``graphistry.bind(dataset_id="...")`` has no
  local dataframe columns, so local validation is limited to syntax, compile,
  and structural checks unless you also bind a declared schema.

Inference uses the same Arrow/GFQL row-schema bridge as declared schemas for
logical property types. The returned ``GraphSchema`` can be passed to
``g.bind(schema=schema)`` and used by ``g.gfql_validate(...)``.

Inferred schemas include descriptive provenance:

.. code-block:: python

   inferred_schema.metadata["source"] == "inferred"

When declared definitions override inferred definitions through
``infer_schema(..., schema=schema)``, the returned schema uses
``metadata["source"] == "mixed"``. This metadata is informational; it does not
change preflight validation, Arrow validation, or schema equality.

Presence And Nullability
------------------------

The public ``GraphSchema`` stores the inferred logical type and scalar
nullability needed by validation. For more detail, request the experimental
report:

.. code-block:: python

   schema, report = g.infer_schema(return_report=True)

The report tracks property presence separately from type:

``required``
  The property has observed values on every row for that node label or edge
  type.

``optional``
  The property has observed values on some rows and nulls on other rows for
  that node label or edge type.

``maybe_absent``
  The column exists on the dataframe but has no observed value for that node
  label or edge type. This commonly means another label/type uses the column.

``unknown``
  No rows were available for that node label or edge type.

Declared Overrides
------------------

Declared schemas stay explicit. Passing ``schema=...`` to ``infer_schema()``
uses declared node and edge definitions in preference to inferred definitions
with the same names, while keeping inferred definitions for other names.

.. code-block:: python

   refined_schema = g.infer_schema(schema=schema)

``g.bind(schema=..., infer_schema=True)`` is rejected. Use either a provided
schema or inferred schema in a single bind call so the validation contract is
unambiguous.

Local vs. Remote GFQL
---------------------

The public schema is consumed by local validation APIs, including:

* ``g.gfql_validate("MATCH ...")``
* ``g.gfql(..., validate=True)``

``gfql_remote(...)`` is different. It compiles Cypher strings locally and sends
the resulting GFQL wire payload to the server, but this release does **not**
serialize a bound ``GraphSchema`` into remote GFQL requests. Remote execution
therefore still depends on the server-side dataset schema and GFQL support. If
you want declared schema checks before a remote call, run
``g.gfql_validate(query)`` locally first, then call ``g.gfql_remote(query)``.

Remote schema transport is planned as a follow-on after the local schema
contract and serialization boundary are stable.

Compatibility Notes
-------------------

The public import path is stable:

.. code-block:: python

   from graphistry.schema import NodeType, EdgeType, GraphSchema

Top-level imports are also available:

.. code-block:: python

   from graphistry import NodeType, EdgeType, GraphSchema

This lane exposes declaration, Arrow row-schema import/export, binder/preflight
integration, and opt-in Arrow boundary validation/coercion. Inference from
existing plottables, schema effects for graph-growing calls, and remote schema
transport remain separate follow-on surfaces.
