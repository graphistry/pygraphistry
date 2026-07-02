Declarative Graph Schemas
=========================

GFQL accepts public schema declarations through the stable
``graphistry.schema`` import path. Use this when application code owns a graph
contract and wants Cypher preflight checks to fail before query execution.
The API is experimental in this release: the import path and core declaration
objects are intended to be stable, while inference, coercion, remote transport,
and planner use are still follow-on surfaces.

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
   assert g.schema is schema
   assert g.schema is not None

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
  still override per call with ``g.gfql_validate(..., strict=True)``. A physical
  node property column must have the same logical type for every node type that
  declares it, and a physical edge property column must have the same logical
  type for every edge type that declares it. Use separate column names when two
  labels or relationship types need incompatible values under the same property
  name.

``g.schema``
  Read back the experimental ``GraphSchema`` bound with ``bind(schema=...)``.
  ``g.schema`` returns the bound object or ``None``. Use
  ``g.schema is not None`` when only a predicate is needed. Use
  ``bind(schema=...)`` to attach schemas, not assignment.
  This is local declaration introspection only. It does not infer schemas from
  data, fetch or hydrate remote dataset schemas, or serialize schemas into
  ``gfql_remote()`` requests in this release.

  Per-type declarations such as Cat, Dog, and Car are represented by
  ``GraphSchema.node_types``. The stable public type identity is
  ``NodeType.name``; ``NodeType.labels`` are the GFQL label predicates that map
  onto label columns such as ``label__Cat``. For example, Cat and Dog can both
  carry an ``Animal`` label while still preserving separate Cat and Dog
  property contracts.

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
  type names and edge topology. When the same column is declared with the same
  Arrow type but different nullability, merged table schemas mark that column as
  nullable while the per-type declarations keep their original nullability.

Nullability is type-local. If ``Cat.lives`` is declared non-nullable and
``House`` does not declare ``lives``, ``Cat.lives`` remains non-nullable in the
``Cat`` declaration. Boundary validation against a full fused node table still
accounts for which labels are active in each row.

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
contract is also the foundation for later inference, coercion, remote transport,
and planner/performance work, but this page covers the declared local contract.

Schema Effects
--------------

Some graph-growing GFQL calls add properties to an existing graph. For example,
``CALL graphistry.degree.write()`` adds degree columns to nodes, and
PageRank-style ``.write()`` procedures add score columns. When a graph has a
bound ``GraphSchema``, PyGraphistry now tracks those successful local effects
internally and attaches the updated schema snapshot to the returned graph:

.. code-block:: python

   enriched = g.gfql("CALL graphistry.degree.write()")
   enriched.gfql_validate("MATCH (n:Person) RETURN n.degree")

This is not a new public API surface. The effect model is internal while schema
inference, remote transport, and planner use continue to evolve. It is scoped to
local graph results with an explicitly bound schema; remote GFQL requests still
do not serialize schema snapshots or effect history.

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

In this release, schemas are **provided**, not inferred. You create
``NodeType``, ``EdgeType``, and ``GraphSchema`` objects directly and attach them
with ``graphistry.bind(..., schema=schema)`` or ``g.bind(schema=schema)``.

Without an explicit ``GraphSchema``:

* ``g.gfql_validate(...)`` can still use local dataframe columns already bound
  on ``g._nodes`` and ``g._edges`` for schema-aware checks.
* It does not infer node types, edge types, Arrow dtypes, nullability, or
  topology from data.
* A remote-only graph such as ``graphistry.bind(dataset_id="...")`` has no
  local dataframe columns, so local validation is limited to syntax, compile,
  and structural checks unless you also bind a declared schema.

Schema inference from existing plottables is tracked separately from this
declared-schema API.

Local vs. Remote GFQL
---------------------

The public schema is consumed by local validation APIs, including:

* ``g.gfql_validate("MATCH ...")``
* ``g.gfql(..., validate=True)``

``gfql_remote(...)`` is different. It compiles Cypher strings locally and sends
the resulting GFQL wire payload to the server, but this release does **not**
serialize a bound ``GraphSchema`` into remote GFQL requests. Remote execution
therefore still depends on the server-side dataset metadata and GFQL support. If
you want local declared-schema checks before a remote call, run
``g.gfql_validate(query)`` locally first, then call ``g.gfql_remote(query)``.

Remote schema transport is planned as a follow-on after the local schema
contract and serialization boundary are stable. The intended direction is a
versioned graph-schema envelope derived from ``GraphSchema.to_arrow()``: exact
Arrow schemas for merged node/edge tables and per-type declarations, plus a
JSON summary for dataset metadata, UI, and REST consumers. That future transport
should live beside ``gfql_query`` / ``gfql_operations`` rather than as fake data
tables.

Compatibility Notes
-------------------

The public import path is stable:

.. code-block:: python

   from graphistry.schema import NodeType, EdgeType, GraphSchema

Top-level imports are also available:

.. code-block:: python

   from graphistry import NodeType, EdgeType, GraphSchema

This lane exposes declaration, Arrow row-schema import/export, binder/preflight
integration, opt-in Arrow boundary validation/coercion, and internal local
schema-effect propagation for graph-growing calls. Inference from existing
plottables and remote schema transport remain separate follow-on surfaces.
