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
       properties=pa.schema([pa.field("since", pa.int32(), nullable=False)]),
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

``NodeType.to_arrow()`` and ``EdgeType.to_arrow()``
  Export declarations as ``pyarrow.Schema`` objects through GFQL's row-schema
  bridge. Label/type columns are included by default so exports line up with
  the table columns used by binder/preflight validation.

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

This lane exposes declaration, Arrow row-schema export, and binder/preflight
integration. Inference from existing plottables, Arrow import/coercion at
plottable boundaries, schema effects for graph-growing calls, and remote schema
transport remain separate follow-on surfaces.
