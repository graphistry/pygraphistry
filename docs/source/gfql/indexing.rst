.. _gfql-indexing:

Indexing Guide
==============

GFQL does not need indexes. Without them, every query scans your dataframes. Queries that
start from known nodes, such as "the neighbors of these 50 accounts", can run faster with
indexes. You build an index once, and later queries on the same graph use it
automatically. Results are the same with or without an index.

.. doc-test: skip

.. code-block:: python

   g = g.gfql("CREATE GFQL INDEX FOR edge_out_adj")   # build once
   g = g.gfql("CREATE GFQL INDEX FOR node_id")
   g.gfql("MATCH (m {id: 0})-[e]->(p) RETURN p")      # later queries from known nodes use it

You can also create the indexes in the same call as the query that uses them:

.. code-block:: python

   g.gfql("CREATE GFQL INDEX FOR edge_out_adj; CREATE GFQL INDEX FOR node_id; "
          "MATCH (m {id: 'a'})-[e]->(p) RETURN p")

Index kinds
-----------

There are four kinds. ``gfql_index_all()`` builds the first three.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Index
     - What it is and what it speeds up
   * - ``edge_out_adj``
     - Analogous to a foreign key index on the edge table's source column. A forward hop
       reads only the edges of its start nodes, instead of every edge.

       | Cypher: ``CREATE GFQL INDEX FOR edge_out_adj``
       | Python: ``g.create_index("edge_out_adj")``
       | JSON: ``{"type": "CreateIndex", "kind": "edge_out_adj"}``
   * - ``edge_in_adj``
     - Analogous to a foreign key index on the edge table's destination column. The same,
       for reverse hops. Undirected hops need both.

       | Cypher: ``CREATE GFQL INDEX FOR edge_in_adj``
       | Python: ``g.create_index("edge_in_adj")``
       | JSON: ``{"type": "CreateIndex", "kind": "edge_in_adj"}``
   * - ``node_id``
     - Analogous to a primary key index on the node table. Finds start nodes and result
       nodes by id without reading the whole node table. It needs unique node ids;
       ``gfql_index_all()`` skips it when the ids repeat.

       | Cypher: ``CREATE GFQL INDEX FOR node_id``
       | Python: ``g.create_index("node_id")``
       | JSON: ``{"type": "CreateIndex", "kind": "node_id"}``
   * - ``node_prop``
     - Analogous to an ordinary column index on the node table. Finds start nodes by a
       column other than the node id, such as an account number. You choose the columns.
       Only integer columns without nulls can be indexed today; queries on other columns
       still return the right rows through a scan.

       | Cypher: ``CREATE GFQL INDEX FOR node_prop ON (account_number)``
       | Python: ``g.create_index("node_prop", column="account_number")``
       | JSON: ``{"type": "CreateIndex", "kind": "node_prop", "column": "account_number"}``

Indexes do not change or copy your dataframes. ``g.show_indexes()`` lists each index and
its memory use (the ``nbytes`` column). GFQL builds an index only when you ask for one,
or when you set ``index_policy`` to ``'auto'`` or ``'force'`` (see `Controlling the planner`_). Index kinds and
column types that are not supported yet raise ``NotImplementedError``, with a link to the
issue that tracks them.

Quick start
-----------

A complete, runnable example:

.. code-block:: python

   import pandas as pd
   import graphistry

   # A small synthetic graph: 6 accounts, 8 transfers
   edges_df = pd.DataFrame({
       "src": [0, 0, 1, 1, 2, 3, 4, 5],
       "dst": [1, 2, 2, 3, 4, 4, 5, 0],
       "amount": [10, 20, 30, 40, 50, 60, 70, 80],
   })
   nodes_df = pd.DataFrame({
       "id": [0, 1, 2, 3, 4, 5],
       "risk": ["low", "high", "low", "high", "low", "low"],
   })
   g = graphistry.edges(edges_df, "src", "dst").nodes(nodes_df, "id")

   # Build once: adjacency over outgoing edges, plus the node-id lookup
   g_indexed = g.gfql("CREATE GFQL INDEX FOR edge_out_adj").gfql("CREATE GFQL INDEX FOR node_id")
   print(g_indexed.gfql("SHOW GFQL INDEXES")[["name", "kind", "key_col", "n_keys", "valid"]])

   # 1-hop from a known node: who did account 0 transfer to?
   out = g_indexed.gfql("MATCH (m {id: 0})-[e]->(p) RETURN p")
   print(sorted(out._nodes["p.id"].tolist()))        # [1, 2]

   # 2-hop from the same node
   out2 = g_indexed.gfql("MATCH (m {id: 0})-[e]->()-[f]->(p) RETURN p")
   print(sorted(out2._nodes["p.id"].tolist()))       # [2, 3, 4]

   # Same lookups as a graph pipeline: GRAPH { ... } keeps nodes AND edges, so the
   # result is a subgraph you can plot or keep querying, not a table of rows
   sub1 = g_indexed.gfql("GRAPH { MATCH (m {id: 0})-[e]->(p) }")
   print(sorted(sub1._nodes["id"].tolist()), len(sub1._edges))   # [0, 1, 2] 2
   sub2 = g_indexed.gfql("GRAPH { MATCH (m {id: 0})-[e]->()-[f]->(p) }")
   print(sorted(sub2._nodes["id"].tolist()), len(sub2._edges))   # [0, 1, 2, 3, 4] 5

   # With indexes switched off, the answer is the same
   out_scan = g_indexed.gfql("MATCH (m {id: 0})-[e]->(p) RETURN p", index_policy="off")
   assert sorted(out._nodes["p.id"].tolist()) == sorted(out_scan._nodes["p.id"].tolist())

   # Was the index used? gfql_explain says so
   assert g_indexed.gfql_explain("MATCH (m {id: 0})-[e]->(p) RETURN p")["used_index"]

   # A seed LIST takes the index too, in both forms
   out_many = g_indexed.gfql("MATCH (m)-[e]->(p) WHERE m.id IN [0, 3] RETURN p")
   print(sorted(out_many._nodes["p.id"].tolist()))   # [1, 2, 4]
   sub_many = g_indexed.gfql("GRAPH { MATCH (m)-[e]->(p) WHERE m.id IN [0, 3] }")
   print(sorted(sub_many._nodes["id"].tolist()), len(sub_many._edges))   # [0, 1, 2, 3, 4] 3
   assert g_indexed.gfql_explain("GRAPH { MATCH (m)-[e]->(p) WHERE m.id IN [0, 3] }")["used_index"]

Both forms use the index for a query from one known node, as ``gfql_explain`` reports.
A list of start nodes, written ``WHERE m.id IN [0, 3]``, uses the index too: in the
row-returning form, in the ``GRAPH { }`` form, and in the native ``is_in`` chain below.
A list that contains ``null``, or an ``IN`` inside ``OR`` or ``NOT``, is applied as a
filter after the match instead. The same hop as a native chain, and as a direct
``hop()`` call:

.. code-block:: python

   from graphistry import n, e_forward, is_in

   out_chain = g_indexed.gfql([n({"id": is_in([0])}), e_forward(), n()])
   hop_out = g_indexed.hop(nodes=pd.DataFrame({"id": [0]}), hops=2, direction="forward")
   print(sorted(hop_out._nodes["id"].tolist()))      # [0, 1, 2, 3, 4]

The lifecycle calls. Each returns a new ``g``, like the rest of the API:

.. code-block:: python

   g = g.gfql_index_all()               # out+in adjacency + node_id
   g = g.gfql_index_edges("forward")    # or one direction: 'forward'|'reverse'|'both'
   g = g.create_index("edge_out_adj")   # or one kind: 'edge_out_adj'|'edge_in_adj'|'node_id'
   g = g.gfql_index_node_props(["id"])  # property indexes on node columns
   g.show_indexes()                     # pandas DataFrame: kind, engine, ..., valid, usable, reason
   g = g.drop_index()                   # drop all (or drop_index("edge_out_adj"))

``gfql_index_all()`` skips the ``node_id`` index when node ids repeat. An explicit
``create_index("node_id")`` raises an error instead.

Start from a property column
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``node_id`` index covers the column bound as the node id. To start queries from a
different column, such as a business key, index that column with ``node_prop``:

.. code-block:: python

   g = g.gfql_index_all().gfql_index_node_props(["id"])   # skips columns it cannot index
   g = g.create_index("node_prop", column="id")           # or one column, raising if it cannot

   # the same with Cypher DDL
   g.gfql('CREATE GFQL INDEX FOR node_prop ON id')
   g = g.drop_index("node_prop", column="id")             # or drop_index("node_prop") for all

Only integer columns without nulls can be indexed today. When one query filters on
several indexed columns, GFQL starts from the most selective one and applies the other
filters to its matches, so results do not depend on which indexes exist.

What uses an index
------------------

These queries use a resident index automatically:

- **Cypher queries that start from known nodes**: ``MATCH (m {id: $x})-[e]->(p)``,
  ``WHERE m.id = $x``, and ``WHERE m.id IN [...]``, over one or more hops, returning
  nodes, properties, or several aliases.
- **The same queries in** ``GRAPH { }`` **form.**
- **Native chains that start from known ids**, such as
  ``[n({"id": is_in([...])}), e_forward(), n()]``.
- **Start nodes found by a property column** that has a ``node_prop`` index.
- **A direct** ``g.hop(nodes=..., hops=..., direction=...)``.

Queries that do not start from known nodes, such as a filter over every node, scan. To
check a specific query, use ``g.gfql_explain(query)`` (see `Controlling the planner`_).
When an index cannot serve a query, the scan answers it, so the slowest case is the
speed you had without the index.

Graph pipelines
~~~~~~~~~~~~~~~

In a multi-stage ``GRAPH { }`` query, stages that run on the indexed graph use its indexes.
A stage that runs on a graph derived earlier in the same query does not. The derived graph
has new tables, so the original indexes do not apply to it, and Cypher cannot yet build an
index on a derived graph inside the query. That stage scans. This is tracked in
`issue 2148 <https://github.com/graphistry/pygraphistry/issues/2148>`_.

In a pipeline, ``used_index`` is ``True`` when any stage used an index. To see which stages
did, read the ``steps`` list that ``gfql_explain`` returns.

The native Python API can index a derived graph inside a ``let()`` pipeline. The index is
built again on every call, so this helps only when several later stages reuse it:

.. code-block:: python

   from graphistry import n, e_forward, is_in, call, let, ref

   pipeline = let({
       "low": [n({"risk": "low"}), e_forward(), n({"risk": "low"})],          # derived graph
       "low_idx": ref("low", [call("create_index", {"kind": "edge_out_adj"}),
                              call("create_index", {"kind": "node_id"})]),   # index it
       "out": ref("low_idx", [n({"id": is_in([0, 2])}), e_forward(), n()]),   # later stage uses it
   })
   out = g_indexed.gfql(pipeline)
   print(sorted(out._nodes["id"].tolist()))   # [0, 2, 4]

When indexes go stale
---------------------

An index serves a graph only while that graph has the tables the index was built from:

- If you bind new edges with ``.edges(...)``, the edge indexes become invalid. If you
  bind new nodes with ``.nodes(...)``, the node indexes become invalid. GFQL ignores an
  invalid index.
- If you change a bound table in place, rebuild its indexes. GFQL may not detect
  in-place edits.
- ``g.show_indexes()`` shows this in the ``valid`` column. The ``usable`` column also
  checks that the index was built for the engine the query will run on; when it is
  ``False``, ``reason`` says why. ``show_indexes(engine=...)`` checks a specific engine.
- To rebuild, call ``gfql_index_all()`` again on the new graph.

.. doc-test: skip

.. code-block:: python

   new_edges_df = edges_df.assign(amount=edges_df["amount"] + 1)
   g2 = g_indexed.edges(new_edges_df, "src", "dst")
   g2.show_indexes()          # edge_out_adj now valid=False; node_id still True
   g2 = g2.gfql_index_all()   # build again for the new edges; all valid=True

Results are the same with or without an index, whether it is missing, invalid, or does
not cover the query. An index changes speed only.

.. note::
   Index kinds and ``show_indexes()`` columns may change between releases. These rules
   stay: you build an index once, queries reuse it automatically, binding new tables
   makes it invalid, and results do not depend on it.

Engines
-------

``gfql_index_all()`` builds indexes for the engine of your dataframes: pandas, cuDF on the
GPU, or Polars. An index built for Polars also serves Polars-GPU queries. Pass
``engine=...`` to build for a different engine.

.. _gfql-adjacency-index:

When indexes help
-----------------

Building an index takes time and memory once, and the time grows with the number of
edges. After that, a query from known nodes reads only the edges of those nodes, so its
cost depends mainly on their neighborhood, not on the size of the graph.

Indexes help most when you:

- **start from specific nodes**, such as a watchlist, a session, or the known members of a
  fraud ring, and hop out one to three steps;
- **run many such queries** against the same graph, so one build serves all of them;
- **need interactive latency** for neighbor expansion.

Queries over the whole graph, such as a filter on every node or PageRank, do not get
faster. For those, choose an engine instead; see :doc:`engines`.

Index DDL
~~~~~~~~~

The DDL forms are ``CREATE GFQL INDEX [name] [IF NOT EXISTS] FOR <kind> [ON (col)]``,
``DROP GFQL INDEX name [IF EXISTS]`` (or ``DROP GFQL INDEX [IF EXISTS] FOR <kind> [ON (col)]``),
and ``SHOW GFQL INDEXES``. The ``GFQL`` token separates them from standard Cypher
``CREATE INDEX``. The optional parts use the Cypher spelling; the older GFQL spellings
(``ON col`` without parentheses, ``IF EXISTS`` before the name) still work. The JSON wire
protocol carries the same ops (``{"type": "CreateIndex", ...}``) and ``index_policy``, so a
``gfql_remote`` call can include them.

In the native Python API, index ops can lead a chain or form a ``let()`` binding:

.. code-block:: python

   from graphistry import n, e_forward, is_in, call, let, ref
   from graphistry.compute.gfql.index.wire import CreateIndex
   g.gfql([CreateIndex("edge_out_adj"), CreateIndex("node_id"), n({"id": is_in([0, 3])}), e_forward(), n()])
   g.gfql([call("create_index", {"kind": "edge_out_adj"}), n({"id": 0}), e_forward(), n()])
   g.gfql(let({"indexed": [CreateIndex("edge_out_adj")], "out": ref("indexed", [n({"id": 0}), e_forward(), n()])}))

Controlling the planner
~~~~~~~~~~~~~~~~~~~~~~~

``gfql(..., index_policy=...)`` decides whether a resident index is used:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``index_policy``
     - Behavior
   * - ``'use'`` *(default)*
     - Use a resident index when one covers the query. Never build one. No cost when no
       index exists.
   * - ``'auto'``
     - Build an index during the query when the planner predicts it pays off (a selective
       set of start nodes).
   * - ``'force'``
     - Require the index path (useful to check that it is used).
   * - ``'off'``
     - Ignore indexes and scan.

``g.gfql_explain(query, index_policy=...)`` shows whether the index path was taken. It
returns ``used_index`` (bool), ``resident_indexes``, a per-step ``steps`` trace, the
planner's ``decision_reason`` (text), and a stable ``decision_code`` for programs and
tests:

- ``index_selected`` -- the index served the query.
- ``policy_off`` -- ``index_policy='off'``.
- ``no_resident_index`` -- no index exists, and ``index_policy='use'`` never builds one.
- ``index_path_unavailable`` -- the index could not serve this query, so the scan
  answered; ``decision_reason`` says what was missing (for example ``index_missing``).
- ``not_index_coverable`` -- indexes do not cover this query shape; it scans.
- ``missing_graph_columns`` -- the graph lacks a bound column the index needs.
- ``index_build_declined`` -- ``index_policy='auto'`` or ``'force'`` ended with no usable
  index.
- ``scan_cost`` -- an index exists, but the planner chose the scan (for example, the start
  nodes cover most of the graph).
- ``engine_mismatch`` -- the index was built for another engine.
- ``col_stats_absent`` / ``col_stats_stale`` / ``col_stats_insufficient`` /
  ``col_stats_served`` -- the column statistics (below) are missing, out of date, not
  enough for this query, or used to answer it.

Column statistics
~~~~~~~~~~~~~~~~~

``gfql_index_col_stats()`` records the minimum, maximum, and null count of the node id
and edge endpoint columns. Only integer columns are recorded today. Some counting
queries use these numbers to skip work. They never change answers: when they are
missing or out of date, the query does the full work. They become invalid under the same
rules as indexes, and ``gfql_index_all()`` builds them. Pass ``node_columns=`` or
``edge_columns=`` to record other integer columns. A column you name that cannot be
recorded raises an error; the default columns are skipped without one.

See also
--------

- :doc:`engines` — choosing pandas / Polars / cuDF / Polars-GPU.
- :doc:`performance` — GFQL measured against graph databases.
