.. _gfql-indexing:

Indexing Guide: Build Once, Query Faster
========================================

GFQL runs without any indexes: every query is a vectorized scan over your dataframes.
When your workload is **seeded** — "expand from these 50 accounts", "look up this id and
hop out" — you can opt into **resident indexes**: build them once with one call, and
seeded queries reuse them automatically after that. This page is the user guide to that
lifecycle: what the indexes are, what engages them, when they go stale, and what they cost.
The :ref:`adjacency index <gfql-adjacency-index>` section below covers the planner
policy knobs, the Cypher DDL forms, and when the index engages.

.. doc-test: skip

.. code-block:: python

   g = g.gfql("CREATE GFQL INDEX FOR edge_out_adj")   # pay once ...
   g = g.gfql("CREATE GFQL INDEX FOR node_id")
   g.gfql("MATCH (m {id: 0})-[e]->(p) RETURN p")      # ... later lookups from a known node use it

The DDL can also travel with the query that uses it, in one call — leading ``CREATE GFQL INDEX``
statements build first and the rest runs on the indexed graph:

.. code-block:: python

   g.gfql("CREATE GFQL INDEX FOR edge_out_adj; CREATE GFQL INDEX FOR node_id; "
          "MATCH (m {id: 'a'})-[e]->(p) RETURN p")

   # the same in the native API: index ops at the front of a chain, or as a let() binding
   from graphistry import n, e_forward, is_in, call, let, ref
   from graphistry.compute.gfql.index.wire import CreateIndex
   g.gfql([CreateIndex("edge_out_adj"), CreateIndex("node_id"), n({"id": is_in(["a", "b"])}), e_forward(), n()])
   g.gfql([call("create_index", {"kind": "edge_out_adj"}), n({"id": "a"}), e_forward(), n()])
   g.gfql(let({"indexed": [CreateIndex("edge_out_adj")], "out": ref("indexed", [n({"id": "a"}), e_forward(), n()])}))

What a resident index is
------------------------

``gfql_index_all()`` builds up to three sidecar structures and returns a new ``g``
carrying them:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Index
     - What it accelerates
   * - ``edge_out_adj``
     - CSR adjacency over outgoing edges: a forward hop becomes an ``O(degree)``
       positional gather instead of an ``O(E)`` scan over every edge.
   * - ``edge_in_adj``
     - The same for incoming edges (reverse hops; undirected needs both).
   * - ``node_id``
     - Sorted node-id lookup: seed-row and endpoint materialization become positional
       gathers instead of ``O(N)`` scans. Requires unique node ids —
       ``gfql_index_all()`` silently skips it otherwise (adjacency is still built).
   * - ``node_prop``
     - Sorted lookup on a node **property** column (a secondary index): a seed
       predicate like ``MATCH (m {id: 42})`` on a column that is not the node-id
       binding becomes a positional gather instead of an ``O(N)`` scan. Duplicate
       values are fine (all matching rows are gathered). String columns and
       integer, categorical, and timestamp columns are supported. Null property rows
       are excluded.
       Other dtypes decline to the scan. Opt-in per column.

   * - ``edge_prop``
     - Sorted lookup on an edge **property** column, such as a transaction or
       message id. Supported property equality and membership predicates gather
       matching edge rows; duplicate values are retained. Null property rows are
       excluded. Opt-in per column,
       like ``node_prop``.

They are **sidecars over row positions**: your ``.edges`` / ``.nodes`` frames are never
reordered or copied, and the resident footprint is visible per index via
``g.show_indexes()`` (the ``nbytes`` column). The model is **pay-as-you-go**: one
``O(E log E)`` build, spread over every later query from known nodes. Nothing is built
unless you ask.

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

   # Pay once: adjacency over outgoing edges, plus the node-id lookup
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

   # Decline safety: with indexes switched off, the SAME answer comes back
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

Both forms take the index path for a lookup from one known node, as ``gfql_explain``
reports. A seed *list* is written ``WHERE m.id IN [0, 3]`` and takes the index path in
the row-returning form, in the ``GRAPH { }`` form, and in the native ``is_in`` chain
below (a list holding ``null``, or an ``IN`` under ``OR`` / ``NOT``, stays a row filter).
The same hop as a native chain, and the direct ``hop()`` call:

.. code-block:: python

   from graphistry import n, e_forward, is_in

   out_chain = g_indexed.gfql([n({"id": is_in([0])}), e_forward(), n()])
   hop_out = g_indexed.hop(nodes=pd.DataFrame({"id": [0]}), hops=2, direction="forward")
   print(sorted(hop_out._nodes["id"].tolist()))      # [0, 1, 2, 3, 4]

The lifecycle calls, all returning a new ``g`` (functional style, like the rest of the
API):

.. code-block:: python

   g = g.gfql_index_all()               # out+in adjacency + node_id (the one-liner)
   g = g.gfql_index_edges("forward")    # or just one direction: 'forward'|'reverse'|'both'
   g = g.create_index("edge_out_adj")   # or one kind: 'edge_out_adj'|'edge_in_adj'|'node_id'
   g = g.gfql_index_node_props(["id"])  # secondary indexes on node property columns
   g.show_indexes()                     # pandas DataFrame: kind, engine, ..., valid, usable, reason
   g = g.drop_index()                   # drop all (or drop_index("edge_out_adj"))

Unlike ``gfql_index_all()``, an explicit ``create_index("node_id")`` **raises** on
non-unique node ids rather than skipping.

Seeding on a property (secondary index)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``node_id`` index covers the column bound as the node id. A query that seeds on
a *different* column — a business key such as ``MATCH (m {id: 42})`` when the graph
is keyed by something else — otherwise scans the whole node table to find its seed.
``node_prop`` indexes that column instead:

.. code-block:: python

   g = g.gfql_index_all().gfql_index_node_props(["id"])   # skips unindexable columns
   g = g.create_index("node_prop", column="id")           # or one column, raising if it cannot

   # equivalently over the Cypher DDL / JSON surfaces
   g.gfql('CREATE GFQL INDEX FOR node_prop ON id')
   g = g.drop_index("node_prop", column="id")             # or drop_index("node_prop") for all

When several indexed columns appear in one seed predicate, the planner gathers on the
**most selective** one (estimated for free from the index's own offsets) and applies
the remaining predicates to those candidates, so results never depend on which index
happens to be resident. As with every kind, a missing, stale, or cost-gated-out index
falls back to the scan.

Nullable integer keys
~~~~~~~~~~~~~~~~~~~~~

Integer property indexes exclude null rows and preserve the original row
positions of every non-null value. Nullable signed and unsigned integer storage
is supported without a floating-point conversion; an all-null integer column
builds an empty index. Null lookup predicates retain canonical filter semantics.

Categorical and timestamp keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Categorical columns keep their labels and ordering. Native category codes locate
all rows for a label; unused labels produce no rows. Ambiguous mixed-type query
lists use the canonical scan so inference and errors remain unchanged. Polars
categorical and enum columns use native text dictionaries.

Timestamp columns retain their dtype, unit, and timezone in the result. Non-null
physical timestamp keys locate candidates, and the original predicate determines
exact matches. Polars nanosecond storage uses microsecond candidate buckets to
cover its canonical Python datetime/string comparison casts; NumPy nanosecond
queries still receive the exact residual comparison. Temporal predicates without
a proven native key encoding use the scan, including Polars temporal membership.
Existing engine errors for incompatible timezones remain authoritative.

String business keys
~~~~~~~~~~~~~~~~~~~~

Emails, usernames, and external IDs can be indexed without changing the graph's
node-id binding. Text uses native string dictionaries and integer row-position
arrays; comparisons preserve exact text, including Unicode and empty strings.
Null text rows match no non-null key. Duplicate keys gather every matching row,
and the remaining query predicates are still applied.

.. code-block:: python

   accounts = pd.DataFrame({
       "id": range(400),
       "email": ["alice@example.test", "bob@example.test"]
           + [f"account-{i}@example.test" for i in range(398)],
   })
   g_accounts = graphistry.nodes(accounts, "id").create_index("node_prop", column="email")
   g_accounts.gfql("MATCH (p {email: 'alice@example.test'}) RETURN p")
   query = "MATCH (p) WHERE p.email IN ['alice@example.test', 'bob@example.test'] RETURN p"
   assert g_accounts.gfql_explain(query)["used_index"]

An edge property can seed a query in the same way:

.. code-block:: python

   transfers = pd.DataFrame({
       "src": range(400), "dst": range(1, 401), "txn_id": range(400),
   })
   g_transfers = graphistry.edges(transfers, "src", "dst").materialize_nodes()
   g_transfers = g_transfers.create_index("edge_prop", column="txn_id")
   query = "MATCH (a)-[e {txn_id: 7}]->(b) RETURN a, b"
   report = g_transfers.gfql_explain(query)
   assert report["used_index"] and report["decision_code"] == "index_selected"

   # DDL and per-column lifecycle are also available
   g_transfers = g_transfers.gfql("CREATE GFQL INDEX FOR edge_prop ON (txn_id)")
   g_transfers = g_transfers.drop_index("edge_prop", column="txn_id")

Equality and ``is_in`` edge filters use a live edge-property index on native
chains and Cypher queries. The most selective indexed column supplies candidate
rows; the full filter still applies, preserving row order and duplicate edges.
A dense lookup can be costed out; inspect ``gfql_explain`` for the actual decision.

What uses the index today
-------------------------

On 0.58.0, a resident index is consumed automatically by:

- **Cypher hops from one known node**: ``MATCH (m {id: $x})-[:T]->(p) RETURN p``, the
  ``WHERE m.id = $x`` spelling, and the single-alias **property RETURN** form
  (``RETURN p.a AS x, p.b``), typed or untyped. The seed lookup, frontier expansion, and
  endpoint materialization all become positional index gathers. A seed *list*
  (``WHERE m.id IN [...]``) takes the index path too.
- **Native chains** such as ``[n({"id": is_in([...])}), e_forward(), n(...)]``: check a
  given shape with ``g.gfql_explain(query)``, which reports ``used_index`` and the
  planner's decision.
- **Lookups by a property value**: the start filter may hit a *property* column (e.g.
  ``MATCH (m {id: $x})`` when the graph is bound on a different key column). The seed
  row falls back to a property scan, but the adjacency and endpoint gathers still
  engage — the common pattern of a synthetic key binding plus an ``id`` property
  filter is covered.
- **Direct** ``g.hop(nodes=..., hops=..., direction=...)`` — the ``O(degree)``
  gather path.

**Not yet covered**: the general Polars chain traversal — multi-hop and multi-alias
queries executed by the Polars chain engine take their scan/join path even with an
index resident. Coverage is decline-gated: anything the index path does not handle
falls back to the scan, so the worst case is the speed you already had.

Staleness and safety
--------------------

The validity contract is simple: **an index serves only while the frames it was built
over are unchanged** (checked by object identity plus a structural fingerprint at use
time). Consequences:

- Rebinding ``.edges(...)`` invalidates the edge adjacency indexes; rebinding
  ``.nodes(...)`` invalidates the node-id index. A stale index is treated as *absent* —
  skipped, never consulted.
- ``g.show_indexes()`` reports liveness in the ``valid`` column, so you can see at a
  glance whether a rebind knocked an index out.
- ``valid`` alone is not "this index will serve your query": indexes are also
  **engine-specific**. The ``usable`` column is True only when the index is fresh AND
  built for the resolved query engine (shown in ``query_engine``); otherwise ``reason``
  explains the decline — e.g. a polars-built index on a graph whose default queries
  resolve to pandas shows ``usable=False`` with an engine-mismatch reason. Pass
  ``show_indexes(engine=...)`` to preview an explicit engine choice.
- Rebuild by calling ``gfql_index_all()`` again on the rebound ``g``.

.. doc-test: skip

.. code-block:: python

   new_edges_df = edges_df.assign(amount=edges_df["amount"] + 1)
   g2 = g_indexed.edges(new_edges_df, "src", "dst")
   g2.show_indexes()          # edge_out_adj now valid=False; node_id still True
   g2 = g2.gfql_index_all()   # pay again for the new frame; all valid=True

**Declines are always safe.** Whether an index is missing, stale, or the query shape is
uncovered, results are identical either way — indexes only ever change speed, never
answers.

.. note::
   **Stability.** The index kinds, sidecar layout, and ``show_indexes()`` columns describe
   the current implementation and may evolve between releases; the stable contract is the
   lifecycle and decline-safety guarantees on this page — pay once, automatic reuse,
   staleness on rebind, and identical results with or without an index.

Engines
-------

- **pandas and cuDF**: build with the default ``gfql_index_all()`` (AUTO resolves to
  the frames' engine — numpy sidecars for pandas, on-device cupy for cuDF).
- **Polars**: on 0.58.0, pass the engine explicitly — ``gfql_index_all(engine='polars')``.
  With AUTO, an index build on Polars frames swaps them to pandas (the same AUTO
  behavior described in :doc:`engines`; a fix is tracked in PR
  `#1767 <https://github.com/graphistry/pygraphistry/pull/1767>`_).
- **Polars-GPU**: rides the Polars-tagged index — an index built with
  ``engine='polars'`` (or ``'polars-gpu'``) serves both.

What it costs, what it buys
---------------------------

**Build (the "pay" side)**: one-time and ``O(E log E)`` — a sort over the edge frame,
spread across every later query from known nodes. ``index_policy='auto'`` only pays it when
the planner predicts a selective query will earn it back.

**Lookup (the "go" side)**: on a covered query, the lookup gets faster twice: once on
the specialized path and again once the index is built, on both CPU engines.

**Flat in graph size**: a direct ``g.hop()`` from known nodes with the index built turns the
``O(E)`` scan into an ``O(degree)`` gather, so its cost tracks the seeds' neighborhood
rather than the graph.

.. _gfql-adjacency-index:

Adjacency index: fast lookups from known nodes
----------------------------------------------

A **seeded** query starts from known nodes — "the neighbors of this account", "2 hops
out from this device" — and by default GFQL answers it with one pass over every edge.
With the adjacency index resident, the same hop reads only the edges the seeds touch, so
its cost tracks the seeds' neighborhood instead of the size of the graph.

When to use it
~~~~~~~~~~~~~~

- **Seeded traversals**: you start from specific node ids (a watchlist, a session, a fraud
  ring's known members) and hop out 1–3 steps.
- **Repeated queries** against the same graph: build once, reuse over many such queries.
- **Interactive latency**: neighbor expansion whose cost tracks the seeds, not the graph.

It does **not** help a full-graph scan (a property filter over every node, a global
PageRank). For those, choose an *engine* instead — see :doc:`engines`.

Build it with Cypher
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import graphistry

   nodes_df = pd.DataFrame({"id": ["a", "b", "c", "d"]})
   edges_df = pd.DataFrame({"src": ["a", "a", "b", "c"], "dst": ["b", "c", "c", "d"]})
   g = graphistry.edges(edges_df, "src", "dst").nodes(nodes_df, "id")

   g = g.gfql("CREATE GFQL INDEX FOR edge_out_adj")      # build once: the adjacency ...
   g = g.gfql("CREATE GFQL INDEX FOR node_id")           # ... and the node-id lookup
   out = g.gfql("MATCH (a {id: 'a'})-[e]->(b) RETURN b")  # gfql_explain: used_index=True
   g.gfql("SHOW GFQL INDEXES")                           # what is resident

The DDL forms are ``CREATE GFQL INDEX [name] [IF NOT EXISTS] FOR <kind> [ON (col)]``,
``DROP GFQL INDEX name [IF EXISTS]`` (or ``DROP GFQL INDEX [IF EXISTS] FOR <kind> [ON (col)]``), and
``SHOW GFQL INDEXES`` — the mandatory ``GFQL`` token distinguishes them from standard property
``CREATE INDEX``; the optional parts follow the Cypher spelling, and the earlier GFQL spellings
(``ON col`` without parentheses, ``IF EXISTS`` before the name) stay accepted. The same intent travels over the JSON wire protocol
(``{"type": "CreateIndex", ...}`` ops plus ``index_policy`` in the request envelope), so a
remote ``gfql_remote`` call can carry it.

Controlling the planner
~~~~~~~~~~~~~~~~~~~~~~~

``gfql(..., index_policy=...)`` decides whether a resident index is used:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``index_policy``
     - Behavior
   * - ``'use'`` *(default)*
     - Use a resident index when one covers the query; never build one. Zero overhead if
       no index exists.
   * - ``'auto'``
     - Build an index on the fly when the planner predicts it pays off (selective seed set).
   * - ``'force'``
     - Require the index path (useful for asserting it is engaged).
   * - ``'off'``
     - Ignore indexes entirely (the plain scan).

Use ``g.gfql_explain(query, index_policy=...)`` to see whether the index path was taken.
It returns ``used_index`` (bool), ``resident_indexes``, the per-step ``steps`` trace, and
the planner's final ``decision_reason`` (human-readable) with a stable ``decision_code``
for programs and tests to match on:

- ``index_selected`` -- the index served the query.
- ``policy_off`` -- ``index_policy='off'``.
- ``no_resident_index`` -- nothing is resident and ``index_policy='use'`` never builds.
- ``index_path_unavailable`` -- the index path could not serve this query as planned, so
  the scan answered; ``decision_reason`` says what was missing (e.g. ``index_missing``).
- ``not_index_coverable`` -- the shape is one the index path does not cover; it scans.
- ``missing_graph_columns`` -- the graph lacks a bound column the index needs.
- ``index_build_declined`` -- ``index_policy='auto'`` / ``'force'`` ended with no usable
  index (the build was declined or did not cover the hop).
- ``scan_cost`` -- the index is resident but the cost gate chose the scan (a seed set
  covering most of the graph, for example).
- ``engine_mismatch`` -- the resident index was built for another engine.
- ``col_stats_absent`` / ``col_stats_stale`` / ``col_stats_insufficient`` /
  ``col_stats_served`` -- the column-stat fact consult (below) could not help, is out of
  date after a rebind, cannot prove what the plan needs, or answered the query.

A fast path's contract is "same answer, faster", so a decline is never an error: the
scan answers, and the code says why the shortcut was not taken.

Column-stat facts
~~~~~~~~~~~~~~~~~

``gfql_index_col_stats()`` records **verified facts** (min/max/null count; integer
columns in v1) for the bound node id and edge endpoint columns. Fast paths use them to
prove a per-query invariant — for example, that every filtered edge endpoint lies inside a
dense id interval — and skip the scan that would re-prove it. A missing or insufficient
fact just means the scan runs; a fact can save work but never change an answer. Facts
follow the same fingerprint validity contract as the physical indexes, and
``gfql_index_all()`` includes them. Pass ``node_columns=`` / ``edge_columns=`` to fact
additional integer columns — explicitly named columns raise if they can't be fact-ed,
while the binding defaults skip silently.

See also
--------

- :doc:`engines` — choosing pandas / Polars / cuDF / Polars-GPU.
- :doc:`performance` — GFQL measured against graph databases.
