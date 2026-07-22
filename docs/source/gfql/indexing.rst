.. _gfql-indexing:

Pay-As-You-Go Resident Indexing
===============================

GFQL runs without any indexes: every query is a vectorized scan over your dataframes.
When your workload is **seeded** — "expand from these 50 accounts", "look up this id and
hop out" — you can opt into **resident indexes**: build them once with one call, and
seeded queries reuse them automatically after that. This page is the user guide to that
lifecycle: what the indexes are, what engages them, when they go stale, and what they cost.
For the planner policy knobs and competitive benchmarks, see
:doc:`Seeded Traversal Indexes <index_adjacency>`.

.. code-block:: python

   g = g.gfql_index_all()   # pay once ...
   g.gfql(...)              # ... every later seeded lookup on g rides the index

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

They are **sidecars over row positions**: your ``.edges`` / ``.nodes`` frames are never
reordered or copied, and the resident footprint is visible per index via
``g.show_indexes()`` (the ``nbytes`` column). The model is **pay-as-you-go**: one
``O(E log E)`` build, amortized over every seeded query afterward. Nothing is built
unless you ask.

Quick start
-----------

A complete, runnable example:

.. code-block:: python

   import pandas as pd
   import graphistry
   from graphistry import n, e_forward, is_in

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

   # Pay once: build out+in adjacency + node-id indexes (resident on the returned g)
   g_indexed = g.gfql_index_all()
   print(g_indexed.show_indexes()[["name", "kind", "key_col", "n_keys", "valid"]])

   # Seeded 1-hop: who did accounts 0 and 3 transfer to?
   out = g_indexed.gfql([n({"id": is_in([0, 3])}), e_forward(), n()])
   print(sorted(out._nodes["id"].tolist()))          # [0, 1, 2, 3, 4]

   # Decline safety: the same query without any index gives the SAME answer
   out_scan = g.gfql([n({"id": is_in([0, 3])}), e_forward(), n()])
   assert sorted(out._nodes["id"].tolist()) == sorted(out_scan._nodes["id"].tolist())

   # Direct hop() uses the index too
   hop_out = g_indexed.hop(nodes=pd.DataFrame({"id": [0]}), hops=2, direction="forward")
   print(sorted(hop_out._nodes["id"].tolist()))      # [0, 1, 2, 3, 4]

The lifecycle calls, all returning a new ``g`` (functional style, like the rest of the
API):

.. code-block:: python

   g = g.gfql_index_all()               # out+in adjacency + node_id (the one-liner)
   g = g.gfql_index_edges("forward")    # or just one direction: 'forward'|'reverse'|'both'
   g = g.create_index("edge_out_adj")   # or one kind: 'edge_out_adj'|'edge_in_adj'|'node_id'
   g.show_indexes()                     # pandas DataFrame: kind, engine, n_keys, nbytes, valid
   g = g.drop_index()                   # drop all (or drop_index("edge_out_adj"))

Unlike ``gfql_index_all()``, an explicit ``create_index("node_id")`` **raises** on
non-unique node ids rather than skipping.

What uses the index today
-------------------------

On 0.58.0, a resident index is consumed automatically by:

- **Seeded typed-hop fast paths** (native chain or Cypher): a seeded typed 1-hop —
  ``[n({"id": is_in([...])}), e_forward(), n(...)]`` or
  ``MATCH (m {id: $x})-[:T]->(p) RETURN p`` — including the single-alias **property
  RETURN** form (``RETURN p.a AS x, p.b``). The seed lookup, frontier expansion, and
  endpoint materialization all become positional index gathers, so the lookup stops
  paying graph-size costs.
- **Property-seeded lookups**: the seed filter may hit a *property* column (e.g.
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
- Rebuild by calling ``gfql_index_all()`` again on the rebound ``g``.

.. code-block:: python

   g2 = g_indexed.edges(new_edges_df, "src", "dst")
   g2.show_indexes()          # edge_out_adj / edge_in_adj now valid=False; node_id still True
   g2 = g2.gfql_index_all()   # pay again for the new frame; all valid=True

**Declines are always safe.** Whether an index is missing, stale, or the query shape is
uncovered, results are identical either way — indexes only ever change speed, never
answers (index-vs-scan parity is differentially tested across engines).

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

All numbers below: 0.58.0 tag sweep on DGX Spark, warm medians over 30 runs.

**Build (the "pay" side)**: one-time; on a 30.6M-edge graph the full
``gfql_index_all()`` build is about 5.7s.

**Seeded lookup (the "go" side)**: a covered-shape seeded Cypher lookup:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Seeded typed lookup
     - pandas
     - Polars
   * - General path (no fast path)
     - 29.9 ms
     - 13.8 ms
   * - Fast path, no index
     - 2.46 ms
     - 2.28 ms
   * - **Fast path + resident index**
     - **1.74 ms**
     - **1.59 ms**

**Flat in graph size**: a direct seeded ``g.hop()`` with the index resident holds
0.159–0.164 ms from 0.25M to 32M edges (pandas) — cost tracks seed degree, not graph
size.

See also
--------

- :doc:`Seeded Traversal Indexes <index_adjacency>` — the planner (``index_policy``),
  Cypher DDL / wire protocol forms, and competitive benchmarks vs Kuzu / Neo4j.
- :doc:`engines` — choosing pandas / Polars / cuDF / Polars-GPU.
- :doc:`performance` — the vectorization + GPU design behind GFQL.
