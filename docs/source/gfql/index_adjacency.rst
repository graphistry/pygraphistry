Seeded Traversal Indexes (CSR Adjacency)
========================================

A **seeded** graph query starts from a known set of nodes — "the neighbors of these
50 accounts", "2 hops out from this device" — rather than scanning the whole graph.
By default GFQL answers a seeded ``hop`` with an ``O(E)`` pass over every edge. With an
opt-in **CSR adjacency index**, the same hop becomes an ``O(degree)`` gather: its cost
depends on how many edges the *seeds* touch, not on how big the graph is. The result is
**flat in graph size** — and it beats embedded graph databases on selective lookups.

Nothing changes about the answer. The index is a pay-as-you-go accelerator: a query either
uses a resident index or falls back to the scan, and any feature the index does not cover
also falls back — never a different result.

When to use it
--------------

- **Seeded traversals**: you start from specific node ids (a watchlist, a session, a fraud
  ring's known members) and hop out 1–3 steps.
- **Repeated queries** against the same graph: build the index once, amortize it over many
  seeded lookups.
- **Interactive / point-lookup latency**: sub-millisecond neighbor expansion.

It does **not** help a full-graph scan (a property filter over every node, a global
PageRank). For those, choose an *engine* instead — see :doc:`engines`.

Quick start
-----------

.. code-block:: python

   import graphistry

   g = graphistry.edges(edges_df, "src", "dst").nodes(nodes_df, "id")

   # Build the indexes once (out+in adjacency, plus a node-id accelerator when ids are unique)
   g = g.gfql_index_all()

   # Seeded query — the index is used automatically (default index_policy='use')
   out = g.gfql("MATCH (a)-[e]->(b) WHERE a.id IN $seeds RETURN a, e, b",
                params={"seeds": my_seed_ids})

``gfql_index_all()`` is the one-liner. For finer control, build a single kind:

.. code-block:: python

   g = g.create_index("edge_out_adj")   # outgoing adjacency (forward hops)
   g = g.create_index("edge_in_adj")    # incoming adjacency (reverse hops)
   g = g.create_index("node_id")        # node-id lookup accelerator (unique ids only)

   g.show_indexes()                     # inspect what's resident
   g = g.drop_index()                   # drop all (or drop_index("edge_out_adj"))

The index is a **sidecar over edge row positions** — it never reorders your ``.edges`` /
``.nodes`` frames, and it is fingerprint-validated: rebinding ``.edges()`` safely
invalidates a stale index (treated as absent, never a wrong answer).

Controlling the planner
-----------------------

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
     - Require the index path (useful for benchmarking / asserting it is engaged).
   * - ``'off'``
     - Ignore indexes entirely (the plain ``O(E)`` scan).

Use ``g.gfql_explain(query, index_policy=...)`` to see whether the index path was taken.

The indexes are **engine-uniform**: numpy host arrays for pandas / Polars, cupy on-device
for cuDF. They are also exposed as **Cypher DDL** (``CREATE GFQL INDEX FOR edge_out_adj``,
``DROP GFQL INDEX``, ``SHOW GFQL INDEXES`` — the mandatory ``GFQL`` token distinguishes them
from standard property ``CREATE INDEX``) and in the **JSON wire protocol**
(``{"type": "CreateIndex", ...}`` ops plus ``index_policy`` in the request envelope), so a
remote ``gfql_remote`` call can carry the same index intent.

Performance
-----------

**Flat in graph size.** A seeded 1-hop stays sub-millisecond as the graph grows 10×, while
the ``O(E)`` scan grows linearly. Synthetic power-law graphs, GFQL-pandas, warm median,
every cell guarded so the index path was taken *and* the indexed result equals the scan
result:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Seeded 1-hop
     - 0.8M nodes / 6.4M edges
     - 8M nodes / 64M edges
   * - **Indexed (O(degree))**
     - **0.124 ms**
     - **0.122 ms** *(flat)*
   * - Scan (O(E))
     - 105 ms
     - 1045 ms

The same holds on real power-law graphs: a typical-seed 1-hop is ~0.13 ms on LiveJournal
(35M edges) and ~0.14 ms on Orkut (117M edges), versus an ``O(E)`` scan of 367 ms → 1208 ms.

**Beats embedded graph databases on selective lookups.** Same graph (0.8M nodes / 6.4M
edges), matched result counts, warm median. GFQL is CPU-pandas with the index; Kuzu and
Neo4j use their native indexes:

.. list-table::
   :header-rows: 1
   :widths: 24 22 18 18 18

   * - Task
     - GFQL (indexed)
     - Kuzu
     - Neo4j
     - GFQL speedup
   * - 1-hop seeded
     - **0.123 ms**
     - 1.15 ms
     - 1.45 ms
     - 9.4× / 11.8×
   * - 1–2-hop seeded
     - **0.150 ms**
     - 4.25 ms
     - 2.54 ms
     - 28× / 16.9×

On a fairer, fully-prepared, in-process Kuzu re-run (LiveJournal 35M), GFQL is still
**17×** on a typical seed (0.126 ms vs 2.13 ms) and **6×** on a hub seed (3.76 ms vs
22.6 ms). *(Kuzu's worst-case-optimal joins can win on cyclic / multi-way-join patterns —
triangles, cliques — which these forward-expansion lookups do not exercise; we do not
claim those.)*

**Selective traversal is CPU's game.** The indexed hop is tiny work, so the GPU's
kernel-launch floor (~3 ms on cuDF) loses to a ~0.13 ms pandas / ~0.16 ms Polars
``searchsorted`` — the clean inverse of *bulk* analytics, where the GPU pulls ahead
(see :doc:`engines`). Pick the index for selective traversal and a **CPU engine** to
drive it.

Reproduce: ``benchmarks/gfql/index_takeover_bench.py``,
``benchmarks/gfql/index_vs_dbs.py``, ``benchmarks/gfql/index_vs_kuzu_prepared.py``.
Hardware: DGX ``dgx-spark``, GB10 GPU.

Honesty and cost
----------------

- **Build cost** is one ``O(E log E)`` sort, amortized over subsequent queries.
  ``index_policy='auto'`` only builds when the planner predicts a selective query will
  pay it back.
- **No change to default behavior.** With no index resident and ``index_policy='use'``
  (the default), queries run exactly as before.
- **Parity-or-fallback.** The index accelerates the seeded scan sites it covers (forward /
  reverse hop, the Polars hop, the single-hop chain fast path). Any uncovered feature —
  edge / source / destination match, ``target_wave_front``, ``min_hops>1``, labeling —
  falls back to the scan/join path. The indexed subgraph is verified equal to the scan
  subgraph in differential tests across pandas / cuDF / Polars / Polars-GPU. It is an
  accelerator, never a source of a different answer.

See also
--------

- :doc:`engines` — choosing pandas / Polars / cuDF / Polars-GPU for non-seeded work.
- :doc:`performance` — the vectorization + GPU design behind GFQL.
- :doc:`benchmark_filter_pagerank` — an end-to-end filter → PageRank → filter comparison vs Neo4j.
