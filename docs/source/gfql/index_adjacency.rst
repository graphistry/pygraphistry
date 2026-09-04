Adjacency Index: Fast Lookups from Known Nodes
==============================================

A **seeded** graph query starts from a known set of nodes — "the neighbors of these
50 accounts", "2 hops out from this device" — rather than scanning the whole graph.
By default GFQL answers a seeded ``hop`` with an ``O(E)`` pass over every edge. With an
opt-in **CSR adjacency index**, the same hop becomes an ``O(degree)`` gather: its cost
depends on how many edges the *seeds* touch, not on how big the graph is — so a seeded
lookup stays interactive as the graph grows.

Nothing changes about the answer. The index is a pay-as-you-go accelerator: a query either
uses a resident index or falls back to the scan, and any feature the index does not cover
also falls back — never a different result.

When to use it
--------------

- **Seeded traversals**: you start from specific node ids (a watchlist, a session, a fraud
  ring's known members) and hop out 1–3 steps.
- **Repeated queries** against the same graph: build the index once, amortize it over many
  seeded lookups.
- **Interactive / point-lookup latency**: neighbor expansion whose cost tracks the
  seeds rather than the graph.

It does **not** help a full-graph scan (a property filter over every node, a global
PageRank). For those, choose an *engine* instead — see :doc:`engines`.

Quick start
-----------

.. code-block:: python

   import graphistry
   from graphistry import n, e_forward, is_in

   g = graphistry.edges(edges_df, "src", "dst").nodes(nodes_df, "id")

   # Build the indexes once (out+in adjacency, plus a node-id accelerator when ids are unique)
   g = g.gfql_index_all()

   # Seeded traversal — the index is used automatically (default index_policy='use')
   my_seed_ids = ["a", "b"]   # your seed node ids
   out = g.gfql([n({"id": is_in(my_seed_ids)}), e_forward(), n()])

``gfql_index_all()`` is the one-liner. For finer control, build a single kind:

.. code-block:: python

   g = g.create_index("edge_out_adj")   # outgoing adjacency (forward hops)
   g = g.create_index("edge_in_adj")    # incoming adjacency (reverse hops)
   g = g.create_index("node_id")        # node-id lookup accelerator (unique ids only)
   g = g.gfql_index_col_stats()         # verified column-stat facts (see below)

   g.show_indexes()                     # inspect what's resident
   g = g.drop_index()                   # drop all (or drop_index("edge_out_adj"))

The index is a **sidecar over edge row positions** — it never reorders your ``.edges`` /
``.nodes`` frames, and it is fingerprint-validated: rebinding ``.edges()`` safely
invalidates a stale index (treated as absent, never a wrong answer).

Column-stat facts
-----------------

``gfql_index_col_stats()`` records **verified facts** (min/max/null count; integer
columns in v1) for the bound node id and edge endpoint columns — the columns
count-shaped query plans consult. Fast paths use them as *under-approximations of
provability*: a valid fact can prove a per-query invariant (e.g. every filtered edge
endpoint lies inside a dense id interval) and skip the O(E) scan that would re-prove
it; a missing or insufficient fact just means the scan runs. A fact can therefore
save work but never change an answer. Facts follow the same fingerprint validity
contract as the physical indexes, and ``gfql_index_all()`` includes them. Pass
``node_columns=`` / ``edge_columns=`` to fact additional integer columns —
explicitly named columns raise if they can't be fact-ed, while the binding defaults
skip silently.

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

**The index changes the complexity class.** An indexed seeded hop is an ``O(degree)``
gather into a sorted adjacency, so its cost tracks the size of the seeds' neighborhood.
The default scan is ``O(E)`` and grows with the whole graph. The bigger the graph is
relative to the seeds' neighborhood, the wider that gap.

**Selective traversal is CPU's game.** The indexed hop is tiny work, so a GPU engine's
kernel-launch floor dominates it and a CPU engine — pandas or Polars, both backed by a
``searchsorted`` gather — wins. That is the clean inverse of *bulk* analytics, where the
GPU pulls ahead (see :doc:`engines`). Pick the index for selective traversal and a **CPU
engine** to drive it.

Latency figures for this lane are not published yet: it has not been run under the
provenance-carrying harness described on :doc:`performance`, and this page publishes
nothing it cannot trace to a committed artifact. Reproducers:
``benchmarks/gfql/index_takeover_bench.py``, ``benchmarks/gfql/index_vs_dbs.py``,
``benchmarks/gfql/index_vs_kuzu_prepared.py``.

Cost and fallback
-----------------

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
