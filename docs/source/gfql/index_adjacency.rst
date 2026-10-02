Adjacency Index: Fast Lookups from Known Nodes
==============================================

GFQL is fast out of the box. When you or your coding agent want more, GFQL supports the
usual answer: an index. A **seeded** query starts from known nodes — "the neighbors of
this account", "2 hops out from this device" — and by default GFQL answers it with one
pass over every edge. With an opt-in **adjacency index**, the same hop reads only the
edges the seeds touch, so its cost tracks the seeds' neighborhood instead of the size of
the graph, and a seeded lookup stays interactive as the graph grows.

Nothing changes about the answer. The index is a pay-as-you-go accelerator: a query either
uses a resident index or falls back to the scan, and any feature the index does not cover
also falls back — never a different result.

When to use it
--------------

- **Seeded traversals**: you start from specific node ids (a watchlist, a session, a fraud
  ring's known members) and hop out 1–3 steps.
- **Repeated queries** against the same graph: build the index once and reuse it over many
  such queries.
- **Interactive / point-lookup latency**: neighbor expansion whose cost tracks the
  seeds rather than the graph.

It does **not** help a full-graph scan (a property filter over every node, a global
PageRank). For those, choose an *engine* instead — see :doc:`engines`.

Quick start
-----------

Build the index with Cypher, then query as usual:

.. code-block:: python

   import pandas as pd
   import graphistry

   nodes_df = pd.DataFrame({"id": ["a", "b", "c", "d"]})
   edges_df = pd.DataFrame({"src": ["a", "a", "b", "c"], "dst": ["b", "c", "c", "d"]})
   g = graphistry.edges(edges_df, "src", "dst").nodes(nodes_df, "id")

   g = g.gfql("CREATE GFQL INDEX FOR edge_out_adj")      # build once
   out = g.gfql("MATCH (a {id: 'a'})-[e]->(b) RETURN b")  # served by the index
   g.gfql("SHOW GFQL INDEXES")                           # what is resident

Check that a query took the index path with ``g.gfql_explain(query)``: it reports
``used_index`` and the decision behind it. The same hop written as a native chain:

.. code-block:: python

   from graphistry import n, e_forward, is_in

   g = g.gfql_index_all()   # out+in adjacency, plus a node-id accelerator when ids are unique
   out = g.gfql([n({"id": is_in(["a", "b"])}), e_forward(), n()])

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

Cost and fallback
-----------------

- **Build cost**: one sort of the edges, paid once and reused by every later query.
  ``index_policy='auto'`` builds only when the planner expects a query to pay it back.
- **Nothing changes until you build one.** With no index resident, queries run exactly
  as before.
- **Same answer either way.** A query the index covers takes the fast path; anything it
  does not cover falls back to the normal scan. The index is an accelerator, never a
  different result.

See also
--------

- :doc:`engines` — choosing pandas / Polars / cuDF / Polars-GPU for queries that scan the graph.
- :doc:`performance` — the vectorization + GPU design behind GFQL.
- :doc:`benchmark_filter_pagerank` — an end-to-end filter → PageRank → filter comparison vs Neo4j.
