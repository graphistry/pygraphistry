.. _gfql-performance:

GFQL Performance: Vectorization and GPU Acceleration
====================================================

Engine speedups at a glance
---------------------------

GFQL runs the **same query** on four interchangeable engines — ``pandas`` (default),
``polars`` (CPU, columnar), ``cudf`` (NVIDIA GPU), and ``polars-gpu`` (GPU) — and returns
**identical results** on each (differential parity is a release gate; every four-engine
number on this page was kept only after the result rows were verified identical across
engines, and the cross-database pairs were validated against expected result rows).
Unsupported engine/query combinations are declined before execution during validation,
compilation, or planning rather than silently falling back. The biggest, easiest win is one
keyword, **no GPU required**:

.. doc-test: skip

.. code-block:: python

   g.gfql(query)                    # engine='pandas' (default)
   g.gfql(query, engine='polars')   # e.g. 12.3x on LDBC SNB SF1 seed-lookup, same build, same results

On the 0.58.0 release build, the LDBC SNB SF1 seed-lookup drops from **1,299.6 ms** on
eager pandas to **106.1 ms** with ``engine='polars'`` — a **12.3×** one-keyword speedup,
no GPU, results identical.

.. _gfql-0580-numbers:

Release-verified numbers (0.58.0)
---------------------------------

All numbers in this section were measured on the **0.58.0 release tag** on an NVIDIA DGX
Spark (GB10), warm medians over N=30 runs. The four-engine numbers (seeded fast paths,
resident index, scaling) were kept only after the result rows were asserted identical
across engines; the competitor pairs (vs Neo4j, vs Kuzu) were validated against expected
result rows and cross-database value/row-count checks.

Seeded typed-hop fast path
~~~~~~~~~~~~~~~~~~~~~~~~~~

A seeded typed hop — Cypher ``MATCH (m {id: ...})-[:T]->(p) RETURN p`` on a 50k-node /
200k-edge graph — is the bread-and-butter selective lookup. 0.58.0's fast path speeds it
up on every engine (before → after on the same tag sweep):

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 15

   * - Engine
     - Before
     - After (0.58.0 fast path)
     - Speedup
   * - ``pandas``
     - 29.9 ms
     - **2.46 ms**
     - 12.1×
   * - ``polars``
     - 13.8 ms
     - **2.28 ms**
     - 6.1×
   * - ``cudf``
     - 30.1 ms
     - **4.89 ms**
     - 6.1×
   * - ``polars-gpu``
     - 25.2 ms
     - **2.49 ms**
     - 10.1×

The native chain form of the same query is faster still: pandas 21.1 → **1.65 ms**
(12.8×), cuDF 23.2 → **3.84 ms** (6.0×).

With a resident index
~~~~~~~~~~~~~~~~~~~~~

Building the opt-in resident index once (``g.gfql_index_all()``) makes the covered-shape
seeded lookup faster again — pandas **1.74 ms**, polars **1.59 ms**, polars-gpu
**1.91 ms**, cudf **5.78 ms**.

.. warning::
   **Polars + index on 0.58.0: pass** ``engine='polars'`` **when building.** On 0.58.0,
   Polars frames need ``g.gfql_index_all(engine='polars')`` explicitly — an AUTO build
   swaps Polars frames to pandas. The fix is tracked in PR #1767.

Scaling: flat in graph size
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the resident adjacency index, a native seeded 1-hop ``g.hop()`` on pandas stays
**flat at 0.159–0.164 ms from 0.25M to 32M edges** (constant average degree 4): the index
turns the O(E) scan into an O(degree) gather, so seeded latency does not grow with the
graph. (This flat-scaling claim is pandas-only on 0.58.0 — the Polars hop path is not yet
index-routed.)

vs Neo4j (LDBC SNB interactive SF1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same box, warm, against Neo4j 5.26 — GFQL wins **4 of 5** clean pairs:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Query
     - GFQL
     - Neo4j 5.26
     - Winner
   * - seed-lookup
     - **106.1 ms**
     - 143.7 ms
     - GFQL
   * - message-content
     - **7.1 ms**
     - 23.0 ms
     - GFQL
   * - message-creator
     - **6.8 ms**
     - 27.7 ms
     - GFQL (flip shipped in 0.58.0, PR #1770)
   * - one-hop-expand
     - **111.9 ms**
     - 180.7 ms
     - GFQL
   * - recent-replies
     - 209.6 ms
     - **104.0 ms**
     - Neo4j

The message-creator flip shipped in 0.58.0 via property-seeded resident-index gathers
(PR #1770). Neo4j still wins recent-replies — reported as-is.

OLAP multi-join
~~~~~~~~~~~~~~~

On the graph-benchmark OLAP multi-join queries at 100k-node scale with
``engine='polars'``: **q8 runs in 5.0 ms vs 1,004 ms for embedded Kuzu (200×)**; q9 is
**14.2×**.

When not to use GFQL
~~~~~~~~~~~~~~~~~~~~

Honesty matters more than a bigger number. In the same cross-DB sweep, **embedded Kuzu
wins single-table aggregates (2–4×) and seeded property-projection lookups (2.4–64×)**.
GFQL's strengths are **traversals, multi-join OLAP, and covered seeded shapes** — route by
workload, and keep a database as the system-of-record where one fits.

Prior-sweep engine comparison (bulk, 117M edges)
------------------------------------------------

An earlier (pre-0.58.0) bulk sweep on **Orkut** (117M edges, SNAP) shows how the engines
compare on large bulk work — retained here as a prior-measurement example; see
:doc:`engines` for its full methodology:

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Workload (117M edges)
     - ``pandas``
     - ``polars``
     - ``cudf``
     - ``polars-gpu``
   * - 1-hop from 10K seeds
     - 2613 ms
     - **68 ms**
     - 1005 ms
     - 63 ms
   * - Full out-degree aggregation
     - 799 ms
     - 205 ms
     - 314 ms
     - **167 ms**

There is **no universal winner**: ``polars`` typically takes over from ~10K edges up
(``pandas`` still wins trivial sub-millisecond operations), and the right GPU
engine depends on the workload. See :doc:`engines` for the full decision matrix, the honest
"when *not* to use Polars", the cuDF-vs-Polars-GPU comparison, and the methodology + reproducer
scripts behind these numbers. The end-to-end CPU/GPU-vs-Neo4j benchmark is in
:doc:`benchmark_filter_pagerank`.

How GFQL is fast
----------------

Three design choices explain the numbers above:

**Collection-oriented execution.** GFQL evaluates whole collections of nodes and edges at
once (set-at-a-time), rather than walking one path at a time like traditional Cypher/Gremlin
engines. A traversal advances by joining edge tables, so the work vectorizes.

**Vectorized columnar processing.** Data is processed in columnar batches on top of
`Apache Arrow <https://arrow.apache.org/>`_, which keeps the CPU path fast and makes moving
data between systems cheap. The ``polars`` engine additionally builds **one fused lazy plan
and collects once**, which is why it outruns both pandas and eager cuDF on bulk work.

**Massive parallelism on GPUs.** On an NVIDIA GPU (``cudf`` / ``polars-gpu``), the same
vectorized work saturates tens of thousands of threads — paying off when there is enough
work to amortize kernel-launch cost (large frontiers, dense joins, full-graph aggregation).

Start on CPU with no special hardware, and move to a GPU engine by changing one keyword when
your workload grows into GPU territory. See :doc:`engines` for exactly when each engine wins.

.. note::
   Same-path constraints (``where``) can be more expensive on dense graphs.
   Prefer selective per-step predicates and see :doc:`/gfql/where` for details.

Next Steps
----------

- **Choose an engine**: :doc:`engines` — the full decision matrix, methodology, and reproducers.
- **Selective lookups**: :doc:`index_adjacency` — the resident index behind the flat-scaling numbers.
- **End-to-end benchmark**: :doc:`benchmark_filter_pagerank` — CPU/GPU vs Neo4j+GDS.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.
- **Ecosystem**: `Apache Arrow <https://arrow.apache.org/>`_ and `NVIDIA RAPIDS <https://rapids.ai/>`_.
