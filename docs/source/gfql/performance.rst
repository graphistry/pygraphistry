.. _gfql-performance:

GFQL Performance: Vectorization and GPU Acceleration
====================================================

This page is the **canonical home for GFQL benchmark numbers** — the measured tables live
here (and, for the resident-index benchmarks, in :doc:`index_adjacency`), while the rest of
the docs make stable qualitative claims and link back here.

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
   g.gfql(query, engine='polars')   # often much faster on query-heavy workloads, same results

For example, in the release-verified sweep below, the LDBC SNB SF1 seed-lookup drops from
**1,299.6 ms** on eager pandas to **106.1 ms** with ``engine='polars'`` — a **12.3×**
one-keyword speedup, no GPU, results identical.

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
200k-edge graph — is the bread-and-butter selective lookup. The release's fast path speeds
it up on every engine (before → after within the sweep):

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 15

   * - Engine
     - Before
     - After (fast path)
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
   **Polars + index: pass** ``engine='polars'`` **when building.** Polars frames currently
   need ``g.gfql_index_all(engine='polars')`` explicitly — an AUTO build swaps Polars
   frames to pandas. The fix is tracked in PR #1767.

Scaling: flat in graph size
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the resident adjacency index, a native seeded 1-hop ``g.hop()`` on pandas stays
**flat at 0.159–0.164 ms from 0.25M to 32M edges** (constant average degree 4): the index
turns the O(E) scan into an O(degree) gather, so seeded latency does not grow with the
graph. (Pandas-only today — the Polars hop path is not yet index-routed.)

vs Neo4j (LDBC SNB interactive SF1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same box, warm, against Neo4j 5.26 — GFQL wins **4 of 5** clean pairs:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 15

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
     - GFQL
   * - one-hop-expand
     - **111.9 ms**
     - 180.7 ms
     - GFQL
   * - recent-replies
     - 209.6 ms
     - **104.0 ms**
     - Neo4j

The message-creator flip shipped in this release via property-seeded resident-index
gathers (PR #1770). Neo4j still wins recent-replies — reported as-is.

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

.. _gfql-bulk-sweep:

Bulk engine comparison (prior sweep)
------------------------------------

The numbers in this section are from an earlier, pre-0.58.0 bulk sweep on SNAP
**com-LiveJournal** (35M edges) and **com-Orkut** (117M edges) — retained as the
bulk-workload reference until rerun on a current tag.

Same query, same answers, four engines — warm-median latency on Orkut (3.1M nodes /
117M edges), measured on a single machine:

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 16 16

   * - Workload (Orkut, 117M edges)
     - ``pandas``
     - ``polars``
     - ``cudf``
     - ``polars-gpu``
   * - 1-hop from 10K seeds
     - 2613 ms
     - **68 ms**
     - 1005 ms
     - 63 ms
   * - 2-hop from 10K seeds
     - 18161 ms
     - 2695 ms
     - 2774 ms
     - **1518 ms**
   * - Full out-degree aggregation
     - 799 ms
     - 205 ms
     - 314 ms
     - **167 ms**
   * - 2-hop from 100K seeds (~85M output rows)
     - 28822 ms
     - 8215 ms
     - **6002 ms**
     - 8559 ms

Reading the table:

- **Polars-CPU beat pandas up to ~38x** on bulk traversal and ~4x on aggregation — **with
  no GPU**. On the 1-hop workload it was ~38x faster than pandas (68 ms vs 2613 ms).
- **Polars-CPU also beat cuDF** on these shapes (68 ms vs 1005 ms on 1-hop). cuDF runs
  GFQL *eagerly*, op by op (a kernel launch + a materialized intermediate per hop), while
  Polars builds **one fused lazy plan and collects once**. The fused plan wins until the
  work is large enough to amortize GPU launch costs.
- **Polars-GPU was fastest on heavy multi-hop** (2-hop from 10K seeds: 1518 ms) and on
  aggregation — the same fused plan, executed on the GPU.
- **cuDF won the one extreme case** — a 2-hop from 100K seeds materializing ~85M output
  rows (6.0 s) — where raw GPU throughput on a single massive join overtakes everything
  and Polars-GPU comes under memory pressure.
- On LiveJournal (35M edges) the pattern held: 1-hop from 10K seeds was pandas 1129 ms →
  polars **37 ms** (~30x).
- The CPU crossover is early: on LiveJournal subsampled (CPU, warm-median), 1-hop
  traversal was 2.7× / 4.5× / 7.6× and ``WHERE``+``ORDER`` 3.0× / 3.0× / 18× over pandas
  at 10K / 100K / 1M edges. The only case pandas edged out was a trivial sub-millisecond
  operation (a bare node-equality filter), where its boolean mask beats Polars' plan
  overhead — immaterial at <1 ms. Reproducer: ``benchmarks/gfql/index_crossover_bench.py``.

Methodology (prior sweep)
~~~~~~~~~~~~~~~~~~~~~~~~~

- Host: NVIDIA DGX Spark (GB10 Grace-Blackwell, unified memory — the memory-pressure
  boundary above is partly a property of this box), RAPIDS container
  ``graphistry/test-rapids-official:26.02-gfql-polars``.
- Datasets: `SNAP <https://snap.stanford.edu/data/>`_ **com-LiveJournal** (35M edges),
  **com-Orkut** (117M edges).
- Measurement: **warm median** after 2 warmups (5 timed runs on Orkut, 8 on LiveJournal);
  every reported cell is **guarded** — the result rows are verified identical across
  engines before any timing is kept.
- Reproduce: ``benchmarks/gfql/index_bulk_olap_bench.py`` (engine comparison),
  ``benchmarks/gfql/pandas_vs_polars.py``, and ``benchmarks/gfql/index_vs_kuzu_prepared.py``
  (vs kuzu). Numbers on this page are rendered from saved runs; the page does not re-run
  them.
- **LadybugDB comparison** (referenced qualitatively in :doc:`engines`): the Ladybug
  figures are **their published results on their hardware**; the GFQL side ran on the host
  above via ``benchmarks/gfql/bench_ladybug_cypher.py`` (5M/20M synthetic per their suite
  shape, native frames per engine, warm medians) — a cross-machine comparison, disclosed
  as such. GFQL won the scan-shaped ops by large margins (full node scan ~65×,
  relationship property/rowid scans ~3.5–3.7×); Ladybug won the two ops backed by
  persistent structure — point lookups (index seek vs columnar scan) and a cached
  relationship ``COUNT(*)``.

There is **no universal winner**: ``polars`` typically takes over from ~10K edges up
(``pandas`` still wins trivial sub-millisecond operations), and the right GPU
engine depends on the workload. See :doc:`engines` for the full decision matrix, the honest
"when *not* to use Polars", and the cuDF-vs-Polars-GPU comparison. The end-to-end
CPU/GPU-vs-Neo4j pipeline benchmark is in :doc:`benchmark_filter_pagerank`, and the
Spark GraphFrames head-to-head is in :doc:`benchmark_graphframes`.

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

- **Choose an engine**: :doc:`engines` — the full decision matrix and qualitative guidance.
- **Selective lookups**: :doc:`index_adjacency` — the resident index behind the flat-scaling numbers.
- **End-to-end benchmark**: :doc:`benchmark_filter_pagerank` — CPU/GPU vs Neo4j+GDS.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.
- **Ecosystem**: `Apache Arrow <https://arrow.apache.org/>`_ and `NVIDIA RAPIDS <https://rapids.ai/>`_.
