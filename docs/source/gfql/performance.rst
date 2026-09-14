.. _gfql-performance:

GFQL Performance: Vectorization and GPU Acceleration
====================================================

This page collects measured GFQL performance results. See :doc:`index_adjacency`
for the adjacency index used by the seeded-lookup tests.

Choose an engine
----------------

GFQL runs the same query on ``pandas`` (the default), ``polars`` (CPU), ``cudf``
(NVIDIA GPU), or ``polars-gpu``. The benchmark checks that each engine returns the
same rows. If an engine cannot run a query, GFQL reports an error before execution
instead of silently changing engines.

.. doc-test: skip

.. code-block:: python

   g.gfql(query)                    # engine='pandas' (default)
   g.gfql(query, engine='polars')   # columnar CPU execution

On the LDBC Social Network Benchmark (SNB) SF1 seed lookup below, changing from
pandas to Polars reduced the time from **1,299.6 ms** to **106.1 ms**, or
**12.3×**, without a GPU.

.. _gfql-0580-numbers:

Measurements on version 0.58.0
------------------------------

These warm medians use the **0.58.0 release tag** on an NVIDIA DGX Spark (GB10),
with 30 measured runs. Tests checked result rows across GFQL engines and against
the expected Neo4j and Kuzu results.

Seeded typed-hop fast path
~~~~~~~~~~~~~~~~~~~~~~~~~~

A seeded typed hop starts at a known node and follows one relationship type. For
the query ``MATCH (m {id: ...})-[:T]->(p) RETURN p`` on a 50k-node, 200k-edge
graph, the fast path reduced the time on every engine:

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

With an adjacency index
~~~~~~~~~~~~~~~~~~~~~~~

Building the optional in-memory adjacency index once with ``g.gfql_index_all()``
reduced the same lookup to pandas **1.74 ms**, Polars **1.59 ms**, Polars GPU
**1.91 ms**, and cuDF **5.78 ms**.

.. warning::
   For a Polars graph, build the index with
   ``g.gfql_index_all(engine='polars')``. Automatic engine selection currently
   converts the frames to pandas. PR #1767 tracks the fix.

Lookup time as the graph grows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the adjacency index, a seeded one-hop ``g.hop()`` on pandas took
**0.159–0.164 ms from 0.25M to 32M edges** at an average degree of four. The index
reads the seed's neighbors instead of scanning every edge. The Polars hop path does
not yet use this index.

Compared with Neo4j (LDBC SNB interactive SF1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the same host after warm-up, GFQL was faster on four of five queries:

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

Neo4j was faster on ``recent-replies``.

Analytical queries with multiple joins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OLAP multi-join comparison against an embedded graph database is the q1–q9 board
below: :ref:`gfql-vs-kuzu-board`.

When not to use GFQL
~~~~~~~~~~~~~~~~~~~~

In the same tests, embedded Kuzu was **2–4× faster for single-table aggregates**
and **2.4–64× faster for lookups that return node properties**. GFQL performed
best on traversals, analytical queries with multiple joins, and indexed seeded
queries. Keep data in a database when the application needs durable shared storage.

.. _gfql-vs-kuzu-board:

The q1–q9 comparison: GFQL and Kuzu
-----------------------------------

These tables cover nine Cypher queries from ``prrao87/graph-benchmark``. They rank
nodes by degree, group and filter records, and count two-hop paths on synthetic
social graphs with 20,000 and 100,000 people. Each table cell passed result-row
validation.

Kuzu parses and executes the query text on each call. GFQL reuses a prepared graph,
so the tables show direct times rather than Kuzu-to-GFQL speedup ratios. The q8 GFQL
cells are marked diagnostic because they reused cached degree data between calls;
do not use those cells as benchmark results.

See :doc:`benchmark_filter_pagerank` for the Neo4j filter → PageRank → filter
comparison.

The 20,000-person board
~~~~~~~~~~~~~~~~~~~~~~~

The q8 GFQL cells are diagnostic; all other cells are benchmark results.

.. list-table::
   :header-rows: 1
   :widths: 10 12 20 20 20

   * - Query
     - Result rows
     - Kuzu
     - GFQL ``pandas``
     - GFQL ``polars``
   * - q1
     - 3
     - :bench:`graphbench.20k.q1.kuzu`
     - :bench:`graphbench.20k.q1.pandas`
     - :bench:`graphbench.20k.q1.polars`
   * - q2
     - 1
     - :bench:`graphbench.20k.q2.kuzu`
     - :bench:`graphbench.20k.q2.pandas`
     - :bench:`graphbench.20k.q2.polars`
   * - q3
     - 5
     - :bench:`graphbench.20k.q3.kuzu`
     - :bench:`graphbench.20k.q3.pandas`
     - :bench:`graphbench.20k.q3.polars`
   * - q4
     - 2
     - :bench:`graphbench.20k.q4.kuzu`
     - :bench:`graphbench.20k.q4.pandas`
     - :bench:`graphbench.20k.q4.polars`
   * - q5
     - 1
     - :bench:`graphbench.20k.q5.kuzu`
     - :bench:`graphbench.20k.q5.pandas`
     - :bench:`graphbench.20k.q5.polars`
   * - q6
     - 5
     - :bench:`graphbench.20k.q6.kuzu`
     - :bench:`graphbench.20k.q6.pandas`
     - :bench:`graphbench.20k.q6.polars`
   * - q7
     - 1
     - :bench:`graphbench.20k.q7.kuzu`
     - :bench:`graphbench.20k.q7.pandas`
     - :bench:`graphbench.20k.q7.polars`
   * - q8
     - 1
     - :bench:`graphbench.20k.q8.kuzu`
     - :bench-diag:`graphbench.20k.q8.pandas`
     - :bench-diag:`graphbench.20k.q8.polars`
   * - q9
     - 1
     - :bench:`graphbench.20k.q9.kuzu`
     - :bench:`graphbench.20k.q9.pandas`
     - :bench:`graphbench.20k.q9.polars`

The 100,000-person board
~~~~~~~~~~~~~~~~~~~~~~~~

The same queries run on a graph with 100,000 people.

.. list-table::
   :header-rows: 1
   :widths: 10 12 20 20 20

   * - Query
     - Result rows
     - Kuzu
     - GFQL ``pandas``
     - GFQL ``polars``
   * - q1
     - 3
     - :bench:`graphbench.100k.q1.kuzu`
     - :bench:`graphbench.100k.q1.pandas`
     - :bench:`graphbench.100k.q1.polars`
   * - q2
     - 1
     - :bench:`graphbench.100k.q2.kuzu`
     - :bench:`graphbench.100k.q2.pandas`
     - :bench:`graphbench.100k.q2.polars`
   * - q3
     - 5
     - :bench:`graphbench.100k.q3.kuzu`
     - :bench:`graphbench.100k.q3.pandas`
     - :bench:`graphbench.100k.q3.polars`
   * - q4
     - 3
     - :bench:`graphbench.100k.q4.kuzu`
     - :bench:`graphbench.100k.q4.pandas`
     - :bench:`graphbench.100k.q4.polars`
   * - q5
     - 1
     - :bench:`graphbench.100k.q5.kuzu`
     - :bench:`graphbench.100k.q5.pandas`
     - :bench:`graphbench.100k.q5.polars`
   * - q6
     - 5
     - :bench:`graphbench.100k.q6.kuzu`
     - :bench:`graphbench.100k.q6.pandas`
     - :bench:`graphbench.100k.q6.polars`
   * - q7
     - 1
     - :bench:`graphbench.100k.q7.kuzu`
     - :bench:`graphbench.100k.q7.pandas`
     - :bench:`graphbench.100k.q7.polars`
   * - q8
     - 1
     - :bench:`graphbench.100k.q8.kuzu`
     - :bench-diag:`graphbench.100k.q8.pandas`
     - :bench-diag:`graphbench.100k.q8.polars`
   * - q9
     - 1
     - :bench:`graphbench.100k.q9.kuzu`
     - :bench:`graphbench.100k.q9.pandas`
     - :bench:`graphbench.100k.q9.polars`

Provenance
~~~~~~~~~~

.. bench-provenance:: graphbench-q1q9-20k-20260726

.. bench-provenance:: graphbench-q1q9-100k-20260726

.. bench-disclosures::

The values come from committed `pyg-bench <https://github.com/graphistry/pyg-bench>`_
artifacts. The documentation build and ``docs/test_bench_numbers.py`` reject missing,
stale, or unpublished values.

.. _gfql-bulk-sweep:

Bulk engine comparison
----------------------

These measurements predate version 0.58.0. They use the SNAP
**com-LiveJournal** (35M edges) and **com-Orkut** (117M edges) graphs.

The table shows the median time after warm-up for the same query and result on
each engine. Orkut has 3.1M nodes and 117M edges.

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

- Polars CPU reduced the one-hop time from 2613 ms to 68 ms and the aggregation
  time from 799 ms to 205 ms, without a GPU.
- Polars builds one lazy query plan. cuDF executes each step separately, so its GPU
  launch and intermediate-frame costs are larger on these workloads.
- Polars GPU was fastest for the 10K-seed two-hop query and the aggregation. cuDF
  was fastest for the 100K-seed query, which produced about 85M rows.
- On LiveJournal, the 10K-seed one-hop query took 1129 ms on pandas and 37 ms on
  Polars. Across 10K, 100K, and 1M-edge samples, Polars became faster as the graph
  grew. Pandas was faster only for a node filter that took less than 1 ms.
  Reproducer: ``benchmarks/gfql/index_crossover_bench.py``.

Method
~~~~~~

- Host: NVIDIA DGX Spark (GB10 Grace-Blackwell, unified memory), RAPIDS container
  ``graphistry/test-rapids-official:26.02-gfql-polars``.
- Datasets: `SNAP <https://snap.stanford.edu/data/>`_ **com-LiveJournal** (35M edges),
  **com-Orkut** (117M edges).
- Measurement: warm median after two warm-ups, with five timed runs on Orkut and
  eight on LiveJournal. Each engine returned the same result rows.
- Reproduce: ``benchmarks/gfql/index_bulk_olap_bench.py`` (engine comparison),
  ``benchmarks/gfql/pandas_vs_polars.py``, and ``benchmarks/gfql/index_vs_kuzu_prepared.py``
  (vs kuzu). Numbers on this page are rendered from saved runs; the page does not re-run
  them.
- **LadybugDB comparison:** LadybugDB 0.18.1 and GFQL with Polars ran in one
  session on the same host and generated 5M-node, 20M-edge graph. The queries use
  `LadybugDB/kuzu-ladybug-benchmark <https://github.com/LadybugDB/kuzu-ladybug-benchmark>`_
  and returned matching values. GFQL was faster for a full node scan (59.0 ms vs
  364.3 ms) and a 1,001-row range scan (5.1 ms vs 7.6 ms). LadybugDB was faster
  for indexed point lookups and a cached relationship count. Reproducer:
  ``benchmarks/gfql/bench_ladybug_cypher.py``.

Engine choice depends on the workload. Polars usually became faster than pandas
as graphs grew past 10K edges, while pandas remained faster for operations under
1 ms. See :doc:`engines` for selection guidance. See
:doc:`benchmark_filter_pagerank` for the Neo4j pipeline comparison and
:doc:`benchmark_graphframes` for the Spark GraphFrames comparison.

How GFQL is fast
----------------

GFQL joins tables of nodes and edges in batches instead of following one path at a
time. This lets dataframe engines process many records together.

GFQL stores graph data in columnar frames based on
`Apache Arrow <https://arrow.apache.org/>`_. Polars combines operations into one lazy
plan. cuDF and Polars GPU run columnar operations in parallel on NVIDIA GPUs. GPUs
help most on large joins, queries that visit many nodes, and full-graph aggregation,
where the work outweighs the cost of starting GPU operations.

Start on CPU with no special hardware, and move to a GPU engine by changing one keyword when
the graph or result becomes large. See :doc:`engines` for selection guidance.

.. note::
   Same-path constraints (``where``) can be more expensive on dense graphs.
   Prefer selective per-step predicates and see :doc:`/gfql/where` for details.

Next steps
----------

- **Choose an engine**: :doc:`engines` — the full decision matrix and qualitative guidance.
- **Selective lookups**: :doc:`index_adjacency` — the adjacency index used above.
- **End-to-end benchmark**: :doc:`benchmark_filter_pagerank` — CPU/GPU vs Neo4j+GDS.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.
- **Ecosystem**: `Apache Arrow <https://arrow.apache.org/>`_ and `NVIDIA RAPIDS <https://rapids.ai/>`_.
