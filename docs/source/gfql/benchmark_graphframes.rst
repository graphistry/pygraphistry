GFQL vs Apache Spark GraphFrames on One Node
============================================

.. image:: _static/gfql-mascot.png
   :alt: GFQL mascot
   :width: 160px
   :align: right

This page benchmarks GFQL against Apache Spark GraphFrames on one machine.
GFQL is Graphistry's open-source graph query language: Cypher and Python
chains that run in-process on dataframes, with no database or cluster.
GraphFrames is Spark's graph library, run here on ``local[*]`` (a single-node
JVM using all cores). The workload is four tasks on two graphs, LiveJournal
(35M edges) and Orkut (117M edges). GFQL's best engine is faster than
GraphFrames in all eight cells. The CPU engine alone is faster in six of the
eight: graph filters and k-hop traversals run 1.3x to 43x faster. The two
exceptions are whole-graph PageRank on CPU, where GraphFrames beats GFQL's
igraph path. On that task the GFQL GPU engine is 10x to 15x faster than
GraphFrames. Use GFQL on CPU for filters and traversals, and on GPU for
PageRank.

.. image:: _static/graphframes/livejournal_tasks.svg
   :alt: LiveJournal task times: GFQL CPU and GPU versus GraphFrames for filter, 1-hop, 2-hop, and PageRank

.. image:: _static/graphframes/orkut_tasks.svg
   :alt: Orkut task times: GFQL CPU and GPU versus GraphFrames for filter, 1-hop, 2-hop, and PageRank

GFQL runs with ``engine="polars"`` (CPU) and ``engine="polars-gpu"`` (GPU).
Every cell is the median of 5 timed runs after 2 warmups, and every task
returns the same result size on all three systems. One cell, LiveJournal GPU
PageRank, is the median of 3 runs after 1 warmup. See
:ref:`graphframes-method` for the full measurement rules.

LiveJournal (35M edges)
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 20

   * - Task
     - GFQL polars (CPU)
     - GFQL polars-gpu (GPU)
     - GraphFrames (local[*])
     - Best GFQL vs GraphFrames
   * - **filter** (degree >= 42)
     - 2.1ms
     - 2.4ms
     - 90.4ms
     - 43x
   * - **1-hop** (50 seeds)
     - 236.8ms
     - 191.4ms
     - 1421.7ms
     - 7.4x
   * - **2-hop** (50 seeds)
     - 1669.3ms
     - 1542.1ms
     - 3583.3ms
     - 2.3x
   * - **PageRank** (full graph)
     - 49.3s
     - 1.11s
     - 16.3s
     - 14.7x (GPU); CPU is 0.33x

Cold load of the SNAP file: 2.4s for GFQL, 10.3s for GraphFrames.

Orkut (117M edges)
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 20

   * - Task
     - GFQL polars (CPU)
     - GFQL polars-gpu (GPU)
     - GraphFrames (local[*])
     - Best GFQL vs GraphFrames
   * - **filter** (degree >= 162)
     - 1.7ms
     - 2.0ms
     - 70.6ms
     - 42x
   * - **1-hop** (50 seeds)
     - 562.9ms
     - 442.0ms
     - 3826.6ms
     - 8.7x
   * - **2-hop** (50 seeds)
     - 9439.8ms
     - 8860.2ms
     - 11582.9ms
     - 1.3x
   * - **PageRank** (full graph)
     - 160.1s
     - 3.50s
     - 36.8s
     - 10.5x (GPU); CPU is 0.23x

Cold load: 5.1s for GFQL, 14.7s for GraphFrames.

Result sizes are identical across the three systems for every task:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Graph
     - filter
     - 1-hop
     - 2-hop
     - PageRank
   * - LiveJournal
     - 403,561
     - 119,877
     - 1,378,430
     - 3,997,962
   * - Orkut
     - 308,666
     - 434,973
     - 1,991,366
     - 3,072,441

Which engine to use
-------------------

- **Filter and traversal**: use GFQL on CPU. The GPU changes these times
  little, because at these result sizes data movement, not compute, sets the
  floor. Spark's per-query scheduling and shuffle cost dominates its time.
- **Whole-graph PageRank**: use GFQL on GPU (``engine="polars-gpu"``, cugraph).
- **PageRank without a GPU**: GFQL routes the CPU path through single-threaded
  igraph. It is 3x to 4x slower than GraphFrames at these sizes, and the gap
  grows with graph size. Use it for convenience, not for speed.
- **Larger than one node's memory**: see :ref:`graphframes-friendster`.

The tasks
---------

**filter**: keep nodes with ``degree >= threshold``. SNAP graphs have no
attributes, so both systems compute ``degree`` during cold load. The load time
carries that cost, not the query. The shared threshold makes the filter
identical across systems.

.. doc-test: skip

.. code-block:: python

   # GFQL
   from graphistry import n
   from graphistry.compute.predicates.numeric import ge
   g.gfql([n(filter_dict={'degree': ge(42)})], engine="polars")  # or "polars-gpu"

   # GraphFrames
   gf.degrees.filter("degree >= 42").count()

**1-hop** and **2-hop**: undirected expansion from a fixed set of 50
high-degree seed nodes.

.. doc-test: skip

.. code-block:: python

   # GFQL
   from graphistry import n, e_undirected
   g.gfql([n(filter_dict={'id': is_in(seeds)}), e_undirected(hops=1), n()], engine="polars")

GraphFrames has no k-hop primitive. Its ``bfs`` finds shortest paths between
predicates and ``find`` matches a fixed motif. The Spark side therefore expands
with one iterated undirected edge join per hop and ends in ``.count()``.

**PageRank**: full graph, damping 0.85. GFQL CPU calls
``g.compute_igraph('pagerank')``. GFQL GPU calls
``g.compute_cugraph('pagerank')``. GraphFrames calls
``gf.pageRank(resetProbability=0.15, maxIter=20)``. The three engines return
the same vertex set and the same ranking. On LiveJournal, pairwise Spearman rho
is 1.00 and the top-100 overlap is 100 of 100
(``_static/graphframes/bench_graphframes_pagerank_parity.json``).

.. _graphframes-friendster:

Friendster (1.8B edges): not measured
-------------------------------------

Friendster has 1,806,067,135 edges and 65,608,366 nodes
(`SNAP <https://snap.stanford.edu/data/com-Friendster.html>`_). No system
completed a task on the test node (about 120 GB unified memory):

- **GFQL polars (CPU)**: the harness loads the full edge list into a pandas
  frame (about 29 GB) and makes a second pass for degrees. This exceeds physical
  RAM before the query runs.
- **GFQL polars-gpu (GPU)**: a direct cudf edge read also exceeds the roughly 120 GB
  unified memory pool.
- **GraphFrames (local[*])**: a 90 GB driver heap swaps and does not finish in
  usable time.

This is a limit of the eager in-memory harness, not a measured engine limit.
GFQL has two opt-in larger-than-memory paths that this benchmark did not use.
``GFQL_POLARS_CPU_STREAMING=1`` selects the Polars streaming engine, which
spills to disk. ``GFQL_POLARS_GPU_EXECUTOR=streaming`` selects the cudf-polars
streaming executor for results larger than device memory. Both need a lazy
source such as ``pl.scan_parquet`` instead of an eager ``pandas.read_parquet``.
Measuring that path at 1.8B edges is follow-up work.

.. _graphframes-method:

Method and limits
-----------------

- **Scope**: single node, in memory. ``local[*]`` is Spark's single-node mode.
  A cluster amortizes scheduling and shuffle cost across machines and changes
  the trade-off at larger scale. Use a Spark cluster when the data already
  lives there or the graph exceeds one node's memory, including the streaming
  paths above.
- **Timing**: median of 5 runs after 2 warmups per cell. LiveJournal GPU
  PageRank is median of 3 after 1 warmup, rerun after a transient GPU fault.
  Cold load is timed once, separately.
- **Materialization**: Spark is lazy, so every task ends in ``.count()`` or
  ``.vertices.count()``. GFQL materializes with ``len(_nodes)`` or
  ``len(_edges)``.
- **Conversion cost**: GFQL holds edges as pandas and converts to Polars inside
  the timed region on each call. This counts against GFQL.
- **PageRank convergence**: GraphFrames runs a fixed ``maxIter=20``. igraph
  stops at ``eps=1e-3`` and cugraph at ``tol=1e-5``. Times compare
  wall-clock to a usable ranking; the rankings agree as shown above.
- **Result parity**: each task returns the same result size on all systems
  (table above). A mismatch is treated as a bug, not a result.
- **Run order**: all GFQL cells ran in one block, then all GraphFrames cells,
  on a shared machine. Only medians are kept.
- **Errors**: each (system, task) cell records a status on error or OOM and the
  matrix continues. Missing pyspark, graphframes, or GPU skips the cell.
- **Environment**: ``dgx-spark`` (GB10 GPU, about 120 GB unified memory);
  GraphFrames ``0.8.4-spark3.5-s_2.12``; PySpark ``3.5.1``.

Reproduce
---------

This page renders saved results from ``_static/graphframes/results.json``. The
harness is ``benchmarks/gfql/bench_graphframes.py`` (design notes in
``benchmarks/gfql/bench_graphframes_DESIGN.md``). From ``benchmarks/gfql/``,
with the GraphFrames jar on the Spark classpath via ``GRAPHFRAMES_JAR``:

.. code-block:: bash

   python bench_graphframes.py --dataset lj \
       --systems gfql-polars,gfql-polars-gpu,graphframes \
       --tasks filter,hop1,hop2,pagerank \
       --filter-threshold 42 --warmups 2 --iters 5

Orkut uses ``--dataset orkut --filter-threshold 162``.

See also
--------

- :doc:`engines`: choosing pandas, Polars, cuDF, or Polars-GPU
- :doc:`benchmark_filter_pagerank`: GFQL CPU/GPU vs Neo4j + GDS
- :doc:`cypher`: Cypher syntax through ``g.gfql("MATCH ...")``
- :doc:`overview`: GFQL design and features
