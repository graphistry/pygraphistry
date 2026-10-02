GFQL vs Apache Spark GraphFrames on One Node
============================================

.. image:: _static/gfql-mascot.png
   :alt: GFQL mascot
   :width: 160px
   :align: right

This page compares GFQL with Apache Spark GraphFrames on one machine. GFQL is
Graphistry's open-source graph query language: Cypher and Python chains that run
in-process on dataframes, with no database or cluster. GraphFrames is Spark's graph
library, run here on ``local[*]``, a single-node JVM using all cores. The workload is
four tasks on two SNAP graphs, LiveJournal and Orkut, with Friendster as the
larger-than-memory size measured last.

**GFQL queries billion-edge graphs on one machine.** On Orkut (117M edges), GFQL
filters the graph and expands one hop in under a second. On Friendster (1.8B edges,
65.6M nodes), GFQL filters in :bench:`graphframes.friendster.filter.gfql_polars` and
expands one hop in :bench:`graphframes.friendster.hop1.gfql_polars`. GraphFrames did
not load Friendster at all.

GFQL wins some tasks and loses others. Both sides are shown here:

- **PageRank: GFQL on GPU wins.** It is
  :bench:`graphframes.lj.pagerank.gfql_polars_gpu_vs_graphframes` faster than
  GraphFrames on LiveJournal and
  :bench:`graphframes.orkut.pagerank.gfql_polars_gpu_vs_graphframes` faster on Orkut.
- **Filter and 1-hop: GFQL on CPU wins** on both graphs.
- **2-hop: GraphFrames wins** on both graphs.
- **PageRank without a GPU: GraphFrames wins** on both graphs.

.. image:: _static/graphframes/livejournal_tasks.svg
   :alt: LiveJournal task times: GFQL and GraphFrames for filter, 1-hop, 2-hop, and PageRank, with the PageRank solver time shaded inside the GFQL bar

.. image:: _static/graphframes/orkut_tasks.svg
   :alt: Orkut task times: GFQL and GraphFrames for filter, 1-hop, 2-hop, and PageRank, with the PageRank solver time shaded inside the GFQL bar

.. image:: _static/graphframes/friendster_tasks.svg
   :alt: Friendster task times: GFQL CPU streaming filter, 1-hop, and 2-hop; PageRank and GraphFrames not measured

LiveJournal
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - Task
     - GFQL polars (CPU)
     - GFQL polars-gpu (GPU)
     - GraphFrames (local[*])
     - GFQL CPU vs GraphFrames
   * - **filter** (degree >= 42)
     - :bench:`graphframes.lj.filter.gfql_polars`
     - :bench:`graphframes.lj.filter.gfql_polars_gpu`
     - :bench:`graphframes.lj.filter.graphframes`
     - :bench:`graphframes.lj.filter.gfql_polars_vs_graphframes`
   * - **1-hop** (50 seeds)
     - :bench:`graphframes.lj.hop1.gfql_polars`
     - :bench:`graphframes.lj.hop1.gfql_polars_gpu`
     - :bench:`graphframes.lj.hop1.graphframes`
     - :bench:`graphframes.lj.hop1.gfql_polars_vs_graphframes`
   * - **2-hop** (50 seeds)
     - :bench:`graphframes.lj.hop2.gfql_polars`
     - :bench:`graphframes.lj.hop2.gfql_polars_gpu`
     - :bench:`graphframes.lj.hop2.graphframes`
     - :bench:`graphframes.lj.hop2.gfql_polars_vs_graphframes` (GraphFrames wins)
   * - **PageRank** (full graph)
     - :bench:`graphframes.lj.pagerank.gfql_polars`; solver :bench-diag:`graphframes.lj.pagerank.gfql_polars_kernel`
     - :bench:`graphframes.lj.pagerank.gfql_polars_gpu`; solver :bench-diag:`graphframes.lj.pagerank.gfql_polars_gpu_kernel`
     - :bench:`graphframes.lj.pagerank.graphframes`
     - GPU: :bench:`graphframes.lj.pagerank.gfql_polars_gpu_vs_graphframes`; CPU: :bench:`graphframes.lj.pagerank.gfql_polars_vs_graphframes` (GraphFrames wins)

Orkut
-----

.. list-table::
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - Task
     - GFQL polars (CPU)
     - GFQL polars-gpu (GPU)
     - GraphFrames (local[*])
     - GFQL CPU vs GraphFrames
   * - **filter** (degree >= 162)
     - :bench:`graphframes.orkut.filter.gfql_polars`
     - :bench:`graphframes.orkut.filter.gfql_polars_gpu`
     - :bench:`graphframes.orkut.filter.graphframes`
     - :bench:`graphframes.orkut.filter.gfql_polars_vs_graphframes`
   * - **1-hop** (50 seeds)
     - :bench:`graphframes.orkut.hop1.gfql_polars`
     - :bench:`graphframes.orkut.hop1.gfql_polars_gpu`
     - :bench:`graphframes.orkut.hop1.graphframes`
     - :bench:`graphframes.orkut.hop1.gfql_polars_vs_graphframes`
   * - **2-hop** (50 seeds)
     - :bench:`graphframes.orkut.hop2.gfql_polars`
     - :bench:`graphframes.orkut.hop2.gfql_polars_gpu`
     - :bench:`graphframes.orkut.hop2.graphframes`
     - :bench:`graphframes.orkut.hop2.gfql_polars_vs_graphframes` (GraphFrames wins)
   * - **PageRank** (full graph)
     - :bench:`graphframes.orkut.pagerank.gfql_polars`; solver :bench-diag:`graphframes.orkut.pagerank.gfql_polars_kernel`
     - :bench:`graphframes.orkut.pagerank.gfql_polars_gpu`; solver :bench-diag:`graphframes.orkut.pagerank.gfql_polars_gpu_kernel`
     - :bench:`graphframes.orkut.pagerank.graphframes`
     - GPU: :bench:`graphframes.orkut.pagerank.gfql_polars_gpu_vs_graphframes`; CPU: :bench:`graphframes.orkut.pagerank.gfql_polars_vs_graphframes` (GraphFrames wins)

Friendster
----------

On the Polars CPU streaming path; the GPU path stopped at the 1-hop and no other system
ran (see :ref:`graphframes-friendster`).

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Task
     - GFQL polars (CPU)
     - Result
   * - **filter** (degree >= 148, the 90th percentile)
     - :bench:`graphframes.friendster.filter.gfql_polars`
     - 6.6M nodes
   * - **1-hop** (50 seeds)
     - :bench:`graphframes.friendster.hop1.gfql_polars`
     - 166.6k nodes
   * - **2-hop** (50 seeds)
     - :bench:`graphframes.friendster.hop2.gfql_polars`
     - 15.9M nodes
   * - **PageRank**
     - not attempted
     - see :ref:`graphframes-friendster`

Result sizes agree across the systems that ran each task, as recorded in the run records:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Graph
     - filter
     - 1-hop
     - 2-hop
     - PageRank
   * - LiveJournal
     - 403.6k
     - 119.9k
     - 1.4M
     - 4.0M
   * - Orkut
     - 308.7k
     - 435.0k
     - 2.0M
     - 3.1M
   * - Friendster (GFQL only)
     - 6.6M
     - 166.6k
     - 15.9M
     - not attempted

Which engine to use
-------------------

- **Whole-graph PageRank**: use GFQL on GPU (``engine="polars-gpu"``, cuGraph). The
  solver is a small share of the GFQL time; the rest is the conversion of the edge
  frame and the join of scores back onto the nodes.
- **Filter and 1-hop**: use GFQL on CPU (``engine="polars"``). It is faster than
  GraphFrames on both graphs, and the GPU streaming executor does not help at these
  result sizes.
- **2-hop from hub seeds**: GraphFrames wins on both graphs today.
- **PageRank without a GPU**: GFQL routes the CPU path through igraph, and loses to
  GraphFrames on both graphs. The igraph solver itself is
  :bench-diag:`graphframes.lj.pagerank.gfql_polars_kernel` of the
  :bench:`graphframes.lj.pagerank.gfql_polars` LiveJournal row; the rest is the
  conversion into igraph and the join-back, tracked in
  `#2032 <https://github.com/graphistry/pygraphistry/issues/2032>`_. Use the CPU path
  for convenience, not for speed.
- **Larger than one node's memory**: see :ref:`graphframes-friendster`.

The tasks
---------

**filter**: keep nodes with ``degree >= threshold``. SNAP graphs have no attributes,
so both systems compute ``degree`` during load.

.. doc-test: skip

.. code-block:: python

   # GFQL
   from graphistry import n
   from graphistry.compute.predicates.numeric import ge
   g.gfql([n(filter_dict={'degree': ge(42)})], engine="polars")  # or "polars-gpu"

   # GraphFrames
   gf.degrees.filter("degree >= 42").count()

**1-hop** and **2-hop**: undirected expansion from a fixed set of 50 high-degree seed
nodes.

.. doc-test: skip

.. code-block:: python

   # GFQL
   from graphistry import n, e_undirected
   g.gfql([n(filter_dict={'id': is_in(seeds)}), e_undirected(hops=1), n()], engine="polars")

GraphFrames has no k-hop primitive. Its ``bfs`` finds shortest paths between predicates
and ``find`` matches a fixed motif. The Spark side therefore expands with one iterated
undirected edge join per hop and ends in ``.count()``.

**PageRank**: full graph, damping 0.85. GFQL GPU calls
``g.compute_cugraph('pagerank')`` on an eager cuDF copy of the graph; GFQL CPU calls
``g.compute_igraph('pagerank')`` on an eager pandas copy. GraphFrames calls
``gf.pageRank(resetProbability=0.15, maxIter=20)``. All return the full vertex set. The
shaded part of a GFQL PageRank bar is the solver alone (cuGraph or igraph) on a graph
object built outside the timer; the light part is the rest of the query, which is the
conversion into that graph object and the join of scores back onto the nodes.

.. _graphframes-friendster:

Friendster (1.8B edges): the ceiling
------------------------------------

Friendster has 1.8B edges and 65.6M nodes
(`SNAP <https://snap.stanford.edu/data/com-Friendster.html>`_). The eager harness that
produced the earlier version of this page could not load it on the test node (about 119
GB unified memory): a pandas edge frame plus a second pass for degrees exceeds physical
RAM, a direct cuDF read exceeds the unified pool, and a 90 GB Spark driver heap swaps.

The harness binds from ``pl.scan_parquet`` and collects through GFQL's streaming paths
(``GFQL_POLARS_CPU_STREAMING=1`` for the Polars streaming engine,
``GFQL_POLARS_GPU_EXECUTOR=streaming`` for the cudf-polars streaming executor), with a
peak-memory record at every size. On Friendster the CPU streaming run loaded the graph
(scan plus degree pass in about 20 seconds, 55.0 GiB resident), answered the degree filter
and the 1-hop from 50 hub seeds (table above), and peaked at 103.6 GiB resident after the
1-hop; a second run answered the 2-hop, a 15.9M-node ball, in
:bench:`graphframes.friendster.hop2.gfql_polars` at 67.9 GiB resident. The streaming
collect keeps the load out of memory, but the traversal still materializes the edges it
touches, and that is where the GPU path stops: the cudf-polars streaming executor
completed the degree filter at 103.8 GiB resident, then the watchdog ended the run during
the 1-hop when host free memory fell to 17 GB against its 20 GB floor, so the GPU column
has no Friendster cell. Whole-graph PageRank does not fit
on either path: the GPU preflight refused it (an estimated 87 GB peak against an 80 GB
budget), and the CPU path was not attempted: its Orkut row peaked at 29.9 GiB resident
for 117M edges, and Friendster has fifteen times the edges. That is the single-server
ceiling this page measured. GraphFrames on ``local[*]`` stays at the boundary it hit
above.

.. _graphframes-method:

Method and limits
-----------------

- **Scope**: single node, in memory. ``local[*]`` is Spark's single-node mode. A cluster
  amortizes scheduling and shuffle cost across machines and changes the trade-off at
  larger scale. Use a Spark cluster when the data already lives there or the graph
  exceeds one node's memory.
- **Timing**: median of 5 runs after 2 warmups per cell, each system loaded once and
  resident across iterations. Load is not timed.
- **Materialization**: Spark is lazy, so every task ends in ``.count()`` or
  ``.vertices.count()``. GFQL materializes with ``len(_nodes)``.
- **Comparability**: a task is comparable only when every system reports the same
  result size; a cell that disagrees is published as a direct time with a disclosure.
  Cells marked diagnostic are never quoted as GFQL's number.
- **PageRank convergence**: GraphFrames runs a fixed ``maxIter=20``; cuGraph runs to
  its default tolerance. Times compare wall-clock to a usable ranking.
- **Run records**: one run at a time under a host lock, after two clean checks five
  minutes apart; a load monitor samples the host every second and a classifier
  invalidates the run if a process outside the benchmark ran during it. Invalidated
  attempts stay in the package under ``stale-attempts/``; one Orkut GraphFrames run is
  valid by reclassification after the classifier learned that Spark's own shutdown
  cleanup is the benchmark's process (``RECLASSIFIED.txt`` in that run's directory).
- **Harness**: the GFQL streaming harness and every run record live in pyg-bench; the
  GraphFrames baseline is ``benchmarks/gfql/bench_graphframes.py --systems graphframes``
  in this repository, run from a host Spark with the GraphFrames assembly jar.

Provenance
----------

Every figure on this page is printed from ``docs/source/_data/gfql_benchmarks.json``,
which pyg-bench publishes. The documentation build and ``docs/test_bench_numbers.py``
reject missing, stale, or unpublished values.

.. bench-provenance:: graphframes-ladder-ship-62df29a8a-20260920 graphframes-ladder-059-hops-20260904
   :fields: measured_at,host
   :disclosures:

See also
--------

- :doc:`engines`: choosing pandas, Polars, cuDF, or Polars-GPU
- :doc:`benchmark_filter_pagerank`: GFQL CPU/GPU vs Neo4j + GDS
- :doc:`performance`: the q1–q9 boards against Kuzu, Memgraph, and Neo4j
- :doc:`cypher`: Cypher syntax through ``g.gfql("MATCH ...")``
- :doc:`overview`: GFQL design and features
