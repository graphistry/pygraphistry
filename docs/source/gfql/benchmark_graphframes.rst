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
larger-than-memory size measured last. Every number below renders from a
committed pyg-bench receipt; the Measurement block at the end names the runs, hosts, and
commits.

**Where it stands.** The single-server ceiling measured here is Friendster:
1,806,067,135 edges bound from a lazy Polars scan, a degree filter in
:bench:`graphframes.friendster.filter.gfql_polars`, a 1-hop from 50 hub seeds in
:bench:`graphframes.friendster.hop1.gfql_polars`, and a 2-hop in
:bench:`graphframes.friendster.hop2.gfql_polars` on the CPU streaming path, with resident
memory peaking at 106 GB of the 119 GB host; the GPU path stops at the 1-hop, PageRank
does not fit on either path, and GraphFrames on ``local[*]`` did not load the graph at
all. Below that ceiling the picture is mixed and both sides are printed: on whole-graph
PageRank GFQL on the GPU is
:bench:`graphframes.lj.pagerank.gfql_polars_gpu_vs_graphframes` faster than GraphFrames
on LiveJournal and :bench:`graphframes.orkut.pagerank.gfql_polars_gpu_vs_graphframes` on
Orkut, while GFQL on the CPU loses PageRank on both
(:bench:`graphframes.lj.pagerank.gfql_polars_vs_graphframes` and
:bench:`graphframes.orkut.pagerank.gfql_polars_vs_graphframes` of GraphFrames' speed):
on both paths the solver is a small part of the time (the shaded bars) and the rest is
the conversion into the solver's graph and the join of scores back onto the nodes; on
degree filters and 1-hop the CPU engine is faster on both graphs; on 2-hop GraphFrames
wins on both (:bench:`graphframes.lj.hop2.gfql_polars_vs_graphframes` and
:bench:`graphframes.orkut.hop2.gfql_polars_vs_graphframes` of its speed). The GFQL
filter and hop rows were measured at the head of the fix for
`#2023 <https://github.com/graphistry/pygraphistry/issues/2023>`_
(`#2024 <https://github.com/graphistry/pygraphistry/pull/2024>`_, not yet on master when
measured); the released code's LiveJournal 2-hop was
:bench-diag:`graphframes_059.lj.hop2.gfql_polars`, the before-state the disclosures keep.

.. image:: _static/graphframes/livejournal_tasks.svg
   :alt: LiveJournal task times: GFQL and GraphFrames for filter, 1-hop, 2-hop, and PageRank, with the PageRank solver time shaded inside the GFQL bar

.. image:: _static/graphframes/orkut_tasks.svg
   :alt: Orkut task times: GFQL and GraphFrames for filter, 1-hop, 2-hop, and PageRank, with the PageRank solver time shaded inside the GFQL bar

.. image:: _static/graphframes/friendster_tasks.svg
   :alt: Friendster task times: GFQL CPU streaming filter, 1-hop, and 2-hop; PageRank and GraphFrames not measured

GFQL binds each graph from a lazy Polars scan of the edge parquet and runs the filter
and hop tasks with ``engine="polars"`` under the Polars CPU streaming collect, or with
``engine="polars-gpu"`` under the cudf-polars streaming executor. PageRank re-binds an
eager copy outside the timer and calls cuGraph on the GPU or igraph on the CPU. The
streaming collect is not a tax: with the same commit and protocol the eager collect
matched it on filter and 2-hop and was slower on 1-hop (the receipts are named in the
Measurement block). Every cell is the median of 5 timed runs after 2 warmups, and every
task returns the same result size on every system that ran it. Times are milliseconds
unless marked; lower is better.

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

The GPU streaming executor is slower than the CPU streaming collect on both hops here
(:bench:`graphframes.lj.hop1.gfql_polars_gpu` against
:bench:`graphframes.lj.hop1.gfql_polars`); at these result sizes the work is data
movement, and the GPU column is a loss for traversal.

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
     - 6,585,312 nodes
   * - **1-hop** (50 seeds)
     - :bench:`graphframes.friendster.hop1.gfql_polars`
     - 166,615 nodes
   * - **2-hop** (50 seeds)
     - :bench:`graphframes.friendster.hop2.gfql_polars`
     - 15,878,312 nodes
   * - **PageRank**
     - not attempted
     - see :ref:`graphframes-friendster`

Result sizes agree across the systems that ran each task, as recorded in the receipts:

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
   * - Friendster (GFQL only)
     - 6,585,312
     - 166,615
     - 15,878,312
     - not attempted

Which engine to use
-------------------

- **Whole-graph PageRank**: use GFQL on GPU (``engine="polars-gpu"``, cuGraph). The
  solver is a small share of the GFQL time; the rest is the conversion of the edge
  frame and the join of scores back onto the nodes, which is where the next gains are.
- **Filter and 1-hop**: use GFQL on CPU (``engine="polars"``). It is faster than
  GraphFrames on both graphs, and the GPU streaming executor does not help at these
  result sizes.
- **2-hop from hub seeds**: GraphFrames wins on both graphs today. GFQL's cost is the
  wavefront seed-rediscovery rule evaluated over the traversed ball; #2024 removed the
  interpreter loop, and the remaining gap is the rule itself.
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
so both systems compute ``degree`` during load. The load carries that cost, not the
query. The shared threshold makes the filter identical across systems.

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

Friendster has 1,806,067,135 edges and 65,608,366 nodes
(`SNAP <https://snap.stanford.edu/data/com-Friendster.html>`_). The eager harness that
produced the earlier version of this page could not load it on the test node (about 119
GB unified memory): a pandas edge frame plus a second pass for degrees exceeds physical
RAM, a direct cuDF read exceeds the unified pool, and a 90 GB Spark driver heap swaps.

The harness binds from ``pl.scan_parquet`` and collects through GFQL's streaming paths
(``GFQL_POLARS_CPU_STREAMING=1`` for the Polars streaming engine,
``GFQL_POLARS_GPU_EXECUTOR=streaming`` for the cudf-polars streaming executor), with a
peak-memory receipt at every size. On Friendster the CPU streaming run loaded the graph
(scan plus degree pass in about 20 seconds, 56 GB resident), answered the degree filter
and the 1-hop from 50 hub seeds (table above), and peaked at 106 GB resident after the
1-hop; a second run answered the 2-hop, a 15,878,312-node ball, in
:bench:`graphframes.friendster.hop2.gfql_polars` at 69.5 GB resident. The streaming
collect keeps the load out of memory, but the traversal still materializes the edges it
touches, and that is where the GPU path stops: the cudf-polars streaming executor
completed the degree filter, then the memory guard ended the run during the 1-hop at 106
GB resident, so the GPU column has no Friendster cell. Whole-graph PageRank does not fit
on either path: the GPU preflight refused it (an estimated 87 GB peak against an 80 GB
budget), and the CPU path was not attempted because its Orkut row already projects the
igraph conversion past the host at fifteen times the edges. That is the single-server
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
- **Receipts**: one run at a time under a host lock, after two clean checks five
  minutes apart; a load monitor samples the host every second and a classifier
  invalidates the run if a process outside the benchmark ran during it. Invalidated
  attempts stay in the package under ``stale-attempts/``; one Orkut GraphFrames run is
  valid by reclassification after the classifier learned that Spark's own shutdown
  cleanup is the benchmark's process (``RECLASSIFIED.txt`` in that run's directory).
- **Harness**: the GFQL streaming harness and every receipt live in pyg-bench; the
  GraphFrames baseline is ``benchmarks/gfql/bench_graphframes.py --systems graphframes``
  in this repository, run from a host Spark with the GraphFrames assembly jar.

Provenance
----------

Every figure on this page is printed from ``docs/source/_data/gfql_benchmarks.json``,
which pyg-bench publishes. The documentation build and ``docs/test_bench_numbers.py``
reject missing, stale, or unpublished values.

.. bench-provenance:: graphframes-ladder-20260904 graphframes-ladder-059-hops-20260904
   :disclosures:

See also
--------

- :doc:`engines`: choosing pandas, Polars, cuDF, or Polars-GPU
- :doc:`benchmark_filter_pagerank`: GFQL CPU/GPU vs Neo4j + GDS
- :doc:`performance`: the q1–q9 boards against Kuzu, Memgraph, and Neo4j
- :doc:`cypher`: Cypher syntax through ``g.gfql("MATCH ...")``
- :doc:`overview`: GFQL design and features
