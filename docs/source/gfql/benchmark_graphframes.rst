GFQL Graph Benchmark: DataFrame-Native vs Apache Spark GraphFrames
==================================================================

.. image:: _static/gfql-mascot.png
   :alt: GFQL mascot
   :width: 160px
   :align: right

.. note::

   LiveJournal and Orkut figures are final: median of 5 timed runs after 2
   warmups, result-size parity enforced per task. One cell — LiveJournal GPU
   PageRank — is median of 3 after 1 warmup (a re-run after a transient GPU
   fault on the first pass); every other cell, including Orkut GPU PageRank, is
   the full 5/2. Friendster (~1.8B edges) was the stretch target; our *eager
   in-memory* harness runs out of RAM loading it (documented below) — this is a
   harness/loader limit, not an engine ceiling. Polars' streaming engine and the
   cudf-polars streaming executor are the larger-than-memory paths, not yet
   benchmarked here.

Run graph filters, k-hop neighborhoods, and PageRank directly on Python
dataframes — no cluster required. This benchmark compares **GFQL**
(Graphistry's dataframe-native graph query language) on CPU
(``engine="polars"``) and GPU (``engine="polars-gpu"``) against **Apache Spark
GraphFrames** (``local[*]``, single-node JVM) on the same tasks over large
SNAP graphs.

The short version: for **filter and traversal**, GFQL wins decisively — even on
CPU — because a single-node columnar engine avoids the JVM startup,
task-serialization, and shuffle overhead that dominate Spark at sub-second
result sizes. For **PageRank**, the honest answer is mixed: GFQL's *CPU* path
routes through igraph and is *slower* than GraphFrames at scale; GFQL's win on
PageRank comes from the *GPU* path (cugraph). We state both plainly below.

Headline (LiveJournal, ~35M edges)
----------------------------------

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
     - **~43x**
   * - **1-hop** (50 seeds)
     - 236.8ms
     - 191.4ms
     - 1421.7ms
     - **~7.4x**
   * - **2-hop** (50 seeds)
     - 1669.3ms
     - 1542.1ms
     - 3583.3ms
     - **~2.3x**
   * - **PageRank** (full graph)
     - 49.3s
     - **1.11s**
     - 16.3s
     - **~14.7x** (GPU) / *0.33x* (CPU)

*Median of 5 after 2 warmups (LiveJournal GPU PageRank is median of 3 — see the
note above). DGX* ``dgx-spark``, *GB10 GPU, single node; Spark* ``local[*]``
*over all cores. Cold load (ETL) of the SNAP file is 2.4s for GFQL vs 10.3s for
GraphFrames — GFQL also loads ~4x faster.*

Result-size parity is enforced per task: filter
returns the identical node count above threshold, 1-hop the identical
neighborhood size (**119,877**), 2-hop the identical size (**1,378,430**), and
PageRank the identical vertex count (**3,997,962**). A size mismatch flags a bug
(directedness or seed-set drift), not a speedup.

When GFQL wins, and when it doesn't
-----------------------------------

This page is written for a Spark GraphFrames user evaluating alternatives.
The point is not to spin — it is to be trustworthy. Two findings, both true:

**1. Filter and traversal: GFQL wins across the board (1.3–43x; most cells 2x+), even on CPU.**
There is no JVM to warm, no task graph to serialize, no shuffle to schedule. A
single-node columnar engine is simply the right tool for sub-second graph
queries. Spark's ``local[*]`` per-query scheduler overhead dominates at these
result sizes — Spark is engineered for distributed throughput across a cluster,
not single-node latency. Note the GPU barely moves these numbers: at this scale
the CPU polars path is already fast enough that data movement, not compute, is
the floor.

**2. PageRank: the honest result is mixed — reach for the GPU.**
GFQL's *CPU* path has no native PageRank, so the polars engine converts to
pandas and calls igraph. Single-threaded igraph is **slower than GraphFrames**
at this scale (49.3s vs 16.3s on LiveJournal, and 160s vs 37s on Orkut — the gap
widens with size): Spark's multicore iterative aggregation genuinely beats it.
GFQL's PageRank advantage comes entirely from the **GPU** path (cugraph,
~1.11s), which beats GraphFrames by ~14.7x. So the
guidance is explicit: for whole-graph analytics like PageRank, use the GPU
engine; the CPU-igraph route is a convenience, not a speed play.

If you take one thing away: **GFQL replaces Spark for interactive single-node
graph queries, and the GPU engine additionally replaces it for whole-graph
analytics — but the CPU engine alone does not win PageRank, and we won't
pretend it does.**

filter — WHERE on a degree column
---------------------------------

A ``WHERE`` on a numeric column: keep nodes with ``degree >= threshold``. SNAP
graphs carry no attributes, so ``degree`` is precomputed at cold-load (charged
to load, not to the query, for *both* systems) and used as the natural
threshold column.

.. doc-test: skip

.. code-block:: python

   # GFQL
   from graphistry import n
   from graphistry.compute.predicates.numeric import ge
   g.gfql([n(filter_dict={'degree': ge(42)})], engine="polars")  # or "polars-gpu"

   # GraphFrames
   gf.degrees.filter("degree >= 42").count()

LiveJournal: GFQL polars **2.1ms**, GFQL polars-gpu **2.4ms**, GraphFrames
**90.4ms** — same node count (**403,561**) on a shared degree threshold. The gap
is almost entirely Spark's per-query scheduling floor; the actual predicate is
trivial on both.

1-hop — neighborhood from a 50-node seed set
--------------------------------------------

Undirected 1-hop expansion from a fixed 50-node high-degree seed set.

.. doc-test: skip

.. code-block:: python

   # GFQL
   from graphistry import n, e_undirected
   g.gfql([n(filter_dict={'id': is_in(seeds)}), e_undirected(hops=1), n()], engine="polars")

GraphFrames has no k-hop-neighborhood primitive (``bfs`` is shortest-path
between predicates, ``find`` is a fixed motif), so the Spark side expands via an
iterated undirected edge join — still pure Spark, ending in ``.count()``.

LiveJournal: GFQL polars **236.8ms**, GFQL polars-gpu **191.4ms**, GraphFrames
**1421.7ms**, identical neighborhood size **119,877**.

2-hop — two-hop neighborhood
----------------------------

Same seed set, two undirected hops (``e_undirected(hops=2)`` for GFQL; two
iterated joins for Spark).

LiveJournal: GFQL polars **1669.3ms**, GFQL polars-gpu **1542.1ms**, GraphFrames
**3583.3ms**, identical size **1,378,430**. As the result grows, real join work
starts to dominate Spark's fixed overhead, so the multiple narrows (~2.3x) — but
GFQL still wins on a single node.

PageRank — full-graph analytics
-------------------------------

Full-graph PageRank (damping 0.85). GFQL CPU routes to igraph
(``g.compute_igraph('pagerank')``); GFQL GPU routes to cugraph
(``g.compute_cugraph('pagerank')``); GraphFrames uses
``gf.pageRank(resetProbability=0.15, maxIter=20)``. GraphFrames runs a fixed
20 iterations; igraph and cugraph iterate to their library-default tolerance
(igraph ``eps=1e-3``, cugraph ``tol=1e-5``). This favors neither side
uniformly — it is disclosed so the times are interpretable, not a hidden knob.

LiveJournal (all return **3,997,962** vertices):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Engine / backend
     - Time
     - vs GraphFrames
   * - GFQL polars / igraph (CPU)
     - 49.3s
     - *0.33x (slower)*
   * - GFQL polars-gpu / cugraph (GPU)
     - **1.11s**
     - **~14.7x faster**
   * - GraphFrames (local[*])
     - 16.3s
     - 1.0x

This is the mixed result, stated plainly. The CPU-igraph route is single
threaded and **loses to Spark's multicore aggregation** here. The GPU-cugraph
route wins by an order of magnitude. Because GraphFrames uses a fixed
``maxIter`` while igraph/cugraph iterate to a tolerance, the raw scores are not
bit-identical, so we compare **wall-clock-to-usable-scores**: the three engines
return the identical vertex set (**3,997,962**), and their PageRank rankings
agree **exactly** — pairwise Spearman rho = **1.00** and top-100 overlap
**100/100** across igraph, cugraph, and GraphFrames (parity check saved to
``bench_graphframes_pagerank_parity.json``). This is a "same ranked result, different cost" comparison, not a raced approximation.

Orkut (~117M edges)
-------------------

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
     - **~42x**
   * - **1-hop** (50 seeds)
     - 562.9ms
     - 442.0ms
     - 3826.6ms
     - **~8.7x**
   * - **2-hop** (50 seeds)
     - 9439.8ms
     - 8860.2ms
     - 11582.9ms
     - **~1.3x**
   * - **PageRank** (full graph)
     - 160.1s
     - **3.50s**
     - 36.8s
     - **~10.5x** (GPU) / *0.23x* (CPU)

*Median of 5 after 2 warmups (all cells, including GPU PageRank).
Result-size parity per task: filter* **308,666**; *1-hop* **434,973**; *2-hop*
**1,991,366**; *PageRank* **3,072,441**. *Cold load 5.1s (GFQL) vs 14.7s
(GraphFrames). The pattern holds at 117M edges: GFQL wins filter/traversal
outright, the GPU wins PageRank by ~10x, and CPU-igraph PageRank falls further
behind Spark (0.23x) as the graph grows.*

Friendster (~1.8B edges) — our eager-load harness stops here; streaming is next
--------------------------------------------------------------------------------

Friendster (1,806,067,135 edges, 65.6M nodes) was the stretch target. Every path
we *ran* ran out of headroom on the **119 GB** node — but the honest framing is
that this is where **our benchmark harness's eager, in-memory load** stops, **not
a hard ceiling of the engines.** The harness reads the whole graph into memory up
front (``pandas.read_parquet`` → a ~29 GB edge frame, plus a second ~29 GB pass to
build the degree/node table) *before the query runs*; that materialization is what
the OS kills.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Path (as configured in this harness)
     - Outcome at 1.8B edges on one 119 GB node
   * - GFQL polars (CPU), eager load
     - **OOM in the load**, before the query: the pandas edge frame + degree build
       peak past physical RAM. The *query* engine never runs.
   * - GFQL polars-gpu (GPU), eager cudf load
     - **Exceeds memory in the load**: even a lean cudf-direct edge read drives the
       119 GB unified pool into swap. The in-memory GPU executor is not the
       larger-than-memory path (see below).
   * - GraphFrames (local[*])
     - **Swap-thrash.** A ``local[*]`` driver with a 90 GB heap on a 1.8B-edge
       GraphFrame saturates memory and does not finish in usable time on one box.

**What we did *not* run — the larger-than-memory paths that exist.** GFQL's Polars
engine already ships opt-in streaming escape hatches, and this harness did not use
them:

- **CPU:** ``GFQL_POLARS_CPU_STREAMING=1`` collects the plan with Polars' streaming
  engine (batched, spills to disk), parity-identical to the default. Paired with a
  **lazy** source (``pl.scan_parquet`` instead of an eager ``pandas.read_parquet``),
  the 1.8B-edge input is never fully materialized.
- **GPU:** ``GFQL_POLARS_GPU_EXECUTOR=streaming`` selects the cudf-polars *streaming*
  executor — explicitly the escape hatch for **larger-than-device-memory** results,
  where the default in-memory executor would OOM.

Both are **off by default** because in-scope GFQL graphs/results fit in memory and
streaming regresses small/interactive sizes — the right default for the 35M–117M
regime this page measures. What we have *not yet* done is wire a lazy
``scan_parquet`` ingestion path through GFQL and benchmark the streaming collect at
1.8B; that is the correct larger-than-memory test (comparable to Ladybug's
out-of-core mode and to a Spark cluster) and is **tracked as follow-up work**, not a
limitation we're conceding. So: GFQL wins decisively *in-memory* through ~10^8 edges
here; at ~10^9 the question is streaming-vs-out-of-core-vs-cluster, which we will
measure rather than assert.

Why this matters
----------------

Most graph work in a notebook or a pipeline is single-node and latency
sensitive: filter to a subgraph, expand a few hops, score it. For that regime,
standing up or paying for a Spark cluster is the wrong shape — the per-query
scheduling and serialization cost swamps the actual work. GFQL runs the same
queries in-process on your dataframe, on CPU, and wins by 1.3–43x here
(most cells 2x+; the closest is Orkut's heavy 2-hop at 1.3x).

When the workload shifts to whole-graph analytics like PageRank, the GPU engine
(``engine="polars-gpu"``, cugraph) is the tool that beats Spark — by ~10–15x
(14.7x on LiveJournal, 10.5x on Orkut) — on the same single node. The CPU
engine's PageRank is a convenience for when no GPU is present, not a performance
claim.

**When to go back to Spark.** These *in-memory* numbers hold while the graph and
its intermediates fit in one machine's memory (here, 119 GB unified host/GPU
memory comfortably holds Orkut's 117M edges). Above that, GFQL has two moves
before a cluster: Polars' **streaming engine** (``GFQL_POLARS_CPU_STREAMING=1``,
disk-spill) and the **cudf-polars streaming executor**
(``GFQL_POLARS_GPU_EXECUTOR=streaming``, larger-than-device-memory) — both
opt-in, both untested at 1.8B here (see the Friendster section). A managed Spark
cluster is the right tool when the data already lives there, or when the graph
outgrows even streaming on one node. This page measures the in-memory single-node
regime; it does not claim GFQL replaces a cluster at every scale, nor that
one node is a hard ceiling.

Fairness and caveats (documented, not hidden)
---------------------------------------------

We benchmark the single-node regime where GFQL lives, and we flag every place
that favors or disfavors either side:

- **local[*] is Spark's single-node configuration.** This measures single-box
  multicore, not a distributed cluster. A real cluster amortizes scheduling and
  shuffle overhead across many machines and would change the trade-off,
  especially at larger scales. We are explicitly benchmarking single-node
  latency, which is where GFQL is designed to run.
- **End-to-end materialization on both sides.** Spark is lazy, so every task
  ends in a materializing action (``.count()`` / ``.vertices.count()``) to force
  honest end-to-end timing. GFQL likewise materializes via
  ``len(_nodes)`` / ``len(_edges)``. Both are timed to a real answer, not a lazy
  plan.
- **The pandas→polars conversion is charged to GFQL.** GFQL holds edges as
  pandas and converts to polars *inside* the timed region on each call. This is
  conservative — it counts against GFQL — and is left in deliberately rather
  than pre-converting.
- **PageRank convergence differs (disclosed).** GraphFrames runs a fixed
  ``maxIter=20``; igraph iterates to ``eps=1e-3`` and cugraph to ``tol=1e-5``.
  The comparison is wall-clock-to-usable-scores; we verify all three return the
  identical vertex set and rank it identically (LiveJournal: pairwise Spearman
  rho = 1.00, top-100 overlap 100/100 — ``bench_graphframes_pagerank_parity.json``),
  not per-iteration cost — the algorithms converge to the same ranking at
  different cost.
- **In-memory by default (streaming is opt-in).** These results are the default
  *in-memory* configuration, which assumes the graph fits in one node's RAM — the
  regime this page measures. GFQL does **not** shard across machines, but it *can*
  spill to disk / stream: Polars' streaming engine
  (``GFQL_POLARS_CPU_STREAMING=1``) and the cudf-polars streaming executor
  (``GFQL_POLARS_GPU_EXECUTOR=streaming``) are larger-than-memory paths, off by
  default and not exercised in these numbers. We report the host memory (119 GB)
  so the in-memory envelope is explicit.
- **Runs are blocked, not interleaved.** On this shared box, GFQL and
  GraphFrames were run in separate blocks (all GFQL cells, then all GraphFrames
  cells), not interleaved, and only medians are retained per cell. Validation and
  final medians agreed within run-to-run noise; we report medians, consistent
  with :doc:`benchmark_filter_pagerank`.
- **Warmups and median.** 2 warmups absorb one-time costs (JIT, lazy-plan
  compilation, JVM class-loading, executor spin-up, filesystem cache priming) so
  the timed runs measure steady state. Median of 5 (not mean) is robust to the
  occasional GC / stop-the-world spike on a shared box. Cold load (ETL) is timed
  separately, once — a different question from warm query latency.
- **Guardrails.** Each (system, task) is wrapped: an error/OOM records a status
  and the matrix continues; missing pyspark/graphframes/GPU is skipped with a
  message, never aborting the run.

Reproducibility
---------------

Results are rendered from saved JSON (``_static/graphframes/results.json``) —
this page does **not** rerun benchmarks. The committed harness is
``benchmarks/gfql/bench_graphframes.py`` (design notes in
``benchmarks/gfql/bench_graphframes_DESIGN.md``). To reproduce the LiveJournal
matrix (from ``benchmarks/gfql/``, with the graphframes jar on the Spark
classpath via ``GRAPHFRAMES_JAR``):

.. code-block:: bash

   python bench_graphframes.py --dataset lj \
       --systems gfql-polars,gfql-polars-gpu,graphframes \
       --tasks filter,hop1,hop2,pagerank \
       --filter-threshold 42 --warmups 2 --iters 5

Orkut uses ``--dataset orkut --filter-threshold 162``. The shared
``--filter-threshold`` makes the filter task bit-identical across systems.

Environment
-----------

- Host: ``dgx-spark``, single node; GPU: ``GB10``
- GFQL engines: ``engine="polars"`` (CPU, PageRank via igraph) and
  ``engine="polars-gpu"`` (GPU, PageRank via cugraph)
- Spark: GraphFrames ``0.8.4-spark3.5-s_2.12``, PySpark ``3.5.1``, ``local[*]``
- Datasets: `SNAP <https://snap.stanford.edu/data/>`_ LiveJournal (~35M edges),
  Orkut (~117M edges), Friendster (~1.8B edges, stretch)
- Measurement: median of 5 runs after 2 warmups; result-size parity enforced
  per task; results rendered from saved JSON

See also
--------

- :doc:`engines` — choosing an engine; four-engine and external-tool comparison
  (including where PuppyGraph / warehouse-federated tools fit — not yet
  benchmarked head-to-head)
- :doc:`benchmark_filter_pagerank` — GFQL CPU/GPU vs Neo4j + GDS
- :doc:`cypher` — Cypher syntax through ``g.gfql("MATCH ...")``
- :doc:`overview` — GFQL design, features, and GPU acceleration
- :doc:`about` — 10-minute introduction to GFQL
