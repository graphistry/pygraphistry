.. _gfql-engines:

Choosing a GFQL Engine: pandas, Polars, cuDF, Polars-GPU
========================================================

GFQL runs the **same query** on four interchangeable execution engines. You pick
the engine with one keyword — ``engine=``, accepted uniformly by ``g.gfql()`` and
``g.hop()`` — and GFQL returns **identical results** on every one (differential parity
is a release gate). Unsupported engine/query combinations are declined during
validation, compilation, or planning before query execution whenever they can be
known statically, so the safety contract is same answer or pre-execution error,
not silent fallback. Pick the engine that fits your hardware and workload; nothing
else changes.

.. note::
   **New to GFQL?** This page assumes you already have a graph ``g`` and a ``query``. If not,
   build one first — see :doc:`about` (10 Minutes to GFQL).

The one-line speedup
--------------------

On real graphs, switching the default ``pandas`` engine to the columnar **Polars**
engine is a one-keyword change — no GPU, same results:

.. doc-test: skip

.. code-block:: python

   import graphistry
   g = graphistry.edges(df, 'src', 'dst')   # df: your edges dataframe (pandas / Polars / cuDF)
   query = "MATCH (a)-[e]->(b) RETURN b"     # any GFQL / Cypher query

   g.gfql(query)                    # engine='pandas' (default)
   g.gfql(query, engine='polars')   # e.g. 12.3x on LDBC SNB SF1 seed-lookup, no GPU, identical results

On the 0.58.0 release build (DGX Spark GB10, warm medians N=30, results verified
identical), the LDBC SNB SF1 seed-lookup drops from **1,299.6 ms** on eager pandas to
**106.1 ms** with ``engine='polars'`` — **12.3×** from one keyword on the same build. See
:ref:`gfql-0580-numbers` for the full release-verified sweep (seeded fast paths, resident
index, vs Neo4j, OLAP).

Your existing pandas, Polars, or cuDF graph works as-is: the input frames are accepted and
coerced once; the only change is the keyword. The catch: a few exotic Cypher features still
require ``engine='pandas'`` (they decline during validation, compilation, or planning rather
than silently bridge), and the GPU engines only pay off on larger work. On CPU,
Polars wins the common graph-query shapes (traversal,
``WHERE``/``ORDER``, aggregation) from ~10K edges up — see *When not to use Polars* below.

.. warning::
   **Already a Polars user? Pass** ``engine='polars'`` **— the default does not.** With the
   default ``engine='auto'``, a graph built from ``polars.DataFrame`` is **silently coerced to
   pandas** (``auto`` resolves to ``cudf`` for cuDF input and ``pandas`` for everything else,
   *including Polars*; it never selects the Polars engine). To stay native end-to-end, pass
   ``engine='polars'`` explicitly:

   .. code-block:: python

      import polars as pl, graphistry
      g = graphistry.edges(edges_pl, 'src', 'dst').nodes(nodes_pl, 'id')  # polars frames
      out = g.gfql(query)                    # auto -> coerced to PANDAS (out._nodes is pandas!)
      out = g.gfql(query, engine='polars')   # native Polars in and out (out._nodes is polars)

.. note::
   **Result frames match the engine.** With ``engine='polars'`` or ``'polars-gpu'`` the
   output is Polars — ``result._nodes`` and ``result._edges`` are ``polars.DataFrame`` (and
   ``cudf.DataFrame`` for ``engine='cudf'``). If downstream code is pandas-specific (``.iloc``,
   ``.loc``, ``groupby().apply()``), call ``result._nodes.to_pandas()`` to convert back.

The four engines
----------------

.. list-table::
   :header-rows: 1
   :widths: 16 14 18 12 40

   * - Engine
     - Hardware
     - Frame type
     - Opt-in?
     - In one line
   * - ``pandas``
     - CPU
     - ``pandas``
     - default
     - Universal default; best on small/interactive graphs.
   * - ``polars``
     - CPU
     - ``polars``
     - explicit
     - Columnar + fused lazy plan; the CPU speed win, **no GPU needed**.
   * - ``cudf``
     - NVIDIA GPU
     - ``cudf``
     - explicit
     - RAPIDS GPU, eager op-by-op; great for one very large materialization.
   * - ``polars-gpu``
     - NVIDIA GPU
     - ``polars``
     - explicit
     - The Polars fused plan executed on GPU (cudf_polars); fastest on heavy multi-hop.

``engine='auto'`` resolves to ``cudf`` for cuDF input and ``pandas`` otherwise. **AUTO
never selects Polars or Polars-GPU** — they are explicit opt-in (see *Why opt-in?* below).

Release-verified snapshot (0.58.0): seeded typed hop, four engines
------------------------------------------------------------------

Measured on the **0.58.0 release tag** (DGX Spark GB10, warm medians N=30, results
verified identical across engines): the seeded typed-hop Cypher fast path —
``MATCH (m {id: ...})-[:T]->(p) RETURN p`` on a 50k-node / 200k-edge graph — before →
after on the same tag sweep:

.. list-table::
   :header-rows: 1
   :widths: 25 20 25 15

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

The native chain form is faster still (pandas 21.1 → **1.65 ms**, 12.8×; cuDF 23.2 →
**3.84 ms**, 6.0×), and the opt-in resident index (``g.gfql_index_all()``) brings the
covered-shape lookup to **1.74 ms** pandas / **1.59 ms** polars / **1.91 ms** polars-gpu /
**5.78 ms** cudf. (Polars index caveat on 0.58.0: build with
``g.gfql_index_all(engine='polars')`` explicitly — an AUTO build swaps Polars frames to
pandas; fix tracked in PR #1767.) The full release-verified sweep — flat scaling to 32M
edges, LDBC SNB vs Neo4j, OLAP multi-join — is in :ref:`gfql-0580-numbers`.

Motivating comparison (real graphs — prior sweep)
-------------------------------------------------

Same query, same answers, four engines. Warm-median latency on **Orkut** (3.1M nodes /
**117M edges**, SNAP), measured on a single machine. *These are prior-release
measurements from an earlier bulk sweep (pre-0.58.0), retained as a bulk-workload
example — the release-verified 0.58.0 numbers are in the snapshot above and in*
:ref:`gfql-0580-numbers`:

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

*Warm median, identical result rows across all four engines. Reproducer:*
``benchmarks/gfql/index_bulk_olap_bench.py``. *See Methodology below.*

Reading the table (prior sweep):

- **Polars-CPU beat pandas up to ~38x** on bulk traversal and ~4x on aggregation in that
  sweep — **with no GPU**. On the 1-hop workload it was ~38x faster than pandas (68 ms vs
  2613 ms).
- **Polars-CPU also beats cuDF** on these shapes (68 ms vs 1005 ms on 1-hop). cuDF runs
  GFQL *eagerly*, op by op (a kernel launch + a materialized intermediate per hop), while
  Polars builds **one fused lazy plan and collects once**. The fused plan wins until the
  work is large enough to amortize GPU launch costs.
- **Polars-GPU is fastest on heavy multi-hop** (2-hop from 10K seeds: 1518 ms) and on
  aggregation — the same fused plan, executed on the GPU.
- **cuDF wins the one extreme case** — a 2-hop from 100K seeds materializing ~85M output rows
  (6.0 s) — where raw GPU throughput on a single massive join overtakes everything and
  Polars-GPU comes under memory pressure (footnote F3).
- On a smaller graph (**LiveJournal**, 35M edges) the pattern held: 1-hop from 10K seeds was
  pandas 1129 ms → polars **37 ms** (~30x). Filter- and lookup-heavy workloads favor Polars
  as well — on the 0.58.0 tag, the **LDBC SNB SF1** seed-lookup is pandas 1,299.6 ms →
  polars **106.1 ms** (**12.3×**, same build; see :ref:`gfql-0580-numbers`).

.. note::
   Route by workload shape and size (next section). **CPU Polars wins the common graph-query
   shapes from ~10K edges up** — on LiveJournal subsampled (CPU, warm-median): 1-hop traversal
   2.7× / 4.5× / 7.6× and ``WHERE``+``ORDER`` 3.0× / 3.0× / 18× over pandas at 10K / 100K / 1M.
   The **GPU** engines (cuDF / Polars-GPU) are the ones with a real small-size floor — they need
   enough work to amortize kernel-launch cost (work-bound, [F2]). The only case pandas edges out
   is a trivial sub-millisecond operation (e.g. a bare node-equality filter), where its boolean
   mask beats Polars' plan overhead — but at <1 ms the difference is immaterial. Reproducer:
   ``benchmarks/gfql/index_crossover_bench.py``.

.. _gfql-vs-external-tools:

GFQL vs external graph tools
----------------------------

GFQL is **dataframe-native**: ``pip install``, then query your existing pandas / Polars /
cuDF frame in-process — no separate database to stand up, no ETL to load, no cluster. Graph
databases (Neo4j, Kuzu) are a **system-of-record** you provision and ingest into first. The
table below is deliberately conservative: every speedup is stated with its condition, ``>``
and did-not-finish markers are kept, and where we have no head-to-head we say **not
benchmarked** rather than guess.

.. list-table::
   :header-rows: 1
   :widths: 14 22 30 34

   * - Tool
     - What it is / Setup
     - Where GFQL wins (with condition)
     - Where it complements / GFQL doesn't claim
   * - **Neo4j + GDS**
     - Server + GDS library; stand up a DB and ETL your data in.
     - **LDBC SNB interactive SF1** vs Neo4j 5.26 (0.58.0 tag, same box, warm): GFQL wins
       **4 of 5** clean pairs — seed-lookup **106.1 vs 143.7 ms**, message-content
       **7.1 vs 23.0 ms**, message-creator **6.8 vs 27.7 ms** (flip shipped in 0.58.0 via
       property-seeded resident-index gathers, PR #1770), one-hop-expand
       **111.9 vs 180.7 ms**. Prior sweep: **filter→PageRank→filter pipeline**, dgx-spark
       GB10, warm median: Twitter 2.4M — 13.83 s Neo4j vs 2.55 s GFQL-CPU /
       **0.30 s GFQL-GPU (46×)**; GPlus 30M — **>187 s (did-not-finish)** vs 75.78 s CPU /
       **3.33 s GPU (>56×)**.
     - **Neo4j wins recent-replies** (104.0 vs 209.6 ms) in the same LDBC pairs — reported
       as-is. Neo4j remains the transactional system-of-record; run the read-heavy
       analytics in GFQL. See :doc:`benchmark_filter_pagerank`.
   * - **Kuzu**
     - Embedded graph DB; still a separate store to load + index.
     - **OLAP multi-join** (graph-benchmark, 100k-node scale, 0.58.0 tag,
       ``engine='polars'``): q8 **5.0 ms vs 1,004 ms embedded Kuzu (200×)**; q9 **14.2×**.
       Prior sweep — **seeded index lookup** (0.8M nodes / 6.4M edges): 1-hop
       **0.123 ms vs 1.15 ms (9.4×)**, 2-hop **0.150 ms vs 4.25 ms (28×)**; prepared-Kuzu
       LiveJournal 35M ~ **17×** typical seed, 6× hub; **bulk frontier expansion**
       (LiveJournal 35M, 1-hop, many seeds): **22× Kuzu**, up to **87× at k=100k**.
       See :doc:`index_adjacency`.
     - **Kuzu wins in the same 0.58.0 sweep:** single-table aggregates (**2–4×**) and
       seeded property-projection lookups (**2.4–64×**) — GFQL's strengths are traversals,
       multi-join OLAP, and covered seeded shapes. Also **not claimed:** cyclic /
       multi-way-join patterns (triangles, cliques) where Kuzu's worst-case-optimal joins
       can win. Use Kuzu as the store; GFQL for bulk read analytics.
   * - **LadybugDB**
     - Actively-maintained **Kuzu fork** (Kuzu is archived); embedded C++, strongly-typed
       Cypher, opt-in ART *or* hash indexing, zero-copy Arrow/CSR scans, and **out-of-core
       billion-scale** (query a 1.8B-edge graph in <8 GB RAM).
     - Against **LadybugDB's published numbers** for their own 5M-node / 20M-edge suite
       (their figures, their hardware; GFQL measured separately on an NVIDIA DGX Spark
       GB10 running the identical Cypher ``MATCH … RETURN`` row pipeline, each engine on
       its **native** frames — a cross-machine comparison, so read the ratios as
       indicative): GFQL **wins the scan-shaped ops** — full node scan **~65×** (polars
       58 ms vs 3789 ms), id **range ~1.2×** (polars 6.1 ms vs 7.5 ms), relationship
       property/rowid scans **~3.5–3.7×** (cuDF 4.2 s vs ~15 s). **Point lookup** (single
       id) is ~4 ms vs Ladybug's ~0.3 ms — a full columnar scan vs a B-tree/hash **index
       seek**; close in absolute terms, and a resident GFQL node-id index (tracked in
       issue #1676) should close it. Ladybug still wins the two ops backed by
       persistent structure: point lookups and a relationship ``COUNT(*)`` (an O(1) cached
       count vs GFQL's O(E) endpoint-validated scan — a dataframe has no referential
       integrity). GFQL's angle is dataframe-native, in-process, and GPU-accelerated with
       no separate store to load/index.
     - **Complement:** Ladybug is a durable embedded store with an out-of-core mode
       (billion-scale in <8 GB RAM); GFQL is a query engine over your dataframes. GFQL's
       *default* is in-memory, but it is **not limited to it** — Polars streaming
       (``GFQL_POLARS_CPU_STREAMING=1``, disk-spill) and the cudf-polars streaming executor
       (``GFQL_POLARS_GPU_EXECUTOR=streaming``) are larger-than-memory paths
       (billion-scale head-to-head not yet benchmarked — see :doc:`benchmark_graphframes`).
       Natural split: Ladybug as the persistent/out-of-core store; pull a subgraph into GFQL
       for GPU analytics — or run GFQL streaming directly on your columnar files.
   * - **igraph**
     - Pure-Python/C graph library.
     - — (not a standalone competitor here)
     - **Complement, not competitor:** igraph is the CPU PageRank backend *inside* GFQL.
       No head-to-head benchmarked.
   * - **networkx**
     - Pure-Python graph library; the floor most analysts start from.
     - **not benchmarked** — expect order-of-magnitude headroom qualitatively (no measured
       head-to-head).
     - Fine for small/interactive graphs; GFQL is the columnar/GPU path when they grow.
   * - **Spark GraphFrames**
     - *Distributed* graph engine on a Spark cluster; provision + tune the cluster.
     - GFQL is *single-node* (CPU or one GPU): 100M+ edges in-process on **one machine**,
       no cluster to stand up, interactive latency — and a single node often matches or beats
       Spark on read-heavy traversal and, with the GPU engine, PageRank at a fraction of the cost.
       Head-to-head on LiveJournal (35M) and Orkut (117M): GFQL wins filter/traversal 1.3–43×
       even on CPU, and the GPU engine wins PageRank ~10–15×; on CPU, PageRank via igraph is
       *slower* than GraphFrames — see :doc:`benchmark_graphframes`.
     - Reach for GraphFrames when the graph genuinely exceeds one machine's memory. Motif /
       triangle / multi-way-join queries **run** in GFQL but are not yet perf-benchmarked.
   * - **PuppyGraph**
     - Graph query layer *over your warehouse tables in place* (zero-ETL, query pushdown).
     - GFQL adds GPU/CPU graph **analytics PuppyGraph does not offer — PageRank, centrality,
       community** — on a pulled subgraph, in one pipeline. *No head-to-head yet.*
     - **Complement:** use PuppyGraph for ad-hoc graph SQL across the whole warehouse; pull the
       relevant subgraph into GFQL when you need GPU-accelerated analytics on it.

GFQL **complements** a graph database more than it replaces one: keep Neo4j or Kuzu as the
system-of-record, and do the read-heavy search + analytics in GFQL so ETL, traversal, and
scoring stay in one in-process dataframe pipeline. Route by shape — **selective** seeded
lookups favor the GFQL index (up to 28× Kuzu, 16.9× Neo4j on 2-hop in prior sweeps;
covered-shape lookups at 1.6–1.9 ms on the 0.58.0 tag), **multi-join OLAP** favors Polars
(q8 **200× Kuzu** on the 0.58.0 tag), and **bulk** frontier expansion and full pipelines
favor Polars / GPU (22–87× Kuzu; **46–56× Neo4j** on the filter→PageRank→filter pipeline,
prior sweeps). The inverse holds too: embedded Kuzu wins single-table aggregates (2–4×)
and seeded property-projection lookups (2.4–64×) in the same 0.58.0 sweep.
Against the **distributed** engines the axis is different:
GFQL trades horizontal scale-out for zero cluster/warehouse setup and interactive latency —
choose it below the single-machine ceiling (100M+ edges fit in-process; a cluster is only
needed once the graph genuinely exceeds one node's memory), and complement PuppyGraph's
zero-ETL warehouse graph with GFQL's GPU analytics. The one case we explicitly **do not**
claim is cyclic / multi-way-join patterns (triangles, cliques): they **run**, but Kuzu's
worst-case-optimal joins can beat a dataframe plan there and we have not yet perf-tuned them.

Decision matrix
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 16 18 22 14

   * - Workload shape
     - Size (edges)
     - Hardware
     - Recommended engine
     - Notes
   * - Filter / ``WHERE`` / aggregation
     - > ~10K
     - CPU
     - ``polars``
     - wins from ~10K; gap grows with size (up to order-of-magnitude) [F1]
   * - Bulk 1-hop frontier expansion
     - > ~10K
     - CPU
     - ``polars``
     - wins from ~10K (2.7x); up to ~38x pandas, ~15x cuDF at 100M (prior sweep) [F1]
   * - Heavy multi-hop (2-hop+)
     - large
     - GPU
     - ``polars-gpu``
     - fastest until extreme materialization [F3]; GPU-or-error [F4]
   * - Full-graph aggregation
     - 100M+
     - GPU
     - ``polars-gpu`` / ``cudf``
     - GPU work-bound [F2]
   * - One very large single materialization
     - 80M+ output rows
     - GPU
     - ``cudf``
     - Polars-GPU can hit memory pressure here [F3]
   * - Trivial sub-ms op (bare equality filter)
     - any
     - CPU
     - ``pandas``
     - boolean mask beats Polars plan overhead; immaterial (<1 ms) [F1]
   * - Selective / seeded traversal
     - any
     - CPU
     - ``pandas``/``polars`` + **CSR index**
     - O(degree), not an engine choice [F5]

**[F1] CPU crossover is ~10K, not ~1M.** For the common graph-query shapes (traversal,
``WHERE``/``ORDER``, aggregation) CPU Polars beats pandas from ~10K edges up (2.7-18× in our
runs). Pandas only edges out on a trivial sub-millisecond operation (a bare equality mask),
where the absolute difference is immaterial. The real small-size floor is **GPU-only** —
cuDF / Polars-GPU need enough work to amortize kernel launch ([F2]).

**[F2] GPU is work-bound, not size-bound.** A GPU wins when there is enough work to amortize
its ~3 ms kernel-launch floor: big frontiers, dense joins, full-graph aggregation. Tiny or
seeded work finishes faster on CPU.

**[F3] Polars-GPU memory pressure.** On an extreme single materialization (~85M output rows,
2-hop from 100K seeds on Orkut) raw ``cudf`` leads (6.0 s) and ``polars-gpu`` slips (8.6 s)
as its in-memory GPU executor comes under memory pressure. Prefer ``cudf`` for that regime.

**[F4] Polars-GPU is GPU-or-error.** It never silently falls back to CPU and reports the
result as a GPU run (see *Honesty* below).

**[F5] Selective traversal is an indexing problem, not an engine choice.** A seeded ``hop``
from a few nodes is fastest with the opt-in **CSR adjacency index** (``g.gfql_index_all()`` /
``g.create_index(...)``, ``index_policy=``), which turns the O(E) scan into an O(degree)
gather — flat in graph size: on the 0.58.0 tag, a native seeded 1-hop ``g.hop()`` on pandas
holds **0.159–0.164 ms from 0.25M to 32M edges** (constant avg degree 4; pandas-only —
the Polars hop path is not yet index-routed). It works on all four engines, but seeded
work is so small that **CPU wins**: in a prior LiveJournal 35M sweep a typical-seed 1-hop
was ~0.13 ms on pandas and ~0.16 ms on Polars (numpy ``searchsorted``) vs ~3 ms on cuDF
(GPU kernel-launch floor) — the clean inverse of bulk, where the GPU pulls ahead. So pick
the index for selective traversal and a CPU engine to drive it. See :doc:`index_adjacency`
for the full guide.

Switching engines
-----------------

The engine is a single keyword on ``g.gfql()`` (and ``g.hop()``). The graph and
the query never change — only ``engine=`` does, and the answer stays identical
(or the compiler/planner declines the unsupported engine before execution rather
than silently changing it).

.. code-block:: python

   import graphistry
   g = graphistry.edges(df, 'src', 'dst')   # your existing graph (any frame type)
   query = "MATCH (a)-[e]->(b) RETURN b"     # any GFQL / Cypher query

   g.gfql(query)                       # engine='pandas' (default)
   g.gfql(query, engine='polars')      # CPU columnar, no GPU, identical results
   g.gfql(query, engine='cudf')        # NVIDIA GPU (RAPIDS)
   g.gfql(query, engine='polars-gpu')  # same fused plan on GPU

Getting results back as pandas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The result's ``._nodes`` / ``._edges`` come back in the engine's frame type: a
``polars.DataFrame`` for ``'polars'`` / ``'polars-gpu'``, a ``cudf.DataFrame``
for ``'cudf'``. When downstream code is pandas-only (matplotlib, scikit-learn,
``.iloc`` / ``groupby().apply()``), convert once with ``.to_pandas()``:

.. code-block:: python

   out = g.gfql(query, engine='polars')       # or 'cudf' / 'polars-gpu'
   nodes_pd = out._nodes.to_pandas()          # -> pandas for matplotlib / sklearn / ...
   nodes_pd.plot.scatter(x='x', y='y')        # pandas-only downstream code, unchanged

Mixing engines
~~~~~~~~~~~~~~~

The build frame type and the run engine are independent — GFQL coerces the input
frames to the engine you ask for. A pandas graph runs on ``engine='polars'``, a
Polars graph runs on ``engine='pandas'``, and so on. The only cost is a
**one-time convert** of the input frames at the start of the call; the query then
runs fully on the chosen engine. Note that ``engine='auto'`` (the default)
resolves to ``cudf`` for cuDF input and ``pandas`` for everything else — **it
never selects Polars or Polars-GPU**, so those two are always an explicit opt-in.

.. tip::
   For selective, seeded traversal, build the CSR adjacency index once with
   ``g.gfql_index_all()`` (or ``index_policy=``) — it works on all four engines
   and turns the O(E) scan into an O(degree) gather. **On 0.58.0, Polars frames need
   the engine passed explicitly** — ``g.gfql_index_all(engine='polars')`` — because an
   AUTO build swaps Polars frames to pandas (fix tracked in PR #1767).
   See :doc:`index_adjacency`.

.. _gfql-offengine-calls:

Analytics under Polars (``umap`` / ``hypergraph`` / ``compute_cugraph`` …)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A GFQL ``call()`` that runs a **whole-graph analytic** — ``umap``, ``hypergraph``,
``compute_cugraph`` / ``compute_igraph``, the ``*_layout`` ops, ``collapse`` — has
**no native Polars implementation** (these wrap pandas / cuDF / GPU libraries and
always will). Under ``engine='polars'`` / ``'polars-gpu'`` GFQL runs them as a
**mode-gated, off-engine modality switch** rather than declining outright:

- **``call_mode='auto'`` (the default):** the analytic runs off-engine — on
  **pandas** for ``polars``, on **cuDF (on device)** for ``polars-gpu`` — and its
  result is coerced back to Polars **losslessly** (via Arrow). A one-time
  ``RuntimeWarning`` per analytic notes the off-engine run. ``polars-gpu`` is
  **GPU-or-error**: it bridges to cuDF and *declines* if the GPU/cuDF stack is
  missing (it never silently drops a GPU analytic to host pandas).
- **``call_mode='strict'``:** decline before running the analytic instead of
  bridging — for benchmark integrity (no hidden modality switch attributed to the
  Polars engine) or a hard memory ceiling.

.. note::
   **Memory on a very large graph.** The bridge materializes a copy of the graph in
   the off-engine format — pandas (host) for ``polars``, cuDF (device / unified
   memory) for ``polars-gpu``. That transient copy is the *same* allocation you'd
   incur running the analytic on ``engine='cudf'`` directly, so GFQL does **not** add
   a per-call size cap (a row count is a poor memory proxy, and the real cap belongs
   at the RMM / container / deployment layer). For a graph large enough that the copy
   is a concern, either set ``call_mode='strict'`` (decline the bridge) or run the
   analytic under an RMM device-memory limit / container memory limit, exactly as you
   would for any cuDF workload.

This is **deliberately narrower** than traversal / filter / row ops (``hop``,
``WHERE``, ``RETURN`` …), which stay **parity-or-static-decline** and are never
bridged — a bridge there would hide a missing native impl and misreport pandas
performance as Polars. Set the mode from Python or the environment (live, Python
override > env > default):

.. doc-test: skip

.. code-block:: python

   from graphistry.compute.gfql.lazy import set_call_mode, CALL_MODES  # ('auto', 'strict')

   set_call_mode('strict')   # decline off-engine analytics (pass None to reset to env/default)
   # or: export GFQL_POLARS_CALL_MODE=strict

cuDF vs Polars-GPU
------------------

Both run on an NVIDIA GPU, so which do you use?

- **cuDF is not deprecated.** It remains a first-class, supported engine and is the right
  choice for one very large materialization (footnote F3).
- **They execute differently.** ``cudf`` runs GFQL eagerly — each hop is a separate kernel
  launch with a materialized intermediate. ``polars-gpu`` runs the **same fused lazy plan as
  the CPU Polars engine**, collected once on the GPU. Fusing the plan is why ``polars-gpu``
  leads on heavy multi-hop and why even **CPU Polars often beats eager cuDF** on bulk work.
- **Frame type.** ``cudf`` operates on ``cudf.DataFrame``; ``polars-gpu`` operates on
  ``polars.DataFrame`` (only the lazy ``.collect()`` runs on the GPU). Either way, a graph
  built from pandas frames is accepted and coerced for you — only the keyword changes.
- **Install.** ``cudf`` and ``polars-gpu`` both need the RAPIDS GPU stack; ``polars-gpu``
  additionally uses ``cudf_polars``. ``polars`` (CPU) only needs ``pip install polars``.

.. _gfql-larger-than-memory:

Larger-than-memory: streaming execution
---------------------------------------

The default Polars engines run **in-memory**: fastest and most stable while the
graph and its query intermediates fit in RAM (or device memory). When a query's
*intermediates* would blow past memory — a wide multi-hop frontier, a large
join, a big aggregation — GFQL has two **opt-in** streaming modes that trade a
little latency for a much larger working set:

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Mode
     - Engine
     - What it does
   * - ``GFQL_POLARS_CPU_STREAMING=1``
     - ``polars``
     - Collects the fused plan with Polars' **streaming engine** — processes in
       batches and **spills to disk**, so intermediates can exceed RAM.
   * - ``GFQL_POLARS_GPU_EXECUTOR=streaming``
     - ``polars-gpu``
     - Uses the **cudf-polars streaming executor** — the escape hatch for
       results **larger than device memory** (the default in-memory executor
       would OOM).

Both are **off by default** on purpose: they add overhead that *regresses*
small/interactive work (~0.86× at 100K edges), and for the in-memory regime this
page measures, the default is faster and more stable. Results are
**parity-identical** to the default — streaming changes *how* the plan runs, not
*what* it returns.

Set them by environment variable:

.. code-block:: bash

   # CPU: batched + disk-spill for larger-than-RAM intermediates
   export GFQL_POLARS_CPU_STREAMING=1

   # GPU: streaming executor for larger-than-device-memory results
   export GFQL_POLARS_GPU_EXECUTOR=streaming

...or from Python at runtime — the setting is read **live** (per collect), and a Python
override takes precedence over the environment variable:

.. doc-test: skip

.. code-block:: python

   from graphistry.compute.gfql.lazy import (
       set_cpu_streaming, set_gpu_executor, GPU_EXECUTORS,
   )

   set_cpu_streaming(True)          # CPU streaming collect (pass None to reset to env/default)
   set_gpu_executor('streaming')    # one of GPU_EXECUTORS == ('in-memory', 'streaming')

Then use ``engine='polars'`` / ``engine='polars-gpu'`` exactly as before — no code
change:

.. doc-test: skip

.. code-block:: python

   import graphistry            # env vars above must be set first
   g = graphistry.edges(edges_df, 'src', 'dst')
   result = g.gfql(query, engine='polars')       # streaming collect (CPU, disk-spill)
   # result = g.gfql(query, engine='polars-gpu')  # streaming executor (GPU)

.. note::
   **What streaming does and does not cover today.** These flags stream the
   **query** (collect), which helps when the *input fits but the intermediates or
   result do not*. They do **not** yet give out-of-core *input*: ``graphistry``
   currently materializes edge/node frames at ingestion (a passed
   ``polars.LazyFrame`` is collected immediately), so the source graph must still
   fit in memory. True out-of-core-from-disk — building GFQL directly on a lazy
   ``pl.scan_parquet`` source so a graph larger than RAM never fully materializes —
   is **work in progress**; see the Friendster (~1.8B edges) discussion in the
   GraphFrames benchmark page.

When **not** to use Polars
--------------------------

Honesty matters more than a bigger number:

- **Trivial sub-millisecond operations** (a bare node-equality filter): pandas' boolean mask
  beats Polars' plan overhead — but at <1 ms it is immaterial. For traversal / ``WHERE`` /
  ``ORDER`` / aggregation, CPU Polars wins from ~10K edges up (footnote F1). The real small-size
  caveat is **GPU-only** (cuDF / Polars-GPU need larger work — footnote F2).
- **A few exotic Cypher features** are not yet native on Polars (e.g. cross-entity same-path
  ``WHERE``, some temporal/entity-text forms). GFQL rejects those shapes during
  validation, compilation, or planning before query execution and points at
  ``engine='pandas'`` — it **never** silently bridges Polars to pandas, because that would
  misreport pandas performance as Polars (see *Honesty*).
- **One extreme materialization (80M+ output rows):** prefer ``cudf`` over ``polars-gpu``
  (footnote F3).
- **vs graph databases:** GFQL-Polars beats embedded kuzu on multi-join OLAP (q8 **200×**,
  q9 **14.2×** on the 0.58.0 tag) and, in prior sweeps, on frontier expansion (up to ~87x
  on LiveJournal 1-hop — reproducer ``benchmarks/gfql/index_vs_kuzu_prepared.py``); it
  separately beats Neo4j+GDS end-to-end (:doc:`benchmark_filter_pagerank`). The honest
  boundary: in the same 0.58.0 sweep, embedded kuzu **wins single-table aggregates (2–4×)
  and seeded property-projection lookups (2.4–64×)**, and its worst-case-optimal joins
  target **cyclic / multi-way join** patterns (triangles, cliques) that we have **not**
  yet benchmarked, and kuzu may lead there.

Parity and honesty
------------------

- **Identical results across engines.** Differential parity — every engine's output must match
  the pandas oracle — is a release gate, exercised across forward/reverse/undirected, 1-3 hop,
  filters, and aggregations.
- **No silent fallback for traversal / filter / row ops — parity-verified.** For ``hop`` /
  ``WHERE`` / ``RETURN`` / aggregation, the Polars engine runs natively or the query is
  declined before execution during validation, compilation, or planning. For string GFQL /
  Cypher queries, known unsupported syntax and unsupported lowering shapes are rejected by
  the compiler/validator before execution starts; Python-built ASTs hit the same safety
  boundary in the local planner before the unsupported engine path runs. GFQL never quietly
  converts to pandas, so a *traversal* latency you measure is real work on the engine you
  asked for. ``polars-gpu`` is **GPU-or-error**: if any step of the plan cannot run on the
  GPU, the plan is rejected rather than silently running on CPU and labelling it a GPU result.
- **Whole-graph analytics are the one mode-gated exception.** ``umap`` / ``hypergraph`` /
  ``compute_cugraph`` and friends have no Polars kernel; under ``call_mode='auto'`` (default)
  they run off-engine and warn once (see
  :ref:`Analytics under Polars <gfql-offengine-calls>`). This is *not* silent — it warns — and
  ``call_mode='strict'`` restores strict parity-or-pre-execution-decline for benchmark
  integrity, so a benchmarked run can guarantee no hidden modality switch.

Methodology
-----------

- **0.58.0 release-tag sweep** (the seeded typed-hop snapshot, resident-index and
  flat-scaling numbers, LDBC SNB SF1 / Neo4j 5.26 pairs, and OLAP q8/q9 above): measured
  on the ``0.58.0`` release tag on an NVIDIA DGX Spark (GB10), **warm medians over
  N=30 runs**. Four-engine numbers were kept only after result rows were asserted
  identical across engines; the cross-database pairs (Neo4j, Kuzu) were validated against
  expected result rows and cross-database value/row-count checks. The Neo4j pairs ran on
  the same box, warm.
- Prior bulk sweep (the Orkut / LiveJournal tables and their derived ratios), pre-0.58.0 —
  the remaining bullets in this section describe that sweep's setup:
- Host: NVIDIA DGX Spark (GB10 Grace-Blackwell, unified memory — the F3 memory-pressure
  boundary is partly a property of this box), RAPIDS container
  ``graphistry/test-rapids-official:26.02-gfql-polars``.
- Datasets: `SNAP <https://snap.stanford.edu/data/>`_ **com-LiveJournal** (35M edges),
  **com-Orkut** (117M edges). The order-of-magnitude filter/lookup figure is from a separate
  **LDBC SNB sf1** benchmark, not the table above.
- Measurement: **warm median** after 2 warmups (5 timed runs on Orkut, 8 on LiveJournal);
  every reported cell is **guarded** — the result rows are verified identical across engines
  before any timing is kept.
- Reproduce: ``benchmarks/gfql/index_bulk_olap_bench.py`` (engine comparison),
  ``benchmarks/gfql/pandas_vs_polars.py``, and ``benchmarks/gfql/index_vs_kuzu_prepared.py``
  (vs kuzu). Numbers on this page are rendered from saved runs; the page does not re-run them.
- **LadybugDB row**: the Ladybug figures are **their published results on their hardware**;
  the GFQL side ran on the host above via ``benchmarks/gfql/bench_ladybug_cypher.py``
  (5M/20M synthetic per their suite shape, native frames per engine, warm medians) — a
  cross-machine comparison, disclosed as such in the row.

Install
-------

.. code-block:: bash

   pip install graphistry          # base; pandas engine works out of the box
   pip install graphistry polars   # adds the CPU 'polars' engine
   # 'cudf' and 'polars-gpu' require the NVIDIA RAPIDS stack (GPU);
   # 'polars-gpu' additionally uses cudf_polars.

Then change one keyword — your existing graph and query are unchanged:

.. doc-test: skip

.. code-block:: python

   import graphistry
   g = graphistry.edges(df, 'src', 'dst')          # your existing pandas, Polars, or cuDF graph
   g.gfql("MATCH (a)-[e]->(b) RETURN b", engine='polars')      # CPU columnar
   g.gfql("MATCH (a)-[e]->(b) RETURN b", engine='polars-gpu')  # same plan on GPU

Why opt-in?
-----------

Polars and Polars-GPU are explicit (``engine='polars'`` / ``'polars-gpu'``; ``auto`` never
picks them). The main reason is robustness, not speed: a few exotic Cypher features still
require ``engine='pandas'`` and are **rejected before execution** rather than silently
bridge, so auto-selecting Polars would turn queries that work today on pandas into hard
errors. (Performance is rarely the
downside — CPU Polars wins common graph queries from ~10K edges; only trivial sub-millisecond
operations favor pandas, immaterially.) Opting in keeps the default behavior unchanged and
guarantees a working result.

See also
--------

- :doc:`performance` — GFQL performance overview
- :doc:`benchmark_filter_pagerank` — end-to-end CPU/GPU vs Neo4j+GDS
- :doc:`/api/gfql/index` — GFQL API reference
- :doc:`remote` — run GFQL on a remote GPU
