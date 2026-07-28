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
   g.gfql(query, engine='polars')   # often much faster on query-heavy workloads, identical results

Switching is often a large speedup on query-heavy workloads, and the margin grows with the
data. The measured board — the ``prrao87/graph-benchmark`` q1–q9 Cypher suite, pandas vs
Polars vs an embedded graph database — lives on the :doc:`performance` page.

Your existing pandas, Polars, or cuDF graph works as-is: the input frames are accepted and
coerced once; the only change is the keyword. The catch: a few exotic Cypher features still
require ``engine='pandas'`` (they decline during validation, compilation, or planning rather
than silently bridge), and the GPU engines only pay off on larger work. On CPU,
Polars takes over the common graph-query shapes (traversal, ``WHERE``/``ORDER``,
aggregation) once graphs get past small/interactive sizes — see *When not to use Polars*
below.

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

How the engines compare
-----------------------

The measured board lives on the :doc:`performance` page; :doc:`index_adjacency` covers the
resident index. The four engines differ in character rather than in degree:

- **Polars-CPU is the everyday win**: a columnar, fused lazy plan that outruns pandas on
  query-heavy workloads (traversal, ``WHERE``/``ORDER``, aggregation) once graphs get past
  small/interactive sizes, with **no GPU** — and the advantage widens as the data grows.
  It also frequently beats eager cuDF on bulk work, because it builds **one fused lazy plan
  and collects once** while cuDF pays a kernel launch and a materialized intermediate per op.
- **Polars-GPU leads heavy multi-hop and full-graph aggregation**: the same fused plan,
  executed on the GPU, once there is enough work to amortize kernel launches.
- **cuDF suits the extreme single materialization**: one very large join/output where raw
  GPU throughput dominates and the in-memory Polars-GPU executor comes under memory
  pressure.
- **pandas keeps trivial sub-millisecond operations**: a bare equality filter's boolean
  mask beats Polars' plan overhead — immaterial in absolute terms.
- **Seeded / selective lookups are an indexing problem**, not an engine race: the opt-in
  resident index turns the ``O(E)`` scan into an ``O(degree)`` gather on every engine, so
  the cost tracks the seeds rather than the graph — see [F5] below and
  :doc:`index_adjacency`.

For the measured board, see :doc:`performance`.

.. _gfql-vs-external-tools:

GFQL vs external graph tools
----------------------------

GFQL is **dataframe-native**: ``pip install``, then query your existing pandas / Polars /
cuDF frame in-process — no separate database to stand up, no ETL to load, no cluster. Graph
databases (Neo4j, Kuzu) are a **system-of-record** you provision and ingest into first. The
table below is deliberately conservative: wins are stated with their conditions, losses are
reported as-is, and where we have no head-to-head we say **not benchmarked** rather than
guess. The measured pairs behind every claim live in :doc:`performance`; comparisons whose
raw artifacts could not be recovered have been withdrawn rather than restated.

.. list-table::
   :header-rows: 1
   :widths: 14 22 30 34

   * - Tool
     - What it is / Setup
     - Where GFQL wins (with condition)
     - Where it complements / GFQL doesn't claim
   * - **Neo4j + GDS**
     - Server + GDS library; stand up a DB and ETL your data in.
     - GFQL's angle is the **end-to-end pipeline**: filter → analytic (PageRank,
       centrality, community) → filter stays in one in-process dataframe call, with no
       projection step and no write-back. The pipeline and its reproducer are described in
       :doc:`benchmark_filter_pagerank`; its previously published head-to-head figures have
       been withdrawn pending a provenance-carrying re-run.
     - **Neo4j remains the transactional system-of-record** and wins on durability,
       concurrency, and write workloads; run the read-heavy analytics in GFQL. No
       currently-publishable head-to-head latency comparison.
   * - **Kuzu**
     - Embedded graph DB; still a separate store to load + index.
     - **Size decides.** On the q1–q9 Cypher board, GFQL with ``engine='polars'`` leads on
       most queries once the graph gets an order of magnitude larger, where scan-and-
       aggregate volume rewards the columnar plan — with no separate store to provision,
       load and index. See :ref:`gfql-vs-kuzu-board`.
     - **On the small graph embedded Kuzu wins most of the same board**: it answers those
       queries below GFQL's per-query planning floor, so there is not enough work to
       amortize. Note the board is the *Polars* engine — GFQL-pandas loses to Kuzu on most
       cells at both sizes. Also **not claimed:** cyclic / multi-way-join patterns
       (triangles, cliques) where Kuzu's worst-case-optimal joins can win. Use Kuzu as the
       store; GFQL for bulk read analytics.
   * - **LadybugDB**
     - Actively-maintained **Kuzu fork** (Kuzu is archived); embedded C++, strongly-typed
       Cypher, opt-in ART *or* hash indexing, zero-copy Arrow/CSR scans, and **out-of-core
       billion-scale** — they advertise querying a 1.8B-edge graph in <8 GB RAM.
     - **No publishable head-to-head.** The comparison that used to sit here has been
       withdrawn: the competitor side was an uncited constant and the GFQL side has no
       surviving artifact. Structurally, GFQL's angle is dataframe-native, in-process and
       GPU-accelerated with no separate store to load or index, and its scan-shaped work
       is columnar; a persistent store's angle is the **index seek** and the structures it
       maintains on disk (a cached relationship ``COUNT(*)`` is ``O(1)`` there, while a
       dataframe has no referential integrity and must validate endpoints). A resident
       GFQL node-id index (tracked in issue #1676) targets the point-lookup shape.
     - **Complement:** Ladybug is a durable embedded store with an out-of-core mode;
       GFQL is a query engine over your dataframes. GFQL's
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
     - GFQL is *single-node* (CPU or one GPU): a large graph in-process on **one machine**,
       no cluster to stand up, interactive latency. The measured head-to-head — where a
       single node holds its own on read-heavy filter/traversal, where the GPU engine pulls
       ahead on PageRank, and where CPU PageRank via igraph trails GraphFrames — is
       published with its raw results in :doc:`benchmark_graphframes`.
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
scoring stay in one in-process dataframe pipeline. Route by shape and by size —
**selective** seeded lookups favor the GFQL index, **scan-and-aggregate** volume favors
Polars, and **bulk** frontier expansion and full pipelines favor Polars / GPU (see the
board on :doc:`performance`). The inverse holds too: on small, latency-bound queries an
embedded engine with persistent indexes is the fair choice.
Against the **distributed** engines the axis is different:
GFQL trades horizontal scale-out for zero cluster/warehouse setup and interactive latency —
choose it below the single-machine ceiling (a cluster is only
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
     - past small/interactive
     - CPU
     - ``polars``
     - takes over past small graphs; gap grows with size [F1]
   * - Bulk 1-hop frontier expansion
     - past small/interactive
     - CPU
     - ``polars``
     - takes over past small graphs; gap grows with size [F1]
   * - Heavy multi-hop (2-hop+)
     - large
     - GPU
     - ``polars-gpu``
     - fastest until extreme materialization [F3]; GPU-or-error [F4]
   * - Full-graph aggregation
     - very large
     - GPU
     - ``polars-gpu`` / ``cudf``
     - GPU work-bound [F2]
   * - One very large single materialization
     - huge output row count
     - GPU
     - ``cudf``
     - Polars-GPU can hit memory pressure here [F3]
   * - Trivial sub-millisecond op (bare equality filter)
     - any
     - CPU
     - ``pandas``
     - boolean mask beats Polars plan overhead; immaterial [F1]
   * - Selective / seeded traversal
     - any
     - CPU
     - ``pandas``/``polars`` + **CSR index**
     - O(degree), not an engine choice [F5]

**[F1] The CPU crossover is early, not exotic.** For the common graph-query shapes
(traversal, ``WHERE``/``ORDER``, aggregation) CPU Polars takes over from pandas once
graphs get past small/interactive sizes, and the gap widens as the data grows (see the
board on :doc:`performance`).
Pandas only edges out on a trivial sub-millisecond operation (a bare equality mask),
where the absolute difference is immaterial. The real small-size floor is **GPU-only** —
cuDF / Polars-GPU need enough work to amortize kernel launch ([F2]).

**[F2] GPU is work-bound, not size-bound.** A GPU wins when there is enough work to amortize
its millisecond-scale kernel-launch floor: big frontiers, dense joins, full-graph
aggregation. Tiny or seeded work finishes faster on CPU.

**[F3] Polars-GPU memory pressure.** On an extreme single materialization (a huge output
row count from one join) raw ``cudf`` leads and ``polars-gpu`` slips as its in-memory
GPU executor comes under memory pressure. Prefer ``cudf`` for that regime.

**[F4] Polars-GPU is GPU-or-error.** It never silently falls back to CPU and reports the
result as a GPU run (see *Honesty* below).

**[F5] Selective traversal is an indexing problem, not an engine choice.** A seeded ``hop``
from a few nodes is fastest with the opt-in **CSR adjacency index** (``g.gfql_index_all()`` /
``g.create_index(...)``, ``index_policy=``), which turns the O(E) scan into an O(degree)
gather — a complexity-class change, so the cost tracks the seeds' neighborhood rather than
the graph (index routing for the native seeded ``g.hop()`` currently engages on pandas, not
yet the Polars hop path). It works on all four engines, but seeded work is so small that
**CPU wins** — sub-millisecond on pandas/Polars vs the GPU kernel-launch floor on cuDF — the
clean inverse of bulk, where the GPU pulls ahead. So pick the index for selective
traversal and a CPU engine to drive it. See :doc:`index_adjacency` for the full guide.

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
   and turns the O(E) scan into an O(degree) gather. **Polars frames currently need
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

Both are **off by default** on purpose: they add overhead that mildly *regresses*
small/interactive work, and for the in-memory regime the benchmarks measure, the
default is faster and more stable. Results are
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
  beats Polars' plan overhead — but in absolute terms it is immaterial. For traversal /
  ``WHERE`` / ``ORDER`` / aggregation, CPU Polars takes over past small/interactive sizes
  (footnote F1). The real small-size caveat is **GPU-only** (cuDF / Polars-GPU need larger
  work — footnote F2).
- **A few exotic Cypher features** are not yet native on Polars (e.g. cross-entity same-path
  ``WHERE``, some temporal/entity-text forms). GFQL rejects those shapes during
  validation, compilation, or planning before query execution and points at
  ``engine='pandas'`` — it **never** silently bridges Polars to pandas, because that would
  misreport pandas performance as Polars (see *Honesty*).
- **One extreme materialization (a huge output row count):** prefer ``cudf`` over
  ``polars-gpu`` (footnote F3).
- **vs graph databases:** on the q1–q9 board (:ref:`gfql-vs-kuzu-board`) GFQL-Polars leads
  on most queries at the larger size and **loses most of them at the smaller size**, where
  embedded kuzu answers below GFQL's per-query planning floor. That board is the *Polars*
  engine — **GFQL-pandas loses to kuzu on most cells at both sizes**. And kuzu's
  worst-case-optimal joins target **cyclic / multi-way join** patterns (triangles,
  cliques) that we have **not** yet benchmarked, where kuzu may lead.

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

The measured board and its full methodology — hosts, datasets, warm-median protocol,
cross-engine result validation, provenance, and reproducer scripts — live with the numbers
on the :doc:`performance` page. Figures whose originating run could not be reproduced from
a committed artifact have been removed from these docs rather than restated.

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
downside — CPU Polars wins common graph queries past small/interactive sizes; only trivial
sub-millisecond operations favor pandas, immaterially.) Opting in keeps the default behavior
unchanged and
guarantees a working result.

See also
--------

- :doc:`performance` — GFQL performance overview
- :doc:`benchmark_filter_pagerank` — end-to-end CPU/GPU vs Neo4j+GDS
- :doc:`/api/gfql/index` — GFQL API reference
- :doc:`remote` — run GFQL on a remote GPU
