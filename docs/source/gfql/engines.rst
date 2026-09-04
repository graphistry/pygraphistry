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

This page assumes you already have a graph ``g`` and a ``query``; if not, start with
:doc:`about`.

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

On the ``prrao87/graph-benchmark`` q1–q9 Cypher suite, Polars beats pandas on all nine
queries at both graph sizes measured, and by more on the larger one. The per-query
numbers are on the :doc:`performance` page.

Your existing pandas, Polars, or cuDF graph works as-is: the input frames are accepted and
coerced once; the only change is the keyword. The catch: a few exotic Cypher features still
require ``engine='pandas'`` (they decline during validation, compilation, or planning rather
than silently bridge), and the GPU engines only pay off on larger work. On CPU,
Polars wins the common graph-query shapes (traversal,
``WHERE``/``ORDER``, aggregation) — see *When not to use Polars* below.

**Already a Polars user?** With the default ``engine='auto'``, a graph whose bound frames
are all ``polars.DataFrame`` runs on the Polars engine and returns Polars frames. If the
query uses a shape the Polars engine declines, ``auto`` falls back to pandas for that call.
Pass ``engine='polars'`` when a decline should raise instead:

.. doc-test: skip

.. code-block:: python

   import polars as pl, graphistry
   g = graphistry.edges(edges_pl, 'src', 'dst').nodes(nodes_pl, 'id')  # polars frames
   out = g.gfql(query)                    # auto -> native Polars (out._nodes is polars)
   out = g.gfql(query, engine='polars')   # same, but a declined shape raises

**Result frames match the engine.** With ``engine='polars'`` or ``'polars-gpu'`` the output
frames are Polars, and ``cudf.DataFrame`` for ``engine='cudf'``. Pandas-only downstream code
(``.iloc``, ``groupby().apply()``) gets a pandas frame with ``result._nodes.to_pandas()``.

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

``engine='auto'`` follows the input frames: Polars frames run on ``polars``, cuDF frames on
``cudf``, everything else on ``pandas``. Two AUTO fast paths go further — all-Polars frames
are tried on ``polars``, and all-cuDF frames are tried on ``polars-gpu`` when a GPU collect
probes usable — each falling back to ``pandas`` / ``cudf`` respectively if the query uses a
shape that engine declines. Passing the engine explicitly turns those declines into errors
instead of a fallback (see *What auto does* below).

How the engines compare
-----------------------

Each engine has a shape it is built for:

- **Polars-CPU is the everyday win.** It beats pandas on all nine queries of the q1–q9
  Cypher suite at both sizes measured (:doc:`performance`), with **no GPU**, because it
  builds **one fused lazy plan and collects once** instead of materializing an
  intermediate per operation.
- **Polars-GPU runs that same fused plan on the GPU.** It pays off once a step carries
  enough work to amortize a kernel launch: large frontiers, dense joins, full-graph
  aggregation.
- **cuDF executes eagerly, op by op.** That suits one very large materialization where a
  single join dominates the query and the in-memory Polars-GPU executor comes under
  memory pressure.
- **pandas carries no plan overhead**, so it stays the right default for trivially small
  operations and the widest-compatibility path.
- **Seeded / selective lookups are an indexing problem**, not an engine race: the opt-in
  resident index turns the ``O(E)`` scan into an ``O(degree)`` gather on every engine, so
  the cost tracks the seeds rather than the graph — see [F5] below and
  :doc:`index_adjacency`.

.. _gfql-vs-external-tools:

Coming from another graph tool
------------------------------

GFQL is **dataframe-native**: ``pip install``, then query the pandas, Polars, or cuDF frame
you already have, in your own process. There is no server to stand up, no ETL to load, no
projection step, no cluster to size. The query, the analytic, and the scoring stay in one
pipeline over one set of frames.

The table names the concrete change for each system and where the measured comparison
lives. Every figure on those pages renders from a committed pyg-bench artifact.

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - Coming from
     - What changes
     - What you gain, and where it is measured
   * - **Neo4j + GDS**
     - Same ``MATCH ... RETURN`` Cypher; no server, no GDS projection, no write-back.
     - One in-process call runs filter, PageRank, and filter over resident frames, on CPU
       or GPU. The measured pipeline times against Neo4j + GDS on the 30M-edge GPlus graph
       are in :doc:`benchmark_filter_pagerank`.
   * - **Memgraph**
     - Same Cypher; no server round trip.
     - Point lookups are Memgraph's strength: on the SNB-derived point queries the graph
       databases, Memgraph first, beat GFQL. GFQL's wins are bulk shapes: traversals from
       seed sets and global aggregates. See :doc:`performance`.
   * - **Kuzu**
     - Same Cypher; query the frame already in memory, nothing to load or index first.
     - The q1–q9 board on :doc:`performance` is the measured comparison, per query, with
       the losses shown.
   * - **LadybugDB**
     - Same dataframe-native path, in process.
     - Polars streaming (``GFQL_POLARS_CPU_STREAMING=1``) and the cudf-polars streaming
       executor (``GFQL_POLARS_GPU_EXECUTOR=streaming``) spill intermediates and results
       beyond RAM. Scan-shaped queries are measured on :doc:`performance`.
   * - **networkx**
     - A declarative query language over the graph, on frames instead of Python objects.
     - Columnar CPU execution and a one-keyword move to GPU.
   * - **igraph**
     - Nothing to give up: igraph is GFQL's CPU PageRank backend.
     - The query layer, the Polars engines, and the GPU path on top of igraph analytics.
   * - **Spark GraphFrames**
     - Cypher instead of a DataFrame API; single node, no cluster.
     - Interactive latency for filters and traversals on CPU, and GPU PageRank. The
       head-to-head, with committed results, is :doc:`benchmark_graphframes`.

Route by shape: **selective** seeded lookups favor the GFQL resident index, **scan and
aggregate** volume favors Polars, and **bulk** frontier expansion and full pipelines favor
Polars or a GPU engine.

What is **not** benchmarked: motif, triangle, and other cyclic multi-way-join patterns.
They run in GFQL, and we publish no performance claim about them.

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
   * - Trivially small op (bare equality filter)
     - any
     - CPU
     - ``pandas``
     - boolean mask beats Polars plan overhead; immaterial [F1]
   * - Selective / seeded traversal
     - any
     - CPU
     - ``pandas``/``polars`` + **CSR index**
     - O(degree), not an engine choice [F5]

**[F1] Polars leads on CPU, and by more as the graph grows.** On the q1–q9 Cypher suite
it beats pandas on all nine queries at both sizes measured, and the pandas-to-Polars gap
is wider on the larger graph for every query (:doc:`performance`).
Pandas only edges out on a trivially small operation (a bare equality mask),
where the absolute difference is immaterial. The real small-size floor is **GPU-only** —
cuDF / Polars-GPU need enough work to amortize kernel launch ([F2]).

**[F2] GPU is work-bound, not size-bound.** A GPU wins when there is enough work to amortize
its millisecond-scale kernel-launch floor: big frontiers, dense joins, full-graph
aggregation. Tiny or seeded work finishes faster on CPU.

**[F3] Polars-GPU memory pressure.** On an extreme single materialization (a huge output
row count from one join) raw ``cudf`` leads and ``polars-gpu`` slips as its in-memory
GPU executor comes under memory pressure. Prefer ``cudf`` for that regime.

**[F4] Polars-GPU is GPU-or-error.** It never silently falls back to CPU and reports the
result as a GPU run (see *Parity and fallback rules* below).

**[F5] Selective traversal is an indexing problem, not an engine choice.** A seeded ``hop``
from a few nodes is fastest with the opt-in **CSR adjacency index** (``g.gfql_index_all()`` /
``g.create_index(...)``, ``index_policy=``), which turns the O(E) scan into an O(degree)
gather — a complexity-class change, so the cost tracks the seeds' neighborhood rather than
the graph (index routing for the native seeded ``g.hop()`` currently engages on pandas, not
yet the Polars hop path). It works on all four engines, but seeded work is so small that
**CPU wins** — the gather is tiny work on pandas/Polars, below the GPU kernel-launch floor
on cuDF — the clean inverse of bulk, where the GPU pulls ahead. So pick the index for selective
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
runs fully on the chosen engine. Note that ``engine='auto'`` (the default) follows
the input frames — Polars frames run natively on ``polars``, cuDF frames on
``cudf`` (or ``polars-gpu`` when that GPU path probes usable), everything else on
``pandas`` — falling back to ``pandas`` / ``cudf`` only for query shapes the native
engine declines.

.. tip::
   For selective, seeded traversal, build the CSR adjacency index once with
   ``g.gfql_index_all()`` (or ``index_policy=``) — it works on all four engines
   and turns the O(E) scan into an O(degree) gather. An AUTO build on Polars frames now
   keeps them native, so ``g.gfql_index_all()`` and ``g.gfql_index_all(engine='polars')``
   build the same index. See :doc:`index_adjacency`.

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

Three cases, stated so you can route around them:

- **Trivially small operations** (a bare node-equality filter): pandas' boolean mask
  beats Polars' plan overhead, and in absolute terms it is immaterial. For traversal /
  ``WHERE`` / ``ORDER`` / aggregation, Polars leads on CPU (footnote F1). The real
  small-size caveat is **GPU-only** (cuDF / Polars-GPU need larger work — footnote F2).
- **A few exotic Cypher features** are not yet native on Polars (e.g. cross-entity same-path
  ``WHERE``, some temporal/entity-text forms). GFQL rejects those shapes during
  validation, compilation, or planning before query execution and points at
  ``engine='pandas'`` — it **never** silently bridges Polars to pandas, because that would
  misreport pandas performance as Polars (see *Parity and fallback rules*).
- **One extreme materialization (a huge output row count):** prefer ``cudf`` over
  ``polars-gpu`` (footnote F3).

Parity and fallback rules
-------------------------

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

Hosts, datasets, warm-median protocol, cross-engine result validation, provenance, and
reproducers live with the numbers on the :doc:`performance` page. A figure that cannot be
traced to a committed benchmark artifact is not published.

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

What auto does
--------------

``auto`` prefers the native engine for your frames and keeps a safety net. A few exotic
Cypher features still require ``engine='pandas'``: the Polars engine **declines them before
execution** rather than silently bridging. Under ``auto`` that decline is caught and the
call is re-served on ``pandas`` (all-cuDF frames decline back to ``cudf``), so a query that
works today keeps working while everything the native engine does support stays native.

Pass the engine explicitly when you would rather know: ``engine='polars'`` /
``'polars-gpu'`` raise ``NotImplementedError`` on a declined shape instead of falling back,
which is what you want in a benchmark or a pipeline that must not silently change engines.
``engine='polars-gpu'`` is additionally GPU-or-error and never quietly runs on CPU.

Performance is rarely the downside — CPU Polars wins common graph queries past
small/interactive sizes; only trivially small operations favor pandas, immaterially.

.. note::
   Non-GFQL surfaces (layouts, plotting, featurization) still consume Polars frames as an
   *input format* and compute in pandas, so ``auto`` coerces there. The native-under-auto
   behavior described above is specific to GFQL query execution.

See also
--------

- :doc:`performance` — GFQL performance overview
- :doc:`benchmark_filter_pagerank` — end-to-end CPU/GPU vs Neo4j+GDS
- :doc:`/api/gfql/index` — GFQL API reference
- :doc:`remote` — run GFQL on a remote GPU
