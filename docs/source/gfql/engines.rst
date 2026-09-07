.. _gfql-engines:

Choosing a GFQL Engine: pandas, Polars, cuDF, Polars-GPU
========================================================

GFQL runs the same query on four execution engines. You choose the engine with one
keyword, ``engine=``, on ``g.gfql()`` and ``g.hop()``. Every engine returns the same
result. When an engine cannot run a query, GFQL raises an error before the query runs
instead of switching engines behind your back.

This page assumes you have a graph ``g`` and a ``query``. If not, start with
:doc:`about`.

Switch engines with one keyword
-------------------------------

.. doc-test: skip

.. code-block:: python

   import graphistry
   g = graphistry.edges(df, 'src', 'dst')   # df: your edges dataframe (pandas / Polars / cuDF)
   query = "MATCH (a)-[e]->(b) RETURN b"     # any GFQL / Cypher query

   g.gfql(query)                       # engine='pandas' (default)
   g.gfql(query, engine='polars')      # CPU, columnar
   g.gfql(query, engine='cudf')        # NVIDIA GPU (RAPIDS)
   g.gfql(query, engine='polars-gpu')  # the Polars plan on the GPU

Polars is the usual first move. On the ``prrao87/graph-benchmark`` q1–q9 Cypher suite it
beats pandas on all nine queries at both graph sizes measured, without a GPU. The
per-query numbers are on the :doc:`performance` page.

Your existing pandas, Polars, or cuDF graph works as-is. GFQL converts the input frames
once, at the start of the call. Results come back in the engine's frame type: Polars
frames for ``'polars'`` and ``'polars-gpu'``, ``cudf.DataFrame`` for ``'cudf'``. Convert
once when downstream code needs pandas:

.. doc-test: skip

.. code-block:: python

   out = g.gfql(query, engine='polars')       # or 'cudf' / 'polars-gpu'
   nodes_pd = out._nodes.to_pandas()          # pandas for matplotlib, scikit-learn, .iloc, ...

**Already a Polars user?** The default ``engine='auto'`` follows your frames: a graph built
from Polars frames runs on the Polars engine and returns Polars frames, and a cuDF graph
runs on ``cudf``. If a query uses a feature the Polars engine does not support, ``auto``
runs that call on pandas. Pass ``engine='polars'`` when you want an error instead:

.. doc-test: skip

.. code-block:: python

   import polars as pl, graphistry
   g = graphistry.edges(edges_pl, 'src', 'dst').nodes(nodes_pl, 'id')  # polars frames
   out = g.gfql(query)                    # auto -> Polars engine (out._nodes is polars)
   out = g.gfql(query, engine='polars')   # same, but an unsupported feature raises

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
     - Works everywhere; best for small, interactive graphs.
   * - ``polars``
     - CPU
     - ``polars``
     - explicit
     - Columnar, one fused plan; the CPU speed win, no GPU needed.
   * - ``cudf``
     - NVIDIA GPU
     - ``cudf``
     - explicit
     - RAPIDS GPU, one operation at a time; best for one very large result.
   * - ``polars-gpu``
     - NVIDIA GPU
     - ``polars``
     - explicit
     - The Polars fused plan run on the GPU (cudf_polars); fastest on heavy multi-hop work.

Polars builds one plan for the whole query and runs it once. pandas and cuDF run the
query one operation at a time and materialize each intermediate result. That difference
is why Polars leads on CPU, why ``polars-gpu`` leads on heavy multi-hop work, and why CPU
Polars often beats cuDF on bulk work.

Which engine for which work
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 16 18 22 14

   * - Workload
     - Size (edges)
     - Hardware
     - Recommended engine
     - Notes
   * - Filter / ``WHERE`` / aggregation
     - past small/interactive
     - CPU
     - ``polars``
     - the gap over pandas grows with graph size
   * - Bulk 1-hop expansion
     - past small/interactive
     - CPU
     - ``polars``
     - the gap over pandas grows with graph size
   * - Heavy multi-hop (2-hop+)
     - large
     - GPU
     - ``polars-gpu``
     - fastest until one step produces an extreme result size
   * - Full-graph aggregation
     - very large
     - GPU
     - ``polars-gpu`` / ``cudf``
     - the GPU wins once there is enough work per step
   * - One very large single result
     - huge output row count
     - GPU
     - ``cudf``
     - ``polars-gpu`` can run short of GPU memory here
   * - Trivially small operation (one equality filter)
     - any
     - CPU
     - ``pandas``
     - pandas avoids the plan overhead; the difference is microseconds
   * - Query starting from a few known nodes
     - any
     - CPU
     - ``pandas`` / ``polars`` + adjacency index
     - cost follows the neighborhood, not the graph; see below
   * - Cypher features the Polars engine does not support yet
     - any
     - CPU
     - ``pandas``
     - the Polars engine raises before running; ``auto`` runs the call on pandas

Three rules cover most decisions:

- **A GPU pays off by work, not by graph size.** Each GPU step costs about a millisecond
  to launch. Large frontiers, dense joins and full-graph aggregation cover that cost;
  small work finishes faster on CPU.
- **Queries that start from a few known nodes are an indexing problem.** Build the
  adjacency index once with ``g.gfql_index_all()`` and a traversal from a watchlist, a
  session, or a seed set reads only those nodes' neighborhoods instead of scanning every
  edge. The work is then small, so drive it from a CPU engine. See :doc:`index_adjacency`.
- **Polars-GPU runs on the GPU or raises.** It never runs a plan on the CPU and reports it
  as a GPU result.

.. _gfql-vs-external-tools:

Coming from another graph tool
------------------------------

GFQL runs inside your Python process on the pandas, Polars, or cuDF frame you already
have. There is no database server to run, no load step, no projection, and no cluster to
size. The query, the analytic and the scoring run in one pipeline over one set of frames.

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - Coming from
     - What changes
     - What you gain, and where it is measured
   * - **Neo4j + GDS**
     - The same ``MATCH ... RETURN`` Cypher.
     - GFQL removes the database server, the GDS projection step, and the write-back. One
       call runs the filter, PageRank and scoring over frames that are already in memory,
       on CPU or GPU, so a pipeline is one function instead of three systems. Measured
       pipeline times against Neo4j + GDS on the 30M-edge GPlus graph:
       :doc:`benchmark_filter_pagerank`.
   * - **Memgraph**
     - The same Cypher.
     - GFQL removes the server round trip and keeps results as dataframes. GFQL is at its
       best on traversals from seed sets and on global aggregates; the SNB-derived
       comparison on :doc:`performance` shows each query, including the ones the graph
       databases win.
   * - **Kuzu**
     - The same Cypher.
     - Query the frame already in memory, with nothing to load or index first. The q1–q9
       board on :doc:`performance` shows every query, wins and losses.
   * - **LadybugDB**
     - The same dataframe-native approach, in process.
     - Streaming modes for results larger than RAM or GPU memory
       (:ref:`gfql-larger-than-memory`). Scan-style queries are measured on
       :doc:`performance`.
   * - **networkx**
     - A query language over frames instead of Python object graphs.
     - Columnar CPU execution and a one-keyword move to the GPU.
   * - **igraph**
     - Nothing to give up: igraph is GFQL's CPU PageRank backend.
     - A query layer, the Polars engines and the GPU path on top of igraph analytics.
   * - **Spark GraphFrames**
     - Cypher instead of a DataFrame API; one machine, no cluster.
     - Interactive latency for filters and traversals on CPU, and GPU PageRank. The
       head-to-head with results: :doc:`benchmark_graphframes`.

Motif, triangle and other cyclic multi-way-join patterns run in GFQL but are not
benchmarked; this documentation makes no performance claim about them.

.. _gfql-offengine-calls:

Analytics under Polars (``umap`` / ``hypergraph`` / ``compute_cugraph`` …)
-------------------------------------------------------------------------------

A GFQL ``call()`` that runs a whole-graph analytic (``umap``, ``hypergraph``,
``compute_cugraph`` / ``compute_igraph``, the ``*_layout`` ops, ``collapse``) wraps a
pandas, cuDF or GPU library and has no Polars implementation. Under ``engine='polars'``
or ``'polars-gpu'`` these calls run off-engine:

- **``call_mode='auto'`` (default):** the analytic runs on pandas for ``polars`` and on
  cuDF for ``polars-gpu``. The result comes back as Polars frames without loss, and GFQL
  warns once per analytic. ``polars-gpu`` raises if the cuDF stack is missing; it never
  moves a GPU analytic to the CPU.
- **``call_mode='strict'``:** raise instead of running the analytic off-engine. Use this
  when a run must stay on one engine, or to hold a hard memory ceiling.

.. note::
   **Memory on a very large graph.** The off-engine run makes one copy of the graph in
   the analytic's format: pandas in host memory for ``polars``, cuDF in device memory for
   ``polars-gpu``. This is the same allocation the analytic makes on ``engine='cudf'``.
   GFQL sets no size cap of its own. If that copy is a concern, set
   ``call_mode='strict'`` or run under an RMM or container memory limit, as for any cuDF
   workload.

Traversal, filter and row operations (``hop``, ``WHERE``, ``RETURN``) are never run
off-engine: they run on the engine you asked for or raise before running. Set the mode
from Python or the environment; a Python setting overrides the environment:

.. doc-test: skip

.. code-block:: python

   from graphistry.compute.gfql.lazy import set_call_mode, CALL_MODES  # ('auto', 'strict')

   set_call_mode('strict')   # raise on off-engine analytics (pass None to reset to env/default)
   # or: export GFQL_POLARS_CALL_MODE=strict

cuDF vs Polars-GPU
------------------

Both run on an NVIDIA GPU.

- ``cudf`` runs the query one operation at a time; each hop is a separate kernel with a
  materialized intermediate. It is a supported, first-class engine and the right choice
  for one very large result.
- ``polars-gpu`` runs the same fused plan as CPU Polars, collected once on the GPU. That
  is why it leads on heavy multi-hop work.
- ``cudf`` operates on ``cudf.DataFrame``; ``polars-gpu`` on ``polars.DataFrame``, with
  only the collect running on the GPU. A graph built from pandas frames works with either.
- Both need the RAPIDS GPU stack; ``polars-gpu`` also uses ``cudf_polars``. CPU
  ``polars`` needs only ``pip install polars``.

.. _gfql-larger-than-memory:

Larger-than-memory: streaming execution
---------------------------------------

The Polars engines run in memory by default, which is fastest while the graph and the
query's intermediate results fit in RAM or GPU memory. Two opt-in streaming modes trade a
little latency for a much larger working set:

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Mode
     - Engine
     - What it does
   * - ``GFQL_POLARS_CPU_STREAMING=1``
     - ``polars``
     - Runs the plan with Polars' streaming engine: batches, with spill to disk, so
       intermediate results can exceed RAM.
   * - ``GFQL_POLARS_GPU_EXECUTOR=streaming``
     - ``polars-gpu``
     - Uses the cudf-polars streaming executor for results larger than GPU memory.

Both are off by default because they slow down small, interactive work. Results are
identical to the default modes. Set them by environment variable:

.. code-block:: bash

   # CPU: batched + disk-spill for larger-than-RAM intermediates
   export GFQL_POLARS_CPU_STREAMING=1

   # GPU: streaming executor for larger-than-device-memory results
   export GFQL_POLARS_GPU_EXECUTOR=streaming

or from Python at runtime; a Python setting overrides the environment:

.. doc-test: skip

.. code-block:: python

   from graphistry.compute.gfql.lazy import (
       set_cpu_streaming, set_gpu_executor, GPU_EXECUTORS,
   )

   set_cpu_streaming(True)          # CPU streaming collect (pass None to reset to env/default)
   set_gpu_executor('streaming')    # one of GPU_EXECUTORS == ('in-memory', 'streaming')

Then call ``g.gfql(query, engine='polars')`` or ``engine='polars-gpu'`` as before.

.. note::
   These modes stream the query, which helps when the input fits in memory but the
   intermediate or final results do not. The source graph must still fit in memory:
   ``graphistry`` materializes edge and node frames at ingestion, and a
   ``polars.LazyFrame`` is collected immediately. Building GFQL directly on a lazy
   ``pl.scan_parquet`` source, so that a graph larger than RAM never fully materializes,
   is work in progress; see the Friendster discussion on the GraphFrames benchmark page.

Same results on every engine
----------------------------

- Every engine returns the same result as pandas. This is tested across forward,
  reverse and undirected traversal, one to three hops, filters and aggregations.
- Traversal, filter and row operations never change engine silently. The Polars engine
  runs them natively or raises before running, so a latency you measure is real work on
  the engine you asked for. ``polars-gpu`` also raises if any step of the plan cannot run
  on the GPU.
- Whole-graph analytics are the one exception, and they warn: see
  :ref:`gfql-offengine-calls`. ``call_mode='strict'`` turns that warning into an error.

Install
-------

.. code-block:: bash

   pip install graphistry          # base; pandas engine works out of the box
   pip install graphistry polars   # adds the CPU 'polars' engine
   # 'cudf' and 'polars-gpu' require the NVIDIA RAPIDS stack (GPU);
   # 'polars-gpu' additionally uses cudf_polars.

.. note::
   Layouts, plotting and featurization accept Polars frames as input and compute in
   pandas. The engine behavior on this page applies to GFQL query execution.

See also
--------

- :doc:`performance` — measured results against graph databases
- :doc:`benchmark_filter_pagerank` — a Cypher + PageRank pipeline vs Neo4j + GDS
- :doc:`index_adjacency` — queries that start from known nodes
- :doc:`/api/gfql/index` — GFQL API reference
- :doc:`remote` — run GFQL on a remote GPU
