.. _gfql-engines:

Choosing a GFQL Engine: pandas, Polars, cuDF, Polars-GPU
========================================================

GFQL runs the same query on four execution engines. You choose the engine with one
keyword, ``engine=``, on ``g.gfql()`` and ``g.hop()``. Supported queries return the
same results across engines. Set an explicit engine to control execution; the default
``engine='auto'`` selects from the input frames.

This page assumes you have a graph ``g`` and a ``query``. If not, start with
:doc:`about`.

Switch engines with one keyword
-------------------------------

.. doc-test: skip

.. code-block:: python

   import graphistry
   g = graphistry.edges(df, 'src', 'dst')   # df: your edges dataframe (pandas / Polars / cuDF)
   query = "MATCH (a)-[e]->(b) RETURN b"     # any GFQL / Cypher query

   g.gfql(query)                       # engine='auto': follows the input frames
   g.gfql(query, engine='polars')      # CPU, columnar
   g.gfql(query, engine='cudf')        # NVIDIA GPU (RAPIDS)
   g.gfql(query, engine='polars-gpu')  # the Polars plan on the GPU

Start with Polars for CPU graph analytics. On the ``prrao87/graph-benchmark`` Cypher
suite it beats pandas on :bench-tally:`graphbench.20k|polars|pandas` queries at
20,000 people and :bench-tally:`graphbench.100k|polars|pandas` at 100,000 people.
See :doc:`performance` for the per-query times.

Your existing pandas, Polars, or cuDF graph works as-is. GFQL converts the input frames
once, at the start of the call. Results come back in the engine's frame type: Polars
frames for ``'polars'`` and ``'polars-gpu'``, ``cudf.DataFrame`` for ``'cudf'``. Convert
once when downstream code needs pandas:

.. doc-test: skip

.. code-block:: python

   out = g.gfql(query, engine='polars')       # or 'cudf' / 'polars-gpu'
   nodes_pd = out._nodes.to_pandas()          # pandas for matplotlib, scikit-learn, .iloc, ...

**Already a Polars user?** The default ``engine='auto'`` follows your frames: a graph built
from Polars frames uses Polars when the query is supported. If the query is unsupported,
``auto`` runs it on pandas. Pass ``engine='polars'`` to require Polars execution.
For cuDF input, ``auto`` can use Polars-GPU when available and returns cuDF frames;
it otherwise uses cuDF. Select an explicit engine for reproducible comparisons:

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
     - Selection
     - In one line
   * - ``pandas``
     - CPU
     - ``pandas``
     - ``auto`` for pandas input, or explicit
     - CPU execution with pandas frames.
   * - ``polars``
     - CPU
     - ``polars``
     - ``auto`` for Polars input, or explicit
     - Columnar CPU execution with combined operations and indexed lookups.
   * - ``cudf``
     - NVIDIA GPU
     - ``cudf``
     - ``auto`` for cuDF input, or explicit
     - RAPIDS operations on cuDF frames.
   * - ``polars-gpu``
     - NVIDIA GPU
     - ``polars``
     - explicit, or selected by ``auto`` for cuDF input
     - Polars plans execute through ``cudf_polars``; some query paths use CPU operations.

Polars combines operations into lazy plans and uses direct row selection for some
indexed queries. A query can execute several plans. With ``polars-gpu``, plans submitted
to the GPU must be supported by ``cudf_polars`` or raise an error. CPU operations can
still run before, between, or instead of those plans. The :doc:`performance` page
identifies measured queries that execute entirely on CPU with this engine setting.

Which engine for which work
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 25 43

   * - Workload
     - Start with
     - What to check
   * - Filters and aggregations on CPU
     - ``polars``
     - Compare the full query, including input conversion.
   * - Bulk one-hop expansion on CPU
     - ``polars``
     - Selective filters can reduce the rows passed into joins.
   * - Large multi-hop queries
     - ``polars-gpu``
     - Compare CPU Polars too; GPU overhead and intermediate result sizes affect the outcome.
   * - Full-graph aggregation on GPU
     - ``polars-gpu`` / ``cudf``
     - Measure execution time and peak memory on your query.
   * - Results larger than available memory
     - Polars streaming modes
     - See :ref:`gfql-larger-than-memory` for input and intermediate-memory limits.
   * - Small filters and interactive queries
     - The engine matching your input frames
     - Input conversion can cost more than the query itself.
   * - Queries starting from a few known nodes
     - ``pandas`` / ``polars`` with an adjacency index
     - Build indexes once and reuse them across queries; see :doc:`index_adjacency`.
   * - Features the Polars engine does not support
     - ``pandas``
     - Explicit Polars raises for unsupported queries; ``auto`` can run them on pandas.

Three rules cover most decisions:

- **Compare CPU and GPU on your query.** GPU launches and data transfers add overhead.
  Large joins and aggregations can benefit from GPU execution; small results often
  finish faster on CPU.
- **Reuse indexes for repeated lookups.** Build them with ``g.gfql_index_all()`` so
  queries from known nodes can read their neighborhoods. Start with a CPU engine for
  small neighborhoods. See :doc:`index_adjacency`.
- **Check which operations use the GPU.** The ``polars-gpu`` setting selects GPU
  execution for submitted lazy plans. It does not guarantee that every query uses the
  GPU. Unsupported GPU plans raise an error.

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
     - GFQL removes the server round trip and keeps results as dataframes. Indexed
       message-content and creator lookups lead the matched SNB-derived comparison
       on :doc:`performance`.
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

``call_mode`` controls whole-graph analytics. It does not force every traversal,
filter, or row operation onto the GPU. Set it from Python or the environment;
a Python setting overrides the environment:

.. doc-test: skip

.. code-block:: python

   from graphistry.compute.gfql.lazy import set_call_mode, CALL_MODES  # ('auto', 'strict')

   set_call_mode('strict')   # raise on off-engine analytics (pass None to reset to env/default)
   # or: export GFQL_POLARS_CALL_MODE=strict

cuDF vs Polars-GPU
------------------

Both can execute dataframe work on an NVIDIA GPU.

- ``cudf`` operates directly on ``cudf.DataFrame`` objects through RAPIDS.
- ``polars-gpu`` keeps Polars frames and submits lazy plans to ``cudf_polars`` for GPU
  execution. Query execution can also include CPU operations.
- A graph built from pandas frames works with either engine after input conversion.
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
- Explicit ``engine='polars'`` runs supported queries in Polars and raises for unsupported
  features. With ``auto``, unsupported Polars queries can run on pandas.
- Plans submitted to ``cudf_polars`` raise if GPU execution is unsupported. Other
  operations in the query can still use CPU execution.
- Whole-graph analytics can convert frames to another engine and warn: see
  :ref:`gfql-offengine-calls`. ``call_mode='strict'`` makes those calls raise.

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

Provenance
----------

The query comparisons above use these datasets and measurement profiles.

.. bench-provenance:: graphbench-q1q9-20k-master-f283a305e-20260917 graphbench-q1q9-100k-master-f283a305e-20260917

See also
--------

- :doc:`performance` — measured results against graph databases
- :doc:`benchmark_filter_pagerank` — a Cypher + PageRank pipeline vs Neo4j + GDS
- :doc:`index_adjacency` — queries that start from known nodes
- :doc:`/api/gfql/index` — GFQL API reference
- :doc:`remote` — run GFQL on a remote GPU
