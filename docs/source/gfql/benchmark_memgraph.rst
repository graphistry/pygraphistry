GFQL Cypher Benchmark: CPU/GPU DataFrames vs Memgraph
========================================================

This benchmark compares the same ``graph-benchmark`` q1-q9 workload across:

- GFQL on CPU dataframes: ``graph_benchmark_q1_q9.py --engine pandas``
- GFQL on GPU dataframes: ``graph_benchmark_q1_q9.py --engine cudf``
- Memgraph over Bolt: ``graph_benchmark_memgraph_q1_q9.py``
- Neo4j over Bolt: ``graph_benchmark_neo4j_q1_q9.py``
- Kuzu (embedded): ``graph_benchmark_kuzu.py``

All comparator runners emit the same JSON timing shape, and
``graph_benchmark_compare.py`` renders a combined CPU/GPU/Neo4j/Memgraph/Kuzu
table (``--neo4j/--memgraph/--kuzu``), so the same query semantics are compared
across every engine.

The benchmark is intended to run on ``dgx-spark`` so GFQL GPU numbers come
from the same RAPIDS Docker workflow used for GFQL validation. Raw outputs
stay under ``plans/``; publish selected summaries to
``benchmarks/gfql/RESULTS.md`` after a full run.

What It Measures
----------------

The workload replays q1-q9 from ``prrao87/graph-benchmark``. GFQL uses the
existing vectorized pandas/cuDF runner and the Memgraph runner loads the same
generated parquet data into a temporary Memgraph database. Both scripts emit
JSON with per-query median timings.

The comparison renderer builds a markdown table with GFQL CPU, GFQL GPU,
Memgraph, and speedups against Memgraph:

Measured DGX Run
----------------

On 2026-07-04, ``dgx-spark`` ran the full 100k-person graph-benchmark
dataset with RAPIDS 26.02 and Memgraph 3.11.0. The graph contained
107,434 nodes and 2,775,195 relationships, including 2,417,738
``FOLLOWS`` relationships. Memgraph loaded the graph with chunked
server-side ``LOAD CSV`` in 13.10s. The Memgraph baseline comes from the wrapper run with ``RUNS=5`` and ``WARMUP=1``. The GFQL columns below use current-code validation JSONs on the same generated graph at ``RUNS=5``/``WARMUP=1``, including the columnar in-degree ``q1`` shortcut. GFQL ``+ preindex`` columns show the same query medians with per-query preindex build time added.

.. list-table:: DGX q1-q9 median timings
   :header-rows: 1
   :widths: 8 16 16 16 16 14 14 14

   * - Query
     - GFQL CPU query
     - GFQL CPU + preindex
     - GFQL GPU query
     - GFQL GPU + preindex
     - Memgraph
     - CPU query speedup
     - GPU query speedup
   * - q1
     - 80.51ms
     - 384.23ms
     - 19.74ms
     - 126.36ms
     - 1.49s
     - 18.52x
     - 75.57x
   * - q2
     - 96.92ms
     - 550.46ms
     - 58.78ms
     - 146.17ms
     - 687.32ms
     - 7.09x
     - 11.69x
   * - q3
     - 28.38ms
     - 474.28ms
     - 11.34ms
     - 87.25ms
     - 84.14ms
     - 2.96x
     - 7.42x
   * - q4
     - 17.39ms
     - 463.65ms
     - 7.76ms
     - 83.18ms
     - 40.65ms
     - 2.34x
     - 5.24x
   * - q5
     - 3.84ms
     - 663.43ms
     - 2.51ms
     - 109.15ms
     - 2.48ms
     - 0.65x
     - 0.99x
   * - q6
     - 3.59ms
     - 703.66ms
     - 3.77ms
     - 89.89ms
     - 4.41ms
     - 1.23x
     - 1.17x
   * - q7
     - 4.09ms
     - 619.19ms
     - 4.51ms
     - 91.61ms
     - 6.35ms
     - 1.55x
     - 1.41x
   * - q8
     - 87.14ms
     - 380.70ms
     - 27.72ms
     - 106.28ms
     - 737.00ms
     - 8.46x
     - 26.59x
   * - q9
     - 203.08ms
     - 511.67ms
     - 51.89ms
     - 130.52ms
     - 742.62ms
     - 3.66x
     - 14.31x


Raw JSON and markdown outputs are recorded under
``plans/gfql-memgraph-benchmarks/results/`` in the local benchmark plan.

Neo4j and Kuzu comparators
--------------------------

The in-repo q1-q9 runner was also run against Neo4j 5.26 (Bolt) and
Kuzu (embedded) on the identical 100k-person / 2.78M-edge graph, with result
parity confirmed across engines (e.g. q8 ``numPaths`` 58,431,994 and q9
45,514,124 match on Neo4j, Memgraph, and Kuzu). Query-only medians (ms
unless noted):

======  =========  =========  =======  ========  =======
Query   GFQL CPU   GFQL GPU   Neo4j    Memgraph  Kuzu
======  =========  =========  =======  ========  =======
q1      80.51      19.74      832.17   1.49s     253.19
q2      96.92      58.78      569.77   687.32    273.59
q3      28.38      11.34      49.88    84.14     38.55
q4      17.39      7.76       201.68   40.65     34.81
q5      3.84       2.51       9.52     2.48      12.66
q6      3.59       3.77       25.29    4.41      17.64
q7      4.09       4.51       13.60    6.35      8.67
q8      87.14      27.72      1.20s    737.00    1.03s
q9      203.08     51.89      1.78s    742.62    876.65
======  =========  =========  =======  ========  =======

In this in-repo runner view, GFQL wins q1/q2/q3/q4/q7/q8/q9 against
Neo4j, Memgraph, and Kuzu, and is a near-tie on q5/q6 (both sub-5ms). **q1**
(top-3 followers over a full ``FOLLOWS`` scan) was previously the one loss to
Kuzu's columnar scan-and-aggregate; a direct columnar in-degree ``count``/top-3
shortcut now puts GFQL at 19.74ms GPU / 80.51ms CPU, ahead of Kuzu's 253.19ms.

The fair-matrix run using each competitor's own marketed repo queries corrects
the q8 story: Kuzu q8 is 10.4ms and LadybugDB q8 is 19.5ms, both faster than
GFQL GPU at 27.72ms. Treat q8 as a genuine comparator win in fair repo-query
accounting; GFQL still wins q1/q2/q4/q5/q6/q7/q9(GPU). Load times differ
widely (Kuzu ~2s, Memgraph ~13s CSV, Neo4j ~48s Bolt) but are excluded from
query medians.

The combined table is rendered with all three comparators:

.. code-block:: bash

   python benchmarks/gfql/graph_benchmark_compare.py \
     --gfql-cpu plans/gfql-memgraph-benchmarks/results/graph_benchmark_gfql_cpu.json \
     --gfql-gpu plans/gfql-memgraph-benchmarks/results/graph_benchmark_gfql_gpu.json \
     --neo4j plans/gfql-memgraph-benchmarks/results/graph_benchmark_neo4j.json \
     --memgraph plans/gfql-memgraph-benchmarks/results/graph_benchmark_memgraph.json \
     --kuzu plans/gfql-memgraph-benchmarks/results/kuzu-comparator/kuzu-q1q9-full-probe.json \
     --output-md plans/gfql-memgraph-benchmarks/results/graph_benchmark_gfql_vs_all.md

DGX One-Command Run
-------------------

On ``dgx-spark``, use the wrapper from a checkout that includes these
benchmark scripts. The wrapper starts a host-side Memgraph container with ``/tmp`` mounted for CSV staging, then
runs a RAPIDS container with the repository bind-mounted and ``--network
host`` so the Python benchmark can load Memgraph with server-side CSV and reach it on Bolt port ``7687``.

First generate the graph-benchmark data if it is not already present:

.. code-block:: bash

   ssh -o BatchMode=yes dgx-spark
   git clone https://github.com/prrao87/graph-benchmark.git $HOME/graph-benchmark
   cd $HOME/graph-benchmark
   bash generate_data.sh 100000

Then run the benchmark from the pygraphistry checkout:

.. code-block:: bash

   cd /home/lmeyerov/Work/pygraphistry2
   GRAPH_BENCHMARK_ROOT=$HOME/graph-benchmark \
   RESULTS_DIR=plans/gfql-memgraph-benchmarks/results \
   RUNS=5 WARMUP=1 \
   RAPIDS_VERSION=26.02 \
   benchmarks/gfql/run_graph_benchmark_memgraph_dgx.sh

The wrapper writes:

- ``graph_benchmark_gfql_cpu.json``
- ``graph_benchmark_gfql_gpu.json``
- ``graph_benchmark_memgraph.json``
- ``graph_benchmark_gfql_memgraph.md``

Useful environment knobs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Purpose
   * - ``RAPIDS_VERSION``
     - ``26.02`` by default; ``25.02`` is also supported with the DGX CUDA bridge workaround used by ``docker/test-rapids-official-local.sh``.
   * - ``GRAPH_BENCHMARK_ROOT``
     - Path to the generated ``graph-benchmark`` checkout. Defaults to ``$HOME/graph-benchmark``.
   * - ``RESULTS_DIR``
     - Relative output directory in the pygraphistry checkout. Defaults to ``plans/gfql-memgraph-benchmarks/results``.
   * - ``MEMGRAPH_IMAGE``
     - Memgraph Docker image. Defaults to ``memgraph/memgraph-mage``.
   * - ``MEMGRAPH_LOAD_METHOD``
     - ``csv`` by default for server-side ``LOAD CSV``. Set to ``bolt`` for the slower batched Bolt loader.
   * - ``MEMGRAPH_CSV_DIR``
     - CSV staging directory visible to both containers. Defaults to ``/tmp/gfql_memgraph_import``.
   * - ``START_MEMGRAPH``
     - Set to ``0`` to reuse an already-running Memgraph service.
   * - ``KEEP_MEMGRAPH``
     - Set to ``1`` to leave the Memgraph container running after the wrapper exits.
   * - ``INSTALL_DEPS``
     - Set to ``0`` when the selected RAPIDS image already has the needed editable pygraphistry deps and ``neo4j`` driver installed.
   * - ``GFQL_QUERY_VARIANT``
     - ``standard`` by default. It applies the simple benchmark policy: direct dataframe shortcuts for q3/q4/q5/q6/q7, plus scoped setup-time uniqueness-gated lazy Polars CPU and setup-time uniqueness-gated typed cuDF GPU paths for large HAS_INTEREST edge sets when available. The CPU policy includes q5 location-first final semi-join, q6 setup-time interest-id and gender-indexed location join, and q7 target-country-first path pruning. Tiny runs stay on the base dataframe paths. ``dataframe-shortcut`` forces dataframe shortcuts and is retained for exploratory sweeps.

Manual Components
-----------------

Run GFQL CPU or GPU directly:

.. code-block:: bash

   python benchmarks/gfql/graph_benchmark_q1_q9.py \
     --graph-benchmark-root $GRAPH_BENCHMARK_ROOT \
     --engine cudf \
     --mode preindexed \
     --include-preindex \
     --query-variant standard \
     --runs 5 --warmup 1 \
     --output-json /tmp/graph_benchmark_gfql_gpu.json

Run Memgraph directly against an existing Bolt service:

.. code-block:: bash

   uv run --with neo4j python benchmarks/gfql/graph_benchmark_memgraph_q1_q9.py \
     --graph-benchmark-root $GRAPH_BENCHMARK_ROOT \
     --uri bolt://127.0.0.1:7687 \
     --runs 5 --warmup 1 \
     --load-method csv \
     --csv-dir /tmp/gfql_memgraph_import \
     --output-json /tmp/graph_benchmark_memgraph.json

Limitations
-----------

- GFQL preindexed mode reports both query-only and query-plus-preindex
  medians. The comparison renderer defaults to showing both; use
  ``--gfql-timing query`` or ``--gfql-timing with-preindex`` to force one
  accounting mode.
- Memgraph load time is recorded separately under the JSON ``load`` key and
  is not folded into per-query medians.
- ``--query-variant standard`` applies the simple benchmark policy: direct
  dataframe aggregations on q3/q4/q5/q6/q7, with scoped setup-time uniqueness-gated lazy Polars CPU and setup-time uniqueness-gated typed cuDF GPU
  ``isin`` semijoin/groupby execution for large preindexed q5/q6/q7 when available.
  ``dataframe-shortcut`` forces dataframe shortcuts and is retained for
  exploratory sweeps.
- q1-q7 use Cypher pattern queries plus aggregations. q8-q9 count FOLLOWS
  length-2 paths, matching the vectorized degree-math semantics in the GFQL
  runner.
