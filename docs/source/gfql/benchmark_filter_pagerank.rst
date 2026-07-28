GFQL Cypher Benchmark: CPU/GPU DataFrames vs Neo4j
==================================================

.. image:: _static/gfql-mascot.png
   :alt: GFQL mascot
   :width: 160px
   :align: right

Run Cypher graph queries and analytics directly on Python dataframes —
no database required. This benchmark compares **Graphistry's local Cypher**
(CPU and GPU) against **Neo4j + GDS** on the same end-to-end pipeline.

.. warning::
   **The figures previously published on this page have been withdrawn.** The raw
   measurement artifacts for that run no longer exist anywhere — nothing was committed,
   and the chart generator (``benchmarks/gfql/filter_pagerank/presentation.py``) reads a
   results directory that is not present in this repository, so even the rendered charts
   cannot be regenerated. That makes those numbers impossible to confirm *or* refute, so
   they are treated as unpublishable rather than assumed correct.

   What remains below is the part that is still verifiable: what the benchmark measures,
   the exact pipeline, the Neo4j+GDS analog, and how to run it yourself. The figures will
   be republished once this pipeline runs under the provenance-carrying harness described
   on :doc:`performance` — committed per-slot artifacts, recorded commit/host/perf-lock/
   reps, and results validated against the competitor before any ratio is published.

The pipeline
------------

One ``g.gfql(...)`` call — search, enrich with PageRank, search again:

.. code-block:: python

   # pip install graphistry
   result = g.gfql("""
       GRAPH g1 = GRAPH {
         MATCH (n)-[e]-(m)
         WHERE n.degree >= $degree_cutoff
       }
       GRAPH g2 = GRAPH {
         USE g1
         CALL graphistry.cugraph.pagerank.write()
       }
       GRAPH {
         USE g2
         MATCH (n)-[e]-(m)
         WHERE n.pagerank >= $pagerank_cutoff
       }
   """,
       params={
           "degree_cutoff": degree_cutoff,
           "pagerank_cutoff": pagerank_cutoff,
       },
       engine="cudf",  # or "pandas" with igraph backend
   )

- ``GRAPH g1``: find high-degree nodes and their neighbors
- ``GRAPH g2``: enrich ``g1`` with PageRank scores (igraph on CPU, cugraph on GPU)
- Final ``GRAPH``: keep high-PageRank nodes and their neighbors

The same pipeline shape, different backends:

- **CPU**: ``engine="pandas"``, ``backend="igraph"``
- **GPU**: ``engine="cudf"``, ``backend="cugraph"``

The Neo4j equivalent requires ~30 lines of Cypher + GDS projection + batched
writes (see :ref:`neo4j-analog` below).

What is measured
----------------

Two comparisons, on two SNAP graphs (Twitter, 2.4M edges; GPlus, 30M edges):

- **Pipeline time** — search + PageRank + search, with the graph already loaded.
- **Full lifecycle** — the same pipeline plus the one-time cost of getting the data into
  each system: import and preparation for Neo4j, load and shaping for GFQL. This is the
  number that reflects an analyst starting from files.

Both are broken out by workload phase — **ETL** (load + shape), **Search** (graph
queries), **Analytics** (PageRank) — so the cost is attributable rather than a single
opaque total.

Why this matters
----------------

You get Cypher-style graph search + PageRank directly on your dataframe, with no database
to stand up or maintain, and the whole pipeline stays in one process.

The GPU path accelerates everything — ETL, search, and analytics — because
``cudf`` and ``cugraph`` are drop-in replacements for ``pandas`` and ``igraph``
under the same GFQL Cypher surface.

.. _neo4j-analog:

Neo4j + GDS analog
------------------

The Neo4j equivalent of the same pipeline:

.. code-block:: cypher

   -- 1. Mark seed nodes by degree
   MATCH (n:Node)
   SET n.seed = n.degree >= $cutoff;

   -- 2. Expand one hop from seeds
   UNWIND $seed_ids AS sid
   MATCH (s:Node) WHERE id(s) = sid
   MATCH (s)-[r:LINK]-(target:Node)
   SET target.in_subgraph = true, r.in_subgraph = true;

   -- 3. Project subgraph and run PageRank
   CALL gds.graph.project.cypher(
     'subgraph',
     'MATCH (n:Node) WHERE n.in_subgraph RETURN id(n) AS id',
     'MATCH (a)-[r:LINK]->(b) WHERE r.in_subgraph
      RETURN id(a) AS source, id(b) AS target
      UNION ALL
      MATCH (a)-[r:LINK]->(b) WHERE r.in_subgraph
      RETURN id(b) AS source, id(a) AS target'
   );
   CALL gds.pageRank.write('subgraph', {writeProperty: 'pagerank'});

   -- 4. Keep high-PageRank core + one hop
   MATCH (n:Node) WHERE n.pagerank >= $cutoff
   SET n.core = true;
   UNWIND $core_ids AS cid
   MATCH (c:Node) WHERE id(c) = cid
   MATCH (c)-[r:LINK]-(target:Node)
   SET target.final = true, r.final = true;

Why the GFQL pipeline is shorter
--------------------------------

The difference in pipeline length above is not accidental. It reflects a
design difference in how graphs flow through the system:

**Graphs as first-class values.** GFQL's ``GRAPH { }`` constructors treat
graphs as composable values that flow between pipeline stages. Each stage
receives a graph, transforms it, and passes a graph to the next stage.
Standard Cypher and GQL are constrained to paths and rows as output values,
which forces the Neo4j pipeline into explicit property-flag marking,
separate GDS projections, and batched write-back steps.

**Multi-language, single engine.** The GFQL engine is being designed to
support Cypher, and over time additional property graph query languages,
all compiled to the same vectorized columnar execution backend. Users write
in whichever declarative syntax they prefer; the engine handles CPU/GPU
dispatch transparently. See :doc:`cypher` for the current Cypher surface
and :doc:`overview` for the broader GFQL design.

**Modern execution without legacy constraints.** Because GFQL does not
inherit a database storage layer or a row-at-a-time execution model, it can
represent intermediate graph results natively in columnar memory (Arrow /
pandas / cuDF). That is what makes the CPU-to-GPU switch a configuration
flag (``engine="cudf"``) rather than a rewrite, and what keeps ETL, search,
and analytics in the same in-process pipeline.

**Same answer on every engine.** CPU and GPU timings of this pipeline are comparable
because the query meaning is fixed: GFQL's engine contract is same result or
pre-execution decline. Unsupported engine/query combinations are rejected during
validation, compilation, or planning before query execution, rather than silently
falling back or returning a different answer. See :doc:`engines` for the full
parity and static-safety contract.

This page is one workload (a filter → PageRank → filter pipeline) against one
external baseline (Neo4j+GDS). For the full four-engine picture — when Polars
beats pandas on CPU, when the GPU pulls ahead, and how to choose — see
:doc:`engines`, and :doc:`performance` for the measured board. For *seeded*
lookups, where an index rather than an engine is the lever, see
:doc:`index_adjacency`.

For more on the GFQL design and supported surface:

- :doc:`engines` — choosing pandas / Polars / cuDF / Polars-GPU
- :doc:`index_adjacency` — seeded-traversal CSR adjacency index
- :doc:`cypher` — Cypher syntax through ``g.gfql("MATCH ...")``
- :doc:`overview` — GFQL design, features, and GPU acceleration
- :doc:`about` — 10-minute introduction to GFQL

Methodology
-----------

- Host: ``dgx-spark``, GPU: ``GB10``, driver ``580.126.09``
- Container: ``graphistry/test-gpu:latest``
- Datasets: `SNAP <https://snap.stanford.edu/data/>`_ Twitter (2.4M edges) and GPlus (30M edges)
- Measurement: warm median of 5 timed runs after 2 warmup iterations
- Neo4j runs the analog Cypher + GDS pipeline below on the same host

Reproduce
---------

Note the caveat at the top of this page: these reproducers print and plot, but do not
yet emit a provenance-carrying artifact, which is why their output is not published here.

- ``benchmarks/gfql/filter_pagerank/load_prepare_cpu_gpu.py`` — load + shape the graphs
- ``benchmarks/gfql/filter_pagerank/filter_pagerank_pipeline_cpu_gpu.py`` — the GFQL CPU/GPU pipeline
- ``benchmarks/gfql/filter_pagerank/filter_pagerank_pipeline_neo4j.py`` — the Neo4j + GDS analog
- ``benchmarks/gfql/filter_pagerank/presentation.py`` — chart generation

Notebook version
----------------

See ``demos/gfql/benchmark_filter_pagerank_cpu_gpu.ipynb`` for a notebook
version of this writeup.
