GFQL Cypher Filter + PageRank Benchmark
========================================

.. image:: _static/gfql-mascot.png
   :alt: GFQL mascot
   :width: 160px
   :align: right

Run Cypher queries and graph analytics directly on Python dataframes, without a
database. This benchmark compares **Graphistry's local Cypher** on CPU and GPU
with **Neo4j + GDS** for the same three-stage pipeline.

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 20

   * -
     - Neo4j + GDS
     - GFQL Cypher (CPU)
     - GFQL Cypher (GPU)
     - GFQL GPU vs CPU
   * - **Twitter** (81,306 nodes / 2.4M edges)
     - :bench:`pagerank.twitter.neo4j_gds`
     - :bench:`pagerank.twitter.gfql_cpu`
     - :bench:`pagerank.twitter.gfql_gpu`
     - :bench:`pagerank.twitter.gfql_gpu_vs_gfql_cpu`
   * - **GPlus** (107,614 nodes / 30M edges)
     - :bench:`pagerank.gplus.neo4j_gds`
     - :bench:`pagerank.gplus.gfql_cpu`
     - :bench:`pagerank.gplus.gfql_gpu`
     - :bench:`pagerank.gplus.gfql_gpu_vs_gfql_cpu`

Each time covers the full search → PageRank → search pipeline after warm-up. GFQL
reuses data already loaded in Python. Neo4j includes server calls and rebuilds the
in-memory graph used by Graph Data Science (GDS) for each timed iteration. The table
therefore shows direct pipeline times, not a GFQL-to-Neo4j speedup ratio.

For the same GFQL query, the GPU path is
:bench:`pagerank.twitter.gfql_gpu_vs_gfql_cpu` faster on Twitter and
:bench:`pagerank.gplus.gfql_gpu_vs_gfql_cpu` faster on the 30M-edge GPlus graph.

The pipeline
------------

One ``g.gfql(...)`` call searches the graph, calculates PageRank, and searches
the result:

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

Choose a CPU or GPU backend without changing the query:

- **CPU**: ``engine="pandas"``, ``backend="igraph"``
- **GPU**: ``engine="cudf"``, ``backend="cugraph"``

The Neo4j version requires Cypher, a separate in-memory graph for GDS, and several
writes. See :ref:`neo4j-analog` below.

Twitter (2.4M edges): reported pipeline timings
------------------------------------------------

.. image:: _static/filter_pagerank/twitter_pipeline.svg
   :alt: Twitter warm pipeline time: Neo4j + GDS 11.72s, GFQL Cypher CPU 1.58s, GFQL Cypher GPU 0.24s

- **Neo4j + GDS**: :bench:`pagerank.twitter.neo4j_gds`

- **GFQL Cypher on CPU** (pandas + igraph): :bench:`pagerank.twitter.gfql_cpu`

- **GFQL Cypher on GPU** (cuDF + cuGraph): :bench:`pagerank.twitter.gfql_gpu` —
  :bench:`pagerank.twitter.gfql_gpu_vs_gfql_cpu` faster than the GFQL CPU path

GPlus (30M edges): larger graph
-------------------------------

.. image:: _static/filter_pagerank/gplus_pipeline.svg
   :alt: GPlus warm pipeline time: Neo4j + GDS 354.47s, GFQL Cypher CPU 32.10s, GFQL Cypher GPU 2.42s

- **Neo4j + GDS**: :bench:`pagerank.gplus.neo4j_gds`
- **GFQL Cypher on CPU** (pandas + igraph): :bench:`pagerank.gplus.gfql_cpu`
- **GFQL Cypher on GPU** (cuDF + cuGraph): :bench:`pagerank.gplus.gfql_gpu` —
  :bench:`pagerank.gplus.gfql_gpu_vs_gfql_cpu` faster than the CPU path

GPlus is 12x the edges of the Twitter graph, and the GPU pipeline still answers in
seconds.

What this shows
---------------

GFQL runs the same query on pandas + igraph or cuDF + cuGraph. The GPU path was
faster on both graphs. GFQL also keeps dataframe processing, graph search, and
analytics in one Python process.

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

The Neo4j version is longer because its stages write flags to database records
and create a separate GDS graph. GFQL passes a graph directly from one stage to
the next.

**Graphs as values.** Each ``GRAPH { }`` block receives a graph, changes it, and
passes a graph to the next block. This removes the property flags, separate GDS
projections, and batched writes used in the Neo4j example.

**One query, multiple engines.** GFQL compiles Cypher to dataframe operations.
Set ``engine="pandas"`` for CPU execution or ``engine="cudf"`` for GPU execution.
See :doc:`cypher` for supported Cypher features and :doc:`overview` for the GFQL
design.

**Columnar data in Python.** Intermediate graphs stay in Arrow, pandas, or cuDF
memory. ETL, search, and analytics can remain in the same Python pipeline.

**Consistent results.** GFQL either returns the same result on an engine or rejects
the query before execution. It does not silently change engines. See :doc:`engines`
for the parity and validation rules.

This page is one workload (a filter → PageRank → filter pipeline) against one
external baseline (Neo4j + GDS). For the full four-engine picture — when Polars
beats pandas on CPU, when the GPU pulls ahead, and how to choose — see
:doc:`engines`. For seeded lookups, see :doc:`index_adjacency`.

For more on GFQL:

- :doc:`engines` — choosing pandas / Polars / cuDF / Polars-GPU
- :doc:`index_adjacency` — seeded-traversal CSR adjacency index
- :doc:`cypher` — Cypher syntax through ``g.gfql("MATCH ...")``
- :doc:`overview` — GFQL design, features, and GPU acceleration
- :doc:`about` — 10-minute introduction to GFQL

.. _pagerank-provenance:

Benchmark environment and provenance
------------------------------------

Every figure is printed from ``docs/source/_data/gfql_benchmarks.json`` (pyg-bench).

.. bench-provenance:: filter-pagerank-20260728

.. bench-provenance:: filter-pagerank-gplus-locked-20260830

.. bench-disclosures::
