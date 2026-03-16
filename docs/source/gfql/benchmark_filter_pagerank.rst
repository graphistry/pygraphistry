End-to-End GFQL Benchmark: Dataframes, Search, Analytics, No Database
======================================================================

This benchmark is meant to answer a simple question:

**If your graph already lives in Python dataframes, how much of a real graph pipeline can you run there directly, and how fast is it on CPU vs GPU?**

The answer here is unusually strong because the benchmark is not just one graph algorithm in isolation. It is an end-to-end workflow that stays in open-source Python tooling:

- load a large graph from a cached edge list
- shape node metadata in a dataframe
- run graph search / subgraph extraction with GFQL
- run PageRank on the selected graph
- keep the resulting graph for downstream analysis or visualization

Benchmark environment
---------------------

- Host: ``dgx-spark``
- GPU: ``GB10``
- NVIDIA driver: ``580.126.09``
- Container/runtime: ``graphistry/test-gpu:latest``
- Presentation mode: this page renders only saved JSON outputs under ``plans/gfql-gpu-pagerank-benchmark/results/`` and does **not** rerun the benchmarks

Why this matters
----------------

- **GFQL** is Graphistry's dataframe-native graph query language.
- It executes directly on Python dataframes and graph objects instead of requiring an external graph database.
- The same workflow can run on:
  - **CPU** with ``pandas + igraph``
  - **GPU** with ``cudf + cugraph``
- The benchmark also includes a **Neo4j + GDS** comparison where we can make an honest apples-to-apples comparison.

This means the benchmark is testing a practical claim, not just a microbenchmark:

**Can you get database-style graph search + graph analytics directly on your dataframe, and does the GPU path materially change the answer?**

What is being benchmarked
-------------------------

The pipeline is intentionally simple and representative:

1. **Data loading**
   Read a cached SNAP edge list into a dataframe.
2. **Data shaping**
   Compute node degree and materialize the node metadata later queried by GFQL.
3. **Graph search**
   Use GFQL to expand around interesting nodes and extract a subgraph.
4. **Graph analytics**
   Run PageRank on the resulting graph.
5. **Graph search again**
   Keep the high-PageRank core and its local neighborhood.
6. **Downstream use**
   Keep the final graph directly in Python for follow-on analysis or visualization.

The important detail is that these are not separate systems stitched together. The Graphistry CPU and GPU paths both keep the workflow dataframe-native, and the GPU path accelerates both the dataframe work and the graph algorithm work.

What the actual benchmarked pipelines look like
--------------------------------------------------------

The Graphistry side of this benchmark is a single compound local Cypher expression
that chains graph-preserving search, enrichment, and search in one ``g.gfql(...)`` call:

.. code-block:: python

   result = g.gfql(
       "GRAPH g1 = GRAPH { "
       "  MATCH (seed)-[reach]-(nbr) "
       "  WHERE seed.degree >= $degree_cutoff "
       "} "
       "GRAPH g2 = GRAPH { "
       "  USE g1 "
       f"  CALL graphistry.{backend}.pagerank.write() "
       "} "
       "GRAPH { "
       "  USE g2 "
       "  MATCH (core)-[halo]-(nbr) "
       "  WHERE core.pagerank >= $pagerank_cutoff "
       "}",
       params={
           "degree_cutoff": degree_cutoff,
           "pagerank_cutoff": pagerank_cutoff,
       },
       engine=engine,
   )

Where:

- CPU uses ``engine="pandas"`` and ``backend="igraph"``
- GPU uses ``engine="cudf"`` and ``backend="cugraph"``
- ``GRAPH g1 = GRAPH { ... }`` binds a named subgraph from the degree-filtered match
- ``GRAPH g2 = GRAPH { USE g1 CALL ... }`` enriches that subgraph with PageRank
- The terminal ``GRAPH { USE g2 MATCH ... }`` returns the final graph in graph state

Neo4j + GDS analog (Cypher + projection + PageRank write):

.. code-block:: cypher

   MATCH (n:Node)
   SET n.seed_keep = n.degree >= $cutoff,
       n.sub1_keep = false,
       n.core_keep = false,
       n.final_keep = false
   REMOVE n.pagerank;

   MATCH (n:Node)
   WHERE n.seed_keep
   SET n.sub1_keep = true;

   UNWIND $seed_ids AS sid
   MATCH (s:Node) WHERE id(s) = sid
   MATCH (s)-[r:LINK]-(nbr:Node)
   SET nbr.sub1_keep = true, r.sub1_keep = true;

   CALL gds.graph.project.cypher(
     'sub1',
     'MATCH (n:Node) WHERE n.sub1_keep RETURN id(n) AS id',
     'MATCH (a:Node)-[r:LINK]->(b:Node) WHERE r.sub1_keep RETURN id(a) AS source, id(b) AS target
      UNION ALL
      MATCH (a:Node)-[r:LINK]->(b:Node) WHERE r.sub1_keep RETURN id(b) AS source, id(a) AS target'
   );
   CALL gds.pageRank.write('sub1', {writeProperty: 'pagerank'});

   MATCH (n:Node)
   SET n.core_keep = coalesce(n.sub1_keep, false)
                     AND coalesce(n.pagerank, 0.0) >= $cutoff,
       n.final_keep = false;

   UNWIND $core_ids AS cid
   MATCH (c:Node) WHERE id(c) = cid
   MATCH (c)-[r:LINK]-(nbr:Node)
   SET nbr.final_keep = true, r.final_keep = true;

That means the comparison is honest about what each system is actually doing:

- Graphistry CPU/GPU: native Python dataframe + graph runtime
- Neo4j: Cypher + GDS projection/write pipeline inside the database

Exact 3-way comparison on Twitter
---------------------------------

Twitter is the cleanest exact comparison: all three engines finish comfortably on the same workload, and we can measure the full lifecycle including data loading.

.. image:: _static/filter_pagerank/twitter_lifecycle.svg
   :alt: Twitter end-to-end lifecycle: Neo4j vs Graphistry CPU vs Graphistry GPU, stacked by ETL, Search, and Analytics

The stacked bars break the lifecycle into three workload phases:

- **ETL**: data loading and shaping (CSV import + degree computation)
- **Search**: graph-preserving subgraph extraction (GFQL ``GRAPH { MATCH ... }``, or Neo4j seed expansion)
- **Analytics**: PageRank computation (igraph / cugraph / GDS)

Takeaways:

- Neo4j total: ``~21.6s`` (5.99s ETL + 10.2s search + 3.5s analytics + 1.7s prep)
- Graphistry CPU total: ``~2.8s`` (0.28s ETL + 2.55s pipeline) — **~8x faster than Neo4j**
- Graphistry GPU total: ``~0.4s`` (0.10s ETL + 0.30s pipeline) — **~54x faster than Neo4j**
- The GPU advantage compounds across every phase: ETL, search, and analytics are all faster.

Larger-graph story on GPlus
---------------------------

GPlus (30M edges) is where the story becomes especially compelling. Neo4j becomes expensive enough that the honest result is only a lower bound.

.. image:: _static/filter_pagerank/gplus_lifecycle.svg
   :alt: GPlus lifecycle: Neo4j (lower bound) vs Graphistry CPU vs Graphistry GPU

Takeaways:

- Neo4j: **>187s** (lower bound — the transaction did not finish)
- Graphistry CPU: ``~85.5s`` (9.7s ETL + 75.8s pipeline) — still faster than Neo4j's incomplete run
- Graphistry GPU: ``~7.1s`` (3.9s ETL + 3.3s pipeline) — **>26x faster than Neo4j**
- On GPlus, the Graphistry GPU path reduces a minute-scale CPU pipeline to a few seconds.

Why the CPU and GPU versions are both interesting
-------------------------------------------------

This benchmark is interesting even before the GPU enters the picture.
The CPU path already shows that you can run a real graph-search + PageRank workflow directly on your dataframe without standing up a graph database.

The GPU path matters because it keeps the same general workflow while accelerating:

- the dataframe-native parts
- the graph-search parts
- the graph-analytics parts

That is why the story is stronger than "GPU PageRank is faster":

**the whole open-source Python graph pipeline is faster, while staying local to your dataframes.**

Notebook version
----------------

For a notebook-oriented version of this writeup, see:

- ``demos/gfql/benchmark_filter_pagerank_cpu_gpu.ipynb``

That notebook is presentation-first and uses the same saved DGX result files used by this page. It does not rerun the benchmarks.
