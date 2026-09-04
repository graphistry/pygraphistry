Speedup Case Study: Cypher + PageRank, GFQL vs Neo4j + GDS
===========================================================

.. image:: _static/gfql-mascot.png
   :alt: GFQL mascot
   :width: 160px
   :align: right

This case study runs one three-stage graph pipeline, filter, PageRank, filter,
on two systems. GFQL is Graphistry's open-source graph query language: Cypher
that executes in-process on Python dataframes with no database. Neo4j + Graph
Data Science (GDS) is the graph database and its analytics library. On both
graphs, Twitter (2.4M edges) and GPlus (30M edges), GFQL finished the pipeline
faster than Neo4j + GDS, on CPU and on GPU. On GPlus the GFQL GPU path takes
:bench:`pagerank.gplus.gfql_gpu`, the GFQL CPU path
:bench:`pagerank.gplus.gfql_cpu`, and Neo4j + GDS
:bench:`pagerank.gplus.neo4j_gds`. Use the GPU engine when one is available:
it is :bench:`pagerank.gplus.gfql_gpu_vs_gfql_cpu` faster than the CPU engine
on GPlus and :bench:`pagerank.twitter.gfql_gpu_vs_gfql_cpu` faster on Twitter.

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

.. image:: _static/filter_pagerank/twitter_pipeline.svg
   :alt: Twitter warm pipeline time: Neo4j + GDS 11.72s, GFQL Cypher CPU 1.58s, GFQL Cypher GPU 0.24s

.. image:: _static/filter_pagerank/gplus_pipeline.svg
   :alt: GPlus warm pipeline time: Neo4j + GDS 354.47s, GFQL Cypher CPU 32.10s, GFQL Cypher GPU 2.42s

The pipeline
------------

A three-phase graph pipeline: filter, run PageRank, filter again. The query is
standard Cypher extended with GFQL's graph pipeline syntax. Each ``GRAPH { }``
block takes a graph in and passes a graph on.

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

- ``GRAPH g1``: keep high-degree nodes and their neighbors.
- ``GRAPH g2``: add PageRank scores to ``g1`` (igraph on CPU, cugraph on GPU).
- Final ``GRAPH``: keep high-PageRank nodes and their neighbors.

The query does not change between engines:

- **CPU**: ``engine="pandas"``, ``backend="igraph"``
- **GPU**: ``engine="cudf"``, ``backend="cugraph"``

Intermediate graphs stay in Arrow, pandas, or cuDF memory in the same Python
process. GFQL
returns the same result on every engine or rejects the query before execution;
see :doc:`engines` for the parity rules.

.. _neo4j-analog:

Neo4j + GDS analog
------------------

The Neo4j version writes marker properties at each stage and projects a
separate in-memory graph for GDS:

.. code-block:: cypher

   // 1. Mark seed nodes by degree
   MATCH (n:Node)
   SET n.seed = n.degree >= $cutoff;

   // 2. Expand one hop from seeds
   UNWIND $seed_ids AS sid
   MATCH (s:Node) WHERE id(s) = sid
   MATCH (s)-[r:LINK]-(target:Node)
   SET target.in_subgraph = true, r.in_subgraph = true;

   // 3. Project subgraph and run PageRank
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

   // 4. Keep high-PageRank core + one hop
   MATCH (n:Node) WHERE n.pagerank >= $cutoff
   SET n.core = true;
   UNWIND $core_ids AS cid
   MATCH (c:Node) WHERE id(c) = cid
   MATCH (c)-[r:LINK]-(target:Node)
   SET target.final = true, r.final = true;

.. _pagerank-method:

Method and limits
-----------------

- **Workload**: one pipeline (filter, PageRank, filter) on two SNAP graphs.
  In the GPlus locked run, the Neo4j and GFQL arms selected exactly the same
  node set.
- **Timing**: warm runs after warm-up. The GPlus Neo4j time comes from a
  later locked run of twelve position-balanced slots on one machine; the
  Measurement block below records both runs.
- **Profiles differ**: GFQL reuses frames already resident in Python. Neo4j
  includes server round trips, writes marker properties in both filter stages,
  and rebuilds the GDS in-memory projection on every timed iteration. The
  Neo4j column is therefore a direct pipeline time, not an engine-primitive
  time. The page states which system finished first but publishes no
  GFQL-vs-Neo4j ratio.
- **Comparable ratio**: the GPU-vs-CPU column compares the same GFQL query and
  the same profile, so that ratio is published.
- **Scope**: for the four-engine CPU/GPU comparison and engine choice, see
  :doc:`engines`. For seeded lookups, see :doc:`index_adjacency`. For the
  Spark GraphFrames comparison, see :doc:`benchmark_graphframes`.

.. _pagerank-provenance:

Provenance
----------

Every figure on this page is printed from ``docs/source/_data/gfql_benchmarks.json``,
which pyg-bench publishes.

.. bench-provenance:: filter-pagerank-20260728 filter-pagerank-gplus-locked-20260830
   :disclosures:
