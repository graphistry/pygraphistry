GFQL Cypher Filter + PageRank Benchmark
========================================

.. image:: _static/gfql-mascot.png
   :alt: GFQL mascot
   :width: 160px
   :align: right

Run Cypher graph queries and analytics directly on Python dataframes —
no database required. This benchmark reports **Graphistry's local Cypher**
(CPU and GPU) and **Neo4j + GDS** timings for the same pipeline shape.

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

Warm pipeline time — search, PageRank, search. The GFQL arms retain resident frames;
the Neo4j arm performs server round trips and rebuilds its GDS projection for every
timed iteration. Those are different measurement profiles, so the direct timings are
reported without a GFQL-vs-Neo4j speedup ratio.

Within the shared resident GFQL profile, moving the same query text to the GPU is
:bench:`pagerank.twitter.gfql_gpu_vs_gfql_cpu` faster on Twitter and
:bench:`pagerank.gplus.gfql_gpu_vs_gfql_cpu` faster on the 30M-edge GPlus graph.

On GPlus, the locked follow-up completed all twelve position-balanced slots and
reports the direct Neo4j + GDS pipeline time.

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

Twitter (2.4M edges): reported pipeline timings
------------------------------------------------

.. image:: _static/filter_pagerank/twitter_pipeline.svg
   :alt: Twitter warm pipeline time: Neo4j + GDS 11.72s, GFQL Cypher CPU 1.58s, GFQL Cypher GPU 0.24s

- **Neo4j + GDS**: :bench:`pagerank.twitter.neo4j_gds`

- **GFQL Cypher on CPU** (pandas + igraph): :bench:`pagerank.twitter.gfql_cpu`

- **GFQL Cypher on GPU** (cuDF + cuGraph): :bench:`pagerank.twitter.gfql_gpu` —
  :bench:`pagerank.twitter.gfql_gpu_vs_gfql_cpu` faster than the GFQL CPU path

The Neo4j timing includes a per-iteration projection rebuild, while both GFQL timings
reuse resident frames. The three direct values describe those pipelines; they do not
support a cross-profile speedup claim.


GPlus (30M edges): larger graph
-------------------------------

.. image:: _static/filter_pagerank/gplus_pipeline.svg
   :alt: GPlus warm pipeline time: Neo4j + GDS 354.47s, GFQL Cypher CPU 32.10s, GFQL Cypher GPU 2.42s

- **Neo4j + GDS**: :bench:`pagerank.gplus.neo4j_gds` — the locked follow-up ran
  six slots after the earlier incomplete attempt
- **GFQL Cypher on CPU** (pandas + igraph): :bench:`pagerank.gplus.gfql_cpu`
- **GFQL Cypher on GPU** (cuDF + cuGraph): :bench:`pagerank.gplus.gfql_gpu` —
  :bench:`pagerank.gplus.gfql_gpu_vs_gfql_cpu` faster than the CPU path

The Neo4j result includes server round trips and a projection rebuild per timed
iteration. The GFQL results retain resident frames, so no cross-profile ratio is valid.

GPlus is 12x the edges of the Twitter graph, and the GPU pipeline still answers in
seconds.

Why this matters
----------------

The benchmark shows one GFQL query surface over pandas + igraph and cuDF + cuGraph.
Within that shared resident in-process profile, the GPU arm is faster on both graphs.
The Neo4j timing remains useful as a disclosed pipeline reference, but its different
projection and round-trip budget prevents an engine-speedup claim.

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

**Same answer on every engine.** The CPU and GPU timings above are comparable
because the query meaning is fixed: GFQL's engine contract is same result or
pre-execution decline. Unsupported engine/query combinations are rejected during
validation, compilation, or planning before query execution, rather than silently
falling back or returning a different answer. See :doc:`engines` for the full
parity and static-safety contract.

This page is one workload (a filter → PageRank → filter pipeline) against one
external baseline (Neo4j + GDS). For the full four-engine picture — when Polars
beats pandas on CPU, when the GPU pulls ahead, and how to choose — see
:doc:`engines`. For seeded lookups, see :doc:`index_adjacency`.

For more on the GFQL design and supported surface:

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

.. bench-disclosures::
