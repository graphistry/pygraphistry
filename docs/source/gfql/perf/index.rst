Performance and Benchmarks
==========================

How to make GFQL fast, and what it measures against other systems.

Start with :doc:`../engines` to pick pandas, Polars, cuDF, or Polars-GPU.
:doc:`../performance` covers vectorization, GPU acceleration, and measured
engine comparisons. :doc:`../indexing` and :doc:`../index_adjacency` speed up
queries that start from known nodes (a watchlist, a session, a seed set). :doc:`../remote` runs the same queries on
a Graphistry server GPU.

The :doc:`../benchmark_graphframes` page compares GFQL with Apache Spark
GraphFrames on one machine. The Start Here
:doc:`case study <../benchmark_filter_pagerank>` compares one Cypher + PageRank
pipeline with Neo4j + GDS.

.. toctree::
   :maxdepth: 1

   Choosing an Engine <../engines>
   CPU and GPU Acceleration <../performance>
   Indexing Guide <../indexing>
   Adjacency Index for Lookups from Known Nodes <../index_adjacency>
   ../remote
   Benchmark: vs Spark GraphFrames <../benchmark_graphframes>
