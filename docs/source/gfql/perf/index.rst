Performance and Benchmarks
==========================

How to make GFQL fast, and what it measures against other systems.

Start with :doc:`../engines` to pick pandas, Polars, cuDF, or Polars-GPU.
:doc:`../performance` covers vectorization, GPU acceleration, and measured
engine comparisons. :doc:`../indexing` and :doc:`../index_adjacency` speed up
queries that start from known nodes. :doc:`../remote` runs the same queries on
a Graphistry server GPU.

The two benchmark pages compare GFQL with Neo4j + GDS and with Apache Spark
GraphFrames on one machine.

.. toctree::
   :maxdepth: 1

   Choosing an Engine <../engines>
   CPU and GPU Acceleration <../performance>
   Pay-As-You-Go Resident Indexing <../indexing>
   Seeded Traversal Indexes <../index_adjacency>
   ../remote
   Benchmark: Filter + PageRank vs Neo4j + GDS <../benchmark_filter_pagerank>
   Benchmark: vs Spark GraphFrames <../benchmark_graphframes>
