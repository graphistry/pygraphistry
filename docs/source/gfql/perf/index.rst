Performance and Benchmarks
==========================

GFQL runs graph queries as dataframe operations, in your process, on CPU or GPU. These
pages show what that is worth against graph databases and Spark, and how to get the
speed on your own data.

See what it is worth
--------------------

Read in this order if you are deciding whether GFQL fits.

1. :doc:`Speedup case study: Cypher + PageRank vs Neo4j + GDS <../benchmark_filter_pagerank>`
   — one real pipeline, measured end to end on a 30M-edge graph, on CPU and GPU.
2. :doc:`Measured against graph databases <../performance>` — the q1–q9 Cypher board
   against Kuzu, Memgraph and Neo4j, and the SNB point-query comparison, with the losses
   shown next to the wins.
3. :doc:`GFQL vs Spark GraphFrames <../benchmark_graphframes>` — one machine against a
   cluster framework on LiveJournal and Orkut.

Get the speed on your data
--------------------------

Read these when you have a graph and want it to run faster.

1. :doc:`Choose an engine <../engines>` — pandas, Polars, cuDF or Polars-GPU with one
   keyword; which one for which work.
2. :doc:`Index for queries that start from known nodes <../index_adjacency>` — build the
   adjacency index once; a traversal from a watchlist or a seed set then reads only those
   nodes' neighborhoods.
3. :doc:`Indexing guide <../indexing>` — property and adjacency indexes, when they engage,
   and what they cost.
4. :doc:`Run on a remote GPU <../remote>` — the same queries on a Graphistry server GPU.

Reference
---------

- Methodology, hosts, datasets and provenance for every number: the *Provenance* section
  of :doc:`../performance`.
- Streaming for results larger than memory: :ref:`gfql-larger-than-memory`.

.. toctree::
   :maxdepth: 1
   :hidden:

   ../performance
   ../benchmark_graphframes
   ../engines
   ../index_adjacency
   ../indexing
   ../remote
