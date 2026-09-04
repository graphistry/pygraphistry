GFQL: The Dataframe-Native Graph Query Language
===============================================

GFQL is a graph query language that runs directly on Python dataframes. It
needs no graph database. The same query runs on pandas, Polars, cuDF (GPU), or
Polars-GPU, and on a remote Graphistry server. Install it with
``pip install graphistry``.

GFQL accepts two syntaxes. The Python chain syntax composes ``n()`` and ``e()``
steps. The Cypher syntax, ``g.gfql("MATCH (a)-[e]->(b) ...")``, covers a
bounded subset of Cypher, the graph query language popularized by Neo4j. Both
compile to vectorized dataframe operations, so one machine handles graphs of
100M+ edges in interactive time (see :doc:`benchmark_graphframes`).

Where to start
--------------

- **New to GFQL**: :doc:`about` (10 minutes), then :doc:`quick`, then the
  :doc:`case study <benchmark_filter_pagerank>`.
- **Coming from Cypher**: :doc:`cypher`, then :doc:`spec/cypher_mapping`.
- **Need speed**: :doc:`engines` picks the engine. ``engine='polars'`` is the
  one-keyword CPU speedup; :doc:`performance` covers GPU and remote GPU.
- **Start from known nodes**: :doc:`indexing` and :doc:`index_adjacency` make
  seeded lookups cost O(degree) instead of O(E).

.. toctree::
   :maxdepth: 1
   :caption: Start Here

   about
   overview
   quick
   cypher
   Case Study: Cypher + PageRank vs Neo4j <benchmark_filter_pagerank>

.. toctree::
   :maxdepth: 1
   :caption: Guides

   Performance and Benchmarks <perf/index>
   Language Reference <reference/index>

.. toctree::
   :maxdepth: 2
   :caption: Developer Resources

   spec/index
   validation/index
