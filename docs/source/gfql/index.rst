GFQL: The Dataframe-Native Graph Query Language
===============================================

Welcome to **GFQL**, the first fully vectorized dataframe-native graph query
language with an open-source GPU runtime. GFQL is part of the
**PyGraphistry** ecosystem and is designed to make graph analytics easier and
faster without requiring a graph database as the execution layer. Whether
you're working with **CPUs** or leveraging **GPU acceleration** for massive
datasets, GFQL integrates directly into Python dataframe workflows through a
simple `pip install graphistry`.

**GFQL bridges the gap** between traditional storage-tier graph databases and
the modern compute tier, allowing you to perform high-performance graph queries
directly on your dataframes. It is built to feel familiar to users of Cypher,
other graph query languages, and popular dataframe libraries. By being native
to accelerated Python data-science technologies such as Apache Arrow, NumPy,
NVIDIA RAPIDS, and Graphistry, it can already handle workloads like 100M+ edges
in interactive time on a single machine.

If you are new to Cypher: Cypher is a graph query language popularized by
Neo4j and related tools. It uses ASCII-art graph patterns such as
``(n1)-[e1]->(n2)`` to describe traversals from one node to another across an
edge. GFQL supports a bounded Cypher surface directly through
``g.gfql("MATCH ...")``, so Cypher users can keep familiar ``MATCH`` /
``WHERE`` / ``RETURN`` patterns while moving execution onto GFQL's vectorized
columnar engine and open-source GPU runtime. Use ``g.gfql_remote([...])`` when
you want the same GFQL model executed remotely.

For Cypher syntax through ``g.gfql("MATCH ...")``, start with
:doc:`Cypher Syntax In GFQL <cypher>`,
:doc:`GFQL Quick Reference <quick>`,
:doc:`GFQL RETURN <return>`,
and :doc:`Cypher to GFQL Mapping <spec/cypher_mapping>`.

Recommended paths:

- New to GFQL: :doc:`overview` -> :doc:`quick` -> :doc:`where` -> :doc:`return`
- Running Cypher syntax in GFQL: :doc:`cypher` -> :doc:`quick` -> :doc:`return` -> :doc:`spec/cypher_mapping`
- Faster on CPU (no GPU): :doc:`engines` -> :doc:`performance` (one keyword, ``engine='polars'``, up to ~38x over pandas)
- Performance path (intro -> engine choice -> GPU -> remote GPU): :doc:`about` -> :doc:`engines` -> :doc:`performance` -> :doc:`remote`
- Fast seeded lookups (start from known nodes, like a DB index): :doc:`index_adjacency` (O(degree), flat in graph size, 9-28x vs Kuzu/Neo4j)
- Translating existing Cypher to native GFQL: :doc:`spec/cypher_mapping`
- Building agents/integrations: :doc:`spec/language` + :doc:`spec/python_embedding` + :doc:`spec/wire_protocol`


See also:

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   about
   overview
   remote
   Choosing an Engine <engines>
   Seeded Traversal Indexes <index_adjacency>
   GFQL CPU & GPU Acceleration <performance>
   End-to-End Benchmark <benchmark_filter_pagerank>
   translate
   combo
   quick
   cypher
   where
   return
   predicates/quick
   datetime_filtering
   builtin_calls
   policy
   strict_mode
   schema
   wire_protocol_examples

.. toctree::
   :maxdepth: 2
   :caption: Developer Resources

   spec/index
   validation/index
