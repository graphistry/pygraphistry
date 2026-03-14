GFQL: The Dataframe-Native Graph Query Language
===============================================

Welcome to **GFQL**, the first dataframe-native graph query language with GPU support. GFQL is part of the **PyGraphistry** ecosystem and is designed to make graph analytics easier and faster without the need for complex external infrastructure such as databases. Whether you're working with **CPUs** or leveraging **GPU acceleration** for massive datasets, GFQL integrates seamlessly with your data science workflows through a simple `pip install graphistry`.

**GFQL bridges the gap** between traditional storage-tier graph databases and the modern compute tier, allowing you to perform your favorite high-performance graph queries directly on your dataframes. It's built to be familiar to users of Cypher, other graph query languages, and popular dataframe libraries. By being native to accelerated Python datascience dataframe technologies such as Apache Arrow, Numpy, Nvidia RAPIDS, and Graphistry, it can already do workloads like 100M+ edges in interactive time on a single machine.

If you are new to Cypher: Cypher is a graph query language popularized by
Neo4j and related tools. It uses ASCII-art graph patterns such as
``(n1)-[e1]->(n2)`` to describe traversals from one node to another across an
edge. GFQL supports a bounded Cypher surface directly through
``g.gfql("MATCH ...")`` on bound graphs, while keeping GFQL's columnar
execution model and optional GPU acceleration. Use ``g.gfql_remote([...])``
when you want remote GFQL execution.

For Cypher syntax through ``g.gfql("MATCH ...")``, start with
:doc:`Cypher Syntax In GFQL <cypher>`,
:doc:`GFQL Quick Reference <quick>`,
:doc:`GFQL RETURN <return>`,
and :doc:`Cypher to GFQL Mapping <spec/cypher_mapping>`.

Recommended paths:

- New to GFQL: :doc:`overview` -> :doc:`quick` -> :doc:`where` -> :doc:`return`
- Running Cypher syntax in GFQL: :doc:`cypher` -> :doc:`quick` -> :doc:`return` -> :doc:`spec/cypher_mapping`
- Performance path (intro -> GPU -> remote GPU): :doc:`about` -> :doc:`performance` -> :doc:`remote`
- Translating existing Cypher to native GFQL: :doc:`spec/cypher_mapping`
- Building agents/integrations: :doc:`spec/language` + :doc:`spec/python_embedding` + :doc:`spec/wire_protocol`


See also:

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   about
   overview
   remote
   GFQL CPU & GPU Acceleration <performance>
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
   wire_protocol_examples

.. toctree::
   :maxdepth: 2
   :caption: Developer Resources

   spec/index
   validation/index
