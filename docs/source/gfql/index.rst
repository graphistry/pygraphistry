GFQL: The Dataframe-Native Graph Query Language
===============================================

Welcome to **GFQL**, the first dataframe-native graph query language with GPU support. GFQL is part of the **PyGraphistry** ecosystem and is designed to make graph analytics easier and faster without the need for complex external infrastructure such as databases. Whether you're working with **CPUs** or leveraging **GPU acceleration** for massive datasets, GFQL integrates seamlessly with your data science workflows through a simple `pip install graphistry`.

**GFQL bridges the gap** between traditional storage-tier graph databases and the modern compute tier, allowing you to perform your favorite high-performance graph queries directly on your dataframes. It's built to be familiar to users of Cypher, other graph query languages, and popular dataframe libraries. By being native to accelerated Python datascience dataframe technologies such as Apache Arrow, Numpy, Nvidia RAPIDS, and Graphistry, it can already do workloads like 100M+ edges in interactive time on a single machine.

For Cypher-style `MATCH ... RETURN` workflows, start with
:doc:`quick` (MATCH/chain), :doc:`where` (same-path MATCH constraints),
:doc:`return` (row pipelines), and :doc:`spec/cypher_mapping`.

Recommended paths:

- New to GFQL: :doc:`overview` -> :doc:`quick` -> :doc:`where` -> :doc:`return`
- Translating Cypher: :doc:`spec/cypher_mapping`
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
