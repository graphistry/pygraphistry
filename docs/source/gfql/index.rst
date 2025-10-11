GFQL: The Dataframe-Native Graph Query Language
===============================================

Welcome to **GFQL**, the first dataframe-native graph query language with GPU support. GFQL is part of the **PyGraphistry** ecosystem and is designed to make graph analytics easier and faster without the need for complex external infrastructure such as databases. Whether you're working with **CPUs** or leveraging **GPU acceleration** for massive datasets, GFQL integrates seamlessly with your data science workflows through a simple `pip install graphistry`.

**GFQL bridges the gap** between traditional storage-tier graph databases and the modern compute tier, allowing you to perform your favorite high-performance graph queries directly on your dataframes. It's built to be familiar to users of Cypher, other graph query languages, and popular dataframe libraries. By being native to accelerated Python datascience dataframe technologies such as Apache Arrow, Numpy, Nvidia RAPIDS, and Graphistry, it can already do workloads like 100M+ edges in interactive time on a single machine.


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
