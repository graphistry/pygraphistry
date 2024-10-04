GFQL: GraphFrame Query Language
===============================

Welcome to **GFQL**, the first dataframe-native graph query language with GPU support. GFQL is part of the **PyGraphistry** ecosystem and is designed to make graph analytics easier and faster without the need for external complex infrastructure such as databases. Whether you're working with **CPUs** or leveraging **GPU acceleration** for massive datasets, GFQL integrates seamlessly with your data science workflows through a simple `pip install graphistry`.

**GFQL bridges the gap** between traditional storage-tier graph databases and the modern compute tier, allowing you to perform your favorite high-performance graph queries directly on your dataframes. It's built to be familiar to users of Cypher, other graph query languages, and popular dataframe libraries. By being native to accelerated Python datascience dataframe technologies such as Apache Arrow, Numpy, Nvidia RAPIDS, and Graphistry, it can already do workloads like 100M+ edges in interactive time on a single machine.

Key Features
------------

- **Dataframe-Native Integration**: Works directly with Pandas, cuDF, and Apache Arrow dataframes.
- **High Performance**: Optimized for both CPU and GPU execution, capable of processing billions of edges.
- **Ease of Use**: Install via `pip` and start querying without the need for external databases.
- **Seamless Visualization**: Integrated with PyGraphistry for GPU-accelerated graph visualization.
- **Flexibility**: Suitable for a wide range of applications, including cybersecurity, fraud detection, financial analysis, and more.

Why GFQL?
---------

GFQL addresses a critical gap in the data community by providing an in-process graph query language that operates at the compute tier. This means you can:

- **Avoid External Infrastructure**: No need to manage external graph databases or additional servers.
- **Leverage Existing Workflows**: Integrate with your current Python data science tools and libraries.
- **Achieve High Performance**: Utilize GPU acceleration for massive speedups in graph processing.
- **Simplify Graph Analytics**: Write expressive and concise graph queries in Python.

Getting Started
---------------

To get started with GFQL, check out our **10 Minutes to GFQL** guide, which provides a quick introduction to the core concepts and capabilities:

- :doc:`about`


Installation
------------

GFQL is built into pygraphistry:

.. code-block:: bash

    pip install graphistry

Ensure you have `pandas` or `cudf` installed, depending on whether you want to run on CPU or GPU.


Understanding GFQL
------------------

GFQL works on the same graphs as the rest of the PyGraphistry library. The operations run on top of the dataframe engine of your choice, with initial support for Pandas dataframes (CPU) and cuDF dataframes (GPU). 

**Basic Concepts**

- **Nodes and Edges**: Represented using dataframes, making integration with Pandas and cuDF seamless
- **Functional**: Build queries by layering operations, similar to functional method chaining in Pandas
- **Query**: Run graph pattern matching using method `chain()` in a style similar to the popoular OpenCypher graph query language
- **Predicates**: Apply conditions to filter nodes and edges based on their properties, reusing the optimized native operations of the underlying dataframe engine
- **GPU & CPU vectorization**: GFQL automatically leverages GPU acceleration and in-memory columnar processing for massive speedups on your queries

Examples
--------

**Find Nodes of a Certain Type**

Example: Find all nodes where the `type` is `"person"`.

.. code-block:: python

    from graphistry import n

    people_nodes_df = g.chain([ n({"type": "person"}) ])._nodes
    print('Number of person nodes:', len(people_nodes_df))

**Visualize 2-Hop Edge Sequences with an Attribute**

Example: Find 2-hop paths where edges have `"interesting": True`.

.. code-block:: python

    from graphistry import e_forward

    g_2_hops = g.chain([n(), e_forward({"interesting": True}, hops=2) ])
    g_2_hops.plot()

**Find Nodes 1-2 Hops Away and Label Each Hop**

Example: Find nodes up to 2 hops away from node `"a"` and label each hop.

.. code-block:: python

    from graphistry import n, e_undirected

    g_2_hops = g.chain([
        n({g._node: "a"}),
        e_undirected(name="hop1"),
        e_undirected(name="hop2")
    ])
    first_hop_edges = g_2_hops._edges[ g_2_hops._edges.hop1 == True ]
    print('Number of first-hop edges:', len(first_hop_edges))

**Query for Transaction Nodes Between Risky Nodes**

Example: Find transaction nodes between two kinds of risky nodes.

.. code-block:: python

    from graphistry import n, e_forward, e_reverse

    g_risky = g.chain([
        n({"risk1": True}),
        e_forward(to_fixed=True),
        n({"type": "transaction"}, name="hit"),
        e_reverse(to_fixed=True),
        n({"risk2": True})
    ])
    hits = g_risky._nodes[ g_risky._nodes.hit == True ]
    print('Number of transaction hits:', len(hits))

**Filter by Multiple Node Types Using `is_in`**

Example: Filter nodes and edges by multiple types.

.. code-block:: python

    from graphistry import n, e_forward, e_reverse, is_in

    g_filtered = g.chain([
        n({"type": is_in(["person", "company"])}),
        e_forward({"e_type": is_in(["owns", "reviews"])}, to_fixed=True),
        n({"type": is_in(["transaction", "account"])}, name="hit"),
        e_reverse(to_fixed=True),
        n({"risk2": True})
    ])
    hits = g_filtered._nodes[ g_filtered._nodes.hit == True ]
    print('Number of filtered hits:', len(hits))

Leveraging GPU Acceleration
---------------------------

GFQL is optimized to take advantage of GPU acceleration using `cudf` and RAPIDS. When you use GPU dataframes, GFQL automatically executes queries on the GPU for massive speedups.

**Automatic GPU Acceleration**

Example: Run GFQL queries with GPU dataframes.

.. code-block:: python

    import cudf
    import graphistry

    # Load data into GPU dataframes
    e_gdf = cudf.read_parquet('edges.parquet')
    n_gdf = cudf.read_parquet('nodes.parquet')

    # Create a graph with GPU dataframes
    g_gpu = graphistry.edges(e_gdf, 'src', 'dst').nodes(n_gdf, 'id')

    # Run GFQL query (executes on GPU)
    g_result = g_gpu.chain([ ... ])  # Your GFQL query here
    print('Number of resulting edges:', len(g_result._edges))

**Forcing GPU Mode**

Example: Explicitly set the engine to ensure GPU execution.

.. code-block:: python

    g_result = g_gpu.chain([ ... ], engine='cudf')

Visualizing Your Graphs
-----------------------

GFQL integrates with PyGraphistry, allowing you to visualize your graphs with GPU-accelerated rendering.

Example: Visualize high PageRank nodes.

.. code-block:: python

    # Compute PageRank using cuGraph (GPU)
    g_enriched = g_result.compute_cugraph('pagerank')

    # Filter nodes with high PageRank
    g_high_pagerank = g_enriched.chain([
        n(query='pagerank > 0.1'), e(), n(query='pagerank > 0.1')
    ])

    # Plot the subgraph
    g_high_pagerank.plot()

Learn More
----------

Explore the following sections to dive deeper into GFQL's capabilities:

- **10 Minutes to GFQL**: A quickstart guide to get you up and running.

  - :doc:`about`

- **Chain Operations**: Learn how to chain multiple operations to build complex queries.

  - :doc:`chain`

- **Hop Functions**: Understand how to traverse the graph using hop functions.

  - :doc:`hop`

- **Predicates**: Apply advanced filtering using predicates.

  - :doc:`predicates/index`

Additional Resources
--------------------

- **API Reference**: Access detailed documentation of GFQL's API to leverage its full potential.

Community and Support
---------------------

- **Join the Community**: Connect with other GFQL users and developers in our Slack channel.
- **Get Support**: For enterprise support and consultation, feel free to reach out to our team.

---

By integrating directly with your existing data science workflows, GFQL empowers you to perform high-performance graph analytics and visualization without the overhead of managing external databases or infrastructure.

Start exploring GFQL today and unlock new possibilities in graph analytics!


.. toctree::
   :maxdepth: 2

   about
   chain
   hop
   predicates/index
