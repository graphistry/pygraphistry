
Overview of GFQL
=================

New to GFQL, the open source dataframe-native graph query language? This article overviews the gaps it fills, special features like GPU accelerations, and where to go next.


Why GFQL?
~~~~~~~~~~~

GFQL addresses a critical gap in the data community by providing an in-process graph query language that operates at the compute tier. This means you can:

- **Graph search**: Easily and efficiently query and filter nodes and edges using a familiar syntax.
- **Avoid External Infrastructure**: Avoid calls to external infrastructures and eliminate the need for extra databases.
- **Leverage Existing Workflows**: Integrate with your current Python data science tools and libraries.
- **Achieve High Performance**: Utilize GPU acceleration for massive speedups in graph processing.
- **Simplify Graph Analytics**: Write expressive and concise graph queries in Python.

Key Features
~~~~~~~~~~~~~

- **Dataframe-Native Integration**: Works directly with Pandas, cuDF, and Apache Arrow dataframes.
- **High Performance**: Optimized for both CPU and GPU execution, capable of processing billions of edges.
- **Ease of Use**: Install via `pip` and start querying without the need for external databases.
- **Seamless Visualization**: Integrated with PyGraphistry for GPU-accelerated graph visualization.
- **Flexibility**: Suitable for a wide range of applications, including cybersecurity, fraud detection, financial analysis, and more.
- **Architectural Freedom**: Use GFQL with your dataframes on your local CPU/GPU, or offload to a remote GPU cluster.

Installation Guide
~~~~~~~~~~~~~~~~~~~
.. toctree::
   :hidden:

GFQL is built into pygraphistry:

.. code-block:: bash

    pip install graphistry

Ensure you have `pandas` or `cudf` installed, depending on whether you want to run on CPU or GPU.

For more information, see :doc:`../install/index` .

Key GFQL Concepts
~~~~~~~~~~~~~~~~~~~~~

GFQL works on the same graphs as the rest of the PyGraphistry library. The operations run on top of the dataframe engine of your choice, with initial support for Pandas dataframes (CPU) and cuDF dataframes (GPU). 

- **Nodes and Edges**: Represented using dataframes, making integration with Pandas and cuDF seamless
- **Functional**: Build queries by layering operations, similar to functional method chaining in Pandas
- **Query**: Run graph pattern matching using method `chain()` in a style similar to the popoular OpenCypher graph query language
- **Predicates**: Apply conditions to filter nodes and edges based on their properties, reusing the optimized native operations of the underlying dataframe engine
- **GPU & CPU vectorization**: GFQL automatically leverages GPU acceleration and in-memory columnar processing for massive speedups on your queries
- **Optional remote mode**: Bind to remote data or upload it quickly as Arrow, and run your same Python and GFQL queries on remote GPU resources when available

Quick Examples
~~~~~~~~~~~~~~~

**Find Nodes of a Certain Type**

Example: Find all nodes where the `type` is `"person"`.

.. code-block:: python

    from graphistry import n

    people_nodes_df = g.gfql([ n({"type": "person"}) ])._nodes
    print('Number of person nodes:', len(people_nodes_df))

**Visualize 2-Hop Edge Sequences with an Attribute**

Example: Find 2-hop paths where edges have `"interesting": True`.

.. code-block:: python

    from graphistry import n, e_forward

    g_2_hops = g.gfql([n(), e_forward({"interesting": True}, hops=2) ])
    g_2_hops.plot()

Example visualization (static):

.. raw:: html

   <figure class="align-center">
     <img src="../_static/gfql/gfql_overview_2_hops.png" alt="GFQL 2-hop example rendered with plot_static" style="width: 90%;" />
     <figcaption>2-hop "interesting" edges rendered with <code>plot_static()</code>.</figcaption>
   </figure>

**Find Nodes 1-2 Hops Away and Label Each Hop**

Example: Find nodes up to 2 hops away from node `"a"` and label each hop.

.. code-block:: python

    from graphistry import n, e_undirected

    g_2_hops = g.gfql([
        n({g._node: "a"}),
        e_undirected(name="hop1"),
        e_undirected(name="hop2")
    ])
    first_hop_edges = g_2_hops._edges[ g_2_hops._edges["hop1"] == True ]
    print('Number of first-hop edges:', len(first_hop_edges))

**Filter by Date/Time**

Example: Find recent transactions using temporal predicates.

.. code-block:: python

    from graphistry import n, e_forward
    from graphistry.compute import gt, between
    from datetime import datetime, date, time
    import pandas as pd

    # Find transactions after a specific date
    recent = g.gfql([
        n(edge_match={"timestamp": gt(pd.Timestamp("2023-01-01"))})
    ])
    
    # Find transactions in a date range during business hours
    business_hours_txns = g.gfql([
        n(edge_match={
            "date": between(date(2023, 6, 1), date(2023, 6, 30)),
            "time": between(time(9, 0), time(17, 0))
        })
    ])

**Query for Transaction Nodes Between Risky Nodes**

Example: Find transaction nodes between two kinds of risky nodes.

.. code-block:: python

    from graphistry import n, e_forward, e_reverse

    g_risky = g.gfql([
        n({"risk1": True}),
        e_forward(to_fixed_point=True),
        n({"type": "transaction"}, name="hit"),
        e_reverse(to_fixed_point=True),
        n({"risk2": True})
    ])
    hits = g_risky._nodes[ g_risky._nodes["hit"] == True ]
    print('Number of transaction hits:', len(hits))

**Filter by Multiple Node Types Using `is_in`**

Example: Filter nodes and edges by multiple types.

.. code-block:: python

    from graphistry import n, e_forward, e_reverse, is_in

    g_filtered = g.gfql([
        n({"type": is_in(["person", "company"])}),
        e_forward({"e_type": is_in(["owns", "reviews"])}, to_fixed_point=True),
        n({"type": is_in(["transaction", "account"])}, name="hit"),
        e_reverse(to_fixed_point=True),
        n({"risk2": True})
    ])
    hits = g_filtered._nodes[ g_filtered._nodes["hit"] == True ]
    print('Number of filtered hits:', len(hits))

**DAG Patterns with Let Bindings**

GFQL's Let bindings enable you to compose complex graph analyses by defining named subgraphs and operations that can reference each other. Like variables in programming, Let bindings make it easy to manipulate multiple graphs and subgraphs within a single query, while maintaining all the benefits of GFQL like GPU acceleration.

Traditional Python approach (manual variable management):

.. code-block:: python

    # Traditional Python: Manually manage intermediate results
    persons = g.gfql([n({'type': 'person'})])
    adults = persons.gfql([n({'age': ge(18)})])
    friends = adults.gfql([e_forward({'type': 'knows'})])
    # Each step requires careful tracking of which graph to operate on

GFQL Let approach (declarative DAG with named bindings):

.. code-block:: python

    from graphistry import let, ref, n, e_forward, ge

    # GFQL Let: Define a DAG of named operations
    result = g.gfql(let({
        'persons': n({'type': 'person'}),
        'adults': ref('persons', [n({'age': ge(18)})]),  # Reference and filter persons
        'connections': [
            n({'type': 'person', 'age': ge(18)}),
            e_forward({'type': 'knows'}),
            n()  # Find connections from adults
        ]
    }))

    # Access any named result from the DAG
    adults = result._nodes[result._nodes['adults']]
    connections = result._edges[result._edges['connections']]

Key advantages of GFQL Let:
- **Named subgraphs**: Create reusable named graph operations like constants in code
- **Dependency management**: Automatically resolves dependencies between operations
- **Composability**: Build complex multi-stage analyses from simpler named operations
- **GPU preservation**: All operations maintain GPU acceleration when available
- **Clean semantics**: Express complex graph analyses as clear, declarative DAGs

Leveraging GPU Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    g_result = g_gpu.gfql([ ... ])  # Your GFQL query here
    print('Number of resulting edges:', len(g_result._edges))

**Forcing GPU Mode**

Example: Explicitly set the engine to ensure GPU execution.

.. code-block:: python

    g_result = g_gpu.gfql([ ... ], engine='cudf')

Run Remotely
~~~~~~~~~~~~~

You may want to run GFQL remotely such as if the data is remote, e.g., in Hub or cloud storage, and you have faster remote GPU servers for acting on it.

**Bind to Remote Data and Query**

Example: Bind to remote data and run queries on remote GPU resources.

.. code-block:: python

    import graphistry
    from graphistry import n, e

    g = graphistry.bind(dataset_id='my-dataset-id')

    nodes_df = g.gfql_remote([ n() ])._nodes

**Upload Data and Run GPU Python Remotely**

Example: Upload local data to a remote GPU server and run full GPU Python tasks on it.

.. code-block:: python

    import graphistry
    from graphistry import n, e

    # Fully self-contained so can be transferred
    def my_remote_trim_graph_task(g):
        # Trick: You can also put database fetch calls here!
        return (g
            .nodes(g._nodes[:10])
            .edges(g._edges[:10])
        )

    # Upload any local graph data to the remote server
    g2 = g1.upload()
    print(g2._dataset_id, g2._nodes_file_id, g2._edges_file_id)

    # Compute on it locally
    g_result = g2.python_remote_g(my_remote_trim_graph_task)
    print('Number of resulting edges:', len(g_result._edges))

See also `python_remote_table()` and `python_remote_json()` for returning other types of data.


Visualizing GFQL Results
~~~~~~~~~~~~~~~~~~~~~~~~~

GFQL integrates with PyGraphistry, allowing you to visualize your graphs with GPU-accelerated rendering.

Example: Visualize high PageRank nodes.

.. code-block:: python

    from graphistry import n, e

    # Compute PageRank using cuGraph (GPU)
    g_enriched = g_result.compute_cugraph('pagerank')

    # Filter nodes with high PageRank
    g_high_pagerank = g_enriched.gfql([
        n(query='pagerank > 0.1'), e(), n(query='pagerank > 0.1')
    ])

    # Plot the subgraph
    g_high_pagerank.plot()

Example visualization (graphviz):

.. graphviz::

   digraph high_pagerank {
       rankdir=LR;
       node [shape=ellipse];

       a [label="a\npagerank=0.18", style="filled,bold", fillcolor="#90EE90", penwidth=3, color="#228B22"];
       b [label="b\npagerank=0.12", style="filled,bold", fillcolor="#90EE90", penwidth=3, color="#228B22"];
       c [label="c\npagerank=0.05", shape=box, style=filled, fillcolor="#D3D3D3", color="#A9A9A9", fontcolor="#696969"];
       tx1 [label="tx1", shape=diamond, style=filled, fillcolor="#D3D3D3", color="#A9A9A9", fontcolor="#696969"];
       tx2 [label="tx2\npagerank=0.16", shape=diamond, style="filled,bold", fillcolor="#90EE90", penwidth=3, color="#228B22"];

       a -> b [color="#228B22", penwidth=2];
       b -> c [color="#A9A9A9"];
       a -> tx1 [color="#A9A9A9"];
       tx1 -> tx2 [color="#228B22", penwidth=2];
       tx2 -> c [color="#A9A9A9"];
   }

Example visualization (interactive):

.. raw:: html

    <iframe src="https://hub.graphistry.com/graph/graph.html?dataset=1d52d9a62e034d9c94e09f5be45e3caa&type=arrow&viztoken=70cfc1a1-9ec0-44af-bbe3-d55d18e5ac4d&usertag=ef9e6f8d-pygraphistry-0.48.0+75.gf422e208&splashAfter=1766373305&info=true" style="width: 100%; height: 500px; border: 0;" loading="lazy"></iframe>

.. rubric:: Learn More

Explore the following sections to dive deeper into GFQL's capabilities:

- **10 Minutes to GFQL**: A quickstart guide to get you up and running.

  - :doc:`about`

- **Hop & Chain Quick Reference**: Learn how to chain multiple operations to build complex queries.

  - :doc:`quick`

- **Predicates Quick Reference**: Apply advanced filtering using predicates.

  - :doc:`predicates/quick`

GFQL APIs
~~~~~~~~~~

Access detailed documentation of GFQL's API:

- **Chain Operations**: Learn how to chain multiple operations to build complex queries.

  - :doc:`../api/gfql/chain`

- **Hop Functions**: Understand how to traverse the graph using hop functions.

  - :doc:`../api/gfql/hop`

- **Predicates**: Apply advanced filtering using predicates.

  - :doc:`../api/gfql/predicates`
