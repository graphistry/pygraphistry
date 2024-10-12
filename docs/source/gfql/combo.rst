.. _gfql-combo:

Combine GFQL with PyGraphistry Loaders, ML, & AI
================================================

.. contents::
   :depth: 2
   :local:


Common Graph Visualization Tasks
------------------------------------


Ex: Simple Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Quickly visualize a graph using the `.plot()` method.

**SQL**

- **Not suitable**: SQL does not provide direct graph visualization capabilities.

**Pandas**

- **Not suitable**: Pandas lacks built-in methods for graph visualization.

**Cypher**

- **Not suitable**: Cypher does not include visualization functions natively.

**GFQL**

.. code-block:: python

    g.plot()

**Explanation**:

This example demonstrates how to create a quick visual representation of GFQL results using the :meth:`graphistry.PlotterBase.plot` method. For more visualization techniques, see :ref:`10min-viz`.

---







Common Data Loading and Shaping Tasks
--------------------------------------

We'll cover a range of common tasks related to data loading and shaping for graph structures:

.. contents::
   :depth: 2
   :local:

Ex: Data Loading with Pandas from CSV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Demonstrate loading data from CSV files using `pandas` and converting to graph structures.

**GFQL**

.. code-block:: python

    import pandas as pd
    import graphistry

    # pd.DataFrame[['src', 'dst', ...]]
    df = pd.read_csv('data.csv')
    
    # g._edges: df[['src', 'dst', ...]]
    g = graphistry.edges(df, 'src', 'dst')

**Explanation**:

This example illustrates how to load data from a CSV file using `pandas` and bind it into a graph structure via :meth:`graphistry.PlotterBase.PlotterBase.edges` for using **GFQL**. For more information on using `pandas`, refer to the `official pandas documentation <https://pandas.pydata.org/docs/>`__.

---

Ex: GPU Data Loading with cuDF from Parquet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Demonstrate loading data from Parquet files using GPU-accelerated `cuDF` and converting to graph structures.

**GFQL**

.. code-block:: python

    import cudf
    import graphistry

    # cudf.DataFrame[['src', 'dst', ...]]
    df = cudf.read_parquet('data.parquet')

    # g._edges: df[['src', 'dst', ...]]
    g = graphistry.edges(df, 'src', 'dst')

**Explanation**:

This example showcases how to load data from a Parquet file using `cuDF` and convert it into a graph structure with **GFQL**. For further details on using `cuDF`, refer to the official `cuDF <https://docs.rapids.ai/api/cudf/stable/>`__ documentation.

---

Ex: Nodes and Edges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Show how to convert loaded data into graph structures using `.edges()` and `.nodes()` when both are available.

**GFQL**

.. code-block:: python

    # pd.DataFrame[['n_id', ...]]
    df1 = pd.read_csv('nodes.csv')

    # pd.DataFrame[['src', 'dst', ...]]
    df2 = pd.read_csv('edges.csv')

    # g._edges: df2[['src', 'dst', ...]]
    # g._nodes: df1[['n_id', ...]] <-- optional
    g = graphistry.edges(df2, 'src', 'dst').nodes(df1, 'n_id')


**Explanation**:

This example demonstrates how to bind graph data for nodes and edges using **GFQL**. The :meth:`graphistry.PlotterBase.PlotterBase.edges` method is used to load edge data. Binding nodes data is optional, and via method  `graphistry.PlotterBase.PlotterBase.nodes`.

---

Ex: Shaping Graph Data - Multiple Node Columns with Hypergraphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Discuss how to create creates from rows with multiple columns representing nodes via hypergraphs with the default `direct=False` parameter.

**GFQL**

.. code-block:: python
    
    g = graphistry.hypergraph(df, entity_cols=['a', 'b', 'c'])['graph']
    # g._node == 'nodeID'
    # g._nodes: df[['nodeTitle', 'type', 'category', 'nodeID', 'a', 'b', 'c', 'd', 'e', 'EventID']]
    # g._source == 'attribID'
    # g._destination == 'EventID'
    # g._nodes.type.unique() == ['a', 'b', 'c', 'EventID']
    # g._edges: df[['EventID', 'attribID', 'a', 'd', 'e', 'c', 'edgeType']]

**Explanation**:

This example explains how to shape graph data into a hypergraph format using the default `direct=False` parameter. In this case, all values in columns `a`, `b`, and `c` become nodes. Additionally, as `direct=False`, each row also becomes a node, with edges to its corresponding values in columns `a`, `b`, and `c`. When `direct=True`, the nodes for columns `a`, `b`, and `c` would be directly connected. Refer to :meth:`graphistry.PlotterBase.PlotterBase.hypergraph` for more variants and advanced usage.

---









Common Graph Machine Learning and Graph AI Tasks
---------------------------------------------------

We'll cover a range of common tasks related to graph machine learning and AI you can do on GFQL results:

.. contents::
   :depth: 2
   :local:

Ex: UMAP Cluster & Dimensionality Reduction for Embeddings & Similarity Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Show how to apply UMAP for dimensionality reduction, turning wide data into for clustering, embeddings, and similarity graphs.

**GFQL**

.. code-block:: python

    g = graphistry.nodes(df).umap()

**Explanation**:

This example demonstrates how to utilize :meth:`graphistry.umap_utils.UMAPMixin.umap`. See its reference docs for many optional overrides and usage modes, such as defining `X=['col1', 'col2', ...]` to specify which columns to cluster.

---

Ex: UMAP Fit/Transform for Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Explain how to use UMAP's fit/transform capabilities for scaling features across datasets.

**GFQL**

.. code-block:: python

    # Train; feature columns X and label column y are optional
    g1 = graphistry.nodes(df_sample).umap(X=['col_1', ..., 'col_n'], y='col_m')

    # Transform new data
    g2 = g1.transform_umap(new_df, return_graph=True)

    # Visualize new data under initial UMAP embedding
    g2.plot()

**Explanation**:

This example illustrates how to fit a UMAP model on one dataset and then use that model to transform another dataset, enabling consistent scaling of features. For more details on using fit/transform with UMAP, consult the :meth:`graphistry.umap_utils.UMAPMixin.umap` documentation.

---

Ex: Basic RGCN for Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Introduce the basic concepts of RGCNs and how to build and train a simple model.

**GFQL**

.. code-block:: python

    g = graphistry.nodes(ndf).edges(edf, src, dst)
    g.build_gnn(X_nodes=['feature_1', 'feature_2'], y_nodes='label')
    g.train()  # Train the model

**Explanation**:

This example provides an introduction to building and training a basic Relational Graph Convolutional Network (RGCN) using **GFQL**. For further information on GNNs, see the **GNN Models** section of our documentation.

---

Ex: Anomaly Detection Using RGCN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Utilize the trained RGCN model to detect anomalies in graph data based on learned representations.

**GFQL**

.. code-block:: python

    anomalies = g.detect_anomalies()  # Detect anomalies in the graph using the trained model

**Explanation**:

This example demonstrates how to leverage a trained RGCN model to identify anomalies in graph data. For additional details on anomaly detection techniques, refer to the relevant sections in our documentation.

---

Ex: Clustering with DBSCAN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Discuss using DBSCAN for clustering nodes or edges based on features.

**GFQL**

.. code-block:: python

    g = graphistry.nodes(ndf).edges(edf, src, dst).umap()
    g.dbscan(eps=0.5, min_samples=5)  # Apply DBSCAN clustering

**Explanation**:

This example illustrates how to apply DBSCAN clustering to graph data after reducing dimensionality with UMAP. For more information on clustering techniques, consult the **Clustering** section in our documentation.

---

Ex: Automated Feature Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Illustrate generating features from raw data for AI applications.

**GFQL**

.. code-block:: python

    g = graphistry.nodes(df).featurize(kind='nodes', X=['raw_feature_1', 'raw_feature_2'])

**Explanation**:

This example demonstrates how to automatically generate features from raw data using **GFQL**. For additional insights into feature generation techniques, refer to the **Feature Generation** section in our documentation.

---

Ex: Semantic Search in Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Implement semantic search using graph embeddings and natural language queries.

**GFQL**

.. code-block:: python

    results_df, query_vector = g.search('natural language query')

**Explanation**:

This example showcases how to perform semantic searches within graph data using embeddings. For further details on implementing semantic search, see the **Semantic Search** section in our documentation.

---

Ex: Knowledge Graph Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Explain training models for knowledge graph embeddings and predicting relationships.

**GFQL**

.. code-block:: python

    g = graphistry.edges(edf, src, dst)
    g2 = g.embed(relation='relationship_column')

**Explanation**:

This example describes how to train models for knowledge graph embeddings with **GFQL** and how to predict relationships between entities. For more information, refer to the **Knowledge Graph Embeddings** section in our documentation.












