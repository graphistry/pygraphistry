.. _gfql-combo:

Combine GFQL with PyGraphistry Loaders, ML, AI, & Visualization
=================================================================

.. contents::
   :depth: 2
   :local:





Common Graph Visualization Tasks
------------------------------------

For an introduction to visualization techniques, see :ref:`10min-viz`.

Simple Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Quickly visualize a graph using the `.plot()` method.

**Code**

.. code-block:: python

    g2 = g1.chain(gfql_query)
    g2.plot()

**Explanation**:

This creates an interactive graph visualization of the GFQL results using the :meth:`graphistry.PlotterBase.PlotterBase.plot` method.


---







Common Data Loading and Shaping Tasks
--------------------------------------

We'll cover a range of common tasks related to data loading and shaping for graph structures:

.. contents::
   :depth: 2
   :local:

Data Loading with Pandas from CSV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Demonstrate loading data from CSV files using `pandas` and converting to graph structures.

**Code**

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

GPU Data Loading with cuDF from Parquet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Demonstrate loading data from Parquet files using GPU-accelerated `cuDF` and converting to graph structures.

**Code**

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

Bind Nodes and Edges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Show how to convert loaded data into graph structures using `.edges()` and `.nodes()` when both are available.

**Code**

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

Handle Multiple Node Columns with Hypergraphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Discuss how to create creates from rows with multiple columns representing nodes via hypergraphs with the default `direct=False` parameter.

**Code**

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

UMAP Cluster & Dimensionality Reduction for Embeddings & Similarity Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Show how to apply UMAP for dimensionality reduction, turning wide data into for clustering, embeddings, and similarity graphs.

**Code**

.. code-block:: python

    df = pd.DataFrame({
        'a': [0, 1, 2, 3, 4],
        'b': [10, 20, 30, 40, 50],
        'c': [100, 200, 300, 400, 500]
    })
    g = graphistry.nodes(df).umap()  # Automatically featurizes & embeds into X, Y space
    g.plot()

    assert set(g._nodes.columns) == {'_n', 'a', 'b', 'c', 'x', 'y'}
    assert set(g._edges.columns) == {'_src_implicit', '_dst_implicit', '_weight'}
    assert isinstance(g._node_features, (pd.DataFrame, cudf.DataFrame))


**Explanation**:

This example demonstrates how to utilize :meth:`graphistry.umap_utils.UMAPMixin.umap`. See its reference docs for many optional overrides and usage modes, such as defining `X=['col1', 'col2', ...]` to specify which columns to cluster.

---

UMAP Fit/Transform for Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Explain how to use UMAP's fit/transform capabilities for scaling features across datasets.

**Code**

.. code-block:: python

    # Train: Feature columns X and label column y are optional
    g1 = graphistry.nodes(df_sample).umap(X=['col_1', ..., 'col_n'], y='col_m')

    # Transform new data under initial UMAP embedding
    g2 = g1.transform_umap(batch_df, return_graph=True)

    # Visualize new data under initial UMAP embedding
    g2.plot()

**Explanation**:

This example illustrates how to fit a UMAP model on one dataset and then use that model to transform another dataset, enabling consistent scaling of features. For more details on using fit/transform with UMAP, consult the :meth:`graphistry.umap_utils.UMAPMixin.umap` documentation.

---

Anomaly Detection using RGCN Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Introduce the basic concepts of RGCNs and how to build and train a simple graph model.

**Code**

.. code-block:: python

    # df: df[['src_ip', 'dst_ip', 'o1', 'dst_port', ...]]

    g = graphistry.edges(df, 'src_ip', 'dst_ip')  # graph
    g2 = g.embed(  # rerun until happy with quality
        device=dev0,

        #relation='dst_port', # always 22, so runs as a GCN instead of RGCN
        relation='o1', # split by sw type

        #==== OPTIONAL: NODE FEATURES ====
        #requires node feature data, ex: g = graphistry.nodes(nodes_df, node_id_col).edges(..
        #use_feat=True
        #X=[g._node] + good_feats_col_names,
        #cardinality_threshold=len(g._edges)+1, #optional: avoid topic modeling on high-cardinality cols
        #min_words=len(g._edges)+1, #optional: avoid topic modeling on high-cardinality cols

        epochs=10
    )

    def to_cpu(tensor, is_gpu=True):
        return tensor.cpu() if is_gpu else tensor

    score2 = pd.Series(to_cpu(g2._score(g2._triplets)).numpy())

    # df[['score', 'is_low_score', ...]]
    df.assign(
        score=score2,
        is_low_score=(score2 < (score2.mean() - 2 * score2.std()))
    )

**Explanation**:

This example provides an introduction to building and training a basic Relational Graph Convolutional Network (RGCN) using :meth:`graphistry.embed_utils.HeterographEmbedModuleMixin.embed`. See the :doc:`SSH logs RGCN demo notebook <../demos/more_examples/graphistry_features/embed/simple-ssh-logs-rgcn-anomaly-detector>` for more on this example.

---

Cluster labeling with DBSCAN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Discuss using DBSCAN for clustering nodes or edges based on features.

**Code**

.. code-block:: python

    g2 = g1.umap().dbscan(eps=0.5, min_samples=5)  # Apply DBSCAN clustering
    print('labels: ', g2._nodes['_dbscan'])

**Explanation**:

This example illustrates how to apply DBSCAN clustering using :meth:`graphistry.compute.cluster.ClusterMixin.dbscan` to label graph data after reducing dimensionality with UMAP.

---

Automated Feature Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Illustrate generating features from raw data for AI applications.

**Code**

.. code-block:: python

    g = graphistry.nodes(df).featurize(kind='nodes', X=['raw_feature_1', 'raw_feature_2'])

**Explanation**:

This example demonstrates how to automatically generate features from raw data returned by GFQL queries for use with ML and AI using :meth:`graphistry.feature_utils.FeatureMixin.featurize` methods. Paramter `feature_engine=` (:data:`graphistry.feature_utils.FeatureEngine`) selects the feature generation engine.

---

Semantic Search in Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Implement semantic search using graph embeddings and natural language queries.

**Code**

.. code-block:: python

    g2 = g1.featurize(
        X = ['text_col_1', 'text_col_n'],  # ... more columns
        kind='nodes',
        model_name = "paraphrase-MiniLM-L6-v2")
    
    results_df, query_vector = g2.search('my natural language query => df hits', ...)

    g3 = g2.search_graph('my natural language query => graph', ...)
    g3.plot()


**Explanation**:

This example showcases how to perform semantic searches within graph data using embeddings. For further details on implementing semantic search, see the **Semantic Search** section in our documentation.

---

Knowledge Graph Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Explain training models for knowledge graph embeddings and predicting relationships.

**Code**

.. code-block:: python

    g2 = g1.embed(relation='relationship_column_of_interest')

    g3 = g2.predict_links_all(threshold=0.95)  # score high confidence predicted edges
    g3.plot()

    # Score over any set of entities and/or relations. 
    g4 = g2.predict_links(
        source=['entity_k'], 
        relation=['relationship_1', 'relationship_4'],  # ... more relationships
        destination=['entity_l', 'entity_m'],  # ... more entities
        threshold=0.9,  # score threshold
        return_dataframe=False)  # return graph vs _edges df

**Explanation**:

This example describes how to train models for knowledge graph embeddings with **GFQL** and how to predict relationships between entities. Shows using both :meth:`graphistry.embed_utils.HeterographEmbedModuleMixin.embed`, :meth:`graphistry.embed_utils.HeterographEmbedModuleMixin.predict_links_all`, and :meth:`graphistry.embed_utils.HeterographEmbedModuleMixin.predict_links`.












