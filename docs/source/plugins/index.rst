cuGraph
---------------

cuGraph is a GPU-accelerated graph library that leverages the Nvidia RAPIDS ecosystem. PyGraphistry provides a more fluent interface to enrich and transform your data with cuGraph methods without the boilerplate.

.. automodule:: graphistry.plugins.cugraph
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :noindex:

.. rubric:: Constants

.. autodata:: graphistry.plugins.cugraph.compute_algs

.. autodata:: graphistry.plugins.cugraph.layout_algs

.. autodata:: graphistry.plugins_types.cugraph_types.CuGraphKind


graphviz
--------

graphviz is a popular graph visualization library that PyGraphistry can interface with. This allows you to leverage graphviz's powerful layout algorithms, and optionally, static picture renderer. It is especially well-known for its "dot" layout algorithm for hierarchical and tree layouts of graphs with less than 10,000 nodes and edges.

.. automodule:: graphistry.plugins.graphviz
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :noindex:

.. rubric:: Constants

.. autodata:: graphistry.plugins_types.graphviz_types.EdgeAttr
   :noindex:

.. autodata:: graphistry.plugins_types.graphviz_types.Format
   :noindex:

.. autodata:: graphistry.plugins_types.graphviz_types.GraphAttr
   :noindex:

.. autodata:: graphistry.plugins_types.graphviz_types.NodeAttr
   :noindex:

.. autodata:: graphistry.plugins_types.graphviz_types.Prog
   :noindex:

.. autodata:: graphistry.plugins_types.graphviz_types.EDGE_ATTRS
   :annotation: typing.List[graphistry.plugins_types.graphviz_types.EdgeAttr]

.. autodata:: graphistry.plugins_types.graphviz_types.FORMATS
   :annotation: typing.List[graphistry.plugins_types.graphviz_types.Format]

.. autodata:: graphistry.plugins_types.graphviz_types.GRAPH_ATTRS
   :annotation: typing.List[graphistry.plugins_types.graphviz_types.GraphAttr]

.. autodata:: graphistry.plugins_types.graphviz_types.NODE_ATTRS
   :annotation: typing.List[graphistry.plugins_types.graphviz_types.NodeAttr]

.. autodata:: graphistry.plugins_types.graphviz_types.PROGS
   :annotation: typing.List[graphistry.plugins_types.graphviz_types.Prog]

igraph
------------------------------------------------

igraph is a popular graph library that PyGraphistry can interface with. This allows you to leverage igraph's layout algorithms, and optionally, algorithmic enrichments. It is CPU-based and can generally handle small/medium-sized graphs.

.. automodule:: graphistry.plugins.igraph
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :noindex:

.. rubric:: Constants

.. autodata:: graphistry.plugins.igraph.compute_algs

.. autodata:: graphistry.plugins.igraph.layout_algs

