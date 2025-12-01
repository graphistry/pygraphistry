.. _graphviz:


graphviz
--------

graphviz is a popular graph visualization library that PyGraphistry can interface with. This allows you to leverage graphviz's powerful layout algorithms, and optionally, static picture renderer. It is especially well-known for its "dot" layout algorithm for hierarchical and tree layouts of graphs with less than 10,000 nodes and edges.

For static outputs in notebooks or docs, you can either call :py:meth:`graphistry.Plottable.plot_static` (preferred, auto-reuses x/y when present) or :py:meth:`graphistry.plugins.graphviz.render_graphviz` for lower-level control.

.. automodule:: graphistry.plugins.graphviz
   :members:
   :undoc-members:
   :show-inheritance:

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
