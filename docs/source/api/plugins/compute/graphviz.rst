.. _graphviz:


graphviz
--------

graphviz is a popular graph visualization library that PyGraphistry can interface with. This allows you to leverage graphviz's powerful layout algorithms, and optionally, static picture renderer. It is especially well-known for its "dot" layout algorithm for hierarchical and tree layouts of graphs with less than 10,000 nodes and edges.

For static outputs in notebooks or docs, you can either call :py:meth:`graphistry.Plottable.plot_static` (preferred, auto-reuses x/y when present) or :py:meth:`graphistry.plugins.graphviz.render_graphviz` for lower-level control.

**Auto-display**: When called in a Jupyter notebook, ``plot_static`` automatically displays the rendered output inline. It returns an SVG or Image object (use ``.data`` for raw bytes).

.. code-block:: python

   # Simplest form - auto-displays in notebook
   g.plot_static()

   # Save to file while also displaying
   svg_obj = g.plot_static(path='graph.svg')  # saves file AND returns SVG object

``plot_static`` engines:

- ``graphviz-svg`` / ``graphviz-png`` (default render image, optional path)
- ``graphviz`` (render to any Graphviz format, e.g., pdf)
- ``graphviz-dot`` (DOT text, optional path)
- ``mermaid-code`` (Mermaid DSL text, optional path)

**Styling**: Use ``graph_attr``, ``node_attr``, and ``edge_attr`` dictionaries:

.. code-block:: python

   g.plot_static(
       graph_attr={'rankdir': 'LR', 'bgcolor': 'white'},
       node_attr={'style': 'filled', 'fillcolor': 'lightblue'},
       edge_attr={'color': 'gray'}
   )

``plot_static()`` works with any layout source—it will use existing x/y positions if available (``reuse_layout=True``), or compute layout via graphviz if not.

See the `static rendering tutorial <../../../demos/demos_databases_apis/graphviz/static_rendering.ipynb>`_ for complete examples.

.. automodule:: graphistry.plugins.graphviz
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Constants

.. autodata:: graphistry.plugins_types.graphviz_types.EdgeAttr
   :noindex:

.. autodata:: graphistry.plugins_types.graphviz_types.GraphvizAttrValue
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
