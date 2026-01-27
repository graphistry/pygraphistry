.. _gexf:

GEXF
-----

GEXF (Graph Exchange XML Format) is commonly used by Gephi and other tools for graph interchange.

Use :func:`graphistry.gexf` (or :meth:`graphistry.PlotterBase.PlotterBase.from_gexf`) to load a GEXF file, URL, or stream into a PyGraphistry plotter.

**Example**

::

    import graphistry

    g = graphistry.gexf("my_graph.gexf")
    g.plot()

    # Keep layout while dropping GEXF colors/sizes
    g_layout_only = graphistry.gexf(
        "my_graph.gexf",
        bind_node_viz=["position"],
        bind_edge_viz=[],
    )
    g_layout_only.plot()

**Export**

You can export a graph to GEXF using :func:`graphistry.to_gexf` or :meth:`graphistry.PlotterBase.PlotterBase.to_gexf`:

::

    xml_str = g.to_gexf()
    g.to_gexf("out.gexf")

**Viz attribute mapping**

The loader maps standard GEXF viz attributes into Graphistry bindings:

- ``label`` (node) → ``point_title``
- ``label`` (edge) → ``edge_title``
- ``viz:color`` → ``point_color`` / ``edge_color`` (hex color strings)
- ``viz:size`` → ``point_size``
- ``viz:position`` → ``point_x`` / ``point_y`` (and auto-sets ``play=0``)
- ``viz:thickness`` → ``edge_size``
- ``viz:color`` alpha → ``point_opacity`` / ``edge_opacity``
- ``viz:shape`` (nodes) → ``point_icon`` (FA4 icon names without the ``fa-`` prefix: disc→circle, square→square, triangle→caret-up, diamond→diamond; image uses the ``uri`` when available)
- edge ``weight`` → ``edge_weight``

**Viz binding controls**

Use ``bind_node_viz`` / ``bind_edge_viz`` to restrict which GEXF viz fields are bound:

- node fields: ``color``, ``size``, ``opacity``, ``position``, ``icon``
- edge fields: ``color``, ``size``, ``opacity``

Passing an empty list disables all viz bindings for that element type.

After loading, you can apply Graphistry's declarative encodings (for example,
``encode_point_color`` or ``encode_point_size``) to override GEXF defaults.

**Validation**

The loader raises ``ValueError`` for common errors such as missing nodes, missing node IDs, or edges that reference unknown nodes.

.. automodule:: graphistry.plugins.gexf
