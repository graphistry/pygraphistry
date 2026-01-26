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
- edge ``weight`` → ``edge_weight``

**Validation**

The loader raises ``ValueError`` for common errors such as missing nodes, missing node IDs, or edges that reference unknown nodes.

.. automodule:: graphistry.plugins.gexf
