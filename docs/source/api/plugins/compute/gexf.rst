.. _gexf:

Gephi (GEXF)
-------------

GEXF (Graph Exchange XML Format) is commonly used by Gephi and other tools for graph interchange.
Graphistry supports GEXF 1.1draft, 1.2draft, and 1.3 for import/export with no extra dependencies.

Use :func:`graphistry.gexf` (or :meth:`graphistry.PlotterBase.PlotterBase.from_gexf`) to load a GEXF file, URL, or stream into a PyGraphistry plotter.
By default, any available GEXF viz fields are bound; if the file has no viz data, Graphistry defaults apply.

**When to use**

- You have a Gephi (or other tool) export and want to preserve its layout/viz.
- You need a lightweight interchange format for graph attributes or layouts.

**Example**

::

    import graphistry

    # Preserve GEXF's layout, size, colors, shape icons
    g = graphistry.gexf("my_graph.gexf")
    g.plot()

    # Keep layout while dropping GEXF colors/sizes
    g_layout_only = graphistry.gexf(
        "my_graph.gexf",
        bind_node_viz=["position"],
        bind_edge_viz=[],
    )
    g_layout_only.plot()

For notebook walkthroughs (small + medium GEXF) and dataset samples, see :ref:`nb-compute`.

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

Use ``bind_node_viz`` / ``bind_edge_viz`` as allowlists to restrict which GEXF viz
fields are bound to Graphistry encodings (unlisted fields still load as columns):

- node fields: ``color``, ``size``, ``opacity``, ``position``, ``icon``
- edge fields: ``color``, ``size``, ``opacity``

``None`` (default) binds all supported fields present in the file. Passing an empty list disables all viz bindings for that element type.
``position`` binds ``point_x``/``point_y`` and sets ``play=0`` to respect precomputed layouts.

**Examples**

::

    # Bind all available GEXF viz fields (default)
    g = graphistry.gexf("my_graph.gexf")

    # Bind no GEXF viz fields (use Graphistry defaults)
    g = graphistry.gexf("my_graph.gexf", bind_node_viz=[], bind_edge_viz=[])

    # Bind only layout positions for nodes, drop edge viz
    g = graphistry.gexf("my_graph.gexf", bind_node_viz=["position"], bind_edge_viz=[])

After loading, you can apply Graphistry's declarative encodings (for example,
``encode_point_color`` or ``encode_point_size``) to override GEXF defaults.

**Validation**

The loader raises ``ValueError`` for common errors such as missing nodes, missing node IDs, or edges that reference unknown nodes.
These checks run inside the GEXF loader (XML well-formedness + basic structural checks), not PyGraphistry's broader
graph validation or full GEXF schema validation.
For untrusted inputs, install ``defusedxml``; it will be used automatically for safer XML parsing.
Use ``parse_engine="stdlib"`` or ``parse_engine="defused"`` to override the parser (useful in tests).

.. automodule:: graphistry.plugins.gexf
