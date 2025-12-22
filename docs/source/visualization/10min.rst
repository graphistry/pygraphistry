.. _10min-viz:

10 Minutes to Graphistry Visualization
======================================

This guide covers core visualization topics like the difference between uploading and viewing graphs, how the client/server architecture works, and how to use PyGraphistry's fluent API to create powerful visualizations by combining ideas like encodings, layouts, and settings. Finally, we overview how to embed visualizations into different workflows.

Key Concepts
------------

- :ref:`Client/Server Architecture <client-server-architecture>`
- :ref:`Fluent API Style <fluent-api-style>`
- :ref:`Shaping Your Data <shaping-your-data>`
- :ref:`Layouts <layouts>`
- :ref:`Node & Edge Encodings <node-edge-encodings>`
- :ref:`Global URL settings <url-settings>`
- :ref:`Plotting: Inline and URL Rendering <plot>`
- :ref:`Additional Resources <extra>`


.. _client-server-architecture:

Client/Server Architecture: Uploading vs. Serving vs. Viewing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyGraphistry uses a **client-server** model. By separating the uploader, server, and viewer, we can achieve better performance, new capabilities, and a variety of usage modes.

- **Upload Client**: In your local environment, you can shape data and call the Graphistry API to upload it to a server (self-hosted or `Graphistry Hub <https://www.graphistry.com/get-started>`__).
- **Visualization Server**: The server processes the data using GPU acceleration to handle large graphs.
- **Visualization Client**: The graph is then explored in your browser, where interactions like zooming and filtering are handled smoothly by using local and remote GPU resources as appropriate.

This split architecture allows scalable, high-performance visualization for even the largest datasets.


.. _fluent-api-style:

Fluent API Style
~~~~~~~~~~~~~~~~

PyGraphistry uses a **fluent style** API, which means that methods can be chained together. This allows for concise and readable code without an extensive setup:

.. code-block:: python

    g1 = graphistry.edges(df, 'src', 'dst')
    g2 = g1.nodes(df2, 'n')
    g3 = g2.encode_point_size('score')
    g3.plot()

    # As shorter fluent lines
    g = graphistry.edges(df, 'src', 'dst').nodes(df2, 'n')
    g.encode_point_size('score').plot()
    
This approach lets you layer operations as needed, keeping code light and intuitive.


.. _shaping-your-data:

Shaping Your Data
-----------------

PyGraphistry supports flexible shaping of your graph data:

- **`.edges()` & `.nodes()`**: Define edges between entities and optional node attributes

  .. code-block:: python

    # df[['src', 'dst', ...]]
    graphistry.edges(df, 'src', 'dst').plot()

    # ... + df2[['n', ...]]
    graphistry.edges(df, 'src', 'dst').nodes(df2, 'n').plot()

  Example visualization:

  .. raw:: html

      <iframe src="https://hub.graphistry.com/graph/graph.html?dataset=706267dc814545dcb1d0dc634e52f07d&type=arrow&viztoken=a6820453-f52e-4480-8922-9f17aa674c50&usertag=ef9e6f8d-pygraphistry-0.48.0+78.g743cd9d3&splashAfter=1766373653&info=true" style="width: 100%; height: 500px; border: 0;" loading="lazy"></iframe>

- **Hypergraph**: Use multiple columns for nodes for more complex visualizations
  
  .. code-block:: python

    # df[['actor', 'event', 'location', ...]]
    hg = graphistry.hypergraph(df, ['actor', 'event', 'location'])
    hg['graph'].plot()

- **UMAP**: Dimensionality reduction & embedding visualization tool based on row similarity

  .. code-block:: python

    # df[['score', 'time', ...]]
    graphistry.nodes(df).umap(X=['score', 'time']).plot()

These methods ensure you can quickly load & shape data and move into visualizing.

.. _layouts:

Layouts
-------

PyGraphistry's :ref:`Layout catalog <layout-catalog>` provides many options, covering:

- **Live Layout**: Graphistry performs GPU-accelerated force-directed layouts at interaction time.
  You can adjust settings, such as gravity, edge weight, and initial clustering time:

  .. code-block:: python

      g.settings(url_params={'play': 7000, 'info': True}).plot()

- **PyGraphistry Layouts**: PyGraphistry ships with special layouts unavailable elsewhere and that work with the rendering engine's special features:

  .. code-block:: python

      g.time_ring_layout('time_col').plot()

- **Plugin Layouts**: Integrated use of external libraries for specific layouts:

  - :ref:`Graphviz <graphviz>` for hierarchical and directed layouts such as the ``"dot"`` engine
  - :ref:`cuGraph <cugraph>` for GPU-accelerated FA2, a weaker version of Graphistry's live layout
  - :ref:`igraph <igraph>` for CPU-based layouts, similar to GraphViz and with layouts that focus more on medium-sized social networks

Static Graphviz render (for docs/notebooks)
-------------------------------------------

When you need a quick static image without an interactive client, render directly with Graphviz. ``plot_static`` auto-displays in Jupyter and returns an SVG/Image object (use ``.data`` for raw bytes):

.. code-block:: python

    # Auto-displays inline in Jupyter notebooks
    g.plot_static(format='svg', max_nodes=200, max_edges=400)

Example visualization (static):

.. figure:: ../_static/visualization/plot_static_example.png
   :alt: Example static plot rendered with plot_static
   :width: 90%

   Static render produced with ``plot_static()``.

Text-only outputs
-----------------

Emit DOT or Mermaid DSL for downstream rendering or embedding:

.. code-block:: python

    dot_text = g.plot_static(engine='graphviz-dot', reuse_layout=True)
    mermaid_text = g.plot_static(engine='mermaid-code', reuse_layout=False)

- **External Layouts**: Pass in ``x``, ``y`` columns, such as from your own edits, external data, or external ML/AI packages:

   .. code-block:: python

      # nodes_df[['x', 'y', 'n', ...]]
      g = graphistry.edges(e_df, 's', 'd').nodes(nodes_df, 'n')
      g2 = g.settings(url_params={'play': 0}) # skip initial loadtime layout
      g2.plot()


.. _node-edge-encodings:

Node & Edge Encodings
---------------------

You can encode your graph attributes visually using colors, sizes, icons, and more:

* **Direct Encoding**: Set attributes like color directly on nodes or edges.

  .. code-block:: python

      g.encode_point_color('type', categorical_mapping={'A': 'red', 'B': 'blue'}, default_mapping='gray').plot()

  Example visualization:

  .. raw:: html

      <iframe src="https://hub.graphistry.com/graph/graph.html?dataset=b055da15eabb4271a98d76e1955fe125&type=arrow&viztoken=0b236032-2639-48e8-a733-f791eb0b4ba9&usertag=ef9e6f8d-pygraphistry-0.48.0+78.g743cd9d3&splashAfter=1766373655&info=true" style="width: 100%; height: 500px; border: 0;" loading="lazy"></iframe>

  Example visualization (static):

  .. figure:: ../_static/visualization/node_edge_encodings_static.png
     :alt: Example static node/edge encoding render
     :width: 90%

     Static render of the categorical encoding example.

* **Categorical & Continuous Mappings**: Handle both discrete and continuous data:

  .. code-block:: python

      g.encode_point_color('score', ['blue', 'yellow', 'red'], as_continuous=True).plot()

* **Encodings List**: Beyond colors, you can also adjust edge thickness, node icon, and add badges using the following methods:

  * Points:

    - :meth:`graphistry.PlotterBase.PlotterBase.encode_point_badge`
    - :meth:`graphistry.PlotterBase.PlotterBase.encode_point_color`
    - :meth:`graphistry.PlotterBase.PlotterBase.encode_point_icon`
    - :meth:`graphistry.PlotterBase.PlotterBase.encode_point_size`

  * Edges:

    - :meth:`graphistry.PlotterBase.PlotterBase.encode_edge_badge`
    - :meth:`graphistry.PlotterBase.PlotterBase.encode_edge_color`
    - :meth:`graphistry.PlotterBase.PlotterBase.encode_edge_icon`

* **Bind**: Simpler data-driven settings are done through :meth:`graphistry.PlotterBase.PlotterBase.bind`:

  .. code-block:: python

    g.bind(point_title='my_node_title_col')


  Where:

  .. code-block:: python
    
    bind(source=None, destination=None, node=None, edge=None,
      edge_title=None, edge_label=None, edge_color=None, edge_weight=None,
      edge_size=None, edge_opacity=None, edge_icon=None,
      edge_source_color=None, edge_destination_color=None,
      point_title=None, point_label=None, point_color=None, point_weight=None,
      point_size=None, point_opacity=None, point_icon=None, point_x=None, point_y=None
    )

.. _url-settings:

Global URL settings    
-------------------

Graphistry visualizations are highly configurable via URL parameters. You can control the look, interaction, and data filters:

.. code-block:: python

    g.settings(url_params={'play': 7000, 'info': True}).plot()

For a complete list of parameters, refer to the `official REST URL params page <https://hub.graphistry.com/docs/api/1/rest/url/#urloptions/>`__.

.. _plot:

Plotting: Inline and URL Rendering
----------------------------------

Once you're ready to visualize, use ``.plot()`` to render:

- **Inline Plotting**: Directly embed interactive visualizations in your notebook or Python environment:

  .. code-block:: python

      g.plot()

- **URL Rendering**: Get a sharable and embeddable URL to view in the browser:

  .. code-block:: python

      url = g.plot(render=False)
      print(f"View your graph at: {url}")

  You can further control the embeded visualization using URL parameters and JavaScript

Static Image Export
~~~~~~~~~~~~~~~~~~~

For non-interactive outputs (documentation, reports, presentations), use ``plot_static()``:

.. code-block:: python

    # Quick static SVG (auto-displays in Jupyter)
    g.plot_static()

    # With styling
    g.plot_static(
        graph_attr={'rankdir': 'LR', 'bgcolor': 'white'},
        node_attr={'style': 'filled', 'fillcolor': 'lightblue'}
    )

    # Save to file
    g.plot_static(format='png', path='graph.png')

Works with any layout source (UMAP, ring, graphviz, or manual x/y positions).

See the `static rendering tutorial <../../demos/demos_databases_apis/graphviz/static_rendering.ipynb>`_ for styling options, output formats, and complete examples.

.. _extra:


Next Steps
----------

- :ref:`10 Minutes to GFQL <10min-gfql>`: Use GFQL to query and manipulate your graph data before visualization.
- :ref:`Layout guide <layout-guide>`: Explore different layouts for your visualizations.
- :ref:`Plugins <plugins>`: Discover more ways to connect to your data and work with your favorite tools. 
- :ref:`Layout catalog <layout-catalog>`: Dive deeper into the layout options available in PyGraphistry.
- :ref:`PyGraphistry API Reference <api>`

External Resources
--------------------

To dive deeper into graph analytics and visualizations, check out the following resources:

- `Graphistry Get Started <https://www.graphistry.com/get-started>`__
- `GraphistryJS Clients: NodeJS, React, & Vanilla <https://github.com/graphistry/graphistry-js>`__
- `Graphistry GitHub <https://github.com/graphistry/pygraphistry>`__
- `Slack Community <https://join.slack.com/t/graphistry-community/shared_invite/zt-53ik36w2-fpP0Ibjbk7IJuVFIRSnr6g>`__

Happy graphing!
