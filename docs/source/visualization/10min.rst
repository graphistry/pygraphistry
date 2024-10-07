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

PyGraphistry offers several layout options:

- **Native Layout**: PyGraphistry’s default is a force-directed layout. 
  You can adjust settings, such as gravity or node influence:

  .. code-block:: python

      g.time_ring_layout('time_col').plot()

- **Plugin Layouts**: Integrated use of external libraries for specific layouts:
  - `GraphViz Plugin <source/plugins/graphviz.rst>`__ for hierarchical and directed layouts.
  - `cuGraph Plugin <source/plugins/cugraph.rst>`__ for GPU-accelerated graph algorithms.

- **External Layouts**: Pass in `x`, `y` columns that you compute

   .. code-block:: python

      # nodes_df[['x', 'y', 'n', ...]]
      g = graphistry.edges(e_df, 's', 'd').nodes(nodes_df, 'n')
      g2 = g.settings(url_params={'play': 0}) # skip initial loadtime layout
      g2.plot()


.. _node-edge-encodings:

Node & Edge Encodings
---------------------

You can encode your graph attributes visually using colors, sizes, icons, and more:

- **Direct Encoding**: Set attributes like color directly on nodes or edges.

  .. code-block:: python

      g.encode_point_color('type', categorical_mapping={'A': 'red', 'B': 'blue'}).plot()

- **Categorical & Continuous Mappings**: Handle both discrete and continuous data:

  .. code-block:: python

      g.encode_point_color('score', ['blue', 'yellow', 'red'], as_continuous=True).plot()

- **Other Encodings**: You can also adjust edge thickness, node icon, and add badges using the following methods:

  - `encode_point_badge()`
  - `encode_point_color()`
  - `encode_point_icon()`
  - `encode_point_size()`
  - The same applies for edges:

    - `encode_edge_badge()`
    - `encode_edge_color()`
    - `encode_edge_icon()`
    - `encode_edge_size()`

 - **Bind**: Some settigs are more typically done through `bind()`:

   - `bind(point_title=)`
   - `bind(edge_title=)`
   - `bind(edge_weight=)`

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

Once you're ready to visualize, use `.plot()` to render:

- **Inline Plotting**: Directly embed interactive visualizations in your notebook or Python environment:

  .. code-block:: python

      g.plot()

- **URL Rendering**: Get a sharable and embeddable URL to view in the browser:

  .. code-block:: python

      url = g.plot(render=False)
      print(f"View your graph at: {url}")

  You can further control the embeded visualization using URL parameters and JavaScript 

.. _extra:


Next Steps
----------

- `10 Minutes to GFQL <10min-gfql>`_: Use GFQL to query and manipulate your graph data before visualization.
- `Layout guide <layout-guide>`_: Explore different layouts for your visualizations.
- `Plugins <plugins>`_: Discover more ways to connect to your data and work with your favorite tools. 
- `Layout catalog <layout-catalog>`_: Dive deeper into the layout options available in PyGraphistry.
- `PyGraphistry API Reference <source/api/index.rst>`_

External Resources
--------------------

To dive deeper into graph analytics and visualizations, check out the following resources:

- `Graphistry Get Started <https://www.graphistry.com/get-started>`__
- `GraphistryJS Clients: NodeJS, React, & Vanilla <https://github.com/graphistry/graphistry-js>`__
- `Graphistry GitHub <https://github.com/graphistry/pygraphistry>`__
- `Slack Community <https://join.slack.com/t/graphistry-community/shared_invite/zt-53ik36w2-fpP0Ibjbk7IJuVFIRSnr6g>`__

Happy graphing!
