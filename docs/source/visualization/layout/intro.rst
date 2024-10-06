.. _layout-guide:

Quick Guide to PyGraphistry layouts
===================================

This guide provides a quick introduction to key layout concepts in PyGraphistry

Key Concepts Covered
--------------------

- :ref:`Precomputed Layouts <precomputed-layouts>`
- :ref:`Internal & Plugin Layouts <internal-plugin-layouts>`
- :ref:`Runtime Dynamic Layouts <runtime-dynamic-layouts>`
- :ref:`Runtime Layout Settings <layout-settings>`

- Further reading and detailed configuration options for:
  - :ref:`Ring Layout API <ring-api>`
  - :ref:`GIB Layout API <gib-api>`
  - :ref:`Modularity Layout API <mod-layout-api>`
  - Plugin layouts: :ref:`GraphViz <graphviz-plugin>`, :ref:`cuGraph <cugraph-plugin>`, :ref:`iGraph <igraph-plugin>`

---

Key Concepts
------------

.. _precomputed-layouts:

Precomputed Layouts
~~~~~~~~~~~~~~~~~~~

Precomputed layouts involve manually calculating node positions (`x`, `y` columns) before rendering your graph.

This is useful such as when you need to manually control a layout, or are visualizing externally provided positions such as from embeddings.

  .. code-block:: python

    # Precomputed 'x', 'y' coordinates in a nodes DataFrame
    g = graphistry.edges(e_df, 'src', 'dst').nodes(n_df, 'n')
    g2 = g.settings(url_params={'play': 0})  # skip initial loadtime layout
    g2.plot()

Precomputed layouts are ideal for handling complex visualizations where precision is key.

.. _internal-plugin-layouts:

Internal & Plugin Layouts
~~~~~~~~~~~~~~~~~~~~~~~~~

PyGraphistry includes a growing number of built-in layouts.

These help with several scenarios, including:

* Faster performance and greater scale
* Leveraging Graphistry runtime layout features
* Combining layouts

**Graphistry Layouts:**

- **Native Force-Directed Layout:** PyGraphistry’s default layout automatically arranges the nodes based on their connectivity.

  .. code-block:: python

      g = graphistry.edges(e_df, 'src', 'dst').plot()
  
- **Ring Layout:** Ideal for visualizing sorted, hierarchical, or time-based data.

  .. code-block:: python

      g.time_ring_layout('my_timestamp').plot()
      g.categorical_ring_layout('my_type').plot()
      g.continuous_ring_layout('my_score').plot()

  For further details, refer to the :ref:`Ring Layout API <ring-api>`.

- **Modularity Weighted Layout:** Weights edges based on modularity.

  .. code-block:: python

    # Separate by precomputed modules
    assert 'partition' in g._nodes
    g.modularity_weighted_layout(community_col='partition').plot()

    # Separate by automatically computed modules
    g.modularity_weighted_layout(community_alg='louvain', engine='cudf').plot()

  Read more in the :ref:`Modularity Layout API <mod-layout-api>`.

- **Group-in-a-Box Layout:** Groups nodes into a grid of clusters.

  Popularized by NodeXL for analyzing large social networks, the PyGraphistry version enables quickly working with larger datasets than possible in other packages

  .. code-block:: python

    g.gib_layout().plot()

  Learn more in the :ref:`Group-in-a-Box Layout API <gib-api>`.

**Plugin Layouts:**

- **cuGraph Plugin (GPU-accelerated force layouts):** Ideal for large-scale graphs requiring performance.

  .. code-block:: python

    g.cugraph_force_layout().plot()

  See the :ref:`cuGraph Plugin <cugraph-plugin>` for more details.

- **GraphViz Plugin (Hierarchical layouts):** Great for tree-like or hierarchical data.

  .. code-block:: python

    g.graphviz_layout(engine='dot').plot()

  Find more details in the :ref:`GraphViz Plugin <graphviz-plugin>`.

- **iGraph Plugin (Kamada-Kawai, Sugiyama, etc.):** Provides classic layout algorithms for a variety of graph types.

  .. code-block:: python

    g.igraph_layout('kamada_kawai').plot()

  See the :ref:`iGraph Plugin <igraph-plugin>` for more information.

.. _runtime-dynamic-layouts:

Runtime Dynamic Layouts
~~~~~~~~~~~~~~~~~~~~~~~

Dynamic layouts allow PyGraphistry to adjust node positions in real-time based on user interactions and graph updates. This provides highly interactive and scalable graph visualizations.

  .. code-block:: python

    # Run the force-directed layout at viz load time for 5 seconds (5,000 milliseconds)
    g = graphistry.edges(e_df, 'src', 'dst')
    g.settings(url_params={'play': 5000}).plot()

For details on runtime settings and customization, explore the :ref:`Layout Settings <layout-settings>` page.

---

Further Reading
---------------

Layout in general:

- :ref:`Layout Catalog <layout-catalog>`
- :ref:`Layout Settings <layout-settings>`

Individaul layouts and plugins:

- :ref:`Ring Layout API <ring-api>`
- :ref:`GIB Layout API <gib-api>`
- :ref:`Modularity Layout API <mod-layout-api>`
- :ref:`GraphViz Plugin <graphviz-plugin>`
- :ref:`cuGraph Plugin <cugraph-plugin>`
- :ref:`iGraph Plugin <igraph-plugin>`



