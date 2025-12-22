.. _layout-catalog:

PyGraphistry Layout Catalog
============================

This page provides an overview of the main layouts available in PyGraphistry, including through plugins like graphviz and igraph. Each optimizes for different use cases. Click on a plugin to jump to its section.

- :ref:`PyGraphistry Plugin <pygraphistry-plugin>`: GPU-accelerated layouts like ForceAtlas2, modularity-weighted, UMAP, and more.
- :ref:`cuGraph Plugin <cugraph-plugin>`: Large-scale graph layouts with GPU-optimized ForceAtlas2.
- :ref:`Graphviz Plugin <graphviz-plugin>`: Hierarchical, directed, and flowchart-like layouts for medium-sized graphs.
- :ref:`igraph Plugin <igraph-plugin>`: Versatile 2D/3D layouts including Fruchterman-Reingold, Kamada-Kawai, and more.
- :ref:`Custom Layouts <custom-layouts>`: Manually compute or post-process custom layouts.

Preview Gallery
----------------

Small static previews of representative layouts. Click a thumbnail to open the live interactive view.

.. list-table::
   :widths: 33 33 33
   :header-rows: 0

   * - .. figure:: /_static/layout/catalog_circle.svg
          :alt: Circle layout preview
          :target: https://hub.graphistry.com/graph/graph.html?dataset=854a81684f274ffdb093b4255e7f767f&type=arrow&viztoken=0b0075c6-7dd3-4376-acba-87eb60469a6f&usertag=ef9e6f8d-pygraphistry-0.48.0+79.g7938ff78&splashAfter=1766373977&info=true&play=0

          Circle (PyGraphistry)
     - .. figure:: /_static/layout/catalog_time_ring.svg
          :alt: Time ring layout preview
          :target: https://hub.graphistry.com/graph/graph.html?dataset=4d5b51c6868c46228c6f43b5572cd9ad&type=arrow&viztoken=5d3c4a8d-aa75-45a9-a1f6-df7247b99505&usertag=ef9e6f8d-pygraphistry-0.48.0+79.g7938ff78&splashAfter=1766373979&info=true&play=0&lockedR=True&bg=%23E2E2E2

          Time Ring (PyGraphistry)
     - .. figure:: /_static/layout/catalog_graphviz_dot.svg
          :alt: Graphviz dot layout preview
          :target: https://hub.graphistry.com/graph/graph.html?dataset=06938e865010412dbf3b3469cf2bc07e&type=arrow&viztoken=61ad0b49-e69b-47e5-9849-320ab32e848b&usertag=ef9e6f8d-pygraphistry-0.48.0+79.g7938ff78&splashAfter=1766373981&info=true&play=0&edgeCurvature=0

          Dot (Graphviz)

These previews use a small toy graph and ``plot_static()`` for lightweight docs.
For GPU-scale layouts (ForceAtlas2/cugraph) and igraph variants, use the linked notebooks or run the layouts on your own data.

.. _pygraphistry-plugin:

PyGraphistry Plugins
---------------------

PyGraphistry supports GPU-accelerated layouts, including ForceAtlas2, modularity-weighted algorithms, and hierarchical ring layouts for large-scale and specialized structures. (:ref:`API reference on Graphistry layouts <pyg-layout-api>`)

**Supported Layouts**:

- **Circle** — Positions nodes in a circular layout, useful for ordinal data, or separately laying out singleton nodes. :ref:`API info on circle layouts <circle-api>`
- **ForceAtlas2** — Optimized for large, dense graphs. Provides smooth clustering and cluster separation using GPU acceleration. PyGraphistry version gives visual and performance improvements upon other systems, and Graphistry server load-time version provides a different set of features focused on interactivity and additional options. :ref:`API info on FA2 layouts <fa2-api>`
- **Modularity-Weighted** — Lays out clusters based on modularity, optimizing for visualizing community structures. :ref:`API info on modularity-weighted layouts <mod-layout-api>` (`notebook <../../demos/more_examples/graphistry_features/layout_modularity_weighted.ipynb>`__)
- **Group-In-A-Box (GIB)** — Organizes nodes into visually distinct boxes based on their group or cluster for clear structure definition. :ref:`API info on group-in-a-box layouts <gib-api>` (`notebook <../../demos/more_examples/graphistry_features/layout_group_in_a_box.ipynb>`__)
- **UMAP** — Reduces high-dimensional data into a 2D layout based on similarity, best for complex datasets needing dimensionality reduction. :py:meth:`API info on UMAP <graphistry.umap_utils.UMAPMixin.umap>`
- **Hierarchical Ring Layouts** — Creates ring layouts that categorize nodes by time, continuous variables, or categorical properties. :ref:`API info on ring layouts <ring-api>` (`notebook <../../demos/more_examples/graphistry_features/layout_time_ring.ipynb>`__)

**Example**:

Visit the :ref:`PyGraphistry visualization tutorial <10min-viz>`.

.. code-block:: python
    
        g.time_ring_layout('time_col').plot()

.. note::
   When building layouts via GFQL or other JSON interfaces, provide
   ``time_start``/``time_end`` as ISO-8601 strings. PyGraphistry converts them
   to ``numpy.datetime64`` before computing the layout, so the experience matches
   direct Python usage where you pass Timestamp objects.

.. _cugraph-plugin:

cuGraph Plugin
---------------

cuGraph provides one GPU-optimized graph layout for scaling large datasets, making it a candidate for massive graphs. (:ref:`API reference on cuGraph <cugraph>`)

**Supported Layouts**:

- **ForceAtlas2** — Designed for very large graphs, scaling with GPU acceleration to maintain interactive performance with 100k+ nodes. Less flexible version of the Graphistry ForceAtlas2 GPU algorithm.

.. code-block:: python

    g.cugraph_layout('force_atlas2').plot()

.. _graphviz-plugin:

Graphviz Plugin
----------------

Graphviz specializes in directed and hierarchical layouts, useful for flowcharts, dependency trees, and acyclic graphs (DAGs). (:ref:`API reference on graphviz layouts <graphviz>`)

**Supported Layouts**:

- **acyclic** — Removes cycles from directed graphs by reversing edges to make the graph acyclic, useful for processing DAGs.
- **ccomps** — Extracts the connected components from a graph and outputs them as subgraphs.
- **circo** — Circular layout, arranging nodes in a radial fashion, ideal for cycle graphs.
- **dot** — Best for directed acyclic graphs (DAGs) like flowcharts, laying out hierarchies in a top-down manner.
- **fdp** — General force-directed layout, good for smaller undirected graphs.
- **gc** — Used for graph coloring, assigning colors to nodes such that no two adjacent nodes have the same color.
- **gvcolor** — Colorizes graphs based on specific attributes, often used for improving visual distinctions between nodes.
- **gvpr** — Graph pattern scanning and rewriting tool used for scripting changes in a graph, allowing custom manipulation of graph structures.
- **neato** — Force-directed layout for undirected graphs, suitable for smaller networks.
- **nop** — A no-op layout that performs no layout calculations, often used as a placeholder or for manual layout adjustments.
- **osage** — Useful for directed layered graphs with hierarchical structures.
- **patchwork** — Visualizes hierarchical clusters as a nested set of rectangles, similar to a treemap visualization.
- **sccmap** — Finds the strongly connected components in a graph and generates a reduced graph of those components.
- **sfdp** — Force-directed layout optimized for large graphs, providing fast and scalable rendering.
- **tred** — Transitive reduction algorithm that minimizes the number of edges while maintaining reachability between nodes in a directed graph.
- **twopi** — Radial layout that positions nodes in concentric circles, useful for radial hierarchies.
- **unflatten** — Improves readability by adjusting node levels to reduce overlap in hierarchical graphs.

**Example**:

Visit the :ref:`API reference on graphviz page <graphviz>` for more examples.

.. code-block:: python

    g.layout_graphviz('dot').plot()

For static image export (SVG, PNG) instead of interactive visualization, see :py:meth:`~graphistry.PlotterBase.PlotterBase.plot_static` and the `static rendering tutorial <../../../demos/demos_databases_apis/graphviz/static_rendering.ipynb>`_.

.. _igraph-plugin:

igraph Plugin
---------------

The igraph plugin offers various layouts forvarious graph types. (:ref:`API reference on igraph <igraph>`)

**Supported Layouts**:

- **auto / automatic** — Automatically chooses the best layout for the given graph based on its structure and size.
- **bipartite** — Positions nodes in two layers, useful for visualizing bipartite graphs (graphs with two distinct sets of nodes).
- **circle / circular** — Positions nodes in a circular layout, suitable for visualizing cycles and small networks.
- **circle_3d / circular_3d** — 3D version of the circular layout, positioning nodes in a 3D circular structure.
- **davidson_harel / dh** — Force-directed layout algorithm with an iterative approach for improving graph aesthetics, especially useful for smaller graphs.
- **drl** — Distributed Recursive Layout, a force-directed layout algorithm optimized for very large graphs.
- **drl_3d** — 3D version of the DRL algorithm, optimized for large graphs in a 3D space.
- **fr / fruchterman_reingold** — Force-directed layout balancing attractive and repulsive forces for clustered yet separated nodes.
- **fr_3d / fruchterman_reingold_3d / fr3d** — 3D version of the Fruchterman-Reingold force-directed layout.
- **grid** — Organizes nodes in a grid structure, useful for matrix-like data.
- **grid_3d** — 3D version of the grid layout, positioning nodes in a 3D grid.
- **graphopt** — Another force-directed layout algorithm, known for its fast convergence on small to medium-sized graphs.
- **kk / kamada_kawai** — Similar to Fruchterman-Reingold, this force-directed layout focuses on preserving geometric distances between nodes.
- **kk_3d / kamada_kawai_3d / kk3d** — 3D version of the Kamada-Kawai algorithm, preserving distances between nodes in a 3D space.
- **lgl / large / large_graph** — Optimized for very large graphs, often used for graphs with thousands of nodes.
- **mds** — Multi-Dimensional Scaling, used for dimensionality reduction and projecting nodes into 2D or 3D space based on similarity.
- **random / random_3d** — Randomly positions nodes in 2D or 3D space, often used for testing or debugging layout algorithms.
- **reingold_tilford / rt / tree** — Specialized for tree structures, arranging nodes hierarchically from top to bottom.
- **reingold_tilford_circular / rt_circular** — Circular version of the Reingold-Tilford tree layout, arranging tree nodes in a radial fashion.
- **sphere / spherical** — 3D layout positioning nodes on the surface of a sphere, useful for 3D graph exploration.
- **star** — Positions nodes in a star configuration, with a central node surrounded by peripheral nodes.
- **sugiyama** — Specialized for hierarchical structures, often used for organizational charts and trees.

Full list: :ref:`More Info <igraph>`

**Example**:  

Visit the :ref:`API reference on graphviz <igraph>` for more examples.

.. code-block:: python

    g.layout_igraph('circle').plot()

.. _custom-layouts:

Custom Layouts
---------------

Users can manually compute layouts from external sources or post-process the results. This allows flexibility in integrating custom embedding algorithms or other specialized layouts into PyGraphistry. (`API reference <pyg-layout-api>`_)

**Example**:

Manually apply a layout and visualize by `custom layouts (notebook) <../../demos/more_examples/graphistry_features/external_layout/simple_manual_layout.ipynb>`_ .

.. code-block:: python

    # Input: Precompute some x and y positions
    nodes_df : pd.DataFrame = ...
    assert 'x' in df.columns and 'y' in df.columns

    g2 = (g1
        .nodes(nodes_df)
        .bind(point_x='x', point_y='y')
        .settings(url_params={'play': 0})  # Prevent loadtime layout from running
    )

Further reading
----------------

- :ref:`PyGraphistry API Reference <pyg-layout-api>`: GPU-accelerated layouts such as ForceAtlas2, modularity-weighted, hierarchical rings, UMAP, and group-in-a-box.
- :ref:`cuGraph API Reference <cugraph>`: ForceAtlas2 optimized for large-scale graphs using GPU acceleration.
- :ref:`Graphviz API Reference <graphviz>`: Best for hierarchical and flowchart/DAG layouts, including options like dot, neato, and circo.
- :ref:`igraph API Reference <igraph>`: Versatile with 2D/3D layouts, including Fruchterman-Reingold, Kamada-Kawai, and Sugiyama.


Visit the respective tutorial links to dive deeper into each plugin’s capabilities and usage.
