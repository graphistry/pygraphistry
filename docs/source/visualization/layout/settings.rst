.. _layout-settings:

Layout Settings & Visualization Embedding
=========================================

This guide shows how to embed and configure Graphistry visualizations using the PyGraphistry Python API. For users interested in using URL parameters for embedding in HTML, refer to the external documentation.

Using PyGraphistry for Customization
-------------------------------------

You can use the PyGraphistry API to programmatically configure visualizations. Below are some examples of how to use the `g.settings` and `g.addStyle` methods to customize visualizations.

Scene Settings
~~~~~~~~~~~~~~~

Use :meth:`graphistry.PlotterBase.PlotterBase.scene_settings` to modify the appearance of the graph, including menus, node sizes, and edge opacity:

.. code-block:: python

   g2 = g.scene_settings(
       # Hide menus
       menu=False,
       info=False,
       # Customize graph appearance
       show_arrows=False,
       point_size=1.0,         # Node size (logarithmic scale: 0.1-10.0 → UI 0-100)
       edge_curvature=0.0,     # 0.0 = straight edges
       edge_opacity=0.5,       # 0.0-1.0 range (50% transparent)
       point_opacity=0.9       # 0.0-1.0 range (90% opaque)
   ).plot()

**Value Ranges:**

- ``point_size``: Range 0.1 to 10.0. The UI uses a logarithmic scale (0-100) for display. For example: 0.2 displays as approximately "15", 0.5 as "35", 1.0 as "50", 2.0 as "65", and 5.0 as "85". This logarithmic mapping provides finer control over smaller point sizes.
- ``edge_curvature``: Range 0.0 to 1.0 (0.0 for straight edges, displayed as 0-100 in UI)
- ``edge_opacity``: Range 0.0 to 1.0 (0.0 fully transparent, 1.0 fully opaque, displayed as 0-100 in UI)
- ``point_opacity``: Range 0.0 to 1.0 (0.0 fully transparent, 1.0 fully opaque, displayed as 0-100 in UI)

Collections
~~~~~~~~~~~

Use :meth:`graphistry.PlotterBase.PlotterBase.collections` to define collections (subsets) with optional encodings:

.. code-block:: python

   import graphistry

   collections = [
       {
           "type": "set",
           "id": "newsletter_subscribers",
           "name": "Newsletter Subscribers",
           "node_color": "#32CD32",
           "expr": [graphistry.n({"subscribed_to_newsletter": True})],
       }
   ]

   g2 = g.collections(
       collections=collections,
       show_collections=True,
       collections_global_node_color="CCCCCC",
       collections_global_edge_color="CCCCCC",
   )
   g2.plot()

The collections list is JSON-minified and URL-encoded automatically. GFQL operations use the
Python AST helpers; see :doc:`GFQL documentation </gfql/index>`.
For full schema details, see `Collections URL options <https://hub.graphistry.com/docs/api/1/rest/url/#url-collections>`_.
For a full walkthrough, see the :doc:`Collections tutorial notebook </demos/more_examples/graphistry_features/collections>`.
For color encoding basics, see the :doc:`Color encodings notebook </demos/more_examples/graphistry_features/encodings-colors>`.

Notes and validation
^^^^^^^^^^^^^^^^^^^^

- Order matters: earlier collections have higher priority and override later ones.
- Use collections for priority-based subsets and overlaps; use encode_* for simple column-driven colors.
- Intersections reference set IDs; provide ``id`` for any set used in an intersection.
- Omitting ``node_color`` or ``edge_color`` leaves encodings as passthrough.
- ``collections_global_node_color`` and ``collections_global_edge_color`` apply to nodes/edges
  not in any collection (leading ``#`` is optional).
- Accepted inputs for ``expr`` include GFQL AST helpers, ``Chain`` objects, and wire-protocol dicts.
- Use ``validate='strict'`` to raise on invalid collections; the default ``autofix`` warns and
  drops invalid entries. Use ``warn=False`` to silence warnings.
- If you already have an encoded collections string, pass ``encode=False``. When using
  ``settings(url_params=...)`` directly, provide an encoded ``collections`` value.

Wire protocol example
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   collections_wire = [
       {
           "type": "set",
           "name": "Wire Protocol Example",
           "node_color": "#AA00AA",
           "expr": {
               "type": "gfql_chain",
               "gfql": [
                   {"type": "Node", "filter_dict": {"status": "purchased"}}
               ]
           }
       }
   ]
   g.collections(collections=collections_wire)


Styling the Background and Foreground
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With :meth:`graphistry.PlotterBase.PlotterBase.addStyle`, you can configure background and foreground styles, including colors, gradients, and images:

.. code-block:: python

   # Set a red background
   g.addStyle(bg={'color': 'red'})

   # Apply a radial gradient background
   g.addStyle(bg={
       'color': '#333',
       'gradient': {
           'kind': 'radial',
           'stops': [
               ["rgba(255,255,255, 0.1)", "10%", "rgba(0,0,0,0)", "20%"]
           ]
       }
   })

   # Use an image as a background with blend mode
   g.addStyle(bg={'image': {'url': 'http://site.com/cool.png', 'blendMode': 'multiply'}})

   # Apply blend mode for the foreground
   g.addStyle(fg={'blendMode': 'color-burn'})

Page and Logo Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

Customize the page title, favicon, and logo using :meth:`graphistry.PlotterBase.PlotterBase.addStyle`, :

.. code-block:: python

   # Set page title and favicon
   g.addStyle(page={'title': 'My Site'})
   g.addStyle(page={'favicon': 'http://site.com/favicon.ico'})

   # Add a logo
   g.addStyle(logo={'url': 'http://www.site.com/transparent_logo.png'})

   # Customize logo dimensions and opacity
   g.addStyle(logo={
       'url': 'http://www.site.com/transparent_logo.png',
       'dimensions': {'maxHeight': 200, 'maxWidth': 200},
       'style': {'opacity': 0.5}
   })

For more advanced Python configuration options, refer to the PyGraphistry REST API documentation on `URL parameters <https://hub.graphistry.com/docs/api/1/rest/url/#urloptions>`_ and `Branding metadata <https://hub.graphistry.com/docs/api/2/rest/upload/metadata/>`_.

HTML/URL-based Configuration
--------------------------------

For users interested in configuring Graphistry visualizations through HTML and URL parameters, please refer to the official documentation:

- `Graphistry URL Configuration Options <https://hub.graphistry.com/docs/api/1/rest/url/#urloptions>`_

This guide covers how to embed Graphistry visualizations in web pages and configure visualizations via URL parameters like background color, layout settings, and more.

IFrame CSS Style Tips
~~~~~~~~~~~~~~~~~~~~~~~

When embedding visualizations in HTML, you can customize the appearance using CSS. Below are some common style tips for `<iframe>` elements:

- **Control the border**:
  
  .. code-block:: css
  
     border: 1px solid black;

- **Control the size**:

  .. code-block:: css
  
     width: 100%; height: 80%; min-height: 400px;

Refer to the full `Graphistry URL Configuration Options <https://hub.graphistry.com/docs/api/1/rest/url/#urloptions>`_ for more details.
