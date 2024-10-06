.. _layout-settings:

Layout Settings & Visualization Embedding
=========================================

This guide shows how to embed and configure Graphistry visualizations using the PyGraphistry Python API. For users interested in using URL parameters for embedding in HTML, refer to the external documentation.

Using PyGraphistry for Customization
-------------------------------------

You can use the PyGraphistry API to programmatically configure visualizations. Below are some examples of how to use the `g.settings` and `g.addStyle` methods to customize visualizations.

Scene Settings
~~~~~~~~~~~~~~~

Use `g.scene_settings()` to modify the appearance of the graph, including menus, node sizes, and edge opacity:

.. code-block:: python

   g2 = g.scene_settings(
       # Hide menus
       menu=False,
       info=False,
       # Customize graph
       show_arrows=False,
       point_size=1.0,
       edge_curvature=0.0,
       edge_opacity=0.5,
       point_opacity=0.9
   ).plot()

Styling the Background and Foreground
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With `g.addStyle()`, you can configure background and foreground styles, including colors, gradients, and images:

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

Customize the page title, favicon, and logo using `g.addStyle()`:

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

For more advanced Python configuration options, refer to the [PyGraphistry API documentation](https://hub.graphistry.com/docs).

HTML/URL-based Configuration
--------------------------------

For users interested in configuring Graphistry visualizations through HTML and URL parameters, please refer to the official documentation:

- **[Graphistry URL Configuration Options](https://hub.graphistry.com/docs/api/1/rest/url/#urloptions)**

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
