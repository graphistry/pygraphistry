.. _maps-guide:

Maps & Geographic Visualization
================================

PyGraphistry provides multiple approaches for visualizing geographic data, from simple latitude/longitude coordinates to interactive Kepler.gl maps.

Overview
--------

Choose the right approach for your use case:

1. **Latitude/Longitude Bindings** - Automatic server-side geographic layout
2. **Mercator Layout** - Client-side projection for local analysis
3. **Kepler.gl Integration** - Full-featured interactive maps with layers and styling

.. contents:: On this page
   :local:
   :depth: 2

Latitude/Longitude Bindings
----------------------------

The simplest approach: bind your lat/lon columns and let Graphistry handle the rest.

**When to use:**

- You have latitude/longitude data
- You want automatic geographic visualization
- You don't need custom projections

Basic Usage
~~~~~~~~~~~

::

    import graphistry
    import pandas as pd

    # Nodes with geographic coordinates
    cities = pd.DataFrame({
        'id': ['NYC', 'LA', 'London'],
        'name': ['New York', 'Los Angeles', 'London'],
        'latitude': [40.7128, 34.0522, 51.5074],
        'longitude': [-74.0060, -118.2437, -0.1278]
    })

    # Bind lat/lon columns
    g = (graphistry
         .nodes(cities, 'id')
         .bind(point_latitude='latitude', point_longitude='longitude')
         .layout_settings(play=0))

    g.plot()

Graphistry automatically detects the geographic bindings and applies server-side map layout.

**Default column names:** If your columns are named ``latitude`` and ``longitude``, bindings are detected automatically.

Edge Geographic Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For edges with geographic endpoints:

::

    flights = pd.DataFrame({
        'origin': ['NYC', 'LA', 'London'],
        'destination': ['LA', 'London', 'NYC']
    })

    g = (graphistry
         .nodes(cities, 'id')
         .edges(flights, 'origin', 'destination')
         .bind(point_latitude='latitude', point_longitude='longitude')
         .layout_settings(play=0))

    g.plot()

See Also
~~~~~~~~

- :ref:`10 Minutes to Graphistry Visualization <10min-viz>`
- API: :meth:`graphistry.PlotterBase.PlotterBase.bind`

Mercator Layout
---------------

Convert lat/lon to 2D Mercator projection coordinates locally.

**When to use:**

- You need projected coordinates for local analysis
- You want to export coordinates to other tools
- You need GPU-accelerated projection (cuDF support)

Basic Usage
~~~~~~~~~~~

::

    import graphistry
    import pandas as pd

    cities = pd.DataFrame({
        'id': ['NYC', 'LA', 'London'],
        'latitude': [40.7128, 34.0522, 51.5074],
        'longitude': [-74.0060, -118.2437, -0.1278]
    })

    # Apply Mercator projection
    g = graphistry.nodes(cities, 'id').mercator_layout()

    # Coordinates now in 'x' and 'y' columns
    print(g._nodes[['id', 'x', 'y']])

Custom Column Names
~~~~~~~~~~~~~~~~~~~

::

    cities = pd.DataFrame({
        'id': ['NYC', 'LA', 'London'],
        'lat': [40.7128, 34.0522, 51.5074],
        'lon': [-74.0060, -118.2437, -0.1278]
    })

    g = (graphistry
         .nodes(cities, 'id')
         .bind(point_latitude='lat', point_longitude='lon')
         .mercator_layout())

Scaling Options
~~~~~~~~~~~~~~~

**Scaled mode (default):** Optimized for Graphistry visualization

::

    g = g.mercator_layout(scale_for_graphistry=True)  # Default

**Unscaled mode:** Standard Earth radius for geographic accuracy

::

    g = g.mercator_layout(scale_for_graphistry=False)

GPU Acceleration
~~~~~~~~~~~~~~~~

Mercator layout automatically uses GPU acceleration (CuPy) when available:

::

    import cudf
    import graphistry

    # cuDF DataFrame
    cities_gpu = cudf.DataFrame({
        'id': ['NYC', 'LA', 'London'],
        'latitude': [40.7128, 34.0522, 51.5074],
        'longitude': [-74.0060, -118.2437, -0.1278]
    })

    # Automatically uses GPU-accelerated projection
    g = graphistry.nodes(cities_gpu, 'id').mercator_layout()

See Also
~~~~~~~~

- API: :ref:`Mercator Layout API <mercator-api>`

Kepler.gl Integration
---------------------

Full-featured interactive maps with multiple layers, styling, and native Kepler.gl controls.

**When to use:**

- You want rich interactive map visualizations
- You need fine-grained control over map styling
- You're visualizing geographic regions (countries, states)
- You need multiple map layers with different visualizations

**What it provides:**

- **Full Kepler.gl passthrough**: Direct access to all native Kepler layers and configurations
- **Graphistry dataset shortcuts**: Simplified dataset creation from Graphistry nodes, edges, and geographic data
- **Type-safe configuration**: Use ``KeplerDataset``, ``KeplerLayer``, and ``KeplerEncoding`` classes

Quick Start
~~~~~~~~~~~

::

    import graphistry
    from graphistry import KeplerLayer
    import pandas as pd

    cities = pd.DataFrame({
        'id': ['NYC', 'LA', 'London'],
        'latitude': [40.7128, 34.0522, 51.5074],
        'longitude': [-74.0060, -118.2437, -0.1278]
    })

    g = (graphistry
         .nodes(cities, 'id')
         .bind(point_latitude='latitude', point_longitude='longitude')
         .encode_kepler_dataset(id="cities", type="nodes")
         .encode_kepler_layer(KeplerLayer({
             "id": "city-points",
             "type": "point",
             "config": {
                 "dataId": "cities",
                 "columns": {"lat": "latitude", "lng": "longitude"},
                 "visConfig": {"radius": 10, "opacity": 0.8}
             }
         }))
         .layout_settings(play=0))

    g.plot()

Point Layers
~~~~~~~~~~~~

Visualize nodes as points on a map with explicit layer configuration:

::

    cities = pd.DataFrame({
        'city': ['NYC', 'LA', 'London', 'Paris', 'Tokyo'],
        'latitude': [40.7128, 34.0522, 51.5074, 48.8566, 35.6762],
        'longitude': [-74.0060, -118.2437, -0.1278, 2.3522, 139.6503]
    })

    g = (graphistry
        .nodes(cities, 'city')
        .encode_kepler_dataset(id='nodes', type='nodes', label='Cities')
        .encode_kepler_layer(KeplerLayer({
            'id': 'node-layer',
            'type': 'point',
            'config': {
                'dataId': 'nodes',
                'label': 'Cities',
                'color': [255, 0, 0],
                'columns': {'lat': 'latitude', 'lng': 'longitude'}
            }
        }))
        .layout_settings(play=0))
    g.plot()

Arc Layers for Edges
~~~~~~~~~~~~~~~~~~~~~

Visualize edges as arcs between locations:

::

    flights = pd.DataFrame({
        'origin': ['NYC', 'LA'],
        'destination': ['LA', 'London']
    })

    g = (graphistry
         .nodes(cities, 'id')
         .edges(flights, 'origin', 'destination')
         .bind(point_latitude='latitude', point_longitude='longitude')
         .encode_kepler_dataset(id="cities", type="nodes")
         .encode_kepler_dataset(id="flights", type="edges", map_node_coords=True)
         .encode_kepler_layer(KeplerLayer({
             "id": "points",
             "type": "point",
             "config": {
                 "dataId": "cities",
                 "columns": {"lat": "latitude", "lng": "longitude"}
             }
         }))
         .encode_kepler_layer(KeplerLayer({
             "id": "arcs",
             "type": "arc",
             "config": {
                 "dataId": "flights",
                 "columns": {
                     "lat0": "edgeSourceLatitude",
                     "lng0": "edgeSourceLongitude",
                     "lat1": "edgeTargetLatitude",
                     "lng1": "edgeTargetLongitude"
                 }
             }
         }))
         .layout_settings(play=0))
    g.plot()

Hexagon Aggregation
~~~~~~~~~~~~~~~~~~~

Aggregate points into hexagonal bins for density visualization:

::

    locations = pd.DataFrame({
        'location': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'],
        'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
        'longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740]
    })

    g = (graphistry
        .nodes(locations, 'location')
        .encode_kepler_dataset(id='nodes', type='nodes', label='Locations')
        .encode_kepler_layer(KeplerLayer({
            'id': 'density-layer',
            'type': 'hexagon',
            'config': {
                'dataId': 'nodes',
                'label': 'Density',
                'columns': {'lat': 'latitude', 'lng': 'longitude'},
                'visConfig': {
                    'worldUnitSize': 1,
                    'elevationScale': 5
                }
            }
        }))
        .layout_settings(play=0))
    g.plot()

Geographic Regions
~~~~~~~~~~~~~~~~~~

Visualize countries and states with built-in geographic data:

::

    countries = pd.DataFrame({
        'country': ['USA', 'GBR', 'FRA'],
        'gdp': [21.43, 2.83, 2.72]
    })

    g = (graphistry
         .nodes(countries, 'country')
         .encode_kepler_dataset(
             id="countries",
             type="countries",
             resolution=10,  # High resolution
             filter_countries_by_col="country"
         )
         .encode_kepler_layer(KeplerLayer({
             "id": "choropleth",
             "type": "geojson",
             "config": {
                 "dataId": "countries",
                 "columns": {"geojson": "_geojson"}
             }
         })))

Options and Config
~~~~~~~~~~~~~~~~~~

Control map behavior and appearance (continuing from examples above):

::

    # Method 1: Direct parameters
    g = (g
         .encode_kepler_options(center_map=True, read_only=False)
         .encode_kepler_config(cull_unused_columns=True, overlay_blending='additive'))

    # Method 2: Using KeplerEncoding builder
    from graphistry import KeplerEncoding

    encoding = (KeplerEncoding()
        .with_options(center_map=True, read_only=False)
        .with_config(cull_unused_columns=True, overlay_blending='additive'))
    g2 = g.encode_kepler(encoding)

    # Method 3: Using KeplerOptions/Config objects
    from graphistry import KeplerOptions, KeplerConfig

    opts = KeplerOptions(center_map=True, read_only=False)
    cfg = KeplerConfig(cull_unused_columns=True, overlay_blending='additive')
    encoding = KeplerEncoding(options=opts, config=cfg)
    g3 = g.encode_kepler(encoding)

Complete Configuration
~~~~~~~~~~~~~~~~~~~~~~

Build full Kepler configuration with multiple datasets, layers, options, and config:

::

    from graphistry import KeplerEncoding, KeplerDataset, KeplerLayer

    cities = pd.DataFrame({
        'city': ['NYC', 'LA', 'London', 'Paris', 'Tokyo'],
        'latitude': [40.7128, 34.0522, 51.5074, 48.8566, 35.6762],
        'longitude': [-74.0060, -118.2437, -0.1278, 2.3522, 139.6503]
    })

    routes = pd.DataFrame({
        'origin': ['NYC', 'LA', 'London'],
        'destination': ['LA', 'London', 'Tokyo']
    })

    config = (
        KeplerEncoding()
        .with_dataset(KeplerDataset(id='nodes', type='nodes', label='Cities'))
        .with_dataset(KeplerDataset(id='edges', type='edges', label='Routes'))
        .with_layer(KeplerLayer({
            'id': 'node-layer',
            'type': 'point',
            'config': {
                'dataId': 'nodes',
                'columns': {'lat': 'latitude', 'lng': 'longitude'}
            }
        }))
        .with_layer(KeplerLayer({
            'id': 'edge-layer',
            'type': 'arc',
            'config': {
                'dataId': 'edges',
                'columns': {
                    'lat0': 'edgeSourceLatitude', 'lng0': 'edgeSourceLongitude',
                    'lat1': 'edgeTargetLatitude', 'lng1': 'edgeTargetLongitude'
                }
            }
        }))
        .with_options(center_map=True, read_only=False)
        .with_config(cull_unused_columns=True, overlay_blending='normal')
    )

    g = (graphistry
         .nodes(cities, 'city')
         .edges(routes, 'origin', 'destination')
         .encode_kepler(config)
         .layout_settings(play=0))
    g.plot()

See Also
~~~~~~~~

- **API reference:** :ref:`Kepler API <kepler-api>`
- **Configuration classes:** :class:`graphistry.kepler.KeplerEncoding`, :class:`graphistry.kepler.KeplerDataset`, :class:`graphistry.kepler.KeplerLayer`
- **External docs:** `Kepler.gl Documentation <https://docs.kepler.gl/>`_

Comparison
----------

.. list-table:: Geographic Visualization Approaches
   :header-rows: 1
   :widths: 20 25 25 30

   * - Approach
     - Best For
     - Complexity
     - Key Features
   * - **Lat/Lon Bindings**
     - Quick geographic viz
     - Simplest
     - Automatic server-side layout
   * - **Mercator Layout**
     - Local coordinate analysis
     - Simple
     - GPU support, exportable coords
   * - **Kepler.gl**
     - Rich interactive maps
     - Advanced
     - Multiple layers, styling, regions

Examples
--------

See the Map Layout Demo notebook for complete examples:

- `Map Layout Demo <../../demos/more_examples/graphistry_features/layout_map.ipynb>`_

API Reference
-------------

- :ref:`Mercator Layout API <mercator-api>`
- :ref:`Kepler API Reference <kepler-api>`
- :meth:`graphistry.PlotterBase.PlotterBase.bind` (lat/lon bindings)
