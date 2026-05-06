.. _kepler-api:

Kepler API Reference
=====================

Geographic visualization with Kepler.gl integration.

The Kepler module provides classes and methods for creating interactive map-based visualizations
using Kepler.gl. This includes configuration for datasets, layers, and complete geographic visualizations.

.. note::
   Graphistry automatically performs geographic visualization when ``point_longitude`` and ``point_latitude``
   are bound. The Kepler API provides additional control for customizing datasets, layers, and map styling.

Quick Links
-----------

- :ref:`maps-guide`: Maps & geographic visualization guide
- :ref:`mercator-api`: Mercator projection layout (optional)
- `Kepler.gl Documentation <https://docs.kepler.gl/>`_

API Components
--------------

Configuration Classes
~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   dataset
   layer
   options
   config
   encoding

Plotter Methods
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   methods

Native Formats
~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   dataset_format
   layer_format

Overview
--------

Geographic Bindings
~~~~~~~~~~~~~~~~~~~

Bind geographic coordinate columns to enable automatic Kepler rendering:

.. code-block:: python

    g = g.bind(
        point_longitude='longitude_column',
        point_latitude='latitude_column'
    )
    g.plot()  # Automatically renders as geographic visualization

Configuration Classes
~~~~~~~~~~~~~~~~~~~~~

- :ref:`KeplerDataset <kepler-dataset-api>`: Configure datasets (nodes, edges, countries, states)
- :ref:`KeplerLayer <kepler-layer-api>`: Configure visualization layers (point, arc, hexagon, etc.)
- :ref:`KeplerOptions <kepler-options-api>`: Configure visualization options (map centering, interactions)
- :ref:`KeplerConfig <kepler-config-api>`: Configure map settings (blending, tile style)
- :ref:`KeplerEncoding <kepler-encoding-api>`: Complete Kepler configuration container

Plotter Methods
~~~~~~~~~~~~~~~

- :ref:`encode_kepler() <kepler-methods-api>`: Apply complete Kepler configuration
- :ref:`encode_kepler_dataset() <kepler-methods-api>`: Add a dataset
- :ref:`encode_kepler_layer() <kepler-methods-api>`: Add a visualization layer
- :ref:`encode_kepler_options() <kepler-methods-api>`: Apply visualization options
- :ref:`encode_kepler_config() <kepler-methods-api>`: Apply map configuration

Layout Methods
~~~~~~~~~~~~~~

- :ref:`mercator_layout() <mercator-api>`: Convert lat/lon to Mercator projection (optional)

Basic Usage Example
-------------------

.. code-block:: python

    import graphistry
    import pandas as pd

    # Prepare data
    nodes_df = pd.DataFrame({
        'id': ['NYC', 'LA', 'Chicago'],
        'lat': [40.7128, 34.0522, 41.8781],
        'lon': [-74.0060, -118.2437, -87.6298]
    })

    # Automatic geographic visualization
    g = graphistry.nodes(nodes_df, 'id')
    g = g.bind(point_longitude='lon', point_latitude='lat')
    g.plot()

Advanced Configuration Example
-------------------------------

.. code-block:: python

    from graphistry import KeplerEncoding, KeplerDataset, KeplerLayer

    # Build custom configuration
    encoding = (
        KeplerEncoding()
        .with_dataset(KeplerDataset(id="cities", type="nodes", label="Cities"))
        .with_dataset(KeplerDataset(id="routes", type="edges", label="Routes"))
        .with_layer(KeplerLayer(
            id="city-points",
            type="point",
            config={
                "dataId": "cities",
                "columns": {"lat": "latitude", "lng": "longitude"},
                "color": [255, 140, 0],
                "visConfig": {"radius": 10, "opacity": 0.8}
            }
        ))
        .with_layer(KeplerLayer(
            id="route-arcs",
            type="arc",
            config={
                "dataId": "routes",
                "columns": {
                    "lat0": "edgeSourceLatitude",
                    "lng0": "edgeSourceLongitude",
                    "lat1": "edgeTargetLatitude",
                    "lng1": "edgeTargetLongitude"
                },
                "color": [0, 200, 255],
                "visConfig": {"opacity": 0.3}
            }
        ))
        .with_options(centerMap=True)
    )

    # Apply to graph
    g = g.encode_kepler(encoding)
    g.plot()

See Also
--------

- :ref:`maps-guide`: Maps & geographic visualization guide
- :ref:`mercator-api`: Mercator projection layout
- :ref:`layout-catalog`: Layout catalog with all available layouts
