.. _kepler-guide:

Kepler.gl Geographic Visualization
====================================

PyGraphistry integrates with Kepler.gl to enable powerful geographic visualizations of graph data with latitude/longitude coordinates. This feature allows you to overlay nodes and edges on interactive maps with customizable layers, colors, and effects.

Overview
--------

The Kepler integration provides:

- **Geographic bindings**: Specify lat/lon columns with ``point_longitude`` and ``point_latitude``
- **Encoding API**: Configure Kepler datasets, layers, and complete visualizations
- **Layout support**: Convert coordinates to Mercator projection with ``mercator_layout()``
- **Type-safe configuration**: Use ``KeplerDataset``, ``KeplerLayer``, and ``KeplerEncoding`` classes
- **GPU/CPU support**: Process geographic data efficiently on either platform

Quick Start
-----------

Basic geographic visualization:

.. code-block:: python

    import graphistry
    import pandas as pd

    # Sample data with coordinates
    nodes_df = pd.DataFrame({
        'nodeId': ['NYC', 'LA', 'Chicago'],
        'lat': [40.7128, 34.0522, 41.8781],
        'lon': [-74.0060, -118.2437, -87.6298]
    })

    edges_df = pd.DataFrame({
        'src': ['NYC', 'NYC'],
        'dst': ['LA', 'Chicago']
    })

    # Create geographic visualization
    g = graphistry.edges(edges_df, 'src', 'dst').nodes(nodes_df, 'nodeId')
    g = g.bind(point_longitude='lon', point_latitude='lat')
    g.plot()

Examples
--------

Point Layer Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize nodes as points on a map:

.. code-block:: python

    # Configure dataset and point layer
    g = (g
        .encode_kepler_dataset(id='nodes', type='nodes', label='Cities')
        .encode_kepler_layer({
            'id': 'node-layer',
            'type': 'point',
            'config': {
                'dataId': 'nodes',
                'label': 'Cities',
                'color': [255, 0, 0],
                'columns': {
                    'lat': 'latitude',
                    'lng': 'longitude'
                }
            }
        })
    )
    g.plot()

Arc Layer for Edges
~~~~~~~~~~~~~~~~~~~

Visualize edges as arcs between locations:

.. code-block:: python

    # Configure dataset and arc layer for edges
    g = (g
        .encode_kepler_dataset(id='edges', type='edges', label='Routes')
        .encode_kepler_layer({
            'id': 'edge-layer',
            'type': 'arc',
            'config': {
                'dataId': 'edges',
                'label': 'Routes',
                'color': [0, 255, 0],
                'columns': {
                    'lat0': 'src_lat',
                    'lng0': 'src_lon',
                    'lat1': 'dst_lat',
                    'lng1': 'dst_lon'
                }
            }
        })
    )
    g.plot()

Hexagon Grid Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregate points into hexagonal bins:

.. code-block:: python

    # Configure dataset and hexagon layer for density visualization
    g = (g
        .encode_kepler_dataset(id='nodes', type='nodes', label='Locations')
        .encode_kepler_layer({
            'id': 'density-layer',
            'type': 'hexagon',
            'config': {
                'dataId': 'nodes',
                'label': 'Density',
                'columns': {
                    'lat': 'latitude',
                    'lng': 'longitude'
                },
                'visConfig': {
                    'worldUnitSize': 1,
                    'elevationScale': 5
                }
            }
        })
    )
    g.plot()

Customizing Options and Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply visualization options and configuration settings:

.. code-block:: python

    from graphistry import KeplerOptions, KeplerConfig

    # Method 1: Using encode_kepler_options/config
    g = (g
        .encode_kepler_options(center_map=True, read_only=False)
        .encode_kepler_config(cull_unused_columns=True, overlay_blending='additive')
    )
    g.plot()

    # Method 2: Using KeplerEncoding builder
    from graphistry import KeplerEncoding

    encoding = (KeplerEncoding()
        .with_options(center_map=True, read_only=False)
        .with_config(cull_unused_columns=True, overlay_blending='additive')
    )
    g = g.encode_kepler(encoding)
    g.plot()

    # Method 3: Using KeplerOptions/Config objects
    opts = KeplerOptions(center_map=True, read_only=False)
    cfg = KeplerConfig(cull_unused_columns=True, overlay_blending='additive')

    encoding = KeplerEncoding(options=opts, config=cfg)
    g = g.encode_kepler(encoding)
    g.plot()

Complete Configuration
~~~~~~~~~~~~~~~~~~~~~~

Apply full Kepler configuration with multiple layers, options, and config:

.. code-block:: python

    from graphistry import KeplerEncoding, KeplerDataset, KeplerLayer, KeplerOptions, KeplerConfig

    # Build complete configuration
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
                    'lat0': 'src_lat',
                    'lng0': 'src_lon',
                    'lat1': 'dst_lat',
                    'lng1': 'dst_lon'
                }
            }
        }))
        .with_options(center_map=True, read_only=False)
        .with_config(cull_unused_columns=True, overlay_blending='normal')
    )

    g = g.encode_kepler(config)
    g.plot()

GPU/CPU Performance
-------------------

When using ``mercator_layout()`` for custom workflows, it automatically uses GPU (CuPy) acceleration if available, otherwise falls back to CPU (pandas) processing:

.. code-block:: python

    # Automatic GPU/CPU selection
    g = g.mercator_layout()

    # Control coordinate scaling
    g = g.mercator_layout(scale_for_graphistry=True)   # Scaled for visualization (default)
    g = g.mercator_layout(scale_for_graphistry=False)  # Standard Earth radius

For large datasets with geographic coordinates, GPU processing can significantly improve performance when generating Mercator projections.

See Also
--------

- :ref:`10 Minutes to Graphistry Visualization <10min-viz>`
- :ref:`Kepler API Reference <kepler-api>`
- :ref:`Mercator Layout API <mercator-api>`
- `Geospatial Network Visualization Notebook <../demos/more_examples/graphistry_features/layout_map.ipynb>`_

Further Reading
---------------

- `Kepler.gl Documentation <https://docs.kepler.gl/>`_
- `Kepler.gl Layer Types <https://docs.kepler.gl/docs/user-guides/b-kepler-gl-workflow/a-add-data-to-the-map>`_
