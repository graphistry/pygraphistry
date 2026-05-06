.. _kepler-encoding-api:

KeplerEncoding
==============

Immutable container for complete Kepler.gl configuration.

.. autoclass:: graphistry.kepler.KeplerEncoding
   :members:
   :undoc-members:
   :show-inheritance:

Examples
-------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from graphistry import KeplerEncoding, KeplerDataset, KeplerLayer

    # Start with empty encoding
    encoding = KeplerEncoding()

    # Add dataset
    encoding = encoding.with_dataset(
        KeplerDataset(id="nodes", type="nodes", label="Companies")
    )

    # Add layer
    encoding = encoding.with_layer(
        KeplerLayer({
            "id": "point-layer",
            "type": "point",
            "config": {
                "dataId": "nodes",
                "columns": {"lat": "latitude", "lng": "longitude"}
            }
        })
    )

    # Configure options
    encoding = encoding.with_options(center_map=True, read_only=False)

    # Apply to graph
    g = g.encode_kepler(encoding)

Chained Builder Pattern
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from graphistry import KeplerEncoding, KeplerDataset, KeplerLayer

    # Chain multiple operations
    encoding = (
        KeplerEncoding()
        .with_dataset(KeplerDataset(id="companies", type="nodes"))
        .with_dataset(KeplerDataset(id="relationships", type="edges"))
        .with_layer(KeplerLayer({
            "id": "nodes",
            "type": "point",
            "config": {"dataId": "companies", "columns": {"lat": "latitude", "lng": "longitude"}}
        }))
        .with_layer(KeplerLayer({
            "id": "edges",
            "type": "arc",
            "config": {
                "dataId": "relationships",
                "columns": {
                    "lat0": "edgeSourceLatitude",
                    "lng0": "edgeSourceLongitude",
                    "lat1": "edgeTargetLatitude",
                    "lng1": "edgeTargetLongitude"
                }
            }
        }))
        .with_options(center_map=True)
        .with_config(cull_unused_columns=False)
    )

    g = g.encode_kepler(encoding)

Constructor Pattern
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from graphistry import KeplerEncoding, KeplerDataset, KeplerLayer

    # Pass all components to constructor
    encoding = KeplerEncoding(
        datasets=[
            KeplerDataset(id="nodes", type="nodes", label="Companies"),
            KeplerDataset(id="edges", type="edges", label="Relationships")
        ],
        layers=[
            KeplerLayer({"id": "point-layer", "type": "point", "config": {...}}),
            KeplerLayer({"id": "arc-layer", "type": "arc", "config": {...}})
        ],
        options=KeplerOptions(center_map=True, read_only=False),
        config=KeplerConfig(cull_unused_columns=False)
    )

    g = g.encode_kepler(encoding)

Multi-Layer Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create complex visualization with multiple layers
    encoding = (
        KeplerEncoding()

        # Datasets
        .with_dataset(KeplerDataset(id="companies", type="nodes"))
        .with_dataset(KeplerDataset(id="relationships", type="edges", map_node_coords=True))
        .with_dataset(KeplerDataset(
            id="countries",
            type="countries",
            resolution=50,
            computed_columns={
                "avg_revenue": {
                    "type": "aggregate",
                    "computeFromDataset": "companies",
                    "sourceKey": "country",
                    "targetKey": "name",
                    "aggregate": "mean",
                    "aggregateCol": "revenue"
                }
            }
        ))

        # Layers
        .with_layer(KeplerLayer({
            "id": "companies-points",
            "type": "point",
            "config": {
                "dataId": "companies",
                "label": "Company Locations",
                "columns": {"lat": "latitude", "lng": "longitude"},
                "isVisible": True,
                "color": [255, 140, 0]
            }
        }))
        .with_layer(KeplerLayer({
            "id": "relationships-arcs",
            "type": "arc",
            "config": {
                "dataId": "relationships",
                "label": "Business Relationships",
                "columns": {
                    "lat0": "edgeSourceLatitude",
                    "lng0": "edgeSourceLongitude",
                    "lat1": "edgeTargetLatitude",
                    "lng1": "edgeTargetLongitude"
                },
                "isVisible": False,
                "color": [100, 200, 200],
                "visConfig": {"opacity": 0.2}
            }
        }))
        .with_layer(KeplerLayer({
            "id": "countries-geojson",
            "type": "geojson",
            "config": {
                "dataId": "countries",
                "label": "Countries by Avg Revenue",
                "columns": {"geojson": "_geometry"},
                "isVisible": True
            },
            "visualChannels": {
                "colorField": {"name": "avg_revenue", "type": "real"},
                "colorScale": "quantile"
            }
        }))

        # Options and config
        .with_options(center_map=True, read_only=False)
        .with_config(cull_unused_columns=False, overlay_blending="additive")
    )

    g = g.encode_kepler(encoding)

Serialization
~~~~~~~~~~~~~

.. code-block:: python

    # Convert to dictionary for inspection or manual editing
    encoding_dict = encoding.to_dict()

    # Result structure:
    # {
    #     'datasets': [...],
    #     'layers': [...],
    #     'options': {...},
    #     'config': {...}
    # }

See Also
--------

- :ref:`kepler-dataset-api`: Dataset configuration
- :ref:`kepler-layer-api`: Layer configuration
- :ref:`maps-guide`: User guide with examples
- :ref:`kepler-methods-api`: Plotter methods (encode_kepler, encode_kepler_dataset, encode_kepler_layer)
