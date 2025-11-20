.. _kepler-layer-format:

Kepler.gl Layer Format
=======================

Native Kepler.gl layer configuration format reference.

When using ``KeplerLayer(raw_dict={...})``, the dictionary should follow the native Kepler.gl layer structure documented here.

Layer Structure
---------------

A Kepler.gl layer configuration has the following structure:

.. code-block:: python

    {
        "id": "layer-id",              # Layer identifier
        "type": "point",               # Layer type (see below)
        "config": {
            "dataId": "dataset-id",    # Dataset this layer references
            "label": "Layer Label",    # Display label
            "color": [255, 0, 0],      # RGB color array
            "columns": {...},          # Column mappings (type-specific)
            "isVisible": True,         # Visibility toggle
            "visConfig": {...}         # Visual configuration (type-specific)
        },
        "visualChannels": {...}        # Optional: data-driven styling
    }

Layer Types
-----------

Point Layer
~~~~~~~~~~~

Displays individual points on the map.

.. code-block:: python

    {
        "type": "point",
        "config": {
            "columns": {
                "lat": "latitude",      # Latitude column
                "lng": "longitude"      # Longitude column
            },
            "visConfig": {
                "radius": 10,           # Point radius
                "opacity": 0.8,         # Opacity (0-1)
                "outline": False,       # Show outline
                "thickness": 2          # Outline thickness
            }
        }
    }

Arc Layer
~~~~~~~~~

Displays curved lines connecting two points (great circle arcs).

.. code-block:: python

    {
        "type": "arc",
        "config": {
            "columns": {
                "lat0": "src_lat",      # Source latitude
                "lng0": "src_lon",      # Source longitude
                "lat1": "dst_lat",      # Target latitude
                "lng1": "dst_lon"       # Target longitude
            },
            "visConfig": {
                "opacity": 0.3,         # Opacity (0-1)
                "thickness": 2          # Arc thickness
            }
        }
    }

Line Layer
~~~~~~~~~~

Displays straight lines between points.

.. code-block:: python

    {
        "type": "line",
        "config": {
            "columns": {
                "lat0": "src_lat",
                "lng0": "src_lon",
                "lat1": "dst_lat",
                "lng1": "dst_lon"
            },
            "visConfig": {
                "opacity": 0.8,
                "thickness": 2
            }
        }
    }

Hexagon Layer
~~~~~~~~~~~~~

Aggregates points into hexagonal bins.

.. code-block:: python

    {
        "type": "hexagon",
        "config": {
            "columns": {
                "lat": "latitude",
                "lng": "longitude"
            },
            "visConfig": {
                "opacity": 0.8,
                "worldUnitSize": 1,      # Hexagon size in km
                "resolution": 8,         # H3 resolution (0-15)
                "coverage": 1,           # Coverage ratio (0-1)
                "enable3d": True,        # Enable 3D extrusion
                "elevationScale": 5,     # Height multiplier
                "colorRange": {...}      # Color palette
            }
        }
    }

Grid Layer
~~~~~~~~~~

Aggregates points into square grid cells.

.. code-block:: python

    {
        "type": "grid",
        "config": {
            "columns": {
                "lat": "latitude",
                "lng": "longitude"
            },
            "visConfig": {
                "opacity": 0.8,
                "worldUnitSize": 1,      # Cell size in km
                "colorRange": {...},
                "coverage": 1,
                "enable3d": True,
                "elevationScale": 5
            }
        }
    }

GeoJSON Layer
~~~~~~~~~~~~~

Displays polygon/multipolygon geometries.

.. code-block:: python

    {
        "type": "geojson",
        "config": {
            "columns": {
                "geojson": "geometry"    # GeoJSON geometry column
            },
            "visConfig": {
                "opacity": 0.8,
                "filled": True,          # Fill polygons
                "stroked": True,         # Show outline
                "thickness": 1,          # Outline thickness
                "strokeColor": [255, 255, 255],
                "colorRange": {...},
                "radius": 10,            # For point features
                "elevationScale": 1,     # Height multiplier
                "enable3d": False
            }
        }
    }

Heatmap Layer
~~~~~~~~~~~~~

Displays data as a continuous heat distribution.

.. code-block:: python

    {
        "type": "heatmap",
        "config": {
            "columns": {
                "lat": "latitude",
                "lng": "longitude"
            },
            "visConfig": {
                "opacity": 0.8,
                "radius": 20,            # Influence radius in pixels
                "intensity": 1,          # Heat intensity
                "threshold": 0.05,       # Visibility threshold
                "colorRange": {...}
            }
        }
    }

Cluster Layer
~~~~~~~~~~~~~

Clusters nearby points together.

.. code-block:: python

    {
        "type": "cluster",
        "config": {
            "columns": {
                "lat": "latitude",
                "lng": "longitude"
            },
            "visConfig": {
                "opacity": 0.8,
                "clusterRadius": 40,     # Cluster radius in pixels
                "colorRange": {...}
            }
        }
    }

Icon Layer
~~~~~~~~~~

Displays custom icons at point locations.

.. code-block:: python

    {
        "type": "icon",
        "config": {
            "columns": {
                "lat": "latitude",
                "lng": "longitude",
                "icon": "icon_name"      # Optional icon name column
            },
            "visConfig": {
                "radius": 10,            # Icon size
                "opacity": 0.8
            }
        }
    }

Trip Layer
~~~~~~~~~~

Displays animated paths/trips over time.

.. code-block:: python

    {
        "type": "trip",
        "config": {
            "columns": {
                "geojson": "path"        # GeoJSON LineString
            },
            "visConfig": {
                "opacity": 0.8,
                "thickness": 2,
                "trailLength": 180       # Trail length in seconds
            }
        }
    }

Visual Channels
---------------

Visual channels enable data-driven styling:

.. code-block:: python

    {
        "visualChannels": {
            "colorField": {
                "name": "column_name",
                "type": "real"
            },
            "colorScale": "quantile",    # quantile, quantize, ordinal
            "sizeField": {
                "name": "size_column",
                "type": "real"
            },
            "sizeScale": "linear"        # linear, sqrt, log
        }
    }

Complete Example
----------------

.. code-block:: python

    from graphistry import KeplerLayer

    layer = KeplerLayer({
        "id": "city-points",
        "type": "point",
        "config": {
            "dataId": "cities",
            "label": "City Locations",
            "color": [255, 140, 0],
            "columns": {
                "lat": "latitude",
                "lng": "longitude"
            },
            "isVisible": True,
            "visConfig": {
                "radius": 10,
                "opacity": 0.8,
                "outline": True,
                "thickness": 2
            }
        },
        "visualChannels": {
            "colorField": {
                "name": "population",
                "type": "real"
            },
            "colorScale": "quantile",
            "sizeField": {
                "name": "gdp",
                "type": "real"
            },
            "sizeScale": "sqrt"
        }
    })

See Also
--------

- :ref:`kepler-layer-api`: KeplerLayer class API
- :ref:`kepler-dataset-format`: Kepler.gl dataset format
- `Kepler.gl Layer Documentation <https://docs.kepler.gl/docs/user-guides/b-kepler-gl-workflow/a-add-data-to-the-map>`_
- `Kepler.gl Layer Types <https://docs.kepler.gl/docs/api-reference/layers>`_
