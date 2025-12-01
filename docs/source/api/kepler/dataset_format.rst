.. _kepler-dataset-format:

Kepler.gl Dataset Format
=========================

Native Kepler.gl dataset configuration format reference.

When using ``KeplerDataset(raw_dict={...})``, the dictionary should follow the native Kepler.gl dataset structure documented here.

Dataset Structure
-----------------

A Kepler.gl dataset configuration has the following structure:

.. code-block:: python

    {
        "version": "v1",
        "data": {
            "id": "dataset-id",          # Dataset identifier
            "label": "Dataset Label",    # Display label
            "color": [255, 0, 0],        # Optional RGB color
            "allData": [...],            # Array of data rows (optional)
            "fields": [...]              # Array of field definitions (optional)
        }
    }

Fields Definition
-----------------

Each field in the ``fields`` array describes a column:

.. code-block:: python

    {
        "name": "column_name",        # Column name
        "type": "string",             # Data type: string, integer, real, boolean, timestamp, geometry
        "format": "",                 # Optional format string
        "analyzerType": "STRING"      # Analyzer type: STRING, INT, FLOAT, BOOLEAN, DATE, GEOMETRY
    }

Common Field Types
~~~~~~~~~~~~~~~~~~

- ``string`` / ``STRING``: Text data
- ``integer`` / ``INT``: Integer numbers
- ``real`` / ``FLOAT``: Floating point numbers
- ``boolean`` / ``BOOLEAN``: True/false values
- ``timestamp`` / ``DATE``: Temporal data
- ``geometry`` / ``GEOMETRY``: Spatial data (GeoJSON)

Examples
--------

Basic Dataset
~~~~~~~~~~~~~

.. code-block:: python

    from graphistry import KeplerDataset

    dataset = KeplerDataset({
        "version": "v1",
        "data": {
            "id": "cities",
            "label": "City Locations",
            "color": [255, 140, 0]
        }
    })

Dataset with Fields
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset = KeplerDataset({
        "version": "v1",
        "data": {
            "id": "points",
            "label": "Points of Interest",
            "fields": [
                {"name": "name", "type": "string", "analyzerType": "STRING"},
                {"name": "latitude", "type": "real", "analyzerType": "FLOAT"},
                {"name": "longitude", "type": "real", "analyzerType": "FLOAT"},
                {"name": "count", "type": "integer", "analyzerType": "INT"},
                {"name": "timestamp", "type": "timestamp", "analyzerType": "DATE"}
            ]
        }
    })

See Also
--------

- :ref:`kepler-dataset-api`: KeplerDataset class API
- :ref:`kepler-layer-format`: Kepler.gl layer format
- `Kepler.gl addDataToMap API <https://docs.kepler.gl/docs/api-reference/actions/actions#adddatatomap>`_
