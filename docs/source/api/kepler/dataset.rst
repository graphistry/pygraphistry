.. _kepler-dataset-api:

KeplerDataset
=============

Configuration class for Kepler.gl datasets.

.. autoclass:: graphistry.kepler.KeplerDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   For the native Kepler.gl dataset format when using ``raw_dict``, see :ref:`kepler-dataset-format`.

Computed Columns
----------------

``computed_columns`` (dict, optional)
  Define computed columns for data enrichment. Each computed column is added as a new
  column to the current dataset (the dataset where ``computed_columns`` is defined). The key in the
  dictionary becomes the new column name.

  Structure:

  .. code-block:: python

      {
          "new_column_name": {          # The key becomes the new column name in THIS dataset
              "type": "aggregate",              # Aggregation type
              "computeFromDataset": "source_dataset_id",
              "sourceKey": "join_column",       # Column in source dataset
              "targetKey": "join_column",       # Column in target (this) dataset
              "aggregate": "mean",              # Aggregation function: mean, sum, min, max, count
              "aggregateCol": "value_column",   # Column to aggregate
              "normalizer": "mean",             # Optional: normalize by another aggregation
              "normalizerCol": "divisor_col",   # Optional: column for normalization
              "bins": [0, 1, 2, 5, 10],        # Optional: bin continuous values
              "right": False,                   # Optional: bin right-inclusivity
              "includeLowest": True             # Optional: include lowest bin edge
          }
      }

  Example: A countries dataset can create ``avg_revenue`` by aggregating company revenue via country name.

  **Computed Column Fields:**

  - ``type`` (str): Currently supports "aggregate"
  - ``computeFromDataset`` (str): ID of the dataset to aggregate from (the source)
  - ``sourceKey`` (str): Join column in the source dataset
  - ``targetKey`` (str): Join column in the target dataset (this dataset)
  - ``aggregate`` (str): Aggregation function name as string. Common options: "mean", "sum", "min", "max", "count", "std", "var", "median", "first", "last", "prod", "nunique". See `cuDF groupby aggregation docs <https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/groupby.html>`_ for full list.
  - ``aggregateCol`` (str): Column name to aggregate from the source dataset
  - ``normalizer`` (str, optional): Secondary aggregation function for normalization (e.g., divide mean by mean). Uses same aggregation function names as ``aggregate``.
  - ``normalizerCol`` (str, optional): Column for normalization denominator
  - ``bins`` (List[float], optional): Bin edges for discretizing continuous values
  - ``right`` (bool, optional): Whether bins are right-inclusivity
  - ``includeLowest`` (bool, optional): Whether to include the lowest bin edge

Example
~~~~~~~

.. code-block:: python

    # Aggregate data from another dataset
    countries_with_stats = KeplerDataset(
        id="countries-stats",
        type="countries",
        resolution=110,
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
    )

See Also
--------

- :ref:`kepler-dataset-format`: Native Kepler.gl dataset format reference
- :ref:`kepler-layer-api`: Layer configuration
- :ref:`kepler-encoding-api`: Complete Kepler configuration
- :ref:`maps-guide`: User guide with examples
