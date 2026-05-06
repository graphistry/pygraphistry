"""
Mercator layout: Convert latitude/longitude coordinates to Mercator projection.

This module provides the mercator_layout method for Plotter objects.
"""

import logging
import math
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable

logger = logging.getLogger(__name__)


def mercator_layout(self: 'Plottable', scale_for_graphistry: bool = True) -> 'Plottable':
    """
    Convert latitude/longitude coordinates to Mercator projection.

    Uses point_latitude and point_longitude bindings if set, otherwise defaults to 'latitude' and 'longitude' columns.
    Uses existing point_x and point_y bindings if available, otherwise defaults to 'x' and 'y'.
    Uses GPU acceleration via CuPy if available, falls back to CPU/pandas otherwise.

    Note: Graphistry automatically performs server-side geographic layout when latitude/longitude columns are detected.
    Use mercator_layout() when you need projected coordinates locally for analysis or need to export coordinates for use with other tools.

    :param scale_for_graphistry: If True (default), use scaled Earth radius (~637) for manageable coordinate values in Graphistry visualizations. If False, use standard Earth radius (~6,378,137 meters) for accurate geographic coordinates.
    :type scale_for_graphistry: bool

    :returns: Plottable with Mercator projection coordinates
    :rtype: Plottable

    **Example**
        ::

            import graphistry
            import pandas as pd

            # Using default column names 'latitude' and 'longitude'
            nodes_df = pd.DataFrame({
                'id': ['NYC', 'LA', 'London'],
                'latitude': [40.7128, 34.0522, 51.5074],
                'longitude': [-74.0060, -118.2437, -0.1278]
            })
            g = graphistry.nodes(nodes_df, 'id').mercator_layout()

            # Or with custom column names
            nodes_df2 = pd.DataFrame({
                'id': ['NYC', 'LA', 'London'],
                'lat': [40.7128, 34.0522, 51.5074],
                'lon': [-74.0060, -118.2437, -0.1278]
            })
            g2 = (graphistry
                  .nodes(nodes_df2, 'id')
                  .bind(point_latitude='lat', point_longitude='lon')
                  .mercator_layout())
    """
    g = self

    # Use existing bindings if available, otherwise use defaults
    lat_col = g._point_latitude if g._point_latitude is not None else 'latitude'
    lon_col = g._point_longitude if g._point_longitude is not None else 'longitude'
    x_col = g._point_x if g._point_x is not None else 'x'
    y_col = g._point_y if g._point_y is not None else 'y'

    if g._nodes is None:
        raise ValueError("No nodes set yet")
    if lat_col not in g._nodes or lon_col not in g._nodes:
        raise ValueError(f'Did not find columns {lat_col} or {lon_col} in nodes. Use .bind(point_latitude="lat_col", point_longitude="lon_col") or ensure "latitude" and "longitude" columns exist')

    is_not_pandas = not isinstance(g._nodes, pd.DataFrame)

    # Earth radius: scaled for Graphistry visualization or standard for geographic accuracy
    if scale_for_graphistry:
        R = 637.8137  # Scaled down by 10000x from 6,378,137 meters for manageable numbers
    else:
        R = 6378137.0  # Standard Earth radius in meters (WGS84)

    use_cupy = False
    if is_not_pandas:
        try:
            import cupy as cp
            use_cupy = True
        except ImportError:
            logger.warning("cuDF DataFrame detected but cupy is not available. Falling back to NumPy (CPU). Install cupy for GPU-accelerated computation.")

    if use_cupy:
        lat_deg = g._nodes[lat_col]
        lon_deg = g._nodes[lon_col]

        lat_rad = cp.radians(lat_deg.values)
        lon_rad = cp.radians(lon_deg.values)

        # Mercator projection formulas (vectorized)
        x_vals = R * lon_rad
        y_vals = R * cp.log(cp.tan(cp.pi / 4 + lat_rad / 2))

    else:
        import numpy as np

        lat_deg = g._nodes[lat_col].values
        lon_deg = g._nodes[lon_col].values

        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)

        # Mercator projection formulas (vectorized)
        x_vals = R * lon_rad
        y_vals = R * np.log(np.tan(np.pi / 4 + lat_rad / 2))

    g2 = g.nodes(
        g._nodes.assign(**{
            x_col: x_vals,
            y_col: y_vals
        })
    )

    return g2.bind(point_x=x_col, point_y=y_col).layout_settings(play=0)
