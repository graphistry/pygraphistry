"""
Mercator layout: Convert latitude/longitude coordinates to Mercator projection.

This module provides the mercator_layout method for Plotter objects.
"""

import logging
import math
import pandas as pd
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable

logger = logging.getLogger(__name__)

# Latitude where Web Mercator becomes a square map: atan(sinh(pi)) in degrees, ~85.051129
WEB_MERCATOR_MAX_LAT = math.degrees(math.atan(math.sinh(math.pi)))


def _project(xp: Any, lat_deg: Any, lon_deg: Any, R: float, lat_col: str, lon_col: str) -> Tuple[Any, Any]:
    """Validate and project float64 degree arrays with array module ``xp`` (numpy or cupy).

    NaN passes through as a missing position. Latitudes outside [-90, 90] and infinite
    coordinates raise; in-range latitudes clamp to the Web-Mercator limit so the poles
    are finite mirror images instead of -inf / +23810.
    """
    bad_lat = xp.isinf(lat_deg) | (xp.abs(lat_deg) > 90)
    if bool(bad_lat.any()):
        raise ValueError(
            f'Column "{lat_col}" has {int(bad_lat.sum())} latitude(s) outside [-90, 90] '
            f'(first: {float(lat_deg[bad_lat][0])}); check for swapped lat/lon, projected coordinates, or sentinel values')
    bad_lon = xp.isinf(lon_deg)
    if bool(bad_lon.any()):
        raise ValueError(
            f'Column "{lon_col}" has {int(bad_lon.sum())} non-finite longitude(s) (first: {float(lon_deg[bad_lon][0])})')

    lat_rad = xp.radians(xp.clip(lat_deg, -WEB_MERCATOR_MAX_LAT, WEB_MERCATOR_MAX_LAT))
    lon_rad = xp.radians(lon_deg)

    # Mercator projection formulas (vectorized)
    x_vals = R * lon_rad
    y_vals = R * xp.log(xp.tan(xp.pi / 4 + lat_rad / 2))
    return x_vals, y_vals


def mercator_layout(self: 'Plottable', scale_for_graphistry: bool = True) -> 'Plottable':
    """
    Convert latitude/longitude coordinates to Mercator projection.

    Uses point_latitude and point_longitude bindings if set, otherwise defaults to 'latitude' and 'longitude' columns.
    Uses existing point_x and point_y bindings if available, otherwise defaults to 'x' and 'y'.
    Uses GPU acceleration via CuPy if available, falls back to CPU/pandas otherwise.

    Latitudes beyond the Web-Mercator limit (about +-85.051129) are clamped to it, as map tiles do, so the poles project to finite positions.
    Missing coordinates give missing positions. Non-numeric columns, latitudes outside [-90, 90] and infinite coordinates raise ``ValueError``.

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

    for col in (lat_col, lon_col):
        if not pd.api.types.is_numeric_dtype(g._nodes[col].dtype):
            raise ValueError(f'Column "{col}" must be numeric degrees for mercator_layout, got dtype {g._nodes[col].dtype}')

    cp = None
    if is_not_pandas:
        try:
            import cupy  # pragma: no cover - cuDF-only; the changed-line-coverage gate has no cuDF lane (validated on dgx)
            cp = cupy  # pragma: no cover - cuDF-only; see above
        except ImportError:
            logger.warning("cuDF DataFrame detected but cupy is not available. Falling back to NumPy (CPU). Install cupy for GPU-accelerated computation.")

    if cp is not None:
        # Nulls become NaN so missing coordinates stay missing positions
        xp = cp  # pragma: no cover - cuDF-only; see above
        lat_deg = g._nodes[lat_col].to_cupy(dtype='float64', na_value=cp.nan)  # pragma: no cover - cuDF-only; see above
        lon_deg = g._nodes[lon_col].to_cupy(dtype='float64', na_value=cp.nan)  # pragma: no cover - cuDF-only; see above
    else:
        import numpy as np
        xp = np
        # to_numpy (not .values) so a cuDF frame without cupy still yields a host array
        lat_deg = g._nodes[lat_col].to_numpy(dtype='float64', na_value=np.nan)
        lon_deg = g._nodes[lon_col].to_numpy(dtype='float64', na_value=np.nan)

    x_vals, y_vals = _project(xp, lat_deg, lon_deg, R, lat_col, lon_col)

    g2 = g.nodes(
        g._nodes.assign(**{
            x_col: x_vals,
            y_col: y_vals
        })
    )

    return g2.bind(point_x=x_col, point_y=y_col).layout_settings(play=0)
