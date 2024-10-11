from typing import Optional, Tuple, List, Union

from graphistry.Engine import EngineAbstract, s_arange, s_concatenate, s_cos, s_full, s_pi, s_series, s_sin, s_sqrt, resolve_engine
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger


logger = setup_logger(__name__)


def circle_layout(
    self: Plottable, 
    bounding_box: Optional[Tuple[float, float, float, float]] = None,
    ring_spacing: Optional[float] = None,
    point_spacing: Optional[float] = None,
    by: Union[str, List[str]] = 'degree',
    ascending: Union[bool, List[bool]] = True,
    na_position: str = 'last',
    ignore_index: bool = True,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
) -> Plottable:
    """
    Arranges nodes in a circular layout, optionally sorted by a specified column (default: 'degree').
    It supports custom ring spacing, point spacing, and sorting options.

    :param g: The graph object containing nodes and edges.
    :type g: graphistry.Plottable.Plottable
    :param bounding_box: (optional) A tuple representing the bounding box (cx, cy, w, h) of the connected component layout.
    :type bounding_box: Optional[Tuple[float, float, float, float]]
    :param ring_spacing: (optional) The spacing between rings. Defaults to a tighter value.
    :type ring_spacing: Optional[float]
    :param point_spacing: (optional) The spacing between points within a ring. Defaults to a higher value than ring_spacing.
    :type point_spacing: Optional[float]
    :param by: (optional) Column name or list of column names to sort nodes by. Defaults to 'degree'.
    :type by: Union[str, List[str]]
    :param ascending: (optional) Boolean or list of booleans to control the sorting order for each column in `by`.
    :type ascending: Union[bool, List[bool]]
    :param na_position: (optional) Whether NaNs appear at the beginning ('first') or end ('last'). Defaults to 'last'.
    :type na_position: str
    :param ignore_index: (optional) Whether to ignore index when sorting. Defaults to True.
    :type ignore_index: bool
    :param engine: (optional) The graphistry engine to use. Defaults to 'auto'.
    :type engine: Union[graphistry.Engine.EngineAbstract, str]

    :returns: A graph object with nodes arranged in a circular layout.
    :rtype: graphistry.Plottable.Plottable

    **Example: Circular Layout by Degree**
        ::

            g_final = circle_layout(g)

    **Example: Circular Layout Sorted by Custom Metric**
        ::
        
            g_final = circle_layout(
                g,
                by='custom_metric',    # Sort by custom_metric column
                ascending=False        # Sort in descending order
            )

    **Example: Circular Layout Sorted by Multiple Columns**
        ::
        
            g_final = circle_layout(
                g,
                by=['custom_metric', 'other_metric'],    # Sort by custom_metric, then by other_metric
                ascending=[True, False]                  # Sort custom_metric in ascending order, other_metric in descending order
            )

    **Example: Custom Ring Spacing and Point Spacing**
        ::
        
            g_final = circle_layout(
                g, 
                ring_spacing=12,        # Custom spacing between rings
                point_spacing=18        # Custom spacing between points in each ring
            )

    **Example: Custom Sort with Ring and Point Spacing**
        ::
        
            g_final = circle_layout(
                g, 
                by='custom_metric',     # Sort by custom_metric column
                ascending=True,         # Sort in ascending order
                ring_spacing=10,        # Custom ring spacing
                point_spacing=15        # Custom point spacing
            )
    """

    if isinstance(engine, str):
        engine = EngineAbstract(engine)
    
    engine_concrete = resolve_engine(engine, self)

    arange = s_arange(engine_concrete)
    concatenate = s_concatenate(engine_concrete)
    full = s_full(engine_concrete)
    pi = s_pi(engine_concrete)
    sqrt = s_sqrt(engine_concrete)
    sin = s_sin(engine_concrete)
    cos = s_cos(engine_concrete)
    Series = s_series(engine_concrete)

    num_nodes = len(self._nodes)
    if num_nodes == 0:
        return self

    # Check if 'by' column exists; default to 'degree' if not provided
    if isinstance(by, str):
        by = [by]
    if isinstance(ascending, bool):
        ascending = [ascending] * len(by)

    for col in by:
        if col not in self._nodes.columns:
            if col == 'degree':
                self = self.get_degrees()  # Ensure g._nodes.degree is populated
            else:
                raise ValueError(f"Sort column '{col}' not found in nodes")

    # Sort the nodes by the specified columns
    sorted_nodes = self._nodes.sort_values(
        by=by,
        ascending=ascending,
        na_position=na_position,
        ignore_index=ignore_index
    )
    self = self.nodes(sorted_nodes)

    # Define default point_spacing if not provided (it should be larger than ring_spacing)
    if point_spacing is None:
        point_spacing = (ring_spacing or 10) * 1.5  # Larger point spacing than ring_spacing

    if bounding_box:
        cx, cy, w, h = bounding_box
        avg_space_per_point = (w * h) / num_nodes
        initial_radius = sqrt(avg_space_per_point)
        min_radius = sqrt(w**2 + h**2) / 2
        start_radius = min_radius + initial_radius
    else:
        cx, cy = 0.0, 0.0
        start_radius = sqrt(num_nodes) * 4  # Heuristic based on node count

    if ring_spacing is None:
        ring_spacing = start_radius * 0.3  # Tighter default ring spacing

    angles: List = []
    radii: List = []

    num_rings, total_points = 0, 0
    while total_points < num_nodes:
        num_rings += 1
        radius = start_radius + ring_spacing * (num_rings - 1)
        circumference = 2 * pi * radius
        points_per_circle = max(12, int(circumference / point_spacing))  # Ensure reasonable min points per circle

        remaining_points = num_nodes - total_points
        dynamic_threshold = max(10, int(points_per_circle * 0.25))  # Dynamic last ring threshold based on ring size

        if remaining_points < dynamic_threshold:
            # Add remaining points to the previous ring
            points_per_circle += remaining_points
        elif remaining_points <= points_per_circle:
            # Create a new, sparser ring with the remaining points
            radius += ring_spacing  # Push the ring further out
            points_per_circle = remaining_points

        angles_ring = arange(points_per_circle) * (2 * pi / points_per_circle)
        angles.append(angles_ring)
        radii.append(full(points_per_circle, radius))
        total_points += points_per_circle
        if total_points >= num_nodes:
            break

    angles = concatenate(angles)[:num_nodes]
    radii = concatenate(radii)[:num_nodes]
    new_x = cx + radii * cos(angles)
    new_y = cy + radii * sin(angles)

    self._nodes['x'] = Series(new_x.get())
    self._nodes['y'] = Series(new_y.get())

    return self
