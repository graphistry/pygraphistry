from typing import Any, Optional, Tuple, List, Union

import numpy as np

from graphistry.Engine import (
    Engine,
    EngineAbstract,
    resolve_engine,
    df_cons,
    s_arange, s_cos, s_floor, s_full, s_isna, s_pi,
    s_series, s_sin, s_sqrt, s_to_arr
)
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
logger = setup_logger(__name__)



def print_gpu_memory_usage(prefix=""):
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    used = mempool.used_bytes() / 1024**3  # Convert bytes to GB
    total = cp.cuda.Device().mem_info[1] / 1024**3  # Total available memory (in GB)
    free = total - used
    print(f"{prefix} GPU Memory Usage: Used: {used:.2f} GB, Free: {free:.2f} GB, Total: {total:.2f} GB")



def circle_layout(
    self: Plottable,
    bounding_box: Optional[Union[Tuple[float, float, float, float], Any]] = None,
    ring_spacing: Optional[float] = None,
    point_spacing: Optional[float] = None,
    partition_by: Optional[Union[str, List[str]]] = None,
    sort_by: Optional[Union[str, List[str]]] = None,
    ascending: Union[bool, List[bool]] = True,
    na_position: str = 'last',
    ignore_index: bool = True,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
) -> Plottable:

    # Leo 10/14/2024, see o1 on closed-form derivation for concentric rings vs current single

    # BUGGY SO NOT ACTIVE:

    # **Formulas (Plain Text)**:

    # - node_start_radius: The starting radius for nodes (same for all nodes)
    # - delta_r: The spacing between successive rings
    # - spacing_s: The distance between nodes within a ring, along the circumference

    # 1. **Ring Number Calculation**:
    # The ring number R for a node with index I is:
    # R = (sqrt(node_start_radius^2 + (spacing_s * I) / pi) - node_start_radius) / delta_r
    # => the ring number is floor(R)

    # 2. **Radius of Node in Ring**:
    # The radius of a node in a given ring is:
    # radius = node_start_radius + ring_number * delta_r

    # 3. **Total Nodes Up to Ring**:
    # The total number of nodes that fit up to and including ring R is:
    # TotalNodes(R) = (pi / spacing_s) * ((node_start_radius + R * delta_r)^2 - node_start_radius^2)

    # 4. **Node Index Within Ring**:
    # The index of a node within its ring is:
    # node_index_in_ring = node_index - TotalNodes(R - 1)

    # 5. **Number of Nodes in Each Ring**:
    # The number of nodes in a given ring is:
    # nodes_in_ring = (2 * pi * radius_of_ring) / spacing_s

    # 6. **Angle of Node in Ring**:
    # The angular position theta for a node in a ring is:
    # theta = (2 * pi * node_index_in_ring) / nodes_in_ring

    """
    Arranges nodes in a circular layout

    If partition_by and and bounding_box df are provided, do as multiple circles
    
    Each circle is sorted, by default by degree
    
    The ring radius is set to circumscribe the bounding box of the nodes

    Parameters
    ----------

    :param self: Plottable
    :type self: Plottable
    
    :param bounding_box: The bounding box for the circular layout, in the format (cx, cy, width, height), or a partition-keyed dataframe of the same. If not provided, the bounding box is determined based on the nodes' positions.
    :type bounding_box: Optional[Tuple[float, float, float, float] | df[[partition_key, cx, cy, w, h]]]
    
    :param ring_spacing: The spacing between successive rings. Defaults to 1.0 if not provided.
    :type ring_spacing: Optional[float]

    :param point_spacing: The distance between nodes within a ring, along the circumference. Defaults to ring_spacing * 0.1 if not provided.
    :type point_spacing: Optional[float]

    :param partition_by: Column name or list of column names to sort nodes by. Defaults to None, in which case no sorting is applied.
    :type partition_by: Optional[Union[str, List[str]]]

    :param sort_by: Column name or list of column names to sort nodes by. Defaults to None, in which case sorting is by degree, in-degree, outdegree.
    :type sort_by: Optional[Union[str, List[str]]]

    :param ascending: Whether to sort ascending or descending.
    :type ascending: Union[bool, List[bool]]

    :param na_position: Where to position NaNs in the sorting order. Defaults to 'last'.
    :type na_position: str

    :param ignore_index: Whether to ignore the index when sorting. Defaults to True.
    :type ignore_index: bool

    :param engine: The engine to use for computations (either 'pandas' or 'cudf'). Defaults to EngineAbstract.AUTO.
    :type engine: Union[EngineAbstract, str]

    Returns
    -------
    Plottable
        A graph object with nodes arranged in a circular layout.
    """
    if isinstance(engine, str):
        engine = EngineAbstract(engine)
    engine_concrete = resolve_engine(engine, self)

    # Import necessary functions based on the engine
    arange = s_arange(engine_concrete)
    full = s_full(engine_concrete)
    sqrt = s_sqrt(engine_concrete)
    #maximum = s_maximum(engine_concrete)
    pi = s_pi(engine_concrete)
    cos = s_cos(engine_concrete)
    sin = s_sin(engine_concrete)
    floor = s_floor(engine_concrete)
    is_na = s_isna(engine_concrete)
    Series = s_series(engine_concrete)
    to_arr = s_to_arr(engine_concrete)
    cons = df_cons(engine_concrete)

    num_nodes = len(self._nodes)
    if num_nodes == 0:
        return self

    g = self.materialize_nodes()
    g = g.nodes(g._nodes.reset_index(drop=True))

    # Optional sorting (if 'by' is specified)

    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_keys = [sort_by]
        elif isinstance(sort_by, list) and all(isinstance(col, str) for col in sort_by):
            sort_keys = sort_by
        else:
            raise ValueError(f"Invalid 'by' argument: must be None, str, or list[str], but got {sort_by}")
    else:
        g = g.get_degrees()
        sort_keys = ['degree', 'degree_in', 'degree_out']
    if partition_by is not None:
        if isinstance(partition_by, str):
            sort_keys = [partition_by] + sort_keys
        else:
            assert isinstance(partition_by, list) and all(isinstance(col, str) for col in partition_by)
            sort_keys = partition_by + sort_keys

    assert sort_keys is not None
    if len(sort_keys) > 0:
        g = g.nodes(g._nodes.sort_values(
            by=sort_keys,
            ascending=ascending,
            na_position=na_position,
            ignore_index=ignore_index,
            kind='mergesort'  # Stable sort to maintain order
        ))

    if partition_by is not None:
        if isinstance(partition_by, str):
            partition_columns = [partition_by]
        elif isinstance(partition_by, list) and all(isinstance(col, str) for col in partition_by):
            partition_columns = partition_by
        else:
            raise ValueError(f"Invalid 'by' argument: must be None, str, or list[str], but got {partition_by}")
    else:
        partition_columns = []

    if partition_by is not None:
        g = g.nodes(g._nodes.sort_values(by=partition_columns + [g._node]).reset_index(drop=True))
        node_idx_relative = g._nodes.groupby(partition_by).cumcount().reset_index(drop=True)
    else:
        g = g.nodes(g._nodes.sort_values(by=g._node).reset_index(drop=True))
        node_idx_relative = Series(arange(num_nodes))
    if node_idx_relative.isna().any():
        raise ValueError('Unexpected NaNs in node indices')

    delta_r = ring_spacing or 50.0  # Ring spacing (scalar)
    spacing_s = point_spacing or 5.0  # Spacing between nodes in a ring (scalar)

    # Handle partitioning
    start_radius_bound_box_ratio = 1.05
    if partition_by is None:
        # No partitioning; treat all nodes as a single group

        if bounding_box is not None:
            assert len(bounding_box) == 4, f'Invalid bounding box: {bounding_box}, types: {[type(val) for val in bounding_box]}'
            center_x, center_y, width, height = bounding_box
        else:
            node_min_x = g._nodes['x'].min()
            node_max_x = g._nodes['x'].max()
            node_min_y = g._nodes['y'].min()
            node_max_y = g._nodes['y'].max()
            center_x = (node_min_x + node_max_x) * 0.5
            center_y = (node_min_y + node_max_y) * 0.5
            width = node_max_x - node_min_x
            height = node_max_y - node_min_y

        node_centers_x = Series(full(num_nodes, center_x))
        node_centers_y = Series(full(num_nodes, center_y))
        node_widths = Series(full(num_nodes, width))
        node_heights = Series(full(num_nodes, height))
        node_partition_sizes = Series(full(num_nodes, num_nodes))

    else:
        # Partitioning logic

        groupby_partition = g._nodes.groupby(partition_columns)

        if bounding_box is not None:

            #we only support partition_by string typed variant rn, throw if not:
            if not isinstance(partition_by, str):
                raise NotImplementedError(f'partition_by only supported for string type, received: {type(partition_by)}')

            assert isinstance(bounding_box, cons), f'Invalid bounding box type, expected {cons}, got {type(bounding_box)}'

            nodes_with_partitions = g._nodes.merge(
                bounding_box.rename(columns={'partition_key': partition_by}),
                how='left',
                on=partition_columns
            ).sort_values(by=partition_columns).reset_index(drop=True)

            node_centers_x = nodes_with_partitions['cx']  # (num_nodes,)
            node_centers_y = nodes_with_partitions['cy']  # (num_nodes,)
            node_widths = nodes_with_partitions['w']  # (num_nodes,)
            node_heights = nodes_with_partitions['h']  # (num_nodes,)
            if engine_concrete == Engine.CUDF:
                node_partition_sizes = groupby_partition.transform('size')[g._node]  # (num_nodes,)
            else:
                node_partition_sizes = groupby_partition.transform('size')  # (num_nodes,)
                #node_partition_sizes = groupby_partition.transform('size')  # (num_nodes,)
                assert len(node_partition_sizes) == num_nodes
                assert isinstance(node_partition_sizes, Series)

            #singleton nodes will not be placed yet, so place now
            node_centers_x = node_centers_x.fillna(0.)  # (num_nodes,)
            node_centers_y = node_centers_y.fillna(0.)  # (num_nodes,)
            node_widths = node_widths.fillna(1.)  # (num_nodes,)
            node_heights = node_heights.fillna(1.)  # (num_nodes,)

            if node_partition_sizes.isna().any():
                raise ValueError('Unexpected NaNs in node partition sizes')
            
        else:
            raise NotImplementedError('Bounding box not provided')

    #TODO move these and others to bounding box and splat...

    diagonals = Series(sqrt(to_arr(node_widths**2) + to_arr(node_heights**2)))
    node_start_radii = start_radius_bound_box_ratio * diagonals / 2

    # Compute ring numbers R for each node
    # Formula: R = (sqrt((node_start_radius)^2 + (4 * spacing_s * I) / pi) - node_start_radius) / (2 * delta_r)
    #numerator = Series(sqrt(
    #    node_start_radii**2 + (spacing_s * node_indices) / pi
    #)) - node_start_radii  # (num_nodes,)
    #R = numerator / delta_r  # (num_nodes,)
    #ring_numbers = Series(floor(R)).astype(int)  # (num_nodes,)
    ring_numbers = Series(full(num_nodes, 0))

    # Calculate radius for each node based on its ring
    # Radius: radius = node_start_radius + ring_number * delta_r
    node_radii = node_start_radii + ring_numbers * delta_r  # (num_nodes,)

    # Total nodes up to previous rings
    # TotalNodes(R) = (2 * pi / spacing_s) * (R * node_start_radius + (delta_r * R^2) / 2)
    R_prev = ring_numbers  # (num_nodes,)
    total_nodes_prev_rings = ((2 * pi) / spacing_s) * (R_prev * node_start_radii + (delta_r * R_prev**2) * 0.5)  # (num_nodes,)
    total_nodes_prev_rings = floor(total_nodes_prev_rings).astype(int)  # (num_nodes,)

    # Node's index within its ring
    # node_index_in_ring = node_index - TotalNodes(R - 1)
    #node_indices_in_ring = node_indices - total_nodes_prev_rings  # (num_nodes,)
    node_indices_in_ring = node_idx_relative - total_nodes_prev_rings  # (num_nodes,)

    if partition_by is not None:
        node_counts_in_rings = node_partition_sizes - total_nodes_prev_rings  # (num_nodes,)
        assert len(node_counts_in_rings) == num_nodes
        assert len(node_indices_in_ring) == num_nodes
        assert len(total_nodes_prev_rings) == num_nodes
        #is_final_ring = node_indices_in_ring.reset_index(drop=True) >= node_counts_in_rings.reset_index(drop=True) - 1  # (num_nodes,)
    else:
        if ring_numbers.max() == 0:
            # Single partial ring case: all nodes are part of the final ring
            node_counts_in_rings = Series(full(num_nodes, num_nodes))
            #is_final_ring = Series(full(num_nodes, True))  # All nodes in this case belong to the final (and only) ring
        else:
            # Multiple rings case: Check which nodes are in the final ring
            node_counts_in_rings = node_partition_sizes - total_nodes_prev_rings 
            #is_final_ring = node_indices_in_ring >= node_counts_in_rings - 1 

        assert len(total_nodes_prev_rings) == num_nodes

    # Number of nodes in the final ring (adjusted for partial rings)
    #nodes_in_final_ring = is_final_ring * node_partition_sizes - total_nodes_prev_rings  # (num_nodes,)
    #nodes_in_final_ring = is_final_ring.reset_index(drop=True) * node_counts_in_rings.reset_index(drop=True)  # (num_nodes,)

    # Recalculate `nodes_in_ring` to handle partial final rings
    #nodes_in_ring = is_final_ring * nodes_in_final_ring + (~is_final_ring) * full_nodes_in_ring  # Adjust for partial rings
    nodes_in_ring = node_partition_sizes

    ##########################################

    # Compute angles for each node in the ring
    #node_angles = (2 * pi * node_indices_in_ring) / nodes_in_ring  # (num_nodes,)
    if engine_concrete in [EngineAbstract.CUDF, 'cudf', 'Engine.CUDF'] or hasattr(node_idx_relative, 'to_pandas'):
        # CUDA OOM bug despite small data
        #node_angles = Series((2 * np.pi * node_idx_relative.to_pandas()) / nodes_in_ring.to_pandas()).fillna(0.0)  # (num_nodes,)
        node_angles = 2 * np.pi * node_idx_relative.reset_index(drop=True) / nodes_in_ring.reset_index(drop=True)  # (num_nodes,)
    else:
        node_angles = ((2 * pi * node_idx_relative) / nodes_in_ring).fillna(0.0)  # (num_nodes,)
    if is_na(node_angles).any():
        raise ValueError('Unexpected NaNs in node angles')

    # Compute final positions in Cartesian coordinates
    node_final_x = node_centers_x + node_radii * Series(cos(to_arr(node_angles)))  # (num_nodes,)
    node_final_y = node_centers_y + node_radii * Series(sin(to_arr(node_angles)))  # (num_nodes,)

    # g = g.nodes(
    #     g._nodes.assign(
    #         node_idx_relative=node_idx_relative.reset_index(drop=True),
    #         nodes_in_ring=nodes_in_ring.reset_index(drop=True),
    #         cx=node_centers_x.reset_index(drop=True),
    #         cy=node_centers_y.reset_index(drop=True),
    #         node_radii=node_radii.reset_index(drop=True),
    #         node_start_radii=node_start_radii.reset_index(drop=True),
    #         ring_numbers=ring_numbers.reset_index(drop=True),
    #         node_widths=node_widths.reset_index(drop=True),
    #         node_heights=node_heights.reset_index(drop=True),
    #         expt=Series(node_idx_relative.to_pandas().astype('float64') / nodes_in_ring.to_pandas().astype('float64')).reset_index(drop=True)
    #     )
    # )

    g_final = g.nodes(
        g._nodes.assign(
            x=node_final_x.reset_index(drop=True),
            y=node_final_y.reset_index(drop=True),
        )
    )
    
    assert not g_final._nodes['x'].isna().any(), "NaN values detected in x positions."
    assert not g_final._nodes['y'].isna().any(), "NaN values detected in y positions."

    return g_final
