from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from graphistry.Engine import Engine, EngineAbstract, df_concat, df_cons, resolve_engine
from graphistry.Plottable import Plottable
from graphistry.layout.circle import circle_layout
from graphistry.util import setup_logger


logger = setup_logger(__name__)


# Approximate Graphistry server settings
GRAPHISTRY_FA2_PARAMS: Dict[str, Any] = {
    'max_iter': 1000,
    'outbound_attraction_distribution': False,
    'scaling_ratio': 1
}

GRAPHISTRY_FR_PARAMS: Dict[str, Any] = {
    #'max_iter': 1000,
    #'outbound_attraction_distribution': False,
    #'scaling_ratio': 1
}


def compute_bounding_boxes(self: Plottable, partition_key: str, engine: Engine) -> Any:
    """
    
    Returns the bounding boxes for each partition based on the center coordinates and dimensions of the nodes.
    DF keys:
    - cx: Center x-coordinate
    - cy: Center y-coordinate
    - w: Width
    - h: Height
    - partition_key: Partition key
    """

    cons = df_cons(engine)

    groupby_partition = self._nodes.groupby(partition_key)

    # Compute per-partition bounding boxes
    min_x_per_partition = groupby_partition['x'].min().reset_index(drop=True)
    max_x_per_partition = groupby_partition['x'].max().reset_index(drop=True)
    min_y_per_partition = groupby_partition['y'].min().reset_index(drop=True)
    max_y_per_partition = groupby_partition['y'].max().reset_index(drop=True)

    # Compute per-partition center coordinates and dimensions
    center_x_per_partition = (min_x_per_partition + max_x_per_partition) * 0.5
    center_y_per_partition = (min_y_per_partition + max_y_per_partition) * 0.5
    width_per_partition = max_x_per_partition - min_x_per_partition
    height_per_partition = max_y_per_partition - min_y_per_partition

    # Prepare the bounding box per partition
    keys = groupby_partition.size().index.to_series()

    bounding_boxes = cons({
        'partition_key': keys.reset_index(drop=True),
        'cx': center_x_per_partition.reset_index(drop=True),
        'cy': center_y_per_partition.reset_index(drop=True),
        'w': width_per_partition.reset_index(drop=True),
        'h': height_per_partition.reset_index(drop=True)
    })

    return bounding_boxes


def fa2_layout(
    g: Plottable, 
    fa2_params: Optional[Dict[str, Any]] = None,
    circle_layout_params: Optional[Dict[str, Any]] = None,
    singleton_layout: Optional[Callable[[Plottable, Union[Tuple[float, float, float, float], Any]], Plottable]] = None,
    partition_key: Optional[str] = None,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
) -> Plottable:
    """
    Applies FA2 layout for connected nodes and circle layout for singleton (edgeless) nodes

    Allows optional parameterization of the circle layout, e.g., sort keys

    :param g: The graph object with nodes and edges, in a format compatible with Graphistry's Plottable object.
    :type g: graphistry.Plottable.Plottable
    :param fa2_params: Optional parameters for customizing the Force-Atlas 2 (FA2) layout, passed through to `fa2_layout`.
    :type fa2_params: Optional[Dict[str, Any]]
    :param circle_layout_params: Optional parameters for customizing the circle layout, passed through to `general_circle_layout`. Can include:
        - `by`: Column name(s) for sorting nodes (default: 'degree').
        - `ascending`: Boolean(s) to control sorting order.
        - `ring_spacing`: Spacing between rings in the circle layout.
        - `point_spacing`: Spacing between points in each ring.
    :type circle_layout_params: Optional[Dict[str, Any]]

    :param singleton_layout: Optional custom layout function for singleton nodes (default: circle_layout).
    :type singleton_layout: Optional[Callable[[Plottable, Tuple[float, float, float, float] | Any], Plottable]]

    :param partition_key: The key for partitioning nodes (used for picking bounding box type). Default is None.
    :type partition_key: Optional[str]

    :returns: A graph object with FA2 and circle layouts applied.
    :rtype: graphistry.Plottable.Plottable
    """

    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    engine_concrete = resolve_engine(engine, g)

    concat = df_concat(engine_concrete)

    g = g.materialize_nodes()

    drop: List[str] = []
    if 'x' in g._nodes:
        drop.append('x')
    if 'y' in g._nodes:
        drop.append('y')
    if len(drop) > 0:
        g = g.nodes(g._nodes.drop(columns=drop))

    # 1. Drop all self-edges (source == destination)
    edges_without_self_loops = g._edges[g._edges[g._source] != g._edges[g._destination]]

    # 2. Identify remaining edgeless nodes (0-degree nodes after removing self-loops)
    nodes_with_edges = concat([edges_without_self_loops[g._source], edges_without_self_loops[g._destination]]).unique()
    edgeless_nodes = g._nodes[~g._nodes[g._node].isin(nodes_with_edges)]

    # 3. Nodes with edges (after removing self-loops)
    connected_nodes = g._nodes[g._nodes[g._node].isin(nodes_with_edges)]

    # 4. Create subgraphs for edgeless and connected nodes
    empty_edges_df = g._edges[:0]
    g_edgeless = g.nodes(edgeless_nodes).edges(empty_edges_df)
    g_connected = g.nodes(connected_nodes).edges(edges_without_self_loops)

    # 5. Apply FA2 layout to connected nodes, handling the empty case
    if len(g_connected._edges) > 0:
        #g_connected = g_connected.edges(g_connected._edges.reset_index(drop=True))

        if engine_concrete == Engine.PANDAS:
            logger.warning("Pandas engine detected. FA2 falling back to igraph fr")
            g_connected_layout = g_connected.layout_igraph(
                'fr',
                directed=False,
                params=fa2_params if fa2_params is not None else GRAPHISTRY_FR_PARAMS
            )
        elif engine_concrete == Engine.CUDF:
            g_connected_layout = g_connected.layout_cugraph(
                'force_atlas2',
                kind='Graph',
                directed=False,
                params=fa2_params if fa2_params is not None else GRAPHISTRY_FA2_PARAMS
            )
        else:
            raise ValueError(f"Unsupported engine: {engine_concrete}")
        # Calculate the bounding box from the FA2 layout
        right, left = g_connected_layout._nodes.x.max(), g_connected_layout._nodes.x.min()
        top, bottom = g_connected_layout._nodes.y.min(), g_connected_layout._nodes.y.max()
        w, h = right - left, bottom - top
        cx, cy = (right + left) / 2, (top + bottom) / 2
    else:
        if len(g_connected._nodes) < 2:
            g_connected_layout = g_connected.nodes(g_connected._nodes.assign(x=0.0, y=0.0))
            cx, cy, w, h = 0.0, 0.0, 1.0, 1.0
        else:
            raise ValueError("Unexpected number of connected nodes with no edges")

    # 6. Apply circle layout to edgeless nodes using optional circle layout params and FA2 bounding box
    if len(g_edgeless._nodes) > 0:
        # Handle normal edgeless case
        layout = singleton_layout or circle_layout
        if partition_key is not None:
            bounding_box = compute_bounding_boxes(g_connected_layout, partition_key, engine_concrete)
        else:
            bounding_box = (cx, cy, w, h)
        g_edgeless_layout = layout(
            g_edgeless,
            bounding_box,
            **(circle_layout_params or {})  # Pass circle layout parameters, e.g., sort keys, spacing
        )
    else:
        # Handle no edgeless nodes case
        g_edgeless_layout = g_edgeless.nodes(g_edgeless._nodes.assign(x=0.0, y=0.0))

    # 7. Combine the layouts
    updated_nodes = concat([g_connected_layout._nodes, g_edgeless_layout._nodes], ignore_index=True)
    g_final = g.nodes(updated_nodes)

    # 8. Assert that there are no NaN values in the final layout
    assert not g_final._nodes['x'].isna().any(), "NaN values detected in x positions."
    assert not g_final._nodes['y'].isna().any(), "NaN values detected in y positions."

    g_final = g_final.layout_settings(play=0)

    return g_final
