from typing import Any, Dict, Optional, Union

from graphistry.Engine import EngineAbstract, df_concat, df_cons, resolve_engine
from graphistry.Plottable import Plottable
from graphistry.layout.circle import circle_layout
from graphistry.util import setup_logger


logger = setup_logger(__name__)


def fa2_layout(
    self: Plottable, 
    fa2_params: Optional[Dict[str, Any]] = None,
    circle_layout_params: Optional[Dict[str, Any]] = None,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
) -> Plottable:
    """
    Combination layout with GPU acceleration.

        - ForceAtlas 2 layout for connected nodes
    
        - Circle layout for singleton (edgeless) nodes

    Currently GPU-only.

    Note: The loadtime Graphistry version uses a different GPU implementation of FA2 with additional features and interactive controls. In contrast, the PyGraphistry version uses the simpler cuGraph implementation.

    :param g: The graph object with nodes and edges, in a format compatible with Graphistry's Plottable object.
    :type g: graphistry.Plottable.Plottable

    :param fa2_params: Optional parameters for customizing the :ref:`cuGraph ForceAtlas2 (FA2) layout_cugraph() <cugraph>`, passed through to `force_atlas2`.
    :type fa2_params: Optional[Dict[str, Any]]

    :param circle_layout_params: Optional parameters for customizing the circle layout, passed through to :meth:`circle_layout <graphistry.layout.circle.circle_layout>`. Can include:

        - `by`: Column name(s) for sorting nodes (default: 'degree').
        - `ascending`: Boolean(s) to control sorting order.
        - `ring_spacing`: Spacing between rings in the circle layout.
        - `point_spacing`: Spacing between points in each ring.

    :type circle_layout_params: Optional[Dict[str, Any]]

    :returns: A graph object with FA2 and circle layouts applied.
    :rtype: graphistry.Plottable.Plottable
    """

    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    engine_concrete = resolve_engine(engine, self)

    concat = df_concat(engine_concrete)
    cons = df_cons(engine_concrete)

    # 1. Drop all self-edges (source == destination)
    edges_without_self_loops = self._edges[self._edges[self._source] != self._edges[self._destination]]

    # 2. Identify remaining edgeless nodes (0-degree nodes after removing self-loops)
    nodes_with_edges = concat([edges_without_self_loops[self._source], edges_without_self_loops[self._destination]]).unique()
    edgeless_nodes = self._nodes[~self._nodes[self._node].isin(nodes_with_edges)]

    # 3. Nodes with edges (after removing self-loops)
    connected_nodes = self._nodes[self._nodes[self._node].isin(nodes_with_edges)]

    # 4. Create subgraphs for edgeless and connected nodes
    empty_edges_df = cons({self._source: [], self._destination: []})  # Empty edges DataFrame
    g_edgeless = self.nodes(edgeless_nodes).edges(empty_edges_df)
    g_connected = self.nodes(connected_nodes).edges(edges_without_self_loops)

    # 5. Apply FA2 layout to connected nodes, handling the empty case
    if len(g_connected._edges) > 0:
        g_connected_layout = g_connected.layout_cugraph(
            'force_atlas2',
            kind='Graph',
            directed=False,
            params={
                **(fa2_params or {}),
                'max_iter': 1000,
                'outbound_attraction_distribution': False,
                'scaling_ratio': 1
            }
        )
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
        g_edgeless_layout = circle_layout(
            g_edgeless, 
            bounding_box=(cx, cy, w, h), 
            **(circle_layout_params or {})  # Pass circle layout parameters, e.g., sort keys, spacing
        )
    else:
        # Handle no edgeless nodes case
        g_edgeless_layout = g_edgeless  # No layout needed

    # 7. Combine the layouts
    try:
        updated_nodes = concat([g_connected_layout._nodes, g_edgeless_layout._nodes], ignore_index=True)
    except ValueError as e:
        logger.error(f"Error combining layouts: {e}\ndtype1:\n{g_connected_layout._nodes.dtypes}\ndtype2:\n{g_edgeless_layout._nodes.dtypes}")
        raise
    g_final = self.nodes(updated_nodes)

    # 8. Assert that there are no NaN values in the final layout
    assert not g_final._nodes['x'].isna().any(), "NaN values detected in x positions."
    assert not g_final._nodes['y'].isna().any(), "NaN values detected in y positions."

    g_final = g_final.layout_settings(play=0)

    return g_final
