from typing import Optional, TYPE_CHECKING
import pandas as pd

from graphistry.Plottable import Plottable


def hop(self: Plottable, nodes,
    hops: Optional[int] = 1,
    to_fixed_point: bool = False,
    direction: str = 'forward',
    edge_match: Optional[dict] = None
) -> Plottable:
    """
    Given a graph and some source nodes, return subgraph of all paths within k-hops from the sources

    g: Plotter
    nodes: dataframe with id column matching g._node
    hops: how many hops to consider, if any bound
    to_fixed_point: keep hopping until no new nodes are found
    direction: 'forward', 'backwards', 'undirected'
    edge_match: dict of kv-pairs to exact match (see also: filter_edges_by_dict)

    - currently only supports forwards hops
    - does not yet support transitive closure and backwards/undirected hops
    """

    if not to_fixed_point and not isinstance(hops, int):
        raise ValueError(f'Must provide hops int when to_fixed_point is False, received: {hops}')

    g2 = self.materialize_nodes()

    edges_indexed = g2.filter_edges_by_dict(edge_match)._edges.reset_index()
    EDGE_ID = 'index'

    hops_remaining = hops
    wave_front = nodes[[ g2._node ]]
    matches_nodes = wave_front
    matches_edges = edges_indexed[[EDGE_ID]][:0]

    while True:
        if not to_fixed_point and hops_remaining is not None:
            if hops_remaining < 1:
                break
            hops_remaining = hops_remaining - 1

        hop_edges_forward = None
        new_node_ids_forward = None
        if direction in ['forward', 'undirected']:
            hop_edges_forward = (
                wave_front.merge(
                    edges_indexed[[g2._source, g2._destination, EDGE_ID]].rename(columns={g2._source: g2._node}),
                    how='inner',
                    on=g2._node)
                [[g2._destination, EDGE_ID]]
            )
            new_node_ids_forward = hop_edges_forward[[g2._destination]].rename(columns={g2._destination: g2._node}).drop_duplicates()

        hop_edges_reverse = None
        new_node_ids_reverse = None
        if direction in ['reverse', 'undirected']:
            hop_edges_reverse = (
                wave_front.merge(
                    edges_indexed[[g2._destination, g2._source, EDGE_ID]].rename(columns={g2._destination: g2._node}),
                    how='inner',
                    on=g2._node)
                [[g2._source, EDGE_ID]]
            )
            new_node_ids_reverse = hop_edges_reverse[[g2._source]].rename(columns={g2._source: g2._node}).drop_duplicates()

        new_node_ids = pd.concat(
            []
                + ( [ new_node_ids_forward ] if new_node_ids_forward is not None else [] )  # noqa: W503
                + ( [ new_node_ids_reverse] if new_node_ids_reverse is not None else [] ),  # noqa: W503
            ignore_index=True, sort=False).drop_duplicates()
        combined_node_ids = pd.concat([matches_nodes, new_node_ids], ignore_index=True, sort=False).drop_duplicates()

        matches_edges = pd.concat(
            [ matches_edges]
            + ([ hop_edges_forward[[ EDGE_ID ]] ] if hop_edges_forward is not None else [])  # noqa: W503
            + ([ hop_edges_reverse[[ EDGE_ID ]] ] if hop_edges_reverse is not None else []),  # noqa: W503
            ignore_index=True, sort=False).drop_duplicates(subset=[EDGE_ID])

        if len(combined_node_ids) == len(matches_nodes):
            #fixedpoint, exit early: future will come to same spot!
            break
    
        wave_front = new_node_ids
        matches_nodes = combined_node_ids

    #hydrate edges
    final_edges = edges_indexed.merge(matches_edges, on=EDGE_ID, how='inner')
    if EDGE_ID not in self._edges:
        final_edges = final_edges.drop(columns=[EDGE_ID])
    g_out = g2.edges(final_edges)

    #hydrate nodes
    if self._nodes is not None:
        final_nodes = self._nodes.merge(matches_nodes, on=self._node, how='inner')
        g_out = g_out.nodes(final_nodes)

    return g_out
