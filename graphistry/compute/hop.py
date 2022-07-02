from typing import List, Optional
import pandas as pd

from graphistry.Plottable import Plottable
from .filter_by_dict import filter_by_dict


def hop(self: Plottable,
    nodes: Optional[pd.DataFrame] = None,
    hops: Optional[int] = 1,
    to_fixed_point: bool = False,
    direction: str = 'forward',
    edge_match: Optional[dict] = None,
    source_node_match: Optional[dict] = None,
    destination_node_match: Optional[dict] = None,
    return_as_wave_front = False
) -> Plottable:
    """
    Given a graph and some source nodes, return subgraph of all paths within k-hops from the sources

    g: Plotter
    nodes: dataframe with id column matching g._node. None signifies all nodes (default).
    hops: how many hops to consider, if any bound (default 1)
    to_fixed_point: keep hopping until no new nodes are found (ignores hops)
    direction: 'forward', 'reverse', 'undirected'
    edge_match: dict of kv-pairs to exact match (see also: filter_edges_by_dict)
    source_node_match: dict of kv-pairs to match nodes before hopping
    destination_node_match: dict of kv-pairs to match nodes after hopping (including intermediate)
    return_as_wave_front: Only return the nodes/edges reached, ignoring past ones (primarily for internal use)
    """

    if not to_fixed_point and not isinstance(hops, int):
        raise ValueError(f'Must provide hops int when to_fixed_point is False, received: {hops}')

    if direction not in ['forward', 'reverse', 'undirected']:
        raise ValueError(f'Invalid direction: "{direction}", must be one of: "forward" (default), "reverse", "undirected"')

    if destination_node_match == {}:
        destination_node_match = None

    g2 = self.materialize_nodes()

    if nodes is None:
        nodes = g2._nodes

    if g2._edge is None:
        if 'index' in g2._edges.columns:
            raise ValueError('Edges cannot have column "index", please remove or set as g._edge via bind() or edges()')
        edges_indexed = g2.filter_edges_by_dict(edge_match)._edges.reset_index()
        EDGE_ID = 'index'
    else:
        edges_indexed = g2.filter_edges_by_dict(edge_match)._edges
        EDGE_ID = g2._edge

    if g2._node is None:
        raise ValueError('Node binding cannot be None, please set g._node via bind() or nodes()')

    if g2._source is None or g2._destination is None:
        raise ValueError('Source and destination binding cannot be None, please set g._source and g._destination via bind() or edges()')

    hops_remaining = hops
    wave_front = filter_by_dict(nodes[[ g2._node ]], source_node_match)
    matches_nodes = None
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
                    edges_indexed[[g2._source, g2._destination, EDGE_ID]].assign(**{g2._node: edges_indexed[g2._source]}),
                    how='inner',
                    on=g2._node)
                [[g2._source, g2._destination, EDGE_ID]]
            )
            new_node_ids_forward = hop_edges_forward[[g2._destination]].rename(columns={g2._destination: g2._node}).drop_duplicates()

            if destination_node_match is not None:
                new_node_ids_forward = filter_by_dict(
                    g2._nodes.merge(new_node_ids_forward, on=g2._node, how='inner'),
                    destination_node_match
                )[[g2._node]]
                hop_edges_forward = hop_edges_forward.merge(
                    new_node_ids_forward.rename(columns={g2._node: g2._destination}),
                    how='inner',
                    on=g2._destination
                )

        hop_edges_reverse = None
        new_node_ids_reverse = None
        if direction in ['reverse', 'undirected']:
            hop_edges_reverse = (
                wave_front.merge(
                    edges_indexed[[g2._destination, g2._source, EDGE_ID]].assign(**{g2._node: edges_indexed[g2._destination]}),
                    how='inner',
                    on=g2._node)
                [[g2._destination, g2._source, EDGE_ID]]
            )
            new_node_ids_reverse = hop_edges_reverse[[g2._source]].rename(columns={g2._source: g2._node}).drop_duplicates()

            if destination_node_match is not None:
                new_node_ids_reverse = filter_by_dict(
                    g2._nodes.merge(new_node_ids_reverse, on=g2._node, how='inner'),
                    destination_node_match
                )[[g2._node]]
                hop_edges_reverse = hop_edges_reverse.merge(
                    new_node_ids_reverse.rename(columns={g2._node: g2._source}),
                    how='inner',
                    on=g2._source
                )

        mt : List[pd.DataFrame] = []  # help mypy

        matches_edges = pd.concat(
            [ matches_edges ]
            + ([ hop_edges_forward[[ EDGE_ID ]] ] if hop_edges_forward is not None else mt)  # noqa: W503
            + ([ hop_edges_reverse[[ EDGE_ID ]] ] if hop_edges_reverse is not None else mt),  # noqa: W503
            ignore_index=True, sort=False).drop_duplicates(subset=[EDGE_ID])

        new_node_ids = pd.concat(
            mt
                + ( [ new_node_ids_forward ] if new_node_ids_forward is not None else mt )  # noqa: W503
                + ( [ new_node_ids_reverse] if new_node_ids_reverse is not None else mt ),  # noqa: W503
            ignore_index=True, sort=False).drop_duplicates()

        # Finally add initial nodes as confirmed also match edge + post-node predicates, not just pre-node predicates 
        if matches_nodes is None:
            if return_as_wave_front:
                matches_nodes = new_node_ids[:0]
            else:
                matches_nodes = pd.concat(
                    mt
                        + ( [hop_edges_forward[[g2._source]].rename(columns={g2._source: g2._node}).drop_duplicates()]  # noqa: W503
                            if hop_edges_forward is not None
                            else mt)
                        + ( [hop_edges_reverse[[g2._destination]].rename(columns={g2._destination: g2._node}).drop_duplicates()]  # noqa: W503
                            if hop_edges_reverse is not None
                            else mt),
                    ignore_index=True, sort=False).drop_duplicates(subset=[g2._node])

        combined_node_ids = pd.concat([matches_nodes, new_node_ids], ignore_index=True, sort=False).drop_duplicates()

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
        final_nodes = self._nodes.merge(
            matches_nodes if matches_nodes is not None else wave_front[:0],
            on=self._node,
            how='inner')
        g_out = g_out.nodes(final_nodes)

    return g_out
