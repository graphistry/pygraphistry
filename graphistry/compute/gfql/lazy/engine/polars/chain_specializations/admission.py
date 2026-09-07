"""Shape admission for the polars chain specializations: the plain single-hop branches and
the seeded lane. The dispatcher and the tests consult the same predicates."""

from typing import Literal, Optional, Sequence

from graphistry.compute.ast import ASTObject, ASTNode, ASTEdge


PolarsPlainSingleHopShape = Literal["seeded-index", "skip-combine"]


def _plain_node(op: ASTObject) -> bool:
    return isinstance(op, ASTNode) and op._name is None and op.query is None


def _plain_edge(op: ASTObject) -> bool:
    return (isinstance(op, ASTEdge) and op.is_simple_single_hop()
            and op.edge_match is None and op.source_node_match is None
            and op.destination_node_match is None and op._name is None
            and op.source_node_query is None and op.destination_node_query is None
            and op.edge_query is None and not op.include_zero_hop_seed)


def polars_plain_single_hop_admits(ops: Sequence[ASTObject], start_nodes: Optional[object]) -> Optional[PolarsPlainSingleHopShape]:
    """The polars chain's plain single-hop branch for ``ops``: ``"seeded-index"`` when the
    resident-index hop is consulted first (seed filter, no destination filter, directed),
    ``"skip-combine"`` when the one-hop endpoint filter serves it without the
    forward/backward/combine passes, None when the full chain runs. The dispatcher calls
    this; unnamed, unqueried nodes and an unnamed, unmatched simple edge are the shape.
    A filtered undirected hop is the one plain shape that still takes the full chain."""
    if start_nodes is not None or len(ops) != 3:
        return None
    n0, e1, n2 = ops
    if not (_plain_node(n0) and _plain_edge(e1) and _plain_node(n2)):
        return None
    assert isinstance(n0, ASTNode) and isinstance(e1, ASTEdge) and isinstance(n2, ASTNode)
    directed = e1.direction in ("forward", "reverse")
    if n0.filter_dict and not n2.filter_dict and directed:
        return "seeded-index"
    unconstrained = not n0.filter_dict and not n2.filter_dict
    return "skip-combine" if (unconstrained or directed) else None


def polars_seeded_lane_admits(ops: Sequence[ASTObject]) -> bool:
    """Whether the polars seeded lane's shape gate admits ``ops``: a 3-op directed simple
    single hop whose seed node carries a filter, with no node queries, endpoint matches,
    endpoint or edge queries, zero-hop seed or endpoint pruning. The dispatcher calls this
    first; the frame conditions (polars frames, matching id dtypes, valid resident indexes,
    scalar-only filters, no colliding aliases) are decided by the body and can still decline
    an admitted shape."""
    if len(ops) != 3:
        return False
    n0, e1, n2 = ops
    if not (isinstance(n0, ASTNode) and isinstance(n2, ASTNode) and isinstance(e1, ASTEdge)):
        return False
    return not (n0.query is not None or n2.query is not None or not n0.filter_dict
                or not e1.is_simple_single_hop() or e1.direction not in ("forward", "reverse")
                or e1.source_node_match is not None or e1.destination_node_match is not None
                or e1.source_node_query is not None or e1.destination_node_query is not None
                or e1.edge_query is not None or e1.include_zero_hop_seed or e1.prune_to_endpoints)
