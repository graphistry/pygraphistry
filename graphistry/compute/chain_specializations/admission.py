"""Shape admission for the pandas/cuDF chain specializations: the dispatcher and the tests
consult the same predicates, so a test that filters a shape corpus with them exercises exactly
what the dispatcher admits."""
# ruff: noqa: E501

from typing import Literal, Optional, Sequence, Tuple, TYPE_CHECKING

from graphistry.compute.ast import ASTObject, ASTNode, ASTEdge, ASTCall
from graphistry.compute.chain_fast_paths import SeedRowsHow
from graphistry.compute.typing import ArrayNamespace, DataFrameT, ScalarFilterDict

if TYPE_CHECKING:
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index.registry import AdjacencyIndex, NodeIdIndex


NativeFastPathShape = Literal["single-node", "seeded-hop"]


def native_fast_path_admits(
    ops: Sequence[ASTObject], engine: "Engine", start_nodes: Optional[DataFrameT],
) -> Optional[NativeFastPathShape]:
    """The shape the pandas/cuDF chain fast path serves for ``ops``, or None when the full
    path must run. This is the dispatcher's own gate (``chain._try_chain_fast_path`` calls
    it first), so a test that filters a shape corpus with it exercises exactly what the
    dispatcher admits: a node-only op without ``query``, or a 3-op plain single hop whose
    node ops carry no ``query``, whose edge carries no node matches, queries, zero-hop seed
    or endpoint pruning, whose aliases are distinct, and which is not an undirected hop
    with names or with node filters. Seeded chains (``start_nodes``) and other engines
    decline. Frame-dependent conditions (missing frames, alias equal to the node binding)
    are checked by the body after materialization."""
    from graphistry.Engine import Engine
    if engine not in (Engine.PANDAS, Engine.CUDF) or start_nodes is not None:
        return None
    if len(ops) == 1:
        n0 = ops[0]
        return "single-node" if isinstance(n0, ASTNode) and n0.query is None else None
    if len(ops) != 3:
        return None
    n0, e1, n2 = ops
    if not (isinstance(n0, ASTNode) and n0.query is None and isinstance(n2, ASTNode) and n2.query is None):
        return None
    if not (isinstance(e1, ASTEdge) and e1.is_simple_single_hop()
            and e1.source_node_match is None and e1.destination_node_match is None
            and e1.source_node_query is None and e1.destination_node_query is None
            and e1.edge_query is None and not e1.include_zero_hop_seed and not e1.prune_to_endpoints):
        return None
    named = [a for a in (n0._name, e1._name, n2._name) if a is not None]
    if len(named) != len(set(named)):
        return None
    if e1.direction == "undirected" and (n0._name is not None or n2._name is not None or n0.filter_dict or n2.filter_dict):
        return None
    return "seeded-hop"



def _indexed_kernel_admits(
    seed_nodes: DataFrameT, gathered_edges: Optional[DataFrameT], n0f: ScalarFilterDict,
    node: str, how: SeedRowsHow, ctx: Tuple["NodeIdIndex", "AdjacencyIndex", ArrayNamespace, "Engine"],
    n_nodes: int, n_edges: int,
) -> bool:
    """Whether the indexed connected-bindings kernel would have served this seeded 1-hop:
    its seed admission (binding-column integer seed, property-index hit, or a scan on a
    graph with fewer nodes than edges) and its frontier and gather cost gates."""
    from graphistry.Engine import POLARS_ENGINES, is_polars_df
    from numbers import Integral
    from graphistry.compute.gfql.index.cost import cost_gate_frac
    _, adj, _, engine = ctx
    seed_val = n0f.get(node)
    seeded_on_binding = isinstance(seed_val, Integral) and not isinstance(seed_val, bool)
    if not (seeded_on_binding or how == "property_index" or n_nodes < n_edges):
        return False
    frac = cost_gate_frac(engine)
    if engine in POLARS_ENGINES:
        import polars as pl
        assert is_polars_df(seed_nodes) and isinstance(seed_nodes, pl.DataFrame)
        seed_ids = seed_nodes.get_column(node)
        n_frontier = len(seed_ids) if len(seed_ids) <= 1 else int(seed_ids.n_unique())
    else:
        n_frontier = int(seed_nodes[node].nunique())
    if n_frontier >= frac * adj.n_keys:
        return False
    return gathered_edges is not None and len(gathered_edges) < frac * n_edges


def point_rows_admits(
    ops: Sequence[ASTObject], engine: "Engine", start_nodes: Optional[DataFrameT],
) -> Optional[int]:
    boundary = 1 if len(ops) in (2, 3) else 3 if len(ops) in (4, 5) else None
    if boundary is None or native_fast_path_admits(ops[:boundary], engine, start_nodes) is None:
        return None
    first = ops[0]
    if not isinstance(first, ASTNode) or not first.filter_dict:
        return None
    for op in ops[:boundary]:
        filters = op.filter_dict if isinstance(op, ASTNode) else op.edge_match if isinstance(op, ASTEdge) else None
        if filters and any(not isinstance(v, (str, int, float, bool)) for v in filters.values()):
            return None
    if boundary == 3:
        edge = ops[1]
        if not isinstance(edge, ASTEdge) or edge.direction == "undirected":
            return None
    row = ops[boundary]
    if not isinstance(row, ASTCall) or row.function != "rows" or set(row.params) - {"table", "source"}:
        return None
    table, source = row.params.get("table"), row.params.get("source")
    kind = ASTNode if table == "nodes" else ASTEdge if table == "edges" else None
    joined = boundary == 3 and table == "nodes" and source is None and len(ops) == 5
    if not joined and (kind is None or not isinstance(source, str) or not any(
        isinstance(op, kind) and op._name == source for op in ops[:boundary]
    )):
        return None
    if len(ops) > boundary + 1:
        projection = ops[-1]
        if not isinstance(projection, ASTCall) or projection.function != "select" or set(projection.params) != {"items"}:
            return None
    return boundary
