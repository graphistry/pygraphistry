"""Native polars lowering for the correlated pattern-existence row-op family
(``EXISTS { }`` subqueries / bare pattern predicates): ``rows(binding_ops=...)``,
``semi_apply_mark``, ``anti_semi_apply``. Split from row_pipeline.py (which is the
expression/projection lowering core) per the degrees.py module-per-family precedent.

NO-CHEATING contract as everywhere in this engine: unsupported shapes return None
and the boundary lane raises an honest NotImplementedError — never a pandas bridge.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence
from graphistry.utils.json import JSONVal
from graphistry.compute.gfql.index.types import HopDirection

from graphistry.Plottable import Plottable

if TYPE_CHECKING:
    import polars as pl
    from graphistry.compute.ast import ASTObject

from .dtypes import is_lazy
from .row_pipeline import _active_table, _rewrap


def _rows_base_graph(g: Plottable) -> Plottable:
    """The ORIGINAL graph threaded by the boundary lane (`_gfql_rows_base_graph`
    setattr convention, mirrored from the pandas lane at compute/chain.py) — the one
    sanctioned dynamic read, contained here; falls back to the current graph."""
    base = getattr(g, "_gfql_rows_base_graph", None)
    return base if base is not None else g


def _binding_ast_ops(binding_ops: Sequence[Dict[str, JSONVal]]) -> Optional[List["ASTObject"]]:
    """Deserialize the semi-apply family's serialized binding ops; None on failure."""
    from graphistry.compute.ast import from_json as ast_from_json
    try:
        return [ast_from_json(op_json, validate=False) for op_json in binding_ops]
    except Exception:
        return None


def rows_binding_ops_polars(g: Plottable, binding_ops: Sequence[Dict[str, JSONVal]]) -> Optional[Plottable]:
    """Native ``rows(binding_ops=[...])`` for the SINGLE named-Node case — the shape the
    boundary rewrite emits for a one-entity MATCH (the EXISTS pipeline's left table).
    Mirrors the pandas ``_gfql_node_alias_lookup_frame`` layout exactly:
    ``[node_id, alias, alias.node_id, alias.<col>...]`` in source column order.
    Anything else (multi-op, edge ops, unnamed, node query=) declines (None -> NIE)."""
    import polars as pl
    from graphistry.compute.ast import ASTNode as _ASTNode
    from .predicates import filter_by_dict_polars
    ops = _binding_ast_ops(binding_ops)
    if ops is None or len(ops) != 1 or not isinstance(ops[0], _ASTNode):
        return None
    op = ops[0]
    alias = op._name
    if not isinstance(alias, str) or op.query:
        return None
    base_graph = _rows_base_graph(g)
    node_id = base_graph._node or "id"  # Plottable._node: Optional[str]
    nodes = base_graph._nodes
    if nodes is None or node_id not in nodes.columns:
        return None
    start_nodes = getattr(g, "_gfql_start_nodes", None)
    if start_nodes is not None:
        # start-nodes constrain the scan like the pandas prev_node_wavefront does.
        # Normalize to EAGER: an eager.join(LazyFrame) raises and would silently
        # decline the whole wavefront-constrained case (wave-2 W2-3).
        sn = start_nodes.collect() if isinstance(start_nodes, pl.LazyFrame) else start_nodes
        try:
            nodes = nodes.join(sn.select(node_id).unique(), on=node_id, how="semi")
        except Exception:
            return None
    try:
        matched = filter_by_dict_polars(nodes, getattr(op, "filter_dict", None))
    except NotImplementedError:
        raise
    except Exception:
        return None
    if matched is None:
        return None
    if alias == node_id:
        # pandas' named-op flag column OVERWRITES the id column in this corner —
        # neither engine has sane semantics; decline honestly (wave-1 I1).
        return None
    # Mirror the pandas twin's quirks (wave-1 I3): ASTNode.execute leaks the
    # named-op FLAG column into its source frame, so pandas emits a fabricated
    # `alias.alias = True` column (shadowing any real property of that name) —
    # emit the same. (Column ORDER differs slightly; the parity sig sorts columns.)
    prop_cols = [c for c in matched.columns if c != node_id and c != alias]
    exprs = [
        pl.col(node_id),
        pl.col(node_id).alias(alias),
        pl.col(node_id).alias(f"{alias}.{node_id}"),
        pl.lit(True).alias(f"{alias}.{alias}"),
    ]
    exprs.extend(pl.col(c).alias(f"{alias}.{c}") for c in prop_cols)
    lookup = matched.select(exprs)
    return _rewrap(g, lookup)


def _pattern_alias_keys_polars(
    g: Plottable, binding_ops: Sequence[Dict[str, JSONVal]], alias: str, neq: Optional[Sequence[str]] = None
) -> Optional["pl.DataFrame"]:
    """Ids of ``alias``'s nodes that participate in the (single) pattern — the semi-apply
    right side — computed by running the binding chain NATIVELY via ``chain_polars`` on the
    base graph and reading the named-op flag column (the chain's backward prune makes the
    flag exactly 'a full pattern match exists through this node'). v1: [Node, Edge, Node]
    single-hop shapes only; the join alias must be a NAMED endpoint. None declines (NIE)."""
    import polars as pl
    from graphistry.compute.ast import ASTNode as _ASTNode, ASTEdge as _ASTEdge
    ops = _binding_ast_ops(binding_ops)
    if (
        ops is None or len(ops) != 3
        or not isinstance(ops[0], _ASTNode) or not isinstance(ops[1], _ASTEdge)
        or not isinstance(ops[2], _ASTNode)
    ):
        return None
    n0, edge_op, n2 = ops[0], ops[1], ops[2]  # locals: mypy narrows these, not ops[i]
    if edge_op.hops not in (None, 1) or edge_op.to_fixed_point:
        return None
    if edge_op.min_hops not in (None, 1) or edge_op.max_hops not in (None, 1):
        return None
    if alias not in (n0._name, n2._name):
        return None
    if n0.query or n2.query:
        return None
    base_graph = _rows_base_graph(g)
    node_id = base_graph._node or "id"  # Plottable._node: Optional[str]
    # GFQL #1658 index fast path (#3 membership): bare EXISTS pattern (no endpoint/
    # edge filters, no drop-self neq) -> participating nodes == "has an edge in this
    # direction" = CSR adjacency membership. Skips the O(E) chain_polars below.
    # Strict guard; anything richer (filters/neq/multi-hop) falls through unchanged.
    from graphistry.compute.gfql.index import get_index_policy
    if (
        neq is None
        and get_index_policy(g) != "off"
        and not n0.filter_dict and not n2.filter_dict
        and not edge_op.edge_match
        and edge_op.edge_query is None
        and base_graph._edges is not None
        and not is_lazy(base_graph._edges)
    ):
        try:
            import numpy as _np
            from graphistry.compute.gfql.index import get_registry
            from graphistry.compute.gfql.index.degrees import adjacency_membership_keys
            from graphistry.Engine import Engine as _Engine
            from graphistry.compute.gfql.lazy import active_target as _active_target, ExecutionTarget as _ExecutionTarget
            _reg = get_registry(g)
            if not _reg.is_empty():
                _edir = edge_op.direction
                _mdir: HopDirection
                if _edir == "undirected":
                    _mdir = "undirected"
                elif (alias == n0._name) == (_edir == "forward"):
                    _mdir = "forward"
                else:
                    _mdir = "reverse"
                _src, _dst = base_graph._source, base_graph._destination
                if isinstance(_src, str) and isinstance(_dst, str):
                    _eng = _Engine.POLARS_GPU if _active_target() == _ExecutionTarget.GPU else _Engine.POLARS
                    _mk = adjacency_membership_keys(_reg, _mdir, base_graph._edges, (_src, _dst), _eng)
                    if _mk is not None:
                        return pl.DataFrame({node_id: pl.Series(node_id, _np.asarray(_mk))})
        except (AttributeError, ImportError, NotImplementedError, TypeError, ValueError):
            pass
    if neq:
        # EXISTS { (n)--(m) WHERE m <> n } — for the single-edge shape, endpoint
        # inequality == "witnessed by a NON-self-loop edge": pre-drop self loops and
        # reuse the same key computation. Any other neq spelling declines.
        endpoint_names = {n0._name, n2._name}
        if set(neq) != endpoint_names or len(set(neq)) != 2:
            return None
        edges = base_graph._edges
        src, dst = base_graph._source, base_graph._destination
        if (
            edges is None
            or not isinstance(src, str) or src not in edges.columns
            or not isinstance(dst, str) or dst not in edges.columns
        ):
            return None
        base_graph = base_graph.edges(edges.filter(pl.col(src) != pl.col(dst)))
    from .chain import chain_polars  # local: chain imports this module at call time
    from graphistry.compute.exceptions import GFQLValidationError
    try:
        out = chain_polars(base_graph, list(ops))
    except NotImplementedError:
        return None
    except GFQLValidationError:
        # e.g. chain's duplicate-alias guard on EXISTS { (n)--(n) } — decline to
        # honest NIE rather than a non-NIE crash (wave-1 T4); pandas may answer.
        return None
    out_nodes = out._nodes
    if out_nodes is None or alias not in out_nodes.columns or node_id not in out_nodes.columns:
        # empty match -> no keys (an empty id frame, NOT a decline)
        if out_nodes is not None and node_id in out_nodes.columns:
            return out_nodes.select(node_id).head(0)
        return None
    return (
        out_nodes.filter(pl.col(alias).fill_null(False))
        .select(node_id)
        .unique()
    )


def _semi_apply_join_col(left: "pl.DataFrame", alias: str, node_id: str) -> Optional[str]:
    """Mirror the pandas join-column choice: prefer ``alias.node_id``, else bare ``alias``."""
    for cand in (f"{alias}.{node_id}", alias):
        if cand in left.columns:
            return cand
    return None


def semi_apply_mark_polars(
    g: Plottable, binding_ops: Sequence[Dict[str, JSONVal]], join_aliases: Sequence[str], out_col: str,
    neq: Optional[Sequence[str]] = None,
) -> Optional[Plottable]:
    """Native polars ``semi_apply_mark``: boolean existence marker per active row.
    v1: exactly ONE join alias (two-alias correlation needs PAIR bindings, which the
    flag-column route cannot express — declines to honest NIE)."""
    import polars as pl
    if len(join_aliases) != 1 or not isinstance(out_col, str) or not out_col:
        return None
    alias = join_aliases[0]
    left = _active_table(g)
    if left is None:
        return None
    base_graph = _rows_base_graph(g)
    node_id = base_graph._node or "id"  # Plottable._node: Optional[str]
    join_col = _semi_apply_join_col(left, alias, node_id)
    if join_col is None:
        return None
    keys = _pattern_alias_keys_polars(g, binding_ops, alias, neq)
    if keys is None:
        return None
    key_series = keys.get_column(keys.columns[0])
    if key_series.null_count() > 0:
        # pandas merge matches NaN==NaN join keys; is_in does not — decline the
        # pathological null-id case rather than silently diverge (wave-1 E1).
        return None
    # is_in (not a join): row ORDER preserved trivially; a null LEFT key marks
    # False, same as the pandas merge-no-match path (null -> fill_null(False)).
    marked = left.with_columns(
        pl.col(join_col).is_in(key_series).fill_null(False).alias(out_col)
    )
    return _rewrap(g, marked)


def anti_semi_apply_polars(
    g: Plottable, binding_ops: Sequence[Dict[str, JSONVal]], join_aliases: Sequence[str],
    neq: Optional[Sequence[str]] = None,
) -> Optional[Plottable]:
    """Native polars ``anti_semi_apply``: drop active rows whose alias node participates
    in the pattern (NOT EXISTS / negated pattern predicate). Same v1 bounds as the mark."""
    if len(join_aliases) != 1:
        return None
    alias = join_aliases[0]
    left = _active_table(g)
    if left is None:
        return None
    base_graph = _rows_base_graph(g)
    node_id = base_graph._node or "id"  # Plottable._node: Optional[str]
    join_col = _semi_apply_join_col(left, alias, node_id)
    if join_col is None:
        return None
    keys = _pattern_alias_keys_polars(g, binding_ops, alias, neq)
    if keys is None:
        return None
    key_series = keys.get_column(keys.columns[0])
    import polars as pl
    if key_series.null_count() > 0:
        return None  # see semi_apply_mark_polars: NaN==NaN merge semantics (wave-1 E1)
    # filter (not an anti-join): order preserved; a null LEFT key row SURVIVES like
    # the pandas merge-no-match path (is_in null -> fill_null(False) -> not_ -> True).
    kept = left.filter(pl.col(join_col).is_in(key_series).fill_null(False).not_())
    return _rewrap(g, kept)
