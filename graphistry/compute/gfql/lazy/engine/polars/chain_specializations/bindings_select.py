"""Seeded fixed-hop bindings feeding a bounded projection, executed on index arrays.

``MATCH (a {id: 1})-[:R]->(b)-[:S]->(c) RETURN c.name, b.id`` lowers to an
alternating traversal, a ``rows(binding_ops=...)`` call and a ``select``. The
canonical route materializes a frame per hop and a full bindings table, then
projects. Here the traversal stays in index arrays and only the projected columns
are ever materialized.

Serves only what it can prove: Polars, directed fixed hops, an indexed seed, and a
projection whose every item is a literal, a bare node alias, or ``alias.column``
over a column that exists. Anything else declines (``None``) to the canonical
route, which stays the oracle for this lane's differential tests.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, cast

if TYPE_CHECKING:
    import polars as pl

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject
from graphistry.compute.gfql.call.support import SelectItem
from graphistry.compute.gfql.index.api import _record_indexed_traversal
from graphistry.compute.typing import DataFrameT
from graphistry.compute.gfql.index.array_bindings import (
    ProjectionItem, project_array_path_bag, try_array_path_bag,
)


def _projection_items(
    middle: Sequence[ASTObject],
    items: Sequence[SelectItem],
    node_columns: Sequence[str],
    edge_columns: Sequence[str],
) -> Optional[List[ProjectionItem]]:
    """Projection items as (name, kind, alias, value), or None if any item is unbounded."""
    node_aliases = {
        op._name for op in middle if isinstance(op, ASTNode) and isinstance(op._name, str)
    }
    edge_aliases = {
        op._name for op in middle if isinstance(op, ASTEdge) and isinstance(op._name, str)
    }
    nodes, edges = set(map(str, node_columns)), set(map(str, edge_columns))
    plan: List[ProjectionItem] = []
    for item in items:
        expression: object
        if isinstance(item, str):
            name, expression = item, item
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            name, expression = str(item[0]), item[1]
        else:
            return None
        if not isinstance(expression, str):
            plan.append((name, "literal", None, expression))
            continue
        if expression in node_aliases:
            plan.append((name, "node_id", expression, None))
            continue
        alias, separator, column = expression.partition(".")
        if not separator:
            return None
        if alias in node_aliases and column in nodes:
            plan.append((name, "node_column", alias, column))
        elif alias in edge_aliases and column in edges:
            plan.append((name, "edge_column", alias, column))
        else:
            return None
    return plan


def _admits_suffix(
    suffix: Sequence[ASTObject], middle: Sequence[ASTObject],
) -> Optional[Sequence[SelectItem]]:
    """The ``select`` items when the suffix is exactly a bare ``rows()`` then a projection."""
    from graphistry.compute.chain import serialize_binding_ops

    if len(suffix) < 2:
        return None
    rows_call, projection = suffix[0], suffix[1]
    if not isinstance(rows_call, ASTCall) or rows_call.function != "rows":
        return None
    if rows_call.params.get("source") is not None or rows_call.params.get("alias_endpoints") is not None:
        return None
    if rows_call.params.get("alias_prefilters") or rows_call.params.get("table", "nodes") != "nodes":
        return None
    binding_ops = rows_call.params.get("binding_ops")
    if not (
        binding_ops == serialize_binding_ops(middle)
        or (binding_ops is None and any(op._name is not None for op in middle))
    ):
        return None
    if not isinstance(projection, ASTCall) or projection.function not in ("select", "return_"):
        return None
    if set(projection.params) != {"items"}:
        return None
    items = projection.params.get("items")
    return items if isinstance(items, list) else None


def rewrap_projected_polars(
    g: Plottable, middle: Sequence[ASTObject], projected: "pl.DataFrame",
) -> Plottable:
    """Publish the projected table the way the canonical route publishes its own.

    ``frame_ops.row_table`` empties the edge frame and synthesizes an alias marker
    column per aliased edge, so a served fast path that skipped the traversal has to
    synthesize the same ones.
    """
    import polars as pl

    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import _rewrap

    out = _rewrap(g, projected)
    edge_aliases = [
        op._name for op in middle if isinstance(op, ASTEdge) and isinstance(op._name, str)
    ]
    if out._edges is not None and edge_aliases:
        missing = [alias for alias in edge_aliases if alias not in out._edges.columns]
        if missing:
            out._edges = out._edges.with_columns([pl.lit(True).alias(alias) for alias in missing])
    out._gfql_rows_edge_aliases = set(edge_aliases)
    return out


def _projection_can_leak_temporal_text(
    g: Plottable, plan: Sequence[ProjectionItem], engine: Engine,
) -> bool:
    """Whether this projection could emit temporal-constructor text.

    Every projected column is a verbatim copy of a source column, so the build-time
    verdict for that column answers it without scanning the result. Without a verdict
    for a frame this returns True, which declines to the canonical route rather than
    guessing. A String literal is checked directly, since it has no source column.
    """
    from graphistry.compute.gfql.index.api import get_registry
    from graphistry.compute.gfql.index.registry import ColStatsRole
    from graphistry.compute.gfql.lazy.engine.polars.projection import (
        _has_temporal_constructor_text_value,
    )

    registry = get_registry(g)
    roles: List[Tuple[ColStatsRole, Optional[DataFrameT]]] = [
        ("nodes", g._nodes), ("edges", g._edges),
    ]
    facts = {role: registry.get_temporal_text_valid(role, frame, engine) for role, frame in roles}
    for _, kind, _, value in plan:
        if kind == "literal":
            if isinstance(value, str) and _has_temporal_constructor_text_value(value):
                return True
            continue
        fact = facts["edges" if kind == "edge_column" else "nodes"]
        if fact is None:
            return True  # no verdict: decline rather than guess
        column = str(g._node) if kind == "node_id" else str(value)
        if fact.verdicts.get(column, False):
            return True
    return False


def try_bindings_select_polars(
    g: Plottable,
    middle: Sequence[ASTObject],
    suffix: Sequence[ASTObject],
    start_nodes: Optional[DataFrameT],
) -> "Optional[pl.DataFrame]":
    """The projected table for a seeded fixed-hop pattern, or None to decline."""
    from graphistry.compute.gfql.lazy import ExecutionTarget, active_target

    if start_nodes is not None or active_target() == ExecutionTarget.GPU:
        return None
    if not middle or not all(isinstance(op, (ASTNode, ASTEdge)) for op in middle):
        return None
    items = _admits_suffix(suffix, middle)
    if items is None:
        return None
    nodes, edges = g._nodes, g._edges
    if nodes is None or edges is None:
        return None
    plan = _projection_items(middle, items, list(nodes.columns), list(edges.columns))
    if plan is None:
        return None
    bag = try_array_path_bag(g, middle, engine=Engine.POLARS)
    if bag is None:
        return None
    bound = {
        (kind, alias)
        for kind, rows in (("node", bag.node_rows), ("edge", bag.edge_rows))
        for alias in rows
    }
    for _, kind, alias, _ in plan:
        if kind != "literal" and ("edge" if kind == "edge_column" else "node", alias) not in bound:
            return None
    if _projection_can_leak_temporal_text(g, plan, Engine.POLARS):
        return None  # raw constructor text belongs to the canonical decline (NIE)
    out = project_array_path_bag(bag, nodes, edges, str(g._node), plan)
    first = middle[0]
    seed_scan = not (
        isinstance(first, ASTNode)
        and isinstance(first.filter_dict, dict)
        and str(g._node) in first.filter_dict
    )
    _record_indexed_traversal(
        seam="connected_bindings",
        engine=Engine.POLARS,
        served=True,
        reason="served",
        hop_count=bag.hop_count,
        public_seed_scan=seed_scan,
        hop_details=[
            {"hop": hop + 1, "estimated_rows": bag.estimated_rows}
            for hop in range(bag.hop_count)
        ],
    )
    return out
