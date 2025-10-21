"""Mark API helpers for GFQL name-first behavior."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from graphistry.Engine import EngineAbstract
from graphistry.Plottable import Plottable
from graphistry.compute.ast import (
    ASTEdge,
    ASTLet,
    ASTNode,
    ASTObject,
    ASTRef,
)
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
from graphistry.compute.gfql_unified import gfql as gfql_base


QueryT = Union[ASTObject, List[ASTObject], ASTLet, Chain, dict]


def _collect_named_matchers(query: QueryT) -> Tuple[Set[str], Set[str]]:
    node_names: Set[str] = set()
    edge_names: Set[str] = set()

    def visit(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, Chain):
            for op in obj.chain:
                visit(op)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)
        elif isinstance(obj, dict):
            visit(ASTLet(obj))
        elif isinstance(obj, ASTLet):
            for binding in obj.bindings.values():
                visit(binding)
        elif isinstance(obj, ASTRef):
            if obj.chain:
                visit(obj.chain)
        elif isinstance(obj, ASTNode):
            if obj._name:
                node_names.add(obj._name)
        elif isinstance(obj, ASTEdge):
            if obj._name:
                edge_names.add(obj._name)
        elif isinstance(obj, ASTObject):
            # Other ASTObjects (e.g., ASTCall) do not contribute matcher names
            pass

    visit(query)
    return node_names, edge_names


def _ensure_bool_column(df, column: str) -> None:
    if column not in df.columns:
        df[column] = False
    else:
        df[column] = df[column].fillna(False).astype('bool')


def _base_node_columns(g: Plottable) -> List[str]:
    cols: List[str] = []
    if g._node is not None:
        cols.append(g._node)
    return cols


def _base_edge_columns(g: Plottable) -> List[str]:
    cols: List[str] = []
    if g._source is not None:
        cols.append(g._source)
    if g._destination is not None:
        cols.append(g._destination)
    return cols


def mark(
    self: Plottable,
    gfql: QueryT,
    *,
    mode: str = 'auto',
    project: Optional[Union[Dict[str, Iterable[str]], Iterable[str]]] = None,
    name_conflicts: str = 'any',
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
    policy: Optional[Dict[str, Callable]] = None,
) -> Plottable:
    """Execute a GFQL query and surface named matchers as boolean mark columns."""

    if gfql is None:
        raise GFQLTypeError(ErrorCode.E201, "gfql argument is required", field='gfql')

    node_names, edge_names = _collect_named_matchers(gfql)
    result = gfql_base(self, gfql, engine=engine, policy=policy, name_conflicts=name_conflicts)

    nodes_df = result._nodes
    edges_df = result._edges

    if nodes_df is None or edges_df is None:
        raise GFQLTypeError(ErrorCode.E201, "gfql() did not return materialized nodes/edges", suggestion="Run materialize_nodes() before mark()")

    if mode not in {'auto', 'project', 'all'}:
        raise GFQLTypeError(ErrorCode.E201, f"Unknown mark mode '{mode}'", field='mode')

    mark_node_cols: Set[str] = set()
    mark_edge_cols: Set[str] = set()

    has_names = bool(node_names or edge_names)

    if mode == 'auto':
        if has_names:
            mark_node_cols |= node_names
            mark_edge_cols |= edge_names
        else:
            mark_node_cols.add('mark_nodes')
            mark_edge_cols.add('mark_edges')
    elif mode == 'project':
        if not has_names:
            raise GFQLTypeError(
                ErrorCode.E201,
                "mode='project' requires named matchers to project",
                field='mode'
            )
        mark_node_cols |= node_names
        mark_edge_cols |= edge_names
    elif mode == 'all':
        mark_node_cols |= node_names
        mark_edge_cols |= edge_names
        mark_node_cols.add('mark_nodes')
        mark_edge_cols.add('mark_edges')

    # Ensure mark columns exist and are boolean
    for col in mark_node_cols:
        _ensure_bool_column(nodes_df, col)
    for col in mark_edge_cols:
        _ensure_bool_column(edges_df, col)

    if 'mark_nodes' in mark_node_cols:
        nodes_df['mark_nodes'] = True
    if 'mark_edges' in mark_edge_cols:
        edges_df['mark_edges'] = True

    node_project: Optional[Iterable[str]] = None
    edge_project: Optional[Iterable[str]] = None
    if project is not None:
        if isinstance(project, dict):
            node_project = project.get('nodes')
            edge_project = project.get('edges')
        else:
            node_project = project

    base_node_cols = _base_node_columns(result)
    base_edge_cols = _base_edge_columns(result)

    if node_project is not None:
        missing = set(node_project) - set(nodes_df.columns)
        if missing:
            raise GFQLTypeError(
                ErrorCode.E201,
                f"Project references missing node columns: {sorted(missing)}",
                field='project'
            )
        keep = base_node_cols + list(dict.fromkeys(node_project))
        nodes_df = nodes_df[keep]
    elif mode == 'project':
        # Default to keeping named columns when project not specified but mode=project
        keep = base_node_cols + list(dict.fromkeys(mark_node_cols))
        nodes_df = nodes_df[keep]

    if edge_project is not None:
        missing = set(edge_project) - set(edges_df.columns)
        if missing:
            raise GFQLTypeError(
                ErrorCode.E201,
                f"Project references missing edge columns: {sorted(missing)}",
                field='project'
            )
        keep = base_edge_cols + list(dict.fromkeys(edge_project))
        edges_df = edges_df[keep]
    elif mode == 'project':
        keep = base_edge_cols + list(dict.fromkeys(mark_edge_cols))
        edges_df = edges_df[keep]

    # Ensure base columns retained when not projecting
    if base_node_cols:
        for col in base_node_cols:
            if col not in nodes_df.columns:
                nodes_df[col] = result._nodes[col]
    if base_edge_cols:
        for col in base_edge_cols:
            if col not in edges_df.columns:
                edges_df[col] = result._edges[col]

    # Re-bind updated DataFrames to return a new Plottable
    out = result.nodes(nodes_df).edges(edges_df)
    return out
