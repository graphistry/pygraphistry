"""
GFQL reference path enumerator / oracle.

Enumerates every fixed-length path for a chain of AST steps, applies local
predicates, enforces same-path comparisons, and returns the nodes/edges that
participate in at least one satisfying path. Designed for very small graphs
and correctness testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import pandas as pd

try:  # Optional dependency for GPU DataFrame support
    import cudf  # type: ignore
except Exception:  # pragma: no cover - executed only when cudf absent
    cudf = None  # type: ignore

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTEdge, ASTNode, ASTObject
from graphistry.compute.chain import Chain
from graphistry.compute.filter_by_dict import filter_by_dict
from graphistry.util import setup_logger

logger = setup_logger(__name__)

ComparisonOp = Literal["==", "!=", "<", "<=", ">", ">="]


@dataclass(frozen=True)
class StepColumnRef:
    """Reference a column on a named step."""

    alias: str
    column: str


@dataclass(frozen=True)
class WhereComparison:
    """Compare two named step columns."""

    left: StepColumnRef
    op: ComparisonOp
    right: StepColumnRef


@dataclass(frozen=True)
class OracleCaps:
    """Safety caps to keep exhaustive enumeration tractable."""

    max_nodes: int = 12
    max_edges: int = 40
    max_length: int = 5
    max_partial_rows: int = 200_000


@dataclass
class OracleResult:
    """Outputs from the enumerator."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    tags: Dict[str, Set[Any]]
    paths: Optional[List[Dict[str, Any]]] = None


@dataclass
class NodePattern:
    index: int
    alias: Optional[str]
    filter_dict: Optional[Dict[str, Any]]
    query: Optional[str]
    required_columns: Set[str] = field(default_factory=set)
    id_column: str = ""


@dataclass
class EdgePattern:
    index: int
    alias: Optional[str]
    direction: str
    filter_dict: Optional[Dict[str, Any]]
    query: Optional[str]
    required_columns: Set[str] = field(default_factory=set)
    id_column: str = ""
    source_column: str = ""
    destination_column: str = ""


def col(alias: str, column: str) -> StepColumnRef:
    """Convenience builder for StepColumnRef."""

    return StepColumnRef(alias=alias, column=column)


def compare(
    left: StepColumnRef, op: ComparisonOp, right: StepColumnRef
) -> WhereComparison:
    """Convenience builder for WhereComparison."""

    return WhereComparison(left=left, op=op, right=right)


def enumerate_chain(
    g: Plottable,
    ops: Sequence[ASTObject],
    where: Optional[Sequence[WhereComparison]] = None,
    include_paths: bool = False,
    caps: Optional[OracleCaps] = None,
) -> OracleResult:
    """
    Enumerate fixed-length GFQL chain on a Plottable.

    Args:
        g: Source Plottable with bound nodes/edges.
        ops: Sequence of ASTNode/ASTEdge objects (alternating, starting/ending with ASTNode).
        where: Optional list of same-path comparisons referencing step names.
        include_paths: Whether to emit explicit path bindings.
        caps: Optional OracleCaps overrides.
    """

    if caps is None:
        caps = OracleCaps()

    if not ops:
        raise ValueError("enumerate_chain requires at least one operation")

    chain_ops = _coerce_ops(ops)

    raw_nodes = g._nodes
    raw_edges = g._edges
    if raw_nodes is None or raw_edges is None:
        raise ValueError("Plottable must have both nodes and edges bound")
    nodes_df = _ensure_pandas_frame(raw_nodes, "nodes")
    edges_df = _ensure_pandas_frame(raw_edges, "edges")
    if len(nodes_df) > caps.max_nodes:
        raise ValueError(
            f"Node count {len(nodes_df)} exceeds enumerator cap {caps.max_nodes}"
        )
    if len(edges_df) > caps.max_edges:
        raise ValueError(
            f"Edge count {len(edges_df)} exceeds enumerator cap {caps.max_edges}"
        )

    node_id_col = g._node
    edge_src_col = g._source
    edge_dst_col = g._destination
    edge_id_col = g._edge

    if node_id_col is None or edge_src_col is None or edge_dst_col is None:
        raise ValueError("Plottable must bind node id, edge source, and edge destination")

    if edge_id_col is None:
        edge_id_col = "__enumerator_edge_id__"
        edges_df = edges_df.copy()
        edges_df[edge_id_col] = edges_df.reset_index().index

    node_steps, edge_steps = _normalize_steps(chain_ops, caps)
    _validate_where(where, node_steps, edge_steps)

    result = _enumerate_internal(
        nodes_df,
        edges_df,
        node_id_col=node_id_col,
        edge_src_col=edge_src_col,
        edge_dst_col=edge_dst_col,
        edge_id_col=edge_id_col,
        node_steps=node_steps,
        edge_steps=edge_steps,
        where=where or [],
        include_paths=include_paths,
        caps=caps,
    )
    return result


def _coerce_ops(ops: Sequence[ASTObject]) -> List[ASTObject]:
    if isinstance(ops, Chain):
        return list(ops.chain)
    if len(ops) == 1 and isinstance(ops[0], Chain):
        return list(ops[0].chain)
    return list(ops)


def _ensure_pandas_frame(df_like: Any, label: str) -> pd.DataFrame:
    """
    Convert an arbitrary DataFrame-like object into a pandas DataFrame copy.

    Supports pandas, cuDF, and any object exposing a ``to_pandas()`` method.
    Falls back to ``pd.DataFrame(...)`` for generic iterables.
    """

    if isinstance(df_like, pd.DataFrame):
        return df_like.copy()

    if cudf is not None and isinstance(df_like, cudf.DataFrame):  # type: ignore[attr-defined]
        return df_like.to_pandas().copy()

    if hasattr(df_like, "to_pandas"):
        converted = df_like.to_pandas()
        if not isinstance(converted, pd.DataFrame):
            converted = pd.DataFrame(converted)
        return converted.copy()

    try:
        return pd.DataFrame(df_like)
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise TypeError(f"{label} data must be convertible to pandas DataFrame") from exc


def _normalize_steps(
    ops: Sequence[ASTObject],
    caps: OracleCaps,
) -> Tuple[List[NodePattern], List[EdgePattern]]:
    node_steps: List[NodePattern] = []
    edge_steps: List[EdgePattern] = []
    expected_node = True

    for idx, op in enumerate(ops):
        if expected_node:
            if not isinstance(op, ASTNode):
                raise ValueError("Chain must start with ASTNode and alternate node/edge")
            node = NodePattern(
                index=len(node_steps),
                alias=op._name,
                filter_dict=op.filter_dict,
                query=op.query,
            )
            node_steps.append(node)
        else:
            if not isinstance(op, ASTEdge):
                raise ValueError("Node steps must be followed by ASTEdge")
            if op.hops not in (None, 1):
                raise ValueError("Enumerator only supports single-hop edge steps")
            edge = EdgePattern(
                index=len(edge_steps),
                alias=op._name,
                direction=op.direction,
                filter_dict=op.edge_match,
                query=op.edge_query,
            )
            edge_steps.append(edge)
        expected_node = not expected_node

    if expected_node:
        raise ValueError("Chain must end with ASTNode")
    if len(node_steps) > caps.max_length:
        raise ValueError(
            f"Chain length {len(node_steps)} exceeds enumerator cap {caps.max_length}"
        )

    for i, node in enumerate(node_steps):
        node.id_column = f"node_{i}_id"
    for i, edge in enumerate(edge_steps):
        edge.id_column = f"edge_{i}_id"
        edge.source_column = f"edge_{i}_src"
        edge.destination_column = f"edge_{i}_dst"

    _ensure_unique_aliases(node_steps, edge_steps)
    return node_steps, edge_steps


def _ensure_unique_aliases(
    node_steps: Sequence[NodePattern], edge_steps: Sequence[EdgePattern]
) -> None:
    seen: Dict[str, str] = {}
    for node in node_steps:
        if node.alias:
            if node.alias in seen:
                raise ValueError(f"Duplicate alias '{node.alias}' detected on steps {seen[node.alias]} and node")
            seen[node.alias] = "node"
    for edge in edge_steps:
        if edge.alias:
            if edge.alias in seen:
                raise ValueError(f"Duplicate alias '{edge.alias}' detected")
            seen[edge.alias] = "edge"


def _validate_where(
    where: Optional[Sequence[WhereComparison]],
    node_steps: Sequence[NodePattern],
    edge_steps: Sequence[EdgePattern],
) -> None:
    if not where:
        return
    aliases = {s.alias for s in node_steps if s.alias} | {
        s.alias for s in edge_steps if s.alias
    }
    for clause in where:
        for ref in (clause.left, clause.right):
            if ref.alias not in aliases:
                raise ValueError(f"WHERE reference alias '{ref.alias}' is undefined")
        if clause.op not in {"==", "!=", "<", "<=", ">", ">="}:
            raise ValueError(f"Unsupported comparison operator '{clause.op}'")


def _enumerate_internal(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_id_col: str,
    edge_src_col: str,
    edge_dst_col: str,
    edge_id_col: str,
    node_steps: Sequence[NodePattern],
    edge_steps: Sequence[EdgePattern],
    where: Sequence[WhereComparison],
    include_paths: bool,
    caps: OracleCaps,
) -> OracleResult:
    alias_requirements = _collect_alias_requirements(where)
    _ensure_alias_id_requirements(alias_requirements, node_steps, edge_steps)
    paths = _build_initial_node_frame(
        nodes_df, node_id_col, node_steps[0], alias_requirements
    )
    current_node_col = node_steps[0].id_column

    for i, edge_step in enumerate(edge_steps):
        node_next = node_steps[i + 1]
        edge_frame = _build_edge_frame(
            edges_df,
            edge_id_col,
            edge_src_col,
            edge_dst_col,
            edge_step,
            alias_requirements,
        )
        paths = paths.merge(
            edge_frame,
            left_on=current_node_col,
            right_on=edge_step.source_column,
            how="inner",
        )
        paths = paths.drop(columns=[edge_step.source_column])
        current_node_col = edge_step.destination_column

        next_frame = _build_initial_node_frame(
            nodes_df, node_id_col, node_next, alias_requirements
        )
        paths = paths.merge(
            next_frame, left_on=current_node_col, right_on=node_next.id_column, how="inner"
        )
        paths = paths.drop(columns=[current_node_col])
        current_node_col = node_next.id_column

        if len(paths) > caps.max_partial_rows:
            raise ValueError(
                f"Path enumeration exceeded cap {caps.max_partial_rows} rows at step {i}"
            )

    if where:
        mask = _apply_where(paths, where)
        paths = paths[mask]

    paths = paths.reset_index(drop=True)

    node_id_columns = [n.id_column for n in node_steps]
    edge_id_columns = [e.id_column for e in edge_steps]

    node_ids: Set[Any] = set()
    for col in node_id_columns:
        node_ids.update(paths[col].tolist())
    nodes_out = nodes_df[nodes_df[node_id_col].isin(node_ids)].reset_index(drop=True)

    edge_ids: Set[Any] = set()
    for col in edge_id_columns:
        if col in paths:
            edge_ids.update(paths[col].tolist())
    edges_out = edges_df[edges_df[edge_id_col].isin(edge_ids)].reset_index(drop=True)

    tags = _build_tags(paths, node_steps, edge_steps)
    path_bindings = _extract_paths(paths, node_steps, edge_steps) if include_paths else None

    return OracleResult(nodes=nodes_out, edges=edges_out, tags=tags, paths=path_bindings)


def _collect_alias_requirements(
    where: Sequence[WhereComparison],
) -> Dict[str, Set[str]]:
    requirements: Dict[str, Set[str]] = {}
    for clause in where:
        for ref in (clause.left, clause.right):
            col_name = "__id__" if ref.column == "id" else ref.column
            requirements.setdefault(ref.alias, set()).add(col_name)
    return requirements


def _ensure_alias_id_requirements(
    requirements: Dict[str, Set[str]],
    node_steps: Sequence[NodePattern],
    edge_steps: Sequence[EdgePattern],
) -> None:
    for node in node_steps:
        if node.alias:
            requirements.setdefault(node.alias, set()).add("__id__")
    for edge in edge_steps:
        if edge.alias:
            requirements.setdefault(edge.alias, set()).add("__id__")


def _build_initial_node_frame(
    nodes_df: pd.DataFrame,
    node_id_col: str,
    pattern: NodePattern,
    alias_requirements: Dict[str, Set[str]],
) -> pd.DataFrame:
    df = nodes_df
    if pattern.filter_dict:
        df = filter_by_dict(df, pattern.filter_dict, engine="pandas")
    if pattern.query:
        df = df.query(pattern.query)
    df = df.copy()
    keep_cols = [node_id_col]
    required = alias_requirements.get(pattern.alias, set()) if pattern.alias else set()
    value_cols = [col for col in required if col != "__id__"]
    for col in value_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' required by alias '{pattern.alias}' not found in nodes dataframe")
        keep_cols.append(col)

    df = df[keep_cols]
    df = df.rename(columns={node_id_col: pattern.id_column})
    if pattern.alias:
        df[f"{pattern.alias}::__id__"] = df[pattern.id_column]
        for col in value_cols:
            df[f"{pattern.alias}::{col}"] = df[col]
        df = df.drop(columns=[c for c in keep_cols[1:] if c in value_cols])
    return df


def _build_edge_frame(
    edges_df: pd.DataFrame,
    edge_id_col: str,
    edge_src_col: str,
    edge_dst_col: str,
    pattern: EdgePattern,
    alias_requirements: Dict[str, Set[str]],
) -> pd.DataFrame:
    df = edges_df
    if pattern.filter_dict:
        df = filter_by_dict(df, pattern.filter_dict, engine="pandas")
    if pattern.query:
        df = df.query(pattern.query)
    df = df.copy()

    if pattern.direction == "reverse":
        df = df.rename(columns={edge_src_col: "__tmp_dst__", edge_dst_col: "__tmp_src__"})
        src_col = "__tmp_src__"
        dst_col = "__tmp_dst__"
    elif pattern.direction == "undirected":
        swapped = df.rename(columns={edge_src_col: "__swap_dst__", edge_dst_col: "__swap_src__"})
        swapped = swapped.rename(columns={"__swap_src__": edge_src_col, "__swap_dst__": edge_dst_col})
        df = pd.concat([df, swapped], ignore_index=True)
        src_col = edge_src_col
        dst_col = edge_dst_col
    else:
        src_col = edge_src_col
        dst_col = edge_dst_col

    keep_cols = [edge_id_col, src_col, dst_col]
    required = alias_requirements.get(pattern.alias, set()) if pattern.alias else set()
    value_cols = [col for col in required if col != "__id__"]
    for col in value_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' required by alias '{pattern.alias}' not found in edges dataframe")
        keep_cols.append(col)

    df = df[keep_cols]
    df = df.rename(
        columns={
            edge_id_col: pattern.id_column,
            src_col: pattern.source_column,
            dst_col: pattern.destination_column,
        }
    )
    if pattern.alias:
        df[f"{pattern.alias}::__id__"] = df[pattern.id_column]
        for col in value_cols:
            df[f"{pattern.alias}::{col}"] = df[col]
        df = df.drop(columns=[c for c in keep_cols[3:] if c in value_cols])
    return df


def _apply_where(
    paths: pd.DataFrame,
    where: Sequence[WhereComparison],
) -> pd.Series:
    mask = pd.Series(True, index=paths.index)
    for clause in where:
        lhs_col = _alias_column_key(clause.left.alias, clause.left.column)
        rhs_col = _alias_column_key(clause.right.alias, clause.right.column)
        if lhs_col not in paths.columns or rhs_col not in paths.columns:
            raise KeyError(f"WHERE comparison references missing column {lhs_col} or {rhs_col}")
        lhs = paths[lhs_col]
        rhs = paths[rhs_col]
        valid = lhs.notna() & rhs.notna()
        comp = _compare(lhs, rhs, clause.op)
        mask &= valid & comp.fillna(False)
    return mask


def _compare(lhs: pd.Series, rhs: pd.Series, op: ComparisonOp) -> pd.Series:
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    raise ValueError(f"Unsupported comparison operator '{op}'")


def _build_tags(
    paths: pd.DataFrame,
    node_steps: Sequence[NodePattern],
    edge_steps: Sequence[EdgePattern],
) -> Dict[str, Set[Any]]:
    tags: Dict[str, Set[Any]] = {}
    for node in node_steps:
        if node.alias:
            col = _alias_column_key(node.alias, "id")
            tags[node.alias] = set(paths[col].tolist()) if col in paths.columns else set()
    for edge in edge_steps:
        if edge.alias:
            col = _alias_column_key(edge.alias, "id")
            tags[edge.alias] = set(paths[col].tolist()) if col in paths.columns else set()
    return tags


def _extract_paths(
    paths: pd.DataFrame,
    node_steps: Sequence[NodePattern],
    edge_steps: Sequence[EdgePattern],
) -> List[Dict[str, Any]]:
    aliases = [n.alias for n in node_steps if n.alias] + [
        e.alias for e in edge_steps if e.alias
    ]
    alias_cols = [_alias_column_key(alias, "id") for alias in aliases]
    available_cols = {col for col in alias_cols if col in paths.columns}
    result: List[Dict[str, Any]] = []
    for _, row in paths.iterrows():
        binding = {
            alias: row[_alias_column_key(alias, "id")]
            for alias in aliases
            if _alias_column_key(alias, "id") in available_cols
        }
        result.append(binding)
    return result


def _alias_column_key(alias: str, column: str) -> str:
    normalized = "__id__" if column == "id" else column
    return f"{alias}::{normalized}"
