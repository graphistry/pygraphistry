"""Minimal GFQL reference enumerator used as the correctness oracle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import pandas as pd

try:  # Optional GPU dependency
    import cudf  # type: ignore
except Exception:  # pragma: no cover
    cudf = None  # type: ignore

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTEdge, ASTNode, ASTObject
from graphistry.compute.chain import Chain
from graphistry.compute.filter_by_dict import filter_by_dict
ComparisonOp = Literal["==", "!=", "<", "<=", ">", ">="]



@dataclass(frozen=True)
class StepColumnRef:
    alias: str
    column: str


@dataclass(frozen=True)
class WhereComparison:
    left: StepColumnRef
    op: ComparisonOp
    right: StepColumnRef


@dataclass(frozen=True)
class OracleCaps:
    max_nodes: int = 12
    max_edges: int = 40
    max_length: int = 5
    max_partial_rows: int = 200_000


@dataclass
class OracleResult:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    tags: Dict[str, Set[Any]]
    paths: Optional[List[Dict[str, Any]]] = None


def col(alias: str, column: str) -> StepColumnRef:
    return StepColumnRef(alias, column)


def compare(left: StepColumnRef, op: ComparisonOp, right: StepColumnRef) -> WhereComparison:
    return WhereComparison(left, op, right)


def enumerate_chain(
    g: Plottable,
    ops: Sequence[ASTObject],
    where: Optional[Sequence[WhereComparison]] = None,
    include_paths: bool = False,
    caps: Optional[OracleCaps] = None,
) -> OracleResult:
    caps = caps or OracleCaps()
    where = tuple(where or [])

    chain_ops = _coerce_ops(ops)
    nodes_raw, edges_raw = g._nodes, g._edges
    if nodes_raw is None or edges_raw is None:
        raise ValueError("Plottable must have both nodes and edges bound")

    nodes_df = _to_pandas(nodes_raw)
    edges_df = _to_pandas(edges_raw)
    if len(nodes_df) > caps.max_nodes or len(edges_df) > caps.max_edges:
        raise ValueError("Enumerator caps exceeded")

    node_id = g._node
    edge_src = g._source
    edge_dst = g._destination
    edge_id = g._edge or "__enumerator_edge_id__"
    if node_id is None or edge_src is None or edge_dst is None:
        raise ValueError("Plottable must bind node id, edge source, and edge destination")
    if g._edge is None:
        edges_df = edges_df.copy()
        edges_df[edge_id] = edges_df.reset_index().index

    node_steps, edge_steps = _prepare_steps(chain_ops, caps)
    _validate_where(where, node_steps, edge_steps, set(nodes_df.columns), set(edges_df.columns))
    alias_requirements = _collect_alias_requirements(where, node_steps, edge_steps)

    paths = _build_node_frame(nodes_df, node_id, node_steps[0], alias_requirements)
    current = node_steps[0]["id_col"]

    for edge_step, node_step in zip(edge_steps, node_steps[1:]):
        edge_frame = _build_edge_frame(
            edges_df, edge_id, edge_src, edge_dst, edge_step, alias_requirements
        )
        paths = paths.merge(
            edge_frame,
            left_on=current,
            right_on=edge_step["src_col"],
            how="inner",
            validate="m:m",
        ).drop(columns=[edge_step["src_col"]])
        current = edge_step["dst_col"]

        node_frame = _build_node_frame(nodes_df, node_id, node_step, alias_requirements)
        paths = paths.merge(
            node_frame,
            left_on=current,
            right_on=node_step["id_col"],
            how="inner",
            validate="m:1",
        )
        paths = paths.drop(columns=[current])
        current = node_step["id_col"]

        if where:
            ready = _ready_where_clauses(paths, where)
            if ready:
                paths = paths[_apply_where(paths, ready)]

        if len(paths) > caps.max_partial_rows:
            raise ValueError("Path enumeration exceeded partial row cap")

    if where:
        paths = paths[_apply_where(paths, where)]
    seq_cols: List[str] = []
    for i, node_step in enumerate(node_steps):
        seq_cols.append(node_step["id_col"])
        if i < len(edge_steps):
            seq_cols.append(edge_steps[i]["id_col"])
    seq_cols = [col for col in seq_cols if col in paths.columns]
    if seq_cols:
        paths = paths.sort_values(by=seq_cols, kind="mergesort").reset_index(drop=True)
    else:
        paths = paths.reset_index(drop=True)

    node_ids: Set[Any] = set()
    for node_step in node_steps:
        node_ids.update(paths[node_step["id_col"]].tolist())
    nodes_out = nodes_df[nodes_df[node_id].isin(node_ids)].reset_index(drop=True)

    edge_ids: Set[Any] = set()
    for edge_step in edge_steps:
        col = edge_step["id_col"]
        if col in paths:
            edge_ids.update(paths[col].tolist())
    edges_out = edges_df[edges_df[edge_id].isin(edge_ids)].reset_index(drop=True)

    tags = _build_tags(paths, node_steps, edge_steps)
    tags = {alias: set(values) for alias, values in tags.items()}
    path_bindings = _extract_paths(paths, node_steps, edge_steps) if include_paths else None
    return OracleResult(nodes_out, edges_out, tags, path_bindings)


def _coerce_ops(ops: Sequence[ASTObject]) -> List[ASTObject]:
    if isinstance(ops, Chain):
        return list(ops.chain)
    if len(ops) == 1 and isinstance(ops[0], Chain):
        return list(ops[0].chain)
    return list(ops)


def _to_pandas(df_like: Any) -> pd.DataFrame:
    if isinstance(df_like, pd.DataFrame):
        return df_like.copy()
    if cudf is not None and isinstance(df_like, cudf.DataFrame):  # type: ignore[attr-defined]
        return df_like.to_pandas().copy()
    if hasattr(df_like, "to_pandas"):
        converted = df_like.to_pandas()
        return converted.copy() if isinstance(converted, pd.DataFrame) else pd.DataFrame(converted)
    return pd.DataFrame(df_like)


def _prepare_steps(
    ops: Sequence[ASTObject], caps: OracleCaps
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    expect_node = True
    for op in ops:
        if expect_node:
            if not isinstance(op, ASTNode):
                raise ValueError("Chain must alternate ASTNode and ASTEdge")
            nodes.append(
                {
                    "alias": op._name,
                    "filter": op.filter_dict,
                    "query": op.query,
                    "id_col": f"node_{len(nodes)}",
                }
            )
        else:
            if not isinstance(op, ASTEdge) or op.hops not in (None, 1):
                raise ValueError("Enumerator only supports single-hop ASTEdge steps")
            edges.append(
                {
                    "alias": op._name,
                    "filter": op.edge_match,
                    "query": op.edge_query,
                    "direction": op.direction,
                    "id_col": f"edge_{len(edges)}",
                    "src_col": f"edge_{len(edges)}_src",
                    "dst_col": f"edge_{len(edges)}_dst",
                }
            )
        expect_node = not expect_node

    if expect_node:
        raise ValueError("Chain must end with ASTNode")
    if len(nodes) > caps.max_length:
        raise ValueError("Chain length exceeds enumerator cap")

    aliases = set()
    for step in [*nodes, *edges]:
        alias = step["alias"]
        if alias:
            if alias in aliases:
                raise ValueError(f"Duplicate alias '{alias}' detected")
            aliases.add(alias)
    return nodes, edges


def _validate_where(
    where: Sequence[WhereComparison],
    node_steps: Sequence[Dict[str, Any]],
    edge_steps: Sequence[Dict[str, Any]],
    node_columns: Set[str],
    edge_columns: Set[str],
) -> None:
    if not where:
        return
    alias_types: Dict[str, str] = {}
    for step in node_steps:
        if step["alias"]:
            alias_types[step["alias"]] = "node"
    for step in edge_steps:
        if step["alias"]:
            alias_types[step["alias"]] = "edge"
    for clause in where:
        for ref in (clause.left, clause.right):
            alias_kind = alias_types.get(ref.alias)
            if alias_kind is None:
                raise ValueError(f"WHERE reference alias '{ref.alias}' is undefined")
            if ref.column != "id":
                source_cols = node_columns if alias_kind == "node" else edge_columns
                if ref.column not in source_cols:
                    raise KeyError(
                        f"Column '{ref.column}' referenced by alias '{ref.alias}' not found in {alias_kind} dataframe"
                    )
        if clause.op not in {"==", "!=", "<", "<=", ">", ">="}:
            raise ValueError(f"Unsupported comparison operator '{clause.op}'")


def _collect_alias_requirements(
    where: Sequence[WhereComparison],
    node_steps: Sequence[Dict[str, Any]],
    edge_steps: Sequence[Dict[str, Any]],
) -> Dict[str, Set[str]]:
    requirements: Dict[str, Set[str]] = {}
    for step in [*node_steps, *edge_steps]:
        if step["alias"]:
            requirements.setdefault(step["alias"], {"id"})
    for clause in where:
        for ref in (clause.left, clause.right):
            requirements.setdefault(ref.alias, {"id"}).add("id" if ref.column == "id" else ref.column)
    return requirements


def _build_node_frame(
    nodes_df: pd.DataFrame,
    id_column: str,
    step: Dict[str, Any],
    alias_requirements: Dict[str, Set[str]],
) -> pd.DataFrame:
    df = nodes_df
    if step["filter"]:
        df = filter_by_dict(df, step["filter"], engine="pandas")
    if step["query"]:
        df = df.query(step["query"])
    df = df.copy()

    keep = [id_column]
    alias = step["alias"]
    required = alias_requirements.get(alias, set()) if alias else set()
    for col in required:
        if col == "id":
            continue
        if col not in df.columns:
            raise KeyError(f"Column '{col}' required by alias '{alias}' not found in nodes dataframe")
        keep.append(col)
    df = df[keep]
    df = df.rename(columns={id_column: step["id_col"]})

    if alias:
        df[_alias_key(alias, "id")] = df[step["id_col"]]
        for col in required:
            if col == "id":
                continue
            df[_alias_key(alias, col)] = df[col]
        df = df.drop(columns=[c for c in keep[1:] if c in df.columns])
    return df


def _build_edge_frame(
    edges_df: pd.DataFrame,
    edge_id: str,
    edge_src: str,
    edge_dst: str,
    step: Dict[str, Any],
    alias_requirements: Dict[str, Set[str]],
) -> pd.DataFrame:
    df = edges_df
    if step["filter"]:
        df = filter_by_dict(df, step["filter"], engine="pandas")
    if step["query"]:
        df = df.query(step["query"])
    df = df.copy()

    if step["direction"] == "reverse":
        df = df.rename(columns={edge_src: "__src", edge_dst: "__dst"})
        src_col, dst_col = "__dst", "__src"
    elif step["direction"] == "undirected":
        swapped = df.rename(columns={edge_src: "__tmp_src__", edge_dst: "__tmp_dst__"})
        swapped = swapped.rename(columns={"__tmp_src__": edge_dst, "__tmp_dst__": edge_src})
        df = pd.concat([df, swapped], ignore_index=True)
        df = df.drop_duplicates(subset=[edge_id, edge_src, edge_dst], keep="first")
        src_col, dst_col = edge_src, edge_dst
    else:
        src_col, dst_col = edge_src, edge_dst

    keep = [edge_id, src_col, dst_col]
    alias = step["alias"]
    required = alias_requirements.get(alias, set()) if alias else set()
    for col in required:
        if col == "id":
            continue
        if col not in df.columns:
            raise KeyError(f"Column '{col}' required by alias '{alias}' not found in edges dataframe")
        keep.append(col)

    df = df[keep]
    df = df.rename(
        columns={edge_id: step["id_col"], src_col: step["src_col"], dst_col: step["dst_col"]}
    )

    if alias:
        df[_alias_key(alias, "id")] = df[step["id_col"]]
        for col in required:
            if col == "id":
                continue
            df[_alias_key(alias, col)] = df[col]
        df = df.drop(columns=[c for c in keep[3:] if c in df.columns])
    return df


def _apply_where(paths: pd.DataFrame, where: Sequence[WhereComparison]) -> pd.Series:
    mask = pd.Series(True, index=paths.index)
    for clause in where:
        left_key = _alias_key(clause.left.alias, clause.left.column)
        right_key = _alias_key(clause.right.alias, clause.right.column)
        if left_key not in paths.columns or right_key not in paths.columns:
            raise KeyError(f"WHERE comparison references missing column {left_key} or {right_key}")
        left = paths[left_key]
        right = paths[right_key]
        valid = left.notna() & right.notna()
        try:
            result = _compare(left, right, clause.op)
        except Exception:
            result = pd.Series(False, index=paths.index)
        mask &= valid & result.fillna(False)
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
    node_steps: Sequence[Dict[str, Any]],
    edge_steps: Sequence[Dict[str, Any]],
) -> Dict[str, Set[Any]]:
    tags: Dict[str, Set[Any]] = {}
    for step in [*node_steps, *edge_steps]:
        alias = step["alias"]
        if not alias:
            continue
        col = _alias_key(alias, "id")
        tags[alias] = set(paths[col].tolist()) if col in paths.columns else set()
    return tags


def _extract_paths(
    paths: pd.DataFrame,
    node_steps: Sequence[Dict[str, Any]],
    edge_steps: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aliases = [step["alias"] for step in [*node_steps, *edge_steps] if step["alias"]]
    out: List[Dict[str, Any]] = []
    for _, row in paths.iterrows():
        binding: Dict[str, Any] = {}
        for alias in aliases:
            col = _alias_key(alias, "id")
            if col in paths.columns:
                binding[alias] = row[col]
        out.append(binding)
    return out


def _alias_key(alias: str, column: str) -> str:
    normalized = "id" if column in {"id", "__id__"} else column
    return f"{alias}::{normalized}"


def _ready_where_clauses(
    paths: pd.DataFrame, clauses: Sequence[WhereComparison]
) -> List[WhereComparison]:
    cols = set(paths.columns)
    ready: List[WhereComparison] = []
    for clause in clauses:
        left_key = _alias_key(clause.left.alias, clause.left.column)
        right_key = _alias_key(clause.right.alias, clause.right.column)
        if left_key in cols and right_key in cols:
            ready.append(clause)
    return ready
