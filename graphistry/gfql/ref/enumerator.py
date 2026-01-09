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
    # Hop labels: node_id -> hop_distance, edge_id -> hop_distance
    node_hop_labels: Optional[Dict[Any, int]] = None
    edge_hop_labels: Optional[Dict[Any, int]] = None


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
        node_frame = _build_node_frame(nodes_df, node_id, node_step, alias_requirements)

        min_hops = edge_step["min_hops"]
        max_hops = edge_step["max_hops"]
        if min_hops == 1 and max_hops == 1:
            paths = paths.merge(
                edge_frame,
                left_on=current,
                right_on=edge_step["src_col"],
                how="inner",
                validate="m:m",
            ).drop(columns=[edge_step["src_col"]])
            current = edge_step["dst_col"]

            paths = paths.merge(
                node_frame,
                left_on=current,
                right_on=node_step["id_col"],
                how="inner",
                validate="m:1",
            )
            paths = paths.drop(columns=[current])
            current = node_step["id_col"]
        else:
            if where:
                raise ValueError("WHERE clauses not supported for multi-hop edges in enumerator")
            if edge_step["alias"] or node_step["alias"]:
                # Alias tagging for multi-hop not yet supported in enumerator
                raise ValueError("Aliases not supported for multi-hop edges in enumerator")

            dest_allowed: Optional[Set[Any]] = None
            if not node_frame.empty:
                dest_allowed = set(node_frame[node_step["id_col"]])

            seeds = paths[current].dropna().tolist()
            bp_result = _bounded_paths(
                seeds, edge_frame, edge_step, dest_allowed, caps
            )

            # Build new path rows preserving prior columns
            new_rows: List[List[Any]] = []
            base_cols = list(paths.columns)
            for row in paths.itertuples(index=False, name=None):
                row_dict = dict(zip(base_cols, row))
                seed_id = row_dict[current]
                for dst in bp_result.seed_to_nodes.get(seed_id, set()):
                    new_rows.append([*row, dst])
            paths = pd.DataFrame(new_rows, columns=[*base_cols, node_step["id_col"]])
            current = node_step["id_col"]

            # Stash edges/nodes and hop labels for final selection
            edge_step["collected_edges"] = bp_result.edges_used
            edge_step["collected_nodes"] = bp_result.nodes_used
            edge_step["collected_edge_hops"] = bp_result.edge_hops
            edge_step["collected_node_hops"] = bp_result.node_hops

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

    # Collect hop labels from all edge steps
    all_node_hops: Dict[Any, int] = {}
    all_edge_hops: Dict[Any, int] = {}
    for edge_step in edge_steps:
        if "collected_node_hops" in edge_step:
            for nid, hop in edge_step["collected_node_hops"].items():
                if nid not in all_node_hops or all_node_hops[nid] > hop:
                    all_node_hops[nid] = hop
        if "collected_edge_hops" in edge_step:
            for eid, hop in edge_step["collected_edge_hops"].items():
                if eid not in all_edge_hops or all_edge_hops[eid] > hop:
                    all_edge_hops[eid] = hop

    # Apply output slicing if specified
    output_node_hops = dict(all_node_hops)
    output_edge_hops = dict(all_edge_hops)
    for edge_step in edge_steps:
        out_min = edge_step.get("output_min_hops")
        out_max = edge_step.get("output_max_hops")
        if out_min is not None or out_max is not None:
            # Filter node hops by output bounds
            output_node_hops = {
                nid: hop for nid, hop in output_node_hops.items()
                if (out_min is None or hop >= out_min) and (out_max is None or hop <= out_max)
            }
            # Filter edge hops by output bounds
            output_edge_hops = {
                eid: hop for eid, hop in output_edge_hops.items()
                if (out_min is None or hop >= out_min) and (out_max is None or hop <= out_max)
            }
            # Also filter collected_edges/collected_nodes for output
            if "collected_edges" in edge_step:
                edge_step["collected_edges"] = {
                    eid for eid in edge_step["collected_edges"]
                    if eid in output_edge_hops
                }
            if "collected_nodes" in edge_step:
                # For node slicing, we need to look at the associated node_step
                pass  # Node filtering handled via output_node_hops

    has_output_slice = any(
        edge_step.get("output_min_hops") is not None or edge_step.get("output_max_hops") is not None
        for edge_step in edge_steps
    )

    # First, collect edges
    edge_ids: Set[Any] = set()
    for edge_step in edge_steps:
        col = edge_step["id_col"]
        if col in paths:
            edge_ids.update(paths[col].tolist())
        if "collected_edges" in edge_step:
            edge_ids.update(edge_step["collected_edges"])
    # If output slicing was applied, filter to edges within output bounds
    if has_output_slice and output_edge_hops:
        edge_ids = edge_ids & set(output_edge_hops.keys())
    edges_out = edges_df[edges_df[edge_id].isin(edge_ids)].reset_index(drop=True)

    # Collect nodes: include endpoints of kept edges (like GFQL does)
    node_ids: Set[Any] = set()
    for node_step in node_steps:
        node_ids.update(paths[node_step["id_col"]].tolist())
    for edge_step in edge_steps:
        if "collected_nodes" in edge_step:
            node_ids.update(edge_step["collected_nodes"])

    # If output slicing, nodes must be endpoints of kept edges (not just within hop range)
    if has_output_slice and not edges_out.empty:
        # Nodes that are endpoints of kept edges
        edge_endpoint_nodes: Set[Any] = set()
        edge_endpoint_nodes.update(edges_out[edge_src].tolist())
        edge_endpoint_nodes.update(edges_out[edge_dst].tolist())
        node_ids = node_ids & edge_endpoint_nodes
    elif has_output_slice:
        # No edges kept, so only nodes within output bounds
        if output_node_hops:
            node_ids = node_ids & set(output_node_hops.keys())
    nodes_out = nodes_df[nodes_df[node_id].isin(node_ids)].reset_index(drop=True)

    tags = _build_tags(paths, node_steps, edge_steps)
    tags = {alias: set(values) for alias, values in tags.items()}
    path_bindings = _extract_paths(paths, node_steps, edge_steps) if include_paths else None

    # Return hop labels (use output-filtered versions if slicing was applied)
    final_node_hops = output_node_hops if has_output_slice else all_node_hops
    final_edge_hops = output_edge_hops if has_output_slice else all_edge_hops

    return OracleResult(
        nodes_out,
        edges_out,
        tags,
        path_bindings,
        node_hop_labels=final_node_hops if final_node_hops else None,
        edge_hop_labels=final_edge_hops if final_edge_hops else None,
    )


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
            if not isinstance(op, ASTEdge):
                raise ValueError("Chain must alternate ASTNode and ASTEdge")
            if op.to_fixed_point:
                raise ValueError("Enumerator does not support to_fixed_point edges")
            # Normalize hop bounds (supports min/max hops for small graphs)
            hop_min = op.min_hops if op.min_hops is not None else (op.hops if isinstance(op.hops, int) else 1)
            hop_max = op.max_hops if op.max_hops is not None else (op.hops if isinstance(op.hops, int) else hop_min)
            if hop_min is None or hop_max is None:
                raise ValueError("Enumerator requires finite hop bounds")
            if hop_min < 0 or hop_max < 0 or hop_min > hop_max:
                raise ValueError(f"Invalid hop bounds min={hop_min}, max={hop_max}")
            edges.append(
                {
                    "alias": op._name,
                    "filter": op.edge_match,
                    "query": op.edge_query,
                    "direction": op.direction,
                    "id_col": f"edge_{len(edges)}",
                    "src_col": f"edge_{len(edges)}_src",
                    "dst_col": f"edge_{len(edges)}_dst",
                    "min_hops": hop_min,
                    "max_hops": hop_max,
                    # New hop label/slice params
                    "output_min_hops": op.output_min_hops,
                    "output_max_hops": op.output_max_hops,
                    "label_node_hops": op.label_node_hops,
                    "label_edge_hops": op.label_edge_hops,
                    "label_seeds": getattr(op, 'label_seeds', False),
                    "source_node_match": op.source_node_match,
                    "destination_node_match": op.destination_node_match,
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
    mask: pd.Series = pd.Series(True, index=paths.index, dtype=bool)
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
        result_bool = result.fillna(False).astype(bool)
        mask &= valid & result_bool
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


@dataclass
class BoundedPathResult:
    """Result from bounded path enumeration with hop tracking."""
    seed_to_nodes: Dict[Any, Set[Any]]
    edges_used: Set[Any]
    nodes_used: Set[Any]
    # Hop labels: entity_id -> minimum hop distance from any seed
    node_hops: Dict[Any, int]
    edge_hops: Dict[Any, int]


def _bounded_paths(
    seeds: Sequence[Any],
    edges_df: pd.DataFrame,
    step: Dict[str, Any],
    dest_allowed: Optional[Set[Any]],
    caps: OracleCaps,
) -> BoundedPathResult:
    """
    Enumerate bounded-hop paths for a single Edge step (direction already normalized in edges_df).
    Returns BoundedPathResult with reachable nodes, edges used, and hop labels.
    """
    src_col, dst_col, edge_id_col = step["src_col"], step["dst_col"], step["id_col"]
    min_hops, max_hops = step["min_hops"], step["max_hops"]
    label_seeds = step.get("label_seeds", False)

    adjacency: Dict[Any, List[Tuple[Any, Any]]] = {}
    for _, row in edges_df.iterrows():
        adjacency.setdefault(row[src_col], []).append((row[edge_id_col], row[dst_col]))

    seed_to_nodes: Dict[Any, Set[Any]] = {}
    edges_used: Set[Any] = set()
    nodes_used: Set[Any] = set()
    # Track minimum hop distance for each node/edge
    node_hops: Dict[Any, int] = {}
    edge_hops: Dict[Any, int] = {}

    for seed in seeds:
        # Phase 1: Explore all paths and find valid destinations (reachable within [min_hops, max_hops])
        # Track both valid paths (for nodes/edges) and all paths (for hop labeling)
        # A path is "valid" if it satisfies min_hops constraint and reaches an allowed destination
        valid_paths: List[Tuple[Any, List[Any], List[Any]]] = []  # (destination, edge_ids, node_ids)
        all_paths: List[Tuple[Any, List[Any], List[Any]]] = []  # for hop labeling
        valid_destinations: Set[Any] = set()

        stack: List[Tuple[Any, int, List[Any], List[Any]]] = [(seed, 0, [], [seed])]
        while stack:
            node, depth, path_edges, path_nodes = stack.pop()
            if depth >= max_hops:
                continue
            for edge_id, dst in adjacency.get(node, []):
                new_depth = depth + 1
                new_path = path_edges + [edge_id]
                new_nodes = path_nodes + [dst]

                # Save every path for hop labeling (minimum hop distance needs all paths)
                all_paths.append((dst, list(new_path), list(new_nodes)))

                # Only mark as valid path/destination if within [min_hops, max_hops] range
                if new_depth >= min_hops:
                    if dest_allowed is None or dst in dest_allowed:
                        valid_destinations.add(dst)
                        seed_to_nodes.setdefault(seed, set()).add(dst)
                        valid_paths.append((dst, list(new_path), list(new_nodes)))

                if new_depth < max_hops:
                    stack.append((dst, new_depth, new_path, new_nodes))

        # Phase 2: Include nodes/edges from valid paths only
        if valid_destinations:
            # Include seed in output since we have valid paths
            nodes_used.add(seed)
            if label_seeds and seed not in node_hops:
                node_hops[seed] = 0

            # Add nodes/edges from valid paths only
            for dst, path_edges, path_nodes in valid_paths:
                edges_used.update(path_edges)
                nodes_used.update(path_nodes)

            # Compute hop labels from ALL paths that reach valid destinations
            for dst, path_edges, path_nodes in all_paths:
                if dst in valid_destinations:
                    # Track hop distances
                    for i, eid in enumerate(path_edges):
                        hop_dist = i + 1
                        if eid not in edge_hops or edge_hops[eid] > hop_dist:
                            edge_hops[eid] = hop_dist
                    for i, nid in enumerate(path_nodes):
                        hop_dist = i
                        if hop_dist == 0 and not label_seeds:
                            continue
                        if nid not in node_hops or node_hops[nid] > hop_dist:
                            node_hops[nid] = hop_dist

        if len(edges_used) > caps.max_edges or len(nodes_used) > caps.max_nodes:
            raise ValueError("Enumerator caps exceeded during bounded hop traversal")

    return BoundedPathResult(
        seed_to_nodes=seed_to_nodes,
        edges_used=edges_used,
        nodes_used=nodes_used,
        node_hops=node_hops,
        edge_hops=edge_hops,
    )


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
