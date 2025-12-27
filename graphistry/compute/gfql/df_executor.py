"""DataFrame-based GFQL executor with same-path WHERE planning.

This module hosts the execution path for GFQL chains that require
same-path predicate enforcement. Works with both pandas and cuDF
DataFrames.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Set, List, Optional, Any, Tuple, cast

import pandas as pd

from graphistry.Engine import Engine, safe_merge
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject
from graphistry.gfql.ref.enumerator import OracleCaps, OracleResult, enumerate_chain
from graphistry.gfql.same_path_plan import SamePathPlan, plan_same_path
from graphistry.gfql.same_path_types import WhereComparison
from graphistry.compute.typing import DataFrameT

AliasKind = Literal["node", "edge"]

__all__ = [
    "AliasBinding",
    "SamePathExecutorInputs",
    "DFSamePathExecutor",
    "build_same_path_inputs",
    "execute_same_path_chain",
]

_CUDF_MODE_ENV = "GRAPHISTRY_CUDF_SAME_PATH_MODE"


@dataclass(frozen=True)
class AliasBinding:
    """Metadata describing which chain step an alias refers to."""

    alias: str
    step_index: int
    kind: AliasKind
    ast: ASTObject


@dataclass(frozen=True)
class SamePathExecutorInputs:
    """Container for all metadata needed by the cuDF executor."""

    graph: Plottable
    chain: Sequence[ASTObject]
    where: Sequence[WhereComparison]
    plan: SamePathPlan
    engine: Engine
    alias_bindings: Dict[str, AliasBinding]
    column_requirements: Dict[str, Set[str]]
    include_paths: bool = False


class DFSamePathExecutor:
    """Runs a forward/backward/forward pass using pandas or cuDF dataframes."""

    def __init__(self, inputs: SamePathExecutorInputs) -> None:
        self.inputs = inputs
        self.forward_steps: List[Plottable] = []
        self.alias_frames: Dict[str, DataFrameT] = {}
        self._node_column = inputs.graph._node
        self._edge_column = inputs.graph._edge
        self._source_column = inputs.graph._source
        self._destination_column = inputs.graph._destination
        self._minmax_summaries: Dict[str, Dict[str, DataFrameT]] = defaultdict(dict)
        self._equality_values: Dict[str, Dict[str, Set[Any]]] = defaultdict(dict)

    def run(self) -> Plottable:
        """Execute full cuDF traversal.

        Currently defaults to an oracle-backed path unless GPU kernels are
        explicitly enabled and available. Alias frames are updated from the
        oracle tags so downstream consumers can inspect per-alias bindings.
        """
        self._forward()
        if self._should_attempt_gpu():
            return self._run_gpu()
        return self._run_oracle()

    def _forward(self) -> None:
        graph = self.inputs.graph
        ops = self.inputs.chain
        self.forward_steps = []

        for idx, op in enumerate(ops):
            if isinstance(op, ASTCall):
                current_g = self.forward_steps[-1] if self.forward_steps else graph
                prev_nodes = None
            else:
                current_g = graph
                prev_nodes = (
                    None if not self.forward_steps else self.forward_steps[-1]._nodes
                )
            g_step = op(
                g=current_g,
                prev_node_wavefront=prev_nodes,
                target_wave_front=None,
                engine=self.inputs.engine,
            )
            self.forward_steps.append(g_step)
            self._capture_alias_frame(op, g_step, idx)

    def _backward(self) -> None:
        raise NotImplementedError

    def _finalize(self) -> Plottable:
        raise NotImplementedError

    def _capture_alias_frame(
        self, op: ASTObject, step_result: Plottable, step_index: int
    ) -> None:
        alias = getattr(op, "_name", None)
        if not alias or alias not in self.inputs.alias_bindings:
            return
        binding = self.inputs.alias_bindings[alias]
        frame = (
            step_result._nodes
            if binding.kind == "node"
            else step_result._edges
        )
        if frame is None:
            kind = "node" if binding.kind == "node" else "edge"
            raise ValueError(
                f"Alias '{alias}' did not produce a {kind} frame"
            )
        required = set(self.inputs.column_requirements.get(alias, set()))
        id_col = self._node_column if binding.kind == "node" else self._edge_column
        if id_col:
            required.add(id_col)
        missing = [col for col in required if col not in frame.columns]
        if missing:
            cols = ", ".join(missing)
            raise ValueError(
                f"Alias '{alias}' missing required columns: {cols}"
            )
        subset_cols = [col for col in required]
        alias_frame = frame[subset_cols].copy()
        self.alias_frames[alias] = alias_frame
        self._capture_minmax(alias, alias_frame, id_col)
        self._capture_equality_values(alias, alias_frame)
        self._apply_ready_clauses()

    # --- Execution selection helpers -------------------------------------------------

    def _should_attempt_gpu(self) -> bool:
        """Decide whether to try GPU kernels for same-path execution."""

        mode = os.environ.get(_CUDF_MODE_ENV, "auto").lower()
        if mode not in {"auto", "oracle", "strict"}:
            mode = "auto"

        # force oracle path
        if mode == "oracle":
            return False

        # only CUDF engine supports GPU fastpath
        if self.inputs.engine != Engine.CUDF:
            return False

        try:  # check cudf presence
            import cudf  # type: ignore  # noqa: F401
        except Exception:
            if mode == "strict":
                raise RuntimeError(
                    "cuDF engine requested with strict mode but cudf is unavailable"
                )
            return False
        return True

    # --- Oracle (CPU) fallback -------------------------------------------------------

    def _run_oracle(self) -> Plottable:
        oracle = enumerate_chain(
            self.inputs.graph,
            self.inputs.chain,
            where=self.inputs.where,
            include_paths=self.inputs.include_paths,
            caps=OracleCaps(
                max_nodes=1000, max_edges=5000, max_length=20, max_partial_rows=1_000_000
            ),
        )
        nodes_df, edges_df = self._apply_oracle_hop_labels(oracle)
        self._update_alias_frames_from_oracle(oracle.tags)
        return self._materialize_from_oracle(nodes_df, edges_df)

    # --- GPU path placeholder --------------------------------------------------------

    def _run_gpu(self) -> Plottable:
        """GPU-style path using captured wavefronts and same-path pruning."""

        allowed_tags = self._compute_allowed_tags()
        path_state = self._backward_prune(allowed_tags)
        # Apply non-adjacent equality constraints after backward prune
        path_state = self._apply_non_adjacent_where_post_prune(path_state)
        return self._materialize_filtered(path_state)

    def _update_alias_frames_from_oracle(
        self, tags: Dict[str, Set[Any]]
    ) -> None:
        """Filter captured frames using oracle tags to ensure path coherence."""

        for alias, binding in self.inputs.alias_bindings.items():
            if alias not in tags:
                # if oracle didn't emit the alias, leave any existing capture intact
                continue
            ids = tags.get(alias, set())
            frame = self._lookup_binding_frame(binding)
            if frame is None:
                continue
            id_col = self._node_column if binding.kind == "node" else self._edge_column
            if id_col is None:
                continue
            filtered = frame[frame[id_col].isin(ids)].copy()
            self.alias_frames[alias] = filtered

    def _lookup_binding_frame(self, binding: AliasBinding) -> Optional[DataFrameT]:
        if binding.step_index >= len(self.forward_steps):
            return None
        step_result = self.forward_steps[binding.step_index]
        return (
            step_result._nodes
            if binding.kind == "node"
            else step_result._edges
        )

    def _materialize_from_oracle(
        self, nodes_df: DataFrameT, edges_df: DataFrameT
    ) -> Plottable:
        """Build a Plottable from oracle node/edge outputs, preserving bindings."""

        g = self.inputs.graph
        edge_id = g._edge
        src = g._source
        dst = g._destination
        node_id = g._node

        if node_id and node_id not in nodes_df.columns:
            raise ValueError(f"Oracle nodes missing id column '{node_id}'")
        if dst and dst not in edges_df.columns:
            raise ValueError(f"Oracle edges missing destination column '{dst}'")
        if src and src not in edges_df.columns:
            raise ValueError(f"Oracle edges missing source column '{src}'")
        if edge_id and edge_id not in edges_df.columns:
            # Enumerators may synthesize an edge id column when original graph lacked one
            if "__enumerator_edge_id__" in edges_df.columns:
                edges_df = edges_df.rename(columns={"__enumerator_edge_id__": edge_id})
            else:
                raise ValueError(f"Oracle edges missing id column '{edge_id}'")

        g_out = g.nodes(nodes_df, node=node_id)
        g_out = g_out.edges(edges_df, source=src, destination=dst, edge=edge_id)
        return g_out

    # --- GPU helpers ---------------------------------------------------------------

    def _compute_allowed_tags(self) -> Dict[str, Set[Any]]:
        """Seed allowed ids from alias frames (post-forward pruning)."""

        out: Dict[str, Set[Any]] = {}
        for alias, binding in self.inputs.alias_bindings.items():
            frame = self.alias_frames.get(alias)
            if frame is None:
                continue
            id_col = self._node_column if binding.kind == "node" else self._edge_column
            if id_col is None or id_col not in frame.columns:
                continue
            out[alias] = self._series_values(frame[id_col])
        return out

    def _are_aliases_adjacent(self, alias1: str, alias2: str) -> bool:
        """Check if two node aliases are exactly one edge apart in the chain."""
        binding1 = self.inputs.alias_bindings.get(alias1)
        binding2 = self.inputs.alias_bindings.get(alias2)
        if binding1 is None or binding2 is None:
            return False
        # Only consider node aliases for adjacency
        if binding1.kind != "node" or binding2.kind != "node":
            return False
        # Adjacent nodes are exactly 2 step indices apart (n-e-n pattern)
        return abs(binding1.step_index - binding2.step_index) == 2

    def _apply_non_adjacent_where_post_prune(
        self, path_state: "_PathState"
    ) -> "_PathState":
        """
        Apply WHERE constraints between non-adjacent aliases after backward prune.

        For equality clauses like a.id == c.id where a and c are 2+ edges apart,
        we need to trace actual paths to find which (start, end) pairs satisfy
        the constraint, then filter nodes/edges accordingly.
        """
        if not self.inputs.where:
            return path_state

        # Find non-adjacent WHERE clauses
        non_adjacent_clauses = []
        for clause in self.inputs.where:
            left_alias = clause.left.alias
            right_alias = clause.right.alias
            if not self._are_aliases_adjacent(left_alias, right_alias):
                left_binding = self.inputs.alias_bindings.get(left_alias)
                right_binding = self.inputs.alias_bindings.get(right_alias)
                if left_binding and right_binding:
                    if left_binding.kind == "node" and right_binding.kind == "node":
                        non_adjacent_clauses.append(clause)

        if not non_adjacent_clauses:
            return path_state

        # Get node and edge indices in chain order
        node_indices: List[int] = []
        edge_indices: List[int] = []
        for idx, op in enumerate(self.inputs.chain):
            if isinstance(op, ASTNode):
                node_indices.append(idx)
            elif isinstance(op, ASTEdge):
                edge_indices.append(idx)

        # Build adjacency for path tracing (forward direction only for now)
        # Maps (src_node_id) -> list of (edge_step_idx, edge_id, dst_node_id)
        src_col = self._source_column
        dst_col = self._destination_column
        edge_id_col = self._edge_column

        if not src_col or not dst_col:
            return path_state

        # For each non-adjacent clause, trace paths and filter
        for clause in non_adjacent_clauses:
            left_alias = clause.left.alias
            right_alias = clause.right.alias
            left_binding = self.inputs.alias_bindings[left_alias]
            right_binding = self.inputs.alias_bindings[right_alias]

            # Ensure left is before right in chain
            if left_binding.step_index > right_binding.step_index:
                left_alias, right_alias = right_alias, left_alias
                left_binding, right_binding = right_binding, left_binding

            start_node_idx = left_binding.step_index
            end_node_idx = right_binding.step_index

            # Get node indices between start and end (inclusive)
            relevant_node_indices = [
                idx for idx in node_indices
                if start_node_idx <= idx <= end_node_idx
            ]
            relevant_edge_indices = [
                idx for idx in edge_indices
                if start_node_idx < idx < end_node_idx
            ]

            # Trace paths from start nodes to end nodes
            start_nodes = path_state.allowed_nodes.get(start_node_idx, set())
            end_nodes = path_state.allowed_nodes.get(end_node_idx, set())

            if not start_nodes or not end_nodes:
                continue

            # Get column values for the constraint
            left_frame = self.alias_frames.get(left_alias)
            right_frame = self.alias_frames.get(right_alias)
            if left_frame is None or right_frame is None:
                continue

            left_col = clause.left.column
            right_col = clause.right.column
            node_id_col = self._node_column
            if not node_id_col:
                continue

            # Build mapping: node_id -> column value for each alias
            left_values_map: Dict[Any, Any] = {}
            for _, row in left_frame.iterrows():
                if node_id_col in row and left_col in row:
                    left_values_map[row[node_id_col]] = row[left_col]

            right_values_map: Dict[Any, Any] = {}
            for _, row in right_frame.iterrows():
                if node_id_col in row and right_col in row:
                    right_values_map[row[node_id_col]] = row[right_col]

            # Trace paths step by step
            # Start with all valid starts
            current_reachable: Dict[Any, Set[Any]] = {
                start: {start} for start in start_nodes
            }  # Maps current_node -> set of original starts that can reach it

            for edge_idx in relevant_edge_indices:
                edges_df = self.forward_steps[edge_idx]._edges
                if edges_df is None:
                    break

                # Filter edges to allowed edges
                allowed_edges = path_state.allowed_edges.get(edge_idx, None)
                if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
                    edges_df = edges_df[edges_df[edge_id_col].isin(list(allowed_edges))]

                edge_op = self.inputs.chain[edge_idx]
                is_reverse = isinstance(edge_op, ASTEdge) and edge_op.direction == "reverse"
                is_undirected = isinstance(edge_op, ASTEdge) and edge_op.direction == "undirected"
                is_multihop = isinstance(edge_op, ASTEdge) and not self._is_single_hop(edge_op)

                if is_multihop:
                    # For multi-hop edges, we need to trace paths through the underlying
                    # graph edges, not just treat it as one hop. Use DFS from current
                    # reachable nodes to find all nodes reachable within min..max hops.
                    min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
                    max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
                        edge_op.hops if edge_op.hops is not None else 1
                    )

                    # Build adjacency from edges
                    adjacency: Dict[Any, List[Any]] = {}
                    for _, row in edges_df.iterrows():
                        if is_undirected:
                            # Undirected: can traverse both ways
                            adjacency.setdefault(row[src_col], []).append(row[dst_col])
                            adjacency.setdefault(row[dst_col], []).append(row[src_col])
                        elif is_reverse:
                            s, d = row[dst_col], row[src_col]
                            adjacency.setdefault(s, []).append(d)
                        else:
                            s, d = row[src_col], row[dst_col]
                            adjacency.setdefault(s, []).append(d)

                    # DFS/BFS to find all reachable nodes within min..max hops
                    next_reachable: Dict[Any, Set[Any]] = {}
                    for start_node, original_starts in current_reachable.items():
                        # BFS from this node
                        # Track: (node, hop_count)
                        queue = [(start_node, 0)]
                        visited_at_hop: Dict[Any, int] = {start_node: 0}

                        while queue:
                            node, hop = queue.pop(0)
                            if hop >= max_hops:
                                continue
                            for neighbor in adjacency.get(node, []):
                                next_hop = hop + 1
                                if neighbor not in visited_at_hop or visited_at_hop[neighbor] > next_hop:
                                    visited_at_hop[neighbor] = next_hop
                                    queue.append((neighbor, next_hop))

                        # Nodes reachable within [min_hops, max_hops] are valid "mid" nodes
                        for node, hop in visited_at_hop.items():
                            if min_hops <= hop <= max_hops:
                                if node not in next_reachable:
                                    next_reachable[node] = set()
                                next_reachable[node].update(original_starts)

                    current_reachable = next_reachable
                else:
                    # Single-hop edge: propagate reachability through one hop
                    next_reachable: Dict[Any, Set[Any]] = {}

                    for _, row in edges_df.iterrows():
                        if is_undirected:
                            # Undirected: can traverse both ways
                            src_val, dst_val = row[src_col], row[dst_col]
                            if src_val in current_reachable:
                                if dst_val not in next_reachable:
                                    next_reachable[dst_val] = set()
                                next_reachable[dst_val].update(current_reachable[src_val])
                            if dst_val in current_reachable:
                                if src_val not in next_reachable:
                                    next_reachable[src_val] = set()
                                next_reachable[src_val].update(current_reachable[dst_val])
                        elif is_reverse:
                            src_val, dst_val = row[dst_col], row[src_col]
                            if src_val in current_reachable:
                                if dst_val not in next_reachable:
                                    next_reachable[dst_val] = set()
                                next_reachable[dst_val].update(current_reachable[src_val])
                        else:
                            src_val, dst_val = row[src_col], row[dst_col]
                            if src_val in current_reachable:
                                if dst_val not in next_reachable:
                                    next_reachable[dst_val] = set()
                                next_reachable[dst_val].update(current_reachable[src_val])

                    current_reachable = next_reachable

            # Now current_reachable maps end_node -> set of starts that can reach it
            # Apply the WHERE clause: filter to (start, end) pairs satisfying constraint
            valid_starts: Set[Any] = set()
            valid_ends: Set[Any] = set()

            for end_node, starts in current_reachable.items():
                if end_node not in end_nodes:
                    continue
                end_value = right_values_map.get(end_node)
                if end_value is None:
                    continue

                for start_node in starts:
                    start_value = left_values_map.get(start_node)
                    if start_value is None:
                        continue

                    # Apply the comparison
                    satisfies = False
                    if clause.op == "==":
                        satisfies = start_value == end_value
                    elif clause.op == "!=":
                        satisfies = start_value != end_value
                    elif clause.op == "<":
                        satisfies = start_value < end_value
                    elif clause.op == "<=":
                        satisfies = start_value <= end_value
                    elif clause.op == ">":
                        satisfies = start_value > end_value
                    elif clause.op == ">=":
                        satisfies = start_value >= end_value

                    if satisfies:
                        valid_starts.add(start_node)
                        valid_ends.add(end_node)

            # Update allowed_nodes for start and end positions
            if start_node_idx in path_state.allowed_nodes:
                path_state.allowed_nodes[start_node_idx] &= valid_starts
            if end_node_idx in path_state.allowed_nodes:
                path_state.allowed_nodes[end_node_idx] &= valid_ends

            # Re-propagate constraints backward from the filtered ends
            # to update intermediate nodes and edges
            self._re_propagate_backward(
                path_state, node_indices, edge_indices,
                start_node_idx, end_node_idx
            )

        return path_state

    def _re_propagate_backward(
        self,
        path_state: "_PathState",
        node_indices: List[int],
        edge_indices: List[int],
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Re-propagate constraints backward after filtering non-adjacent nodes."""
        src_col = self._source_column
        dst_col = self._destination_column
        edge_id_col = self._edge_column

        if not src_col or not dst_col:
            return

        # Walk backward from end to start
        relevant_node_indices = [idx for idx in node_indices if start_idx <= idx <= end_idx]
        relevant_edge_indices = [idx for idx in edge_indices if start_idx < idx < end_idx]

        for edge_idx in reversed(relevant_edge_indices):
            # Find the node indices this edge connects
            edge_pos = edge_indices.index(edge_idx)
            left_node_idx = node_indices[edge_pos]
            right_node_idx = node_indices[edge_pos + 1]

            edges_df = self.forward_steps[edge_idx]._edges
            if edges_df is None:
                continue

            original_len = len(edges_df)

            # Filter by allowed edges
            allowed_edges = path_state.allowed_edges.get(edge_idx, None)
            if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
                edges_df = edges_df[edges_df[edge_id_col].isin(list(allowed_edges))]

            # Get edge direction and check if multi-hop
            edge_op = self.inputs.chain[edge_idx]
            is_reverse = isinstance(edge_op, ASTEdge) and edge_op.direction == "reverse"
            is_multihop = isinstance(edge_op, ASTEdge) and not self._is_single_hop(edge_op)

            # Filter edges by allowed left (src) and right (dst) nodes
            left_allowed = path_state.allowed_nodes.get(left_node_idx, set())
            right_allowed = path_state.allowed_nodes.get(right_node_idx, set())

            is_undirected = isinstance(edge_op, ASTEdge) and edge_op.direction == "undirected"
            if is_multihop:
                # For multi-hop edges, we need to trace valid paths from left_allowed
                # to right_allowed, keeping all edges that participate in valid paths.
                # Simple src/dst filtering would incorrectly remove intermediate edges.
                edges_df = self._filter_multihop_edges_by_endpoints(
                    edges_df, edge_op, left_allowed, right_allowed, is_reverse, is_undirected
                )
            else:
                # Single-hop: filter by src/dst directly
                if is_undirected:
                    # Undirected: edge connects left and right in either direction
                    if left_allowed and right_allowed:
                        left_set = list(left_allowed)
                        right_set = list(right_allowed)
                        # Keep edges where (src in left and dst in right) OR (dst in left and src in right)
                        mask = (
                            (edges_df[src_col].isin(left_set) & edges_df[dst_col].isin(right_set)) |
                            (edges_df[dst_col].isin(left_set) & edges_df[src_col].isin(right_set))
                        )
                        edges_df = edges_df[mask]
                    elif left_allowed:
                        left_set = list(left_allowed)
                        edges_df = edges_df[
                            edges_df[src_col].isin(left_set) | edges_df[dst_col].isin(left_set)
                        ]
                    elif right_allowed:
                        right_set = list(right_allowed)
                        edges_df = edges_df[
                            edges_df[src_col].isin(right_set) | edges_df[dst_col].isin(right_set)
                        ]
                elif is_reverse:
                    # Reverse: src is right side, dst is left side
                    if right_allowed:
                        edges_df = edges_df[edges_df[src_col].isin(list(right_allowed))]
                    if left_allowed:
                        edges_df = edges_df[edges_df[dst_col].isin(list(left_allowed))]
                else:
                    # Forward: src is left side, dst is right side
                    if left_allowed:
                        edges_df = edges_df[edges_df[src_col].isin(list(left_allowed))]
                    if right_allowed:
                        edges_df = edges_df[edges_df[dst_col].isin(list(right_allowed))]

            # Update allowed edges
            if edge_id_col and edge_id_col in edges_df.columns:
                new_edge_ids = set(edges_df[edge_id_col].tolist())
                if edge_idx in path_state.allowed_edges:
                    path_state.allowed_edges[edge_idx] &= new_edge_ids
                else:
                    path_state.allowed_edges[edge_idx] = new_edge_ids

            # Update allowed left (src) nodes based on filtered edges
            if is_multihop:
                # For multi-hop, the "left" nodes are those that can START paths
                # to reach right_allowed within the hop constraints
                new_src_nodes = self._find_multihop_start_nodes(
                    edges_df, edge_op, right_allowed, is_reverse, is_undirected
                )
            else:
                if is_undirected:
                    # Undirected: source nodes can be either src or dst
                    new_src_nodes = set(edges_df[src_col].tolist()) | set(edges_df[dst_col].tolist())
                elif is_reverse:
                    new_src_nodes = set(edges_df[dst_col].tolist())
                else:
                    new_src_nodes = set(edges_df[src_col].tolist())

            if left_node_idx in path_state.allowed_nodes:
                path_state.allowed_nodes[left_node_idx] &= new_src_nodes
            else:
                path_state.allowed_nodes[left_node_idx] = new_src_nodes

            # Persist filtered edges to forward_steps (important when no edge ID column)
            if len(edges_df) < original_len:
                self.forward_steps[edge_idx]._edges = edges_df

    def _filter_multihop_edges_by_endpoints(
        self,
        edges_df: DataFrameT,
        edge_op: ASTEdge,
        left_allowed: Set[Any],
        right_allowed: Set[Any],
        is_reverse: bool,
        is_undirected: bool = False,
    ) -> DataFrameT:
        """
        Filter multi-hop edges to only those participating in valid paths
        from left_allowed to right_allowed.
        """
        src_col = self._source_column
        dst_col = self._destination_column
        edge_id_col = self._edge_column

        if not src_col or not dst_col or not left_allowed or not right_allowed:
            return edges_df

        min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
        max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
            edge_op.hops if edge_op.hops is not None else 1
        )

        # Build adjacency from edges
        adjacency: Dict[Any, List[Tuple[Any, Any]]] = {}
        for row_idx, row in edges_df.iterrows():
            src_val, dst_val = row[src_col], row[dst_col]
            eid = row[edge_id_col] if edge_id_col and edge_id_col in edges_df.columns else row_idx
            if is_undirected:
                # Undirected: can traverse both ways
                adjacency.setdefault(src_val, []).append((eid, dst_val))
                adjacency.setdefault(dst_val, []).append((eid, src_val))
            elif is_reverse:
                adjacency.setdefault(dst_val, []).append((eid, src_val))
            else:
                adjacency.setdefault(src_val, []).append((eid, dst_val))

        # DFS from left_allowed to find paths reaching right_allowed
        valid_edge_ids: Set[Any] = set()

        for start in left_allowed:
            # Track (current_node, path_edges)
            stack: List[Tuple[Any, List[Any]]] = [(start, [])]
            while stack:
                node, path_edges = stack.pop()
                if len(path_edges) >= max_hops:
                    continue
                for eid, next_node in adjacency.get(node, []):
                    new_edges = path_edges + [eid]
                    if next_node in right_allowed and len(new_edges) >= min_hops:
                        # Valid path found - include all edges
                        valid_edge_ids.update(new_edges)
                    if len(new_edges) < max_hops:
                        stack.append((next_node, new_edges))

        # Filter edges to only those in valid paths
        if edge_id_col and edge_id_col in edges_df.columns:
            return edges_df[edges_df[edge_id_col].isin(list(valid_edge_ids))]
        else:
            return edges_df.loc[list(valid_edge_ids)] if valid_edge_ids else edges_df.iloc[:0]

    def _find_multihop_start_nodes(
        self,
        edges_df: DataFrameT,
        edge_op: ASTEdge,
        right_allowed: Set[Any],
        is_reverse: bool,
        is_undirected: bool = False,
    ) -> Set[Any]:
        """
        Find nodes that can start multi-hop paths reaching right_allowed.
        """
        src_col = self._source_column
        dst_col = self._destination_column

        if not src_col or not dst_col or not right_allowed:
            return set()

        min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
        max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
            edge_op.hops if edge_op.hops is not None else 1
        )

        # Build reverse adjacency to trace backward from endpoints
        # For forward edges: we need to find which src nodes can reach dst nodes in right_allowed
        # For reverse edges: we need to find which dst nodes can reach src nodes in right_allowed
        # For undirected: bidirectional so reverse adjacency is same as forward
        reverse_adj: Dict[Any, List[Any]] = {}
        for _, row in edges_df.iterrows():
            src_val, dst_val = row[src_col], row[dst_col]
            if is_undirected:
                # Undirected: bidirectional, so both directions are valid for tracing back
                reverse_adj.setdefault(src_val, []).append(dst_val)
                reverse_adj.setdefault(dst_val, []).append(src_val)
            elif is_reverse:
                # Reverse: traversal goes dst->src, so to trace back we go src->dst
                reverse_adj.setdefault(src_val, []).append(dst_val)
            else:
                # Forward: traversal goes src->dst, so to trace back we go dst->src
                reverse_adj.setdefault(dst_val, []).append(src_val)

        # BFS backward from right_allowed to find all nodes that can reach them
        valid_starts: Set[Any] = set()
        for end_node in right_allowed:
            # Track (node, hops_from_end)
            queue = [(end_node, 0)]
            visited: Dict[Any, int] = {end_node: 0}

            while queue:
                node, hops = queue.pop(0)
                if hops >= max_hops:
                    continue
                for prev_node in reverse_adj.get(node, []):
                    next_hops = hops + 1
                    if prev_node not in visited or visited[prev_node] > next_hops:
                        visited[prev_node] = next_hops
                        queue.append((prev_node, next_hops))

            # Nodes that are min_hops to max_hops away (backward) can be starts
            for node, hops in visited.items():
                if min_hops <= hops <= max_hops:
                    valid_starts.add(node)

        return valid_starts

    def _capture_minmax(
        self, alias: str, frame: DataFrameT, id_col: Optional[str]
    ) -> None:
        if not id_col:
            return
        cols = self.inputs.column_requirements.get(alias, set())
        target_cols = [
            col for col in cols if self.inputs.plan.requires_minmax(alias) and col in frame.columns
        ]
        if not target_cols:
            return
        grouped = frame.groupby(id_col)
        for col in target_cols:
            summary = grouped[col].agg(["min", "max"]).reset_index()
            self._minmax_summaries[alias][col] = summary

    def _capture_equality_values(
        self, alias: str, frame: DataFrameT
    ) -> None:
        cols = self.inputs.column_requirements.get(alias, set())
        participates = any(
            alias in bitset.aliases for bitset in self.inputs.plan.bitsets.values()
        )
        if not participates:
            return
        for col in cols:
            if col in frame.columns:
                self._equality_values[alias][col] = self._series_values(frame[col])

    @dataclass
    class _PathState:
        allowed_nodes: Dict[int, Set[Any]]
        allowed_edges: Dict[int, Set[Any]]

    def _backward_prune(self, allowed_tags: Dict[str, Set[Any]]) -> "_PathState":
        """Propagate allowed ids backward across edges to enforce path coherence."""

        node_indices: List[int] = []
        edge_indices: List[int] = []
        for idx, op in enumerate(self.inputs.chain):
            if isinstance(op, ASTNode):
                node_indices.append(idx)
            elif isinstance(op, ASTEdge):
                edge_indices.append(idx)
        if not node_indices:
            raise ValueError("Same-path executor requires at least one node step")
        if len(node_indices) != len(edge_indices) + 1:
            raise ValueError("Chain must alternate node/edge steps for same-path execution")

        allowed_nodes: Dict[int, Set[Any]] = {}
        allowed_edges: Dict[int, Set[Any]] = {}

        # Seed node allowances from tags or full frames
        for idx in node_indices:
            node_alias = self._alias_for_step(idx)
            frame = self.forward_steps[idx]._nodes
            if frame is None or self._node_column is None:
                continue
            if node_alias and node_alias in allowed_tags:
                allowed_nodes[idx] = set(allowed_tags[node_alias])
            else:
                allowed_nodes[idx] = self._series_values(frame[self._node_column])

        # Walk edges backward
        for edge_idx, right_node_idx in reversed(list(zip(edge_indices, node_indices[1:]))):
            edge_alias = self._alias_for_step(edge_idx)
            left_node_idx = node_indices[node_indices.index(right_node_idx) - 1]
            edges_df = self.forward_steps[edge_idx]._edges
            if edges_df is None:
                continue

            filtered = edges_df
            edge_op = self.inputs.chain[edge_idx]
            is_multihop = isinstance(edge_op, ASTEdge) and not self._is_single_hop(edge_op)
            is_reverse = isinstance(edge_op, ASTEdge) and edge_op.direction == "reverse"

            # For single-hop edges, filter by allowed dst first
            # For multi-hop, defer dst filtering to _filter_multihop_by_where
            # For reverse edges, "dst" in traversal = "src" in edge data
            if not is_multihop:
                allowed_dst = allowed_nodes.get(right_node_idx)
                if allowed_dst is not None:
                    if is_reverse:
                        if self._source_column and self._source_column in filtered.columns:
                            filtered = filtered[
                                filtered[self._source_column].isin(list(allowed_dst))
                            ]
                    else:
                        if self._destination_column and self._destination_column in filtered.columns:
                            filtered = filtered[
                                filtered[self._destination_column].isin(list(allowed_dst))
                            ]

            # Apply value-based clauses between adjacent aliases
            left_alias = self._alias_for_step(left_node_idx)
            right_alias = self._alias_for_step(right_node_idx)
            if isinstance(edge_op, ASTEdge) and left_alias and right_alias:
                if self._is_single_hop(edge_op):
                    # Single-hop: filter edges directly
                    filtered = self._filter_edges_by_clauses(
                        filtered, left_alias, right_alias, allowed_nodes, is_reverse
                    )
                else:
                    # Multi-hop: filter nodes first, then keep connecting edges
                    filtered = self._filter_multihop_by_where(
                        filtered, edge_op, left_alias, right_alias, allowed_nodes
                    )

            if edge_alias and edge_alias in allowed_tags:
                allowed_edge_ids = allowed_tags[edge_alias]
                if self._edge_column and self._edge_column in filtered.columns:
                    filtered = filtered[
                        filtered[self._edge_column].isin(list(allowed_edge_ids))
                    ]

            # Update allowed_nodes based on filtered edges
            # For reverse edges, swap src/dst semantics
            if is_reverse:
                # Reverse: right node reached via src, left node via dst
                if self._source_column and self._source_column in filtered.columns:
                    allowed_dst_actual = self._series_values(filtered[self._source_column])
                    current_dst = allowed_nodes.get(right_node_idx, set())
                    allowed_nodes[right_node_idx] = (
                        current_dst & allowed_dst_actual if current_dst else allowed_dst_actual
                    )
                if self._destination_column and self._destination_column in filtered.columns:
                    allowed_src = self._series_values(filtered[self._destination_column])
                    current = allowed_nodes.get(left_node_idx, set())
                    allowed_nodes[left_node_idx] = current & allowed_src if current else allowed_src
            else:
                # Forward: right node reached via dst, left node via src
                if self._destination_column and self._destination_column in filtered.columns:
                    allowed_dst_actual = self._series_values(filtered[self._destination_column])
                    current_dst = allowed_nodes.get(right_node_idx, set())
                    allowed_nodes[right_node_idx] = (
                        current_dst & allowed_dst_actual if current_dst else allowed_dst_actual
                    )
                if self._source_column and self._source_column in filtered.columns:
                    allowed_src = self._series_values(filtered[self._source_column])
                    current = allowed_nodes.get(left_node_idx, set())
                    allowed_nodes[left_node_idx] = current & allowed_src if current else allowed_src

            if self._edge_column and self._edge_column in filtered.columns:
                allowed_edges[edge_idx] = self._series_values(filtered[self._edge_column])

            # Store filtered edges back to ensure WHERE-pruned edges are removed from output
            if len(filtered) < len(edges_df):
                self.forward_steps[edge_idx]._edges = filtered

        return self._PathState(allowed_nodes=allowed_nodes, allowed_edges=allowed_edges)

    def _filter_edges_by_clauses(
        self,
        edges_df: DataFrameT,
        left_alias: str,
        right_alias: str,
        allowed_nodes: Dict[int, Set[Any]],
        is_reverse: bool = False,
    ) -> DataFrameT:
        """Filter edges using WHERE clauses that connect adjacent aliases.

        For forward edges: left_alias matches src, right_alias matches dst.
        For reverse edges: left_alias matches dst, right_alias matches src.
        """

        relevant = [
            clause
            for clause in self.inputs.where
            if {clause.left.alias, clause.right.alias} == {left_alias, right_alias}
        ]
        if not relevant or not self._source_column or not self._destination_column:
            return edges_df

        left_frame = self.alias_frames.get(left_alias)
        right_frame = self.alias_frames.get(right_alias)
        if left_frame is None or right_frame is None or self._node_column is None:
            return edges_df

        out_df = edges_df
        left_allowed = allowed_nodes.get(self.inputs.alias_bindings[left_alias].step_index)
        right_allowed = allowed_nodes.get(self.inputs.alias_bindings[right_alias].step_index)

        lf = left_frame
        rf = right_frame
        if left_allowed is not None:
            lf = lf[lf[self._node_column].isin(list(left_allowed))]
        if right_allowed is not None:
            rf = rf[rf[self._node_column].isin(list(right_allowed))]

        left_cols = list(self.inputs.column_requirements.get(left_alias, []))
        right_cols = list(self.inputs.column_requirements.get(right_alias, []))
        if self._node_column in left_cols:
            left_cols.remove(self._node_column)
        if self._node_column in right_cols:
            right_cols.remove(self._node_column)

        lf = lf[[self._node_column] + left_cols].rename(columns={self._node_column: "__left_id__"})
        rf = rf[[self._node_column] + right_cols].rename(columns={self._node_column: "__right_id__"})

        # For reverse edges, left_alias is reached via dst column, right_alias via src column
        # For forward edges, left_alias is reached via src column, right_alias via dst column
        if is_reverse:
            left_merge_col = self._destination_column
            right_merge_col = self._source_column
        else:
            left_merge_col = self._source_column
            right_merge_col = self._destination_column

        out_df = out_df.merge(
            lf,
            left_on=left_merge_col,
            right_on="__left_id__",
            how="inner",
        )
        out_df = out_df.merge(
            rf,
            left_on=right_merge_col,
            right_on="__right_id__",
            how="inner",
            suffixes=("", "__r"),
        )

        for clause in relevant:
            left_col = clause.left.column if clause.left.alias == left_alias else clause.right.column
            right_col = clause.right.column if clause.right.alias == right_alias else clause.left.column
            if clause.op in {">", ">=", "<", "<="}:
                out_df = self._apply_inequality_clause(
                    out_df, clause, left_alias, right_alias, left_col, right_col
                )
            else:
                col_left_name = f"__val_left_{left_col}"
                col_right_name = f"__val_right_{right_col}"

                # When left_col == right_col, the right merge adds __r suffix
                # We need to rename them to distinct names for comparison
                rename_map = {}
                if left_col in out_df.columns:
                    rename_map[left_col] = col_left_name
                # Handle right column: could be right_col or right_col__r depending on merge
                right_col_with_suffix = f"{right_col}__r"
                if right_col_with_suffix in out_df.columns:
                    rename_map[right_col_with_suffix] = col_right_name
                elif right_col in out_df.columns and right_col != left_col:
                    rename_map[right_col] = col_right_name

                if rename_map:
                    out_df = out_df.rename(columns=rename_map)

                if col_left_name in out_df.columns and col_right_name in out_df.columns:
                    mask = self._evaluate_clause(out_df[col_left_name], clause.op, out_df[col_right_name])
                    out_df = out_df[mask]

        return out_df

    def _filter_multihop_by_where(
        self,
        edges_df: DataFrameT,
        edge_op: ASTEdge,
        left_alias: str,
        right_alias: str,
        allowed_nodes: Dict[int, Set[Any]],
    ) -> DataFrameT:
        """
        Filter multi-hop edges by WHERE clauses connecting start/end aliases.

        For multi-hop traversals, edges_df contains all edges in the path. The src/dst
        columns represent intermediate connections, not the start/end aliases directly.

        Strategy:
        1. Identify which (start, end) pairs satisfy WHERE clauses
        2. Trace paths to find valid edges: start nodes connect via hop 1, end nodes via last hop
        3. Keep only edges that participate in valid paths
        """
        relevant = [
            clause
            for clause in self.inputs.where
            if {clause.left.alias, clause.right.alias} == {left_alias, right_alias}
        ]
        if not relevant or not self._source_column or not self._destination_column:
            return edges_df

        left_frame = self.alias_frames.get(left_alias)
        right_frame = self.alias_frames.get(right_alias)
        if left_frame is None or right_frame is None or self._node_column is None:
            return edges_df

        # Get hop label column to identify first/last hop edges
        node_label, edge_label = self._resolve_label_cols(edge_op)
        if edge_label is None or edge_label not in edges_df.columns:
            # No hop labels - can't distinguish first/last hop edges
            return edges_df

        # Identify first-hop edges and valid endpoint edges
        hop_col = edges_df[edge_label]
        min_hop = hop_col.min()
        max_hop = hop_col.max()

        first_hop_edges = edges_df[hop_col == min_hop]

        # Get chain min_hops to find valid endpoints
        chain_min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
        # Valid endpoints are at hop >= chain_min_hops (hop label is 1-indexed)
        valid_endpoint_edges = edges_df[hop_col >= chain_min_hops]

        # For reverse edges, the logical direction is opposite to physical direction
        # Forward: start -> hop 1 -> hop 2 -> end (start=src of hop 1, end=dst of last hop)
        # Reverse: start <- hop 1 <- hop 2 <- end (start=dst of hop 1, end=src of last hop)
        # Undirected: edges can be traversed both ways, so both src and dst are potential starts/ends
        is_reverse = edge_op.direction == "reverse"
        is_undirected = edge_op.direction == "undirected"
        if is_undirected:
            # Undirected: start can be either src or dst of first hop
            start_nodes = set(first_hop_edges[self._source_column].tolist()) | \
                          set(first_hop_edges[self._destination_column].tolist())
            # End can be either src or dst of edges at hop >= min_hops
            end_nodes = set(valid_endpoint_edges[self._source_column].tolist()) | \
                        set(valid_endpoint_edges[self._destination_column].tolist())
        elif is_reverse:
            # Reverse: start is dst of first hop, end is src of edges at hop >= min_hops
            start_nodes = set(first_hop_edges[self._destination_column].tolist())
            end_nodes = set(valid_endpoint_edges[self._source_column].tolist())
        else:
            # Forward: start is src of first hop, end is dst of edges at hop >= min_hops
            start_nodes = set(first_hop_edges[self._source_column].tolist())
            end_nodes = set(valid_endpoint_edges[self._destination_column].tolist())

        # Filter to allowed nodes
        left_step_idx = self.inputs.alias_bindings[left_alias].step_index
        right_step_idx = self.inputs.alias_bindings[right_alias].step_index
        if left_step_idx in allowed_nodes and allowed_nodes[left_step_idx]:
            start_nodes &= allowed_nodes[left_step_idx]
        if right_step_idx in allowed_nodes and allowed_nodes[right_step_idx]:
            end_nodes &= allowed_nodes[right_step_idx]

        if not start_nodes or not end_nodes:
            return edges_df.iloc[:0]  # Empty dataframe

        # Build (start, end) pairs that satisfy WHERE
        lf = left_frame[left_frame[self._node_column].isin(list(start_nodes))]
        rf = right_frame[right_frame[self._node_column].isin(list(end_nodes))]

        left_cols = list(self.inputs.column_requirements.get(left_alias, []))
        right_cols = list(self.inputs.column_requirements.get(right_alias, []))
        if self._node_column in left_cols:
            left_cols.remove(self._node_column)
        if self._node_column in right_cols:
            right_cols.remove(self._node_column)

        lf = lf[[self._node_column] + left_cols].rename(columns={self._node_column: "__start_id__"})
        rf = rf[[self._node_column] + right_cols].rename(columns={self._node_column: "__end_id__"})

        # Cross join to get all (start, end) combinations
        lf = lf.assign(__cross_key__=1)
        rf = rf.assign(__cross_key__=1)
        pairs_df = lf.merge(rf, on="__cross_key__", suffixes=("", "__r")).drop(columns=["__cross_key__"])

        # Apply WHERE clauses to filter valid (start, end) pairs
        for clause in relevant:
            left_col = clause.left.column if clause.left.alias == left_alias else clause.right.column
            right_col = clause.right.column if clause.right.alias == right_alias else clause.left.column
            # Handle column name collision from merge - when left_col == right_col,
            # pandas adds __r suffix to the right side columns to avoid collision
            actual_right_col = right_col
            if left_col == right_col and f"{right_col}__r" in pairs_df.columns:
                actual_right_col = f"{right_col}__r"
            if left_col in pairs_df.columns and actual_right_col in pairs_df.columns:
                mask = self._evaluate_clause(pairs_df[left_col], clause.op, pairs_df[actual_right_col])
                pairs_df = pairs_df[mask]

        if len(pairs_df) == 0:
            return edges_df.iloc[:0]

        # Get valid start and end nodes
        valid_starts = set(pairs_df["__start_id__"].tolist())
        valid_ends = set(pairs_df["__end_id__"].tolist())

        # Trace paths from valid_starts to valid_ends to find valid edges
        # Build adjacency from edges_df, tracking row indices for filtering
        src_col = self._source_column
        dst_col = self._destination_column
        edge_id_col = self._edge_column

        # Use row index as edge identifier if no edge ID column
        # For reverse edges, build adjacency in the opposite direction (dst -> src)
        # For undirected edges, build bidirectional adjacency
        adjacency: Dict[Any, List[Tuple[Any, Any]]] = {}
        for row_idx, row in edges_df.iterrows():
            src_val, dst_val = row[src_col], row[dst_col]
            eid = row[edge_id_col] if edge_id_col and edge_id_col in edges_df.columns else row_idx
            if is_undirected:
                # Undirected: can traverse both directions
                adjacency.setdefault(src_val, []).append((eid, dst_val))
                adjacency.setdefault(dst_val, []).append((eid, src_val))
            elif is_reverse:
                # Reverse: traverse from dst to src
                adjacency.setdefault(dst_val, []).append((eid, src_val))
            else:
                # Forward: traverse from src to dst
                adjacency.setdefault(src_val, []).append((eid, dst_val))

        # DFS from valid_starts to find paths to valid_ends
        valid_edge_ids: Set[Any] = set()
        # Use edge_op.max_hops instead of max_hop from hop column, because hop column
        # is unreliable when all nodes can be starts (all edges get labeled as hop 1)
        chain_max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
            edge_op.hops if edge_op.hops is not None else 10
        )
        max_hops_val = int(chain_max_hops)

        for start in valid_starts:
            # Track (current_node, path_edges)
            stack: List[Tuple[Any, List[Any]]] = [(start, [])]
            while stack:
                node, path_edges = stack.pop()
                if len(path_edges) >= max_hops_val:
                    continue
                for eid, dst_val in adjacency.get(node, []):
                    new_edges = path_edges + [eid]
                    if dst_val in valid_ends:
                        # Valid path found - include all edges
                        valid_edge_ids.update(new_edges)
                    if len(new_edges) < max_hops_val:
                        stack.append((dst_val, new_edges))

        # Filter edges to only those in valid paths
        if edge_id_col and edge_id_col in edges_df.columns:
            return edges_df[edges_df[edge_id_col].isin(list(valid_edge_ids))]
        else:
            # Filter by row index
            return edges_df.loc[list(valid_edge_ids)] if valid_edge_ids else edges_df.iloc[:0]

    @staticmethod
    def _is_single_hop(op: ASTEdge) -> bool:
        hop_min = op.min_hops if op.min_hops is not None else (
            op.hops if isinstance(op.hops, int) else 1
        )
        hop_max = op.max_hops if op.max_hops is not None else (
            op.hops if isinstance(op.hops, int) else hop_min
        )
        if hop_min is None or hop_max is None:
            return False
        return hop_min == 1 and hop_max == 1

    def _apply_inequality_clause(
        self,
        out_df: DataFrameT,
        clause: WhereComparison,
        left_alias: str,
        right_alias: str,
        left_col: str,
        right_col: str,
    ) -> DataFrameT:
        left_summary = self._minmax_summaries.get(left_alias, {}).get(left_col)
        right_summary = self._minmax_summaries.get(right_alias, {}).get(right_col)

        # Fall back to raw values if summaries are missing
        lsum = None
        rsum = None
        if left_summary is not None:
            lsum = left_summary.rename(
                columns={
                    left_summary.columns[0]: "__left_id__",
                    "min": f"{left_col}__min",
                    "max": f"{left_col}__max",
                }
            )
        if right_summary is not None:
            rsum = right_summary.rename(
                columns={
                    right_summary.columns[0]: "__right_id__",
                    "min": f"{right_col}__min_r",
                    "max": f"{right_col}__max_r",
                }
            )
        merged = out_df
        if lsum is not None:
            merged = merged.merge(lsum, on="__left_id__", how="inner")
        if rsum is not None:
            merged = merged.merge(rsum, on="__right_id__", how="inner")

        if lsum is None or rsum is None:
            col_left = left_col if left_col in merged.columns else left_col
            col_right = (
                f"{right_col}__r" if f"{right_col}__r" in merged.columns else right_col
            )
            if col_left in merged.columns and col_right in merged.columns:
                mask = self._evaluate_clause(merged[col_left], clause.op, merged[col_right])
                return merged[mask]
            return merged

        l_min = merged.get(f"{left_col}__min")
        l_max = merged.get(f"{left_col}__max")
        r_min = merged.get(f"{right_col}__min_r")
        r_max = merged.get(f"{right_col}__max_r")

        if (
            l_min is None
            or l_max is None
            or r_min is None
            or r_max is None
            or f"{left_col}__min" not in merged.columns
            or f"{left_col}__max" not in merged.columns
            or f"{right_col}__min_r" not in merged.columns
            or f"{right_col}__max_r" not in merged.columns
        ):
            return merged

        if clause.op == ">":
            return merged[merged[f"{left_col}__min"] > merged[f"{right_col}__max_r"]]
        if clause.op == ">=":
            return merged[merged[f"{left_col}__min"] >= merged[f"{right_col}__max_r"]]
        if clause.op == "<":
            return merged[merged[f"{left_col}__max"] < merged[f"{right_col}__min_r"]]
        # <=
        return merged[merged[f"{left_col}__max"] <= merged[f"{right_col}__min_r"]]

    @staticmethod
    def _evaluate_clause(series_left: Any, op: str, series_right: Any) -> Any:
        if op == "==":
            return series_left == series_right
        if op == "!=":
            return series_left != series_right
        if op == ">":
            return series_left > series_right
        if op == ">=":
            return series_left >= series_right
        if op == "<":
            return series_left < series_right
        if op == "<=":
            return series_left <= series_right
        return False

    def _materialize_filtered(self, path_state: "_PathState") -> Plottable:
        """Build result graph from allowed node/edge ids and refresh alias frames."""

        nodes_df = self.inputs.graph._nodes
        node_id = self._node_column
        edge_id = self._edge_column
        src = self._source_column
        dst = self._destination_column

        edge_frames = [
            self.forward_steps[idx]._edges
            for idx, op in enumerate(self.inputs.chain)
            if isinstance(op, ASTEdge) and self.forward_steps[idx]._edges is not None
        ]
        concatenated_edges = self._concat_frames(edge_frames)
        edges_df = concatenated_edges if concatenated_edges is not None else self.inputs.graph._edges

        if nodes_df is None or edges_df is None or node_id is None or src is None or dst is None:
            raise ValueError("Graph bindings are incomplete for same-path execution")

        allowed_node_ids: Set[Any] = (
            set().union(*path_state.allowed_nodes.values()) if path_state.allowed_nodes else set()
        )
        allowed_edge_ids: Set[Any] = (
            set().union(*path_state.allowed_edges.values()) if path_state.allowed_edges else set()
        )

        # For multi-hop edges, include all intermediate nodes from the edge frames
        # (path_state.allowed_nodes only tracks start/end of multi-hop traversals)
        has_multihop = any(
            isinstance(op, ASTEdge) and not self._is_single_hop(op)
            for op in self.inputs.chain
        )
        if has_multihop and src in edges_df.columns and dst in edges_df.columns:
            # Include all nodes referenced by edges
            edge_src_nodes = set(edges_df[src].tolist())
            edge_dst_nodes = set(edges_df[dst].tolist())
            allowed_node_ids = allowed_node_ids | edge_src_nodes | edge_dst_nodes

        filtered_nodes = (
            nodes_df[nodes_df[node_id].isin(list(allowed_node_ids))]
            if allowed_node_ids
            else nodes_df.iloc[0:0]
        )
        filtered_edges = edges_df
        filtered_edges = (
            filtered_edges[filtered_edges[dst].isin(list(allowed_node_ids))]
            if allowed_node_ids
            else filtered_edges.iloc[0:0]
        )
        if allowed_edge_ids and edge_id and edge_id in filtered_edges.columns:
            filtered_edges = filtered_edges[filtered_edges[edge_id].isin(list(allowed_edge_ids))]

        filtered_nodes = self._merge_label_frames(
            filtered_nodes,
            self._collect_label_frames("node"),
            node_id,
        )
        if edge_id is not None:
            filtered_edges = self._merge_label_frames(
                filtered_edges,
                self._collect_label_frames("edge"),
                edge_id,
            )

        filtered_edges = self._apply_output_slices(filtered_edges, "edge")

        has_output_slice = any(
            isinstance(op, ASTEdge)
            and (op.output_min_hops is not None or op.output_max_hops is not None)
            for op in self.inputs.chain
        )
        if has_output_slice:
            if len(filtered_edges) > 0:
                endpoint_ids = set(filtered_edges[src].tolist()) | set(
                    filtered_edges[dst].tolist()
                )
                filtered_nodes = filtered_nodes[
                    filtered_nodes[node_id].isin(list(endpoint_ids))
                ]
            else:
                filtered_nodes = self._apply_output_slices(filtered_nodes, "node")
        else:
            filtered_nodes = self._apply_output_slices(filtered_nodes, "node")

        for alias, binding in self.inputs.alias_bindings.items():
            frame = filtered_nodes if binding.kind == "node" else filtered_edges
            id_col = self._node_column if binding.kind == "node" else self._edge_column
            if id_col is None or id_col not in frame.columns:
                continue
            required = set(self.inputs.column_requirements.get(alias, set()))
            required.add(id_col)
            subset = frame[[c for c in frame.columns if c in required]].copy()
            self.alias_frames[alias] = subset

        return self._materialize_from_oracle(filtered_nodes, filtered_edges)

    @staticmethod
    def _needs_auto_labels(op: ASTEdge) -> bool:
        return bool(
            (op.output_min_hops is not None or op.output_max_hops is not None)
            or (op.min_hops is not None and op.min_hops > 0)
        )

    @staticmethod
    def _resolve_label_cols(op: ASTEdge) -> Tuple[Optional[str], Optional[str]]:
        node_label = op.label_node_hops
        edge_label = op.label_edge_hops
        if DFSamePathExecutor._needs_auto_labels(op):
            node_label = node_label or "__gfql_output_node_hop__"
            edge_label = edge_label or "__gfql_output_edge_hop__"
        return node_label, edge_label

    def _collect_label_frames(self, kind: AliasKind) -> List[DataFrameT]:
        frames: List[DataFrameT] = []
        id_col = self._node_column if kind == "node" else self._edge_column
        if id_col is None:
            return frames
        for idx, op in enumerate(self.inputs.chain):
            if not isinstance(op, ASTEdge):
                continue
            step = self.forward_steps[idx]
            df = step._nodes if kind == "node" else step._edges
            if df is None or id_col not in df.columns:
                continue
            node_label, edge_label = self._resolve_label_cols(op)
            label_col = node_label if kind == "node" else edge_label
            if label_col is None or label_col not in df.columns:
                continue
            frames.append(df[[id_col, label_col]])
        return frames

    @staticmethod
    def _merge_label_frames(
        base_df: DataFrameT,
        label_frames: Sequence[DataFrameT],
        id_col: str,
    ) -> DataFrameT:
        out_df = base_df
        for frame in label_frames:
            label_cols = [c for c in frame.columns if c != id_col]
            if not label_cols:
                continue
            merged = safe_merge(out_df, frame[[id_col] + label_cols], on=id_col, how="left")
            for col in label_cols:
                col_x = f"{col}_x"
                col_y = f"{col}_y"
                if col_x in merged.columns and col_y in merged.columns:
                    merged = merged.assign(**{col: merged[col_x].fillna(merged[col_y])})
                    merged = merged.drop(columns=[col_x, col_y])
            out_df = merged
        return out_df

    def _apply_output_slices(self, df: DataFrameT, kind: AliasKind) -> DataFrameT:
        out_df = df
        for op in self.inputs.chain:
            if not isinstance(op, ASTEdge):
                continue
            if op.output_min_hops is None and op.output_max_hops is None:
                continue
            label_col = self._select_label_col(out_df, op, kind)
            if label_col is None or label_col not in out_df.columns:
                continue
            mask = out_df[label_col].notna()
            if op.output_min_hops is not None:
                mask = mask & (out_df[label_col] >= op.output_min_hops)
            if op.output_max_hops is not None:
                mask = mask & (out_df[label_col] <= op.output_max_hops)
            out_df = out_df[mask]
        return out_df

    def _select_label_col(
        self, df: DataFrameT, op: ASTEdge, kind: AliasKind
    ) -> Optional[str]:
        node_label, edge_label = self._resolve_label_cols(op)
        label_col = node_label if kind == "node" else edge_label
        if label_col and label_col in df.columns:
            return label_col
        hop_like = [c for c in df.columns if "hop" in c]
        return hop_like[0] if hop_like else None

    def _apply_oracle_hop_labels(self, oracle: "OracleResult") -> Tuple[DataFrameT, DataFrameT]:
        nodes_df = oracle.nodes
        edges_df = oracle.edges
        node_id = self._node_column
        edge_id = self._edge_column
        node_labels = oracle.node_hop_labels or {}
        edge_labels = oracle.edge_hop_labels or {}

        node_frames: List[DataFrameT] = []
        edge_frames: List[DataFrameT] = []
        for op in self.inputs.chain:
            if not isinstance(op, ASTEdge):
                continue
            node_label, edge_label = self._resolve_label_cols(op)
            if node_label and node_id and node_id in nodes_df.columns and node_labels:
                node_series = nodes_df[node_id].map(node_labels)
                node_frames.append(pd.DataFrame({node_id: nodes_df[node_id], node_label: node_series}))
            if edge_label and edge_id and edge_id in edges_df.columns and edge_labels:
                edge_series = edges_df[edge_id].map(edge_labels)
                edge_frames.append(pd.DataFrame({edge_id: edges_df[edge_id], edge_label: edge_series}))

        if node_id is not None and node_frames:
            nodes_df = self._merge_label_frames(nodes_df, node_frames, node_id)
        if edge_id is not None and edge_frames:
            edges_df = self._merge_label_frames(edges_df, edge_frames, edge_id)

        return nodes_df, edges_df

    def _alias_for_step(self, step_index: int) -> Optional[str]:
        for alias, binding in self.inputs.alias_bindings.items():
            if binding.step_index == step_index:
                return alias
        return None

    @staticmethod
    def _concat_frames(frames: Sequence[DataFrameT]) -> Optional[DataFrameT]:
        if not frames:
            return None
        first = frames[0]
        if first.__class__.__module__.startswith("cudf"):
            import cudf  # type: ignore

            return cudf.concat(frames, ignore_index=True)
        return pd.concat(frames, ignore_index=True)


    def _apply_ready_clauses(self) -> None:
        if not self.inputs.where:
            return
        ready = [
            clause
            for clause in self.inputs.where
            if clause.left.alias in self.alias_frames
            and clause.right.alias in self.alias_frames
        ]
        for clause in ready:
            self._prune_clause(clause)

    def _prune_clause(self, clause: WhereComparison) -> None:
        if clause.op == "!=":
            return  # No global prune for inequality-yet
        lhs = self.alias_frames[clause.left.alias]
        rhs = self.alias_frames[clause.right.alias]
        left_col = clause.left.column
        right_col = clause.right.column

        if clause.op == "==":
            allowed = self._common_values(lhs[left_col], rhs[right_col])
            self.alias_frames[clause.left.alias] = self._filter_by_values(
                lhs, left_col, allowed
            )
            self.alias_frames[clause.right.alias] = self._filter_by_values(
                rhs, right_col, allowed
            )
        elif clause.op == ">":
            right_min = self._safe_min(rhs[right_col])
            left_max = self._safe_max(lhs[left_col])
            if right_min is not None:
                self.alias_frames[clause.left.alias] = lhs[lhs[left_col] > right_min]
            if left_max is not None:
                self.alias_frames[clause.right.alias] = rhs[rhs[right_col] < left_max]
        elif clause.op == ">=":
            right_min = self._safe_min(rhs[right_col])
            left_max = self._safe_max(lhs[left_col])
            if right_min is not None:
                self.alias_frames[clause.left.alias] = lhs[lhs[left_col] >= right_min]
            if left_max is not None:
                self.alias_frames[clause.right.alias] = rhs[
                    rhs[right_col] <= left_max
                ]
        elif clause.op == "<":
            right_max = self._safe_max(rhs[right_col])
            left_min = self._safe_min(lhs[left_col])
            if right_max is not None:
                self.alias_frames[clause.left.alias] = lhs[lhs[left_col] < right_max]
            if left_min is not None:
                self.alias_frames[clause.right.alias] = rhs[
                    rhs[right_col] > left_min
                ]
        elif clause.op == "<=":
            right_max = self._safe_max(rhs[right_col])
            left_min = self._safe_min(lhs[left_col])
            if right_max is not None:
                self.alias_frames[clause.left.alias] = lhs[
                    lhs[left_col] <= right_max
                ]
            if left_min is not None:
                self.alias_frames[clause.right.alias] = rhs[
                    rhs[right_col] >= left_min
                ]

    @staticmethod
    def _filter_by_values(
        frame: DataFrameT, column: str, values: Set[Any]
    ) -> DataFrameT:
        if not values:
            return frame.iloc[0:0]
        allowed = list(values)
        mask = frame[column].isin(allowed)
        return frame[mask]

    @staticmethod
    def _common_values(series_a: Any, series_b: Any) -> Set[Any]:
        vals_a = DFSamePathExecutor._series_values(series_a)
        vals_b = DFSamePathExecutor._series_values(series_b)
        return vals_a & vals_b

    @staticmethod
    def _series_values(series: Any) -> Set[Any]:
        pandas_series = DFSamePathExecutor._to_pandas_series(series)
        return set(pandas_series.dropna().unique().tolist())

    @staticmethod
    def _safe_min(series: Any) -> Optional[Any]:
        pandas_series = DFSamePathExecutor._to_pandas_series(series).dropna()
        if pandas_series.empty:
            return None
        value = pandas_series.min()
        if pd.isna(value):
            return None
        return value

    @staticmethod
    def _safe_max(series: Any) -> Optional[Any]:
        pandas_series = DFSamePathExecutor._to_pandas_series(series).dropna()
        if pandas_series.empty:
            return None
        value = pandas_series.max()
        if pd.isna(value):
            return None
        return value

    @staticmethod
    def _to_pandas_series(series: Any) -> pd.Series:
        if hasattr(series, "to_pandas"):
            return series.to_pandas()
        if isinstance(series, pd.Series):
            return series
        return pd.Series(series)


def build_same_path_inputs(
    g: Plottable,
    chain: Sequence[ASTObject],
    where: Sequence[WhereComparison],
    engine: Engine,
    include_paths: bool = False,
) -> SamePathExecutorInputs:
    """Construct executor inputs, deriving planner metadata and validations."""

    bindings = _collect_alias_bindings(chain)
    _validate_where_aliases(bindings, where)
    required_columns = _collect_required_columns(where)
    plan = plan_same_path(where)

    return SamePathExecutorInputs(
        graph=g,
        chain=list(chain),
        where=list(where),
        plan=plan,
        engine=engine,
        alias_bindings=bindings,
        column_requirements=required_columns,
        include_paths=include_paths,
    )


def execute_same_path_chain(
    g: Plottable,
    chain: Sequence[ASTObject],
    where: Sequence[WhereComparison],
    engine: Engine,
    include_paths: bool = False,
) -> Plottable:
    """Convenience wrapper used by Chain execution once hooked up."""

    inputs = build_same_path_inputs(g, chain, where, engine, include_paths)
    executor = DFSamePathExecutor(inputs)
    return executor.run()


def _collect_alias_bindings(chain: Sequence[ASTObject]) -> Dict[str, AliasBinding]:
    bindings: Dict[str, AliasBinding] = {}
    for idx, step in enumerate(chain):
        alias = getattr(step, "_name", None)
        if not alias:
            continue
        if not isinstance(alias, str):
            continue
        if isinstance(step, ASTNode):
            kind: AliasKind = "node"
        elif isinstance(step, ASTEdge):
            kind = "edge"
        else:
            continue

        if alias in bindings:
            raise ValueError(f"Duplicate alias '{alias}' detected in chain")
        bindings[alias] = AliasBinding(alias, idx, kind, step)
    return bindings


def _collect_required_columns(
    where: Sequence[WhereComparison],
) -> Dict[str, Set[str]]:
    requirements: Dict[str, Set[str]] = defaultdict(set)
    for clause in where:
        requirements[clause.left.alias].add(clause.left.column)
        requirements[clause.right.alias].add(clause.right.column)
    return {alias: set(cols) for alias, cols in requirements.items()}


def _validate_where_aliases(
    bindings: Dict[str, AliasBinding],
    where: Sequence[WhereComparison],
) -> None:
    if not where:
        return
    referenced = {clause.left.alias for clause in where} | {
        clause.right.alias for clause in where
    }
    missing = sorted(alias for alias in referenced if alias not in bindings)
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"WHERE references aliases with no node/edge bindings: {missing_str}"
        )
