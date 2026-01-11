"""DataFrame-based GFQL executor with same-path WHERE planning.

Implements Yannakakis-style semijoin pruning for graph queries.
Works with both pandas (CPU) and cuDF (GPU) via vectorized operations.

All operations use DataFrame merge/groupby/masks - no row iteration.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Set, List, Optional, Any, Tuple

import pandas as pd

from graphistry.Engine import Engine, safe_merge
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject
from graphistry.gfql.ref.enumerator import OracleCaps, OracleResult, enumerate_chain
from graphistry.compute.gfql.same_path_plan import SamePathPlan, plan_same_path
from graphistry.compute.gfql.same_path_types import WhereComparison
from graphistry.compute.gfql.same_path.chain_meta import ChainMeta
from graphistry.compute.gfql.same_path.edge_semantics import EdgeSemantics
from graphistry.compute.gfql.same_path.df_utils import series_values, concat_frames
from graphistry.compute.gfql.same_path.post_prune import (
    apply_non_adjacent_where_post_prune,
    apply_edge_where_post_prune,
)
from graphistry.compute.gfql.same_path.where_filter import (
    filter_edges_by_clauses,
    filter_multihop_by_where,
)
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
        self.meta = ChainMeta.from_chain(inputs.chain, inputs.alias_bindings)
        self.forward_steps: List[Plottable] = []
        self.alias_frames: Dict[str, DataFrameT] = {}
        self._node_column = inputs.graph._node
        self._edge_column = inputs.graph._edge
        self._source_column = inputs.graph._source
        self._destination_column = inputs.graph._destination

    def run(self) -> Plottable:
        """Execute same-path traversal with Yannakakis-style pruning.

        Uses native vectorized implementation for both pandas and cuDF.
        The oracle path is only used for testing/debugging via environment variable.

        Environment variable GRAPHISTRY_CUDF_SAME_PATH_MODE controls behavior:
        - 'auto' (default): Use native path for all engines
        - 'strict': Require cudf when Engine.CUDF is requested, raise if unavailable
        - 'oracle': Use O(n!) reference implementation (TESTING ONLY - never use in production)
        """
        self._forward()
        import os
        mode = os.environ.get(_CUDF_MODE_ENV, "auto").lower()

        if mode == "oracle":
            return self._unsafe_run_test_only_oracle()

        # Check strict mode before running native
        # _should_attempt_gpu() will raise RuntimeError if strict + cudf requested but unavailable
        if mode == "strict":
            self._should_attempt_gpu()  # Raises if cudf unavailable in strict mode

        return self._run_native()

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

        # Forward pruning: apply WHERE clause constraints to captured frames
        self._apply_forward_where_pruning()

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

    def _apply_forward_where_pruning(self) -> None:
        """Apply WHERE clause constraints to prune alias frames forward.

        For each WHERE clause, if one alias has known values from pattern filters,
        propagate those constraints to other aliases in the clause.

        This handles cases like:
        - Chain: a:account -> r -> c:user{id=user1}
        - WHERE: a.owner_id == c.id
        - Since c.id is constrained to {user1}, we prune a to owner_id IN {user1}
        """
        if not self.inputs.where:
            return

        # Iterate until no more pruning happens (fixed-point)
        changed = True
        while changed:
            changed = False
            for clause in self.inputs.where:
                left_alias = clause.left.alias
                right_alias = clause.right.alias
                left_col = clause.left.column
                right_col = clause.right.column

                left_frame = self.alias_frames.get(left_alias)
                right_frame = self.alias_frames.get(right_alias)

                if left_frame is None or right_frame is None:
                    continue
                if left_col not in left_frame.columns or right_col not in right_frame.columns:
                    continue

                if clause.op == "==":
                    # Equality: values must match
                    left_values = series_values(left_frame[left_col])
                    right_values = series_values(right_frame[right_col])
                    common = left_values & right_values

                    # Prune left frame
                    if left_values != common:
                        new_left = left_frame[left_frame[left_col].isin(common)]
                        if len(new_left) < len(left_frame):
                            self.alias_frames[left_alias] = new_left
                            changed = True

                    # Prune right frame
                    if right_values != common:
                        new_right = right_frame[right_frame[right_col].isin(common)]
                        if len(new_right) < len(right_frame):
                            self.alias_frames[right_alias] = new_right
                            changed = True

                elif clause.op == "!=":
                    # Inequality: no simple pruning possible without full join
                    pass

                elif clause.op in {"<", "<=", ">", ">="}:
                    # Min/max constraints: prune based on range overlap
                    self._apply_minmax_forward_prune(
                        clause, left_alias, right_alias, left_col, right_col
                    )
                    # Don't set changed for minmax - it's a one-shot prune

    def _apply_minmax_forward_prune(
        self,
        clause: "WhereComparison",
        left_alias: str,
        right_alias: str,
        left_col: str,
        right_col: str,
    ) -> None:
        """Apply min/max constraint pruning for inequality comparisons.

        For a.score < c.score:
        - Prune a to rows where a.score < max(c.score)
        - Prune c to rows where c.score > min(a.score)
        """
        left_frame = self.alias_frames.get(left_alias)
        right_frame = self.alias_frames.get(right_alias)
        if left_frame is None or right_frame is None:
            return

        left_vals = left_frame[left_col]
        right_vals = right_frame[right_col]

        # Get bounds
        left_min, left_max = left_vals.min(), left_vals.max()
        right_min, right_max = right_vals.min(), right_vals.max()

        if clause.op == "<":
            # left < right: left must be < max(right), right must be > min(left)
            new_left = left_frame[left_vals < right_max]
            new_right = right_frame[right_vals > left_min]
        elif clause.op == "<=":
            new_left = left_frame[left_vals <= right_max]
            new_right = right_frame[right_vals >= left_min]
        elif clause.op == ">":
            # left > right: left must be > min(right), right must be < max(left)
            new_left = left_frame[left_vals > right_min]
            new_right = right_frame[right_vals < left_max]
        elif clause.op == ">=":
            new_left = left_frame[left_vals >= right_min]
            new_right = right_frame[right_vals <= left_max]
        else:
            return

        if len(new_left) < len(left_frame):
            self.alias_frames[left_alias] = new_left
        if len(new_right) < len(right_frame):
            self.alias_frames[right_alias] = new_right

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

    def _unsafe_run_test_only_oracle(self) -> Plottable:
        """O(n!) reference implementation - TESTING ONLY, never call from production code."""
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

    def _run_native(self) -> Plottable:
        """Native vectorized path using backward-prune for same-path filtering."""
        allowed_tags = self._compute_allowed_tags()
        path_state = self._backward_prune(allowed_tags)
        path_state = apply_non_adjacent_where_post_prune(self, path_state)
        path_state = apply_edge_where_post_prune(self, path_state)
        return self._materialize_filtered(path_state)

    # Alias for backwards compatibility
    _run_gpu = _run_native

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
            out[alias] = series_values(frame[id_col])
        return out

    @dataclass
    class _PathState:
        allowed_nodes: Dict[int, Set[Any]]
        allowed_edges: Dict[int, Set[Any]]

    def _backward_prune(self, allowed_tags: Dict[str, Set[Any]]) -> "_PathState":
        """Propagate allowed ids backward across edges to enforce path coherence."""

        self.meta.validate()  # Raises if chain structure is invalid
        node_indices = self.meta.node_indices
        edge_indices = self.meta.edge_indices

        allowed_nodes: Dict[int, Set[Any]] = {}
        allowed_edges: Dict[int, Set[Any]] = {}

        # Seed node allowances from tags or full frames
        for idx in node_indices:
            node_alias = self.meta.alias_for_step(idx)
            frame = self.forward_steps[idx]._nodes
            if frame is None or self._node_column is None:
                continue
            if node_alias and node_alias in allowed_tags:
                allowed_nodes[idx] = set(allowed_tags[node_alias])
            else:
                allowed_nodes[idx] = series_values(frame[self._node_column])

        # Walk edges backward
        for edge_idx, right_node_idx in reversed(list(zip(edge_indices, node_indices[1:]))):
            edge_alias = self.meta.alias_for_step(edge_idx)
            left_node_idx = node_indices[node_indices.index(right_node_idx) - 1]
            edges_df = self.forward_steps[edge_idx]._edges
            if edges_df is None:
                continue

            filtered = edges_df
            edge_op = self.inputs.chain[edge_idx]
            if not isinstance(edge_op, ASTEdge):
                continue
            sem = EdgeSemantics.from_edge(edge_op)

            # For single-hop edges, filter by allowed dst first
            # For multi-hop, defer dst filtering to _filter_multihop_by_where
            # For reverse edges, "dst" in traversal = "src" in edge data
            # For undirected edges, "dst" can be either src or dst column
            if not sem.is_multihop:
                allowed_dst = allowed_nodes.get(right_node_idx)
                if allowed_dst is not None:
                    if sem.is_undirected:
                        # Undirected: right node can be reached via either src or dst column
                        if self._source_column and self._destination_column:
                            dst_list = list(allowed_dst)
                            filtered = filtered[
                                filtered[self._source_column].isin(dst_list)
                                | filtered[self._destination_column].isin(dst_list)
                            ]
                    else:
                        # For directed edges, filter by the "end" column
                        _, end_col = sem.endpoint_cols(self._source_column or '', self._destination_column or '')
                        if end_col and end_col in filtered.columns:
                            filtered = filtered[
                                filtered[end_col].isin(list(allowed_dst))
                            ]

            # Apply value-based clauses between adjacent aliases
            left_alias = self.meta.alias_for_step(left_node_idx)
            right_alias = self.meta.alias_for_step(right_node_idx)
            if left_alias and right_alias:
                if not sem.is_multihop:
                    # Single-hop: filter edges directly
                    filtered = filter_edges_by_clauses(
                        self, filtered, left_alias, right_alias, allowed_nodes, sem
                    )
                else:
                    # Multi-hop: filter nodes first, then keep connecting edges
                    filtered = filter_multihop_by_where(
                        self, filtered, edge_op, left_alias, right_alias, allowed_nodes
                    )

            if edge_alias and edge_alias in allowed_tags:
                allowed_edge_ids = allowed_tags[edge_alias]
                if self._edge_column and self._edge_column in filtered.columns:
                    filtered = filtered[
                        filtered[self._edge_column].isin(list(allowed_edge_ids))
                    ]

            # Update allowed_nodes based on filtered edges
            # For reverse edges, swap src/dst semantics
            # For undirected edges, both src and dst can be either left or right node
            if sem.is_undirected:
                # Undirected: both src and dst can be left or right nodes
                if self._source_column and self._destination_column:
                    all_nodes_in_edges = (
                        series_values(filtered[self._source_column])
                        | series_values(filtered[self._destination_column])
                    )
                    # Right node is constrained by allowed_dst already filtered above
                    current_dst = allowed_nodes.get(right_node_idx, set())
                    allowed_nodes[right_node_idx] = (
                        current_dst & all_nodes_in_edges if current_dst else all_nodes_in_edges
                    )
                    # Left node is any node in the filtered edges
                    current = allowed_nodes.get(left_node_idx, set())
                    allowed_nodes[left_node_idx] = current & all_nodes_in_edges if current else all_nodes_in_edges
            else:
                # Directed: use endpoint_cols to get proper column mapping
                start_col, end_col = sem.endpoint_cols(self._source_column or '', self._destination_column or '')
                if end_col and end_col in filtered.columns:
                    allowed_dst_actual = series_values(filtered[end_col])
                    current_dst = allowed_nodes.get(right_node_idx, set())
                    allowed_nodes[right_node_idx] = (
                        current_dst & allowed_dst_actual if current_dst else allowed_dst_actual
                    )
                if start_col and start_col in filtered.columns:
                    allowed_src = series_values(filtered[start_col])
                    current = allowed_nodes.get(left_node_idx, set())
                    allowed_nodes[left_node_idx] = current & allowed_src if current else allowed_src

            if self._edge_column and self._edge_column in filtered.columns:
                allowed_edges[edge_idx] = series_values(filtered[self._edge_column])

            # Store filtered edges back to ensure WHERE-pruned edges are removed from output
            if len(filtered) < len(edges_df):
                self.forward_steps[edge_idx]._edges = filtered

        return self._PathState(allowed_nodes=allowed_nodes, allowed_edges=allowed_edges)

    def backward_propagate_constraints(
        self,
        path_state: "_PathState",
        start_node_idx: int,
        end_node_idx: int,
    ) -> None:
        """Re-propagate constraints backward through a range of edges.

        Updates path_state in-place by filtering edges and nodes between
        start_node_idx and end_node_idx to reflect new constraints.
        Does NOT apply WHERE clauses - only propagates endpoint constraints.

        This is called after post-prune WHERE evaluation to tighten intermediate
        nodes/edges in the affected range.

        Args:
            path_state: Current path state with allowed_nodes/allowed_edges (modified in-place)
            start_node_idx: Start node index for re-propagation (exclusive)
            end_node_idx: End node index for re-propagation (exclusive)
        """
        from graphistry.compute.gfql.same_path.multihop import (
            filter_multihop_edges_by_endpoints,
            find_multihop_start_nodes,
        )

        src_col = self._source_column
        dst_col = self._destination_column
        edge_id_col = self._edge_column
        node_indices = self.meta.node_indices
        edge_indices = self.meta.edge_indices

        if not src_col or not dst_col:
            return

        relevant_edge_indices = [
            idx for idx in edge_indices if start_node_idx < idx < end_node_idx
        ]

        for edge_idx in reversed(relevant_edge_indices):
            edge_pos = edge_indices.index(edge_idx)
            left_node_idx = node_indices[edge_pos]
            right_node_idx = node_indices[edge_pos + 1]

            edges_df = self.forward_steps[edge_idx]._edges
            if edges_df is None:
                continue

            original_len = len(edges_df)
            allowed_edges = path_state.allowed_edges.get(edge_idx, None)
            if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
                edges_df = edges_df[edges_df[edge_id_col].isin(list(allowed_edges))]

            edge_op = self.inputs.chain[edge_idx]
            if not isinstance(edge_op, ASTEdge):
                continue
            sem = EdgeSemantics.from_edge(edge_op)

            left_allowed = path_state.allowed_nodes.get(left_node_idx, set())
            right_allowed = path_state.allowed_nodes.get(right_node_idx, set())

            if sem.is_multihop:
                edges_df = filter_multihop_edges_by_endpoints(
                    edges_df, edge_op, left_allowed, right_allowed, sem,
                    src_col, dst_col
                )
            else:
                if sem.is_undirected:
                    if left_allowed and right_allowed:
                        left_set = list(left_allowed)
                        right_set = list(right_allowed)
                        mask = (
                            (edges_df[src_col].isin(left_set) & edges_df[dst_col].isin(right_set))
                            | (edges_df[dst_col].isin(left_set) & edges_df[src_col].isin(right_set))
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
                else:
                    start_col, end_col = sem.endpoint_cols(src_col, dst_col)
                    if left_allowed:
                        edges_df = edges_df[edges_df[start_col].isin(list(left_allowed))]
                    if right_allowed:
                        edges_df = edges_df[edges_df[end_col].isin(list(right_allowed))]

            if edge_id_col and edge_id_col in edges_df.columns:
                new_edge_ids = set(edges_df[edge_id_col].tolist())
                if edge_idx in path_state.allowed_edges:
                    path_state.allowed_edges[edge_idx] &= new_edge_ids
                else:
                    path_state.allowed_edges[edge_idx] = new_edge_ids

            if sem.is_multihop:
                new_src_nodes = find_multihop_start_nodes(
                    edges_df, edge_op, right_allowed, sem, src_col, dst_col
                )
            else:
                new_src_nodes = sem.start_nodes(edges_df, src_col, dst_col)

            if left_node_idx in path_state.allowed_nodes:
                path_state.allowed_nodes[left_node_idx] &= new_src_nodes
            else:
                path_state.allowed_nodes[left_node_idx] = new_src_nodes

            # Persist filtered edges
            if len(edges_df) < original_len:
                self.forward_steps[edge_idx]._edges = edges_df

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
        concatenated_edges = concat_frames(edge_frames)
        edges_df = concatenated_edges if concatenated_edges is not None else self.inputs.graph._edges

        if nodes_df is None or edges_df is None or node_id is None or src is None or dst is None:
            raise ValueError("Graph bindings are incomplete for same-path execution")

        # If any node step has an explicitly empty allowed set, the path is broken
        # (e.g., WHERE clause filtered out all nodes at some step)
        if path_state.allowed_nodes:
            for node_set in path_state.allowed_nodes.values():
                if node_set is not None and len(node_set) == 0:
                    # Empty set at a step means no valid paths exist
                    return self._materialize_from_oracle(
                        nodes_df.iloc[0:0], edges_df.iloc[0:0]
                    )

        # Build allowed node/edge DataFrames (vectorized - avoid Python sets where possible)
        # Collect allowed node IDs from path_state
        allowed_node_frames: List[DataFrameT] = []
        if path_state.allowed_nodes:
            for node_set in path_state.allowed_nodes.values():
                if node_set:
                    allowed_node_frames.append(pd.DataFrame({'__node__': list(node_set)}))

        allowed_edge_frames: List[DataFrameT] = []
        if path_state.allowed_edges:
            for edge_set in path_state.allowed_edges.values():
                if edge_set:
                    allowed_edge_frames.append(pd.DataFrame({'__edge__': list(edge_set)}))

        # For multi-hop edges, include all intermediate nodes from the edge frames
        # (path_state.allowed_nodes only tracks start/end of multi-hop traversals)
        has_multihop = any(
            isinstance(op, ASTEdge) and EdgeSemantics.from_edge(op).is_multihop
            for op in self.inputs.chain
        )
        if has_multihop and src in edges_df.columns and dst in edges_df.columns:
            # Include all nodes referenced by edges (vectorized)
            allowed_node_frames.append(
                edges_df[[src]].rename(columns={src: '__node__'})
            )
            allowed_node_frames.append(
                edges_df[[dst]].rename(columns={dst: '__node__'})
            )

        # Combine and dedupe allowed nodes
        if allowed_node_frames:
            allowed_nodes_df = pd.concat(allowed_node_frames, ignore_index=True).drop_duplicates()
            filtered_nodes = nodes_df[nodes_df[node_id].isin(allowed_nodes_df['__node__'])]
        else:
            filtered_nodes = nodes_df.iloc[0:0]

        # Filter edges by allowed nodes (both src AND dst must be in allowed nodes)
        # This ensures that edges from filtered-out paths don't appear in the result
        filtered_edges = edges_df
        if allowed_node_frames:
            filtered_edges = filtered_edges[
                filtered_edges[src].isin(allowed_nodes_df['__node__'])
                & filtered_edges[dst].isin(allowed_nodes_df['__node__'])
            ]
        else:
            filtered_edges = filtered_edges.iloc[0:0]

        # Filter by allowed edge IDs
        if allowed_edge_frames and edge_id and edge_id in filtered_edges.columns:
            allowed_edges_df = pd.concat(allowed_edge_frames, ignore_index=True).drop_duplicates()
            filtered_edges = filtered_edges[filtered_edges[edge_id].isin(allowed_edges_df['__edge__'])]

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
                # Build endpoint IDs DataFrame (vectorized - no Python sets)
                endpoint_ids_df = pd.concat([
                    filtered_edges[[src]].rename(columns={src: '__node__'}),
                    filtered_edges[[dst]].rename(columns={dst: '__node__'})
                ], ignore_index=True).drop_duplicates()
                filtered_nodes = filtered_nodes[
                    filtered_nodes[node_id].isin(endpoint_ids_df['__node__'])
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
