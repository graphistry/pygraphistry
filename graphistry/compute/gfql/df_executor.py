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
from graphistry.compute.gfql.same_path.df_utils import (
    series_values,
    common_values,
    safe_min,
    safe_max,
    filter_by_values,
    evaluate_clause,
    concat_frames,
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


def _build_edge_pairs_from_semantics(
    edges_df: DataFrameT, src_col: str, dst_col: str, sem: EdgeSemantics
) -> DataFrameT:
    """Build normalized edge pairs for BFS traversal based on EdgeSemantics."""
    if sem.is_undirected:
        fwd = edges_df[[src_col, dst_col]].copy()
        fwd.columns = pd.Index(['__from__', '__to__'])
        rev = edges_df[[dst_col, src_col]].copy()
        rev.columns = pd.Index(['__from__', '__to__'])
        return pd.concat([fwd, rev], ignore_index=True).drop_duplicates()
    else:
        join_col, result_col = sem.join_cols(src_col, dst_col)
        pairs = edges_df[[join_col, result_col]].copy()
        pairs.columns = pd.Index(['__from__', '__to__'])
        return pairs


def _build_edge_pairs(
    edges_df: DataFrameT, src_col: str, dst_col: str, is_reverse: bool, is_undirected: bool
) -> DataFrameT:
    """Build normalized edge pairs for BFS traversal based on direction.

    DEPRECATED: Use _build_edge_pairs_from_semantics with EdgeSemantics instead.
    """
    if is_undirected:
        fwd = edges_df[[src_col, dst_col]].copy()
        fwd.columns = pd.Index(['__from__', '__to__'])
        rev = edges_df[[dst_col, src_col]].copy()
        rev.columns = pd.Index(['__from__', '__to__'])
        return pd.concat([fwd, rev], ignore_index=True).drop_duplicates()
    elif is_reverse:
        pairs = edges_df[[dst_col, src_col]].copy()
        pairs.columns = pd.Index(['__from__', '__to__'])
        return pairs
    else:
        pairs = edges_df[[src_col, dst_col]].copy()
        pairs.columns = pd.Index(['__from__', '__to__'])
        return pairs


def _bfs_reachability(
    edge_pairs: DataFrameT, start_nodes: Set[Any], max_hops: int, hop_col: str
) -> DataFrameT:
    """Compute BFS reachability with hop distance tracking. Returns DataFrame with __node__ and hop_col."""
    result = pd.DataFrame({'__node__': list(start_nodes), hop_col: 0})
    all_visited = result.copy()
    for hop in range(1, max_hops + 1):
        frontier = result[result[hop_col] == hop - 1][['__node__']].rename(columns={'__node__': '__from__'})
        if len(frontier) == 0:
            break
        next_df = edge_pairs.merge(frontier, on='__from__', how='inner')[['__to__']].drop_duplicates()
        next_df = next_df.rename(columns={'__to__': '__node__'})
        next_df[hop_col] = hop
        merged = next_df.merge(all_visited[['__node__']], on='__node__', how='left', indicator=True)
        new_nodes = merged[merged['_merge'] == 'left_only'][['__node__', hop_col]]
        if len(new_nodes) == 0:
            break
        result = pd.concat([result, new_nodes], ignore_index=True)
        all_visited = pd.concat([all_visited, new_nodes], ignore_index=True)
    return result


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
        self._minmax_summaries: Dict[str, Dict[str, DataFrameT]] = defaultdict(dict)
        self._equality_values: Dict[str, Dict[str, Set[Any]]] = defaultdict(dict)

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
        path_state = self._apply_non_adjacent_where_post_prune(path_state)
        path_state = self._apply_edge_where_post_prune(path_state)
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

    def _apply_non_adjacent_where_post_prune(
        self, path_state: "_PathState"
    ) -> "_PathState":
        """Apply WHERE on non-adjacent node aliases by tracing paths."""
        if not self.inputs.where:
            return path_state

        non_adjacent_clauses = []
        for clause in self.inputs.where:
            left_alias = clause.left.alias
            right_alias = clause.right.alias
            left_binding = self.inputs.alias_bindings.get(left_alias)
            right_binding = self.inputs.alias_bindings.get(right_alias)
            if left_binding and right_binding:
                if left_binding.kind == "node" and right_binding.kind == "node":
                    # Non-adjacent = step indices differ by more than 2
                    if not self.meta.are_steps_adjacent_nodes(
                        left_binding.step_index, right_binding.step_index
                    ):
                        non_adjacent_clauses.append(clause)

        if not non_adjacent_clauses:
            return path_state

        node_indices = self.meta.node_indices
        edge_indices = self.meta.edge_indices

        src_col = self._source_column
        dst_col = self._destination_column
        edge_id_col = self._edge_column

        if not src_col or not dst_col:
            return path_state

        for clause in non_adjacent_clauses:
            left_alias = clause.left.alias
            right_alias = clause.right.alias
            left_binding = self.inputs.alias_bindings[left_alias]
            right_binding = self.inputs.alias_bindings[right_alias]

            if left_binding.step_index > right_binding.step_index:
                left_alias, right_alias = right_alias, left_alias
                left_binding, right_binding = right_binding, left_binding

            start_node_idx = left_binding.step_index
            end_node_idx = right_binding.step_index

            relevant_edge_indices = [
                idx for idx in edge_indices
                if start_node_idx < idx < end_node_idx
            ]

            start_nodes = path_state.allowed_nodes.get(start_node_idx, set())
            end_nodes = path_state.allowed_nodes.get(end_node_idx, set())
            if not start_nodes or not end_nodes:
                continue

            left_col = clause.left.column
            right_col = clause.right.column
            node_id_col = self._node_column
            if not node_id_col:
                continue

            nodes_df = self.inputs.graph._nodes
            if nodes_df is None or node_id_col not in nodes_df.columns:
                continue

            left_values_df = None
            if left_col in nodes_df.columns:
                if node_id_col == left_col:
                    left_values_df = nodes_df[nodes_df[node_id_col].isin(start_nodes)][[node_id_col]].drop_duplicates().copy()
                    left_values_df.columns = ['__start__']
                    left_values_df['__start_val__'] = left_values_df['__start__']
                else:
                    left_values_df = nodes_df[nodes_df[node_id_col].isin(start_nodes)][[node_id_col, left_col]].drop_duplicates().rename(
                        columns={node_id_col: '__start__', left_col: '__start_val__'}
                    )

            right_values_df = None
            if right_col in nodes_df.columns:
                if node_id_col == right_col:
                    right_values_df = nodes_df[nodes_df[node_id_col].isin(end_nodes)][[node_id_col]].drop_duplicates().copy()
                    right_values_df.columns = ['__current__']
                    right_values_df['__end_val__'] = right_values_df['__current__']
                else:
                    right_values_df = nodes_df[nodes_df[node_id_col].isin(end_nodes)][[node_id_col, right_col]].drop_duplicates().rename(
                        columns={node_id_col: '__current__', right_col: '__end_val__'}
                    )

            # State table propagation: (current_node, start_node) pairs
            if left_values_df is not None and len(left_values_df) > 0:
                state_df = left_values_df[['__start__']].copy()
                state_df['__current__'] = state_df['__start__']
            else:
                state_df = pd.DataFrame(columns=['__current__', '__start__'])

            for edge_idx in relevant_edge_indices:
                edges_df = self.forward_steps[edge_idx]._edges
                if edges_df is None or len(state_df) == 0:
                    break

                allowed_edges = path_state.allowed_edges.get(edge_idx, None)
                if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
                    edges_df = edges_df[edges_df[edge_id_col].isin(list(allowed_edges))]

                edge_op = self.inputs.chain[edge_idx]
                if not isinstance(edge_op, ASTEdge):
                    continue
                sem = EdgeSemantics.from_edge(edge_op)

                if sem.is_multihop:
                    # Build edge pairs based on direction
                    edge_pairs = _build_edge_pairs_from_semantics(edges_df, src_col, dst_col, sem)

                    # Propagate state through hops
                    all_reachable = [state_df.copy()]
                    current_state = state_df.copy()

                    for hop in range(1, sem.max_hops + 1):
                        # Propagate current_state through one hop
                        next_state = edge_pairs.merge(
                            current_state, left_on='__from__', right_on='__current__', how='inner'
                        )[['__to__', '__start__']].rename(columns={'__to__': '__current__'}).drop_duplicates()

                        if len(next_state) == 0:
                            break

                        if hop >= sem.min_hops:
                            all_reachable.append(next_state)
                        current_state = next_state

                    # Combine all reachable states
                    if len(all_reachable) > 1:
                        state_df = pd.concat(all_reachable[1:], ignore_index=True).drop_duplicates()
                    else:
                        state_df = pd.DataFrame(columns=['__current__', '__start__'])
                else:
                    # Single-hop: propagate state through one hop
                    join_col, result_col = sem.join_cols(src_col, dst_col)
                    if sem.is_undirected:
                        # Both directions
                        next1 = edges_df.merge(
                            state_df, left_on=src_col, right_on='__current__', how='inner'
                        )[[dst_col, '__start__']].rename(columns={dst_col: '__current__'})
                        next2 = edges_df.merge(
                            state_df, left_on=dst_col, right_on='__current__', how='inner'
                        )[[src_col, '__start__']].rename(columns={src_col: '__current__'})
                        state_df = pd.concat([next1, next2], ignore_index=True).drop_duplicates()
                    else:
                        state_df = edges_df.merge(
                            state_df, left_on=join_col, right_on='__current__', how='inner'
                        )[[result_col, '__start__']].rename(columns={result_col: '__current__'}).drop_duplicates()

            # state_df now has (current_node=end_node, start_node) pairs
            # Filter to valid end nodes
            state_df = state_df[state_df['__current__'].isin(end_nodes)]

            if len(state_df) == 0:
                # No valid paths found
                if start_node_idx in path_state.allowed_nodes:
                    path_state.allowed_nodes[start_node_idx] = set()
                if end_node_idx in path_state.allowed_nodes:
                    path_state.allowed_nodes[end_node_idx] = set()
                continue

            # Join with start and end values to apply WHERE clause
            # left_values_df and right_values_df were built earlier (vectorized)
            if left_values_df is None or right_values_df is None:
                continue

            pairs_df = state_df.merge(left_values_df, on='__start__', how='inner')
            pairs_df = pairs_df.merge(right_values_df, on='__current__', how='inner')

            # Apply the comparison vectorized
            mask = evaluate_clause(pairs_df['__start_val__'], clause.op, pairs_df['__end_val__'])
            valid_pairs = pairs_df[mask]

            valid_starts = set(valid_pairs['__start__'].tolist())
            valid_ends = set(valid_pairs['__current__'].tolist())

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

    def _apply_edge_where_post_prune(
        self, path_state: "_PathState"
    ) -> "_PathState":
        """Apply WHERE on edge columns by enumerating paths."""
        if not self.inputs.where:
            return path_state

        edge_clauses = [
            clause for clause in self.inputs.where
            if (b1 := self.inputs.alias_bindings.get(clause.left.alias))
            and (b2 := self.inputs.alias_bindings.get(clause.right.alias))
            and (b1.kind == "edge" or b2.kind == "edge")
        ]
        if not edge_clauses:
            return path_state

        src_col = self._source_column
        dst_col = self._destination_column
        node_id_col = self._node_column
        if not src_col or not dst_col or not node_id_col:
            return path_state

        node_indices = self.meta.node_indices
        edge_indices = self.meta.edge_indices

        seed_nodes = path_state.allowed_nodes.get(node_indices[0], set())
        if not seed_nodes:
            return path_state

        paths_df = pd.DataFrame({f'n{node_indices[0]}': list(seed_nodes)})

        for i, edge_idx in enumerate(edge_indices):
            left_node_idx = node_indices[i]
            right_node_idx = node_indices[i + 1]

            edges_df = self.forward_steps[edge_idx]._edges
            if edges_df is None or len(edges_df) == 0:
                paths_df = paths_df.iloc[0:0]  # Empty paths
                break

            edge_op = self.inputs.chain[edge_idx]
            if not isinstance(edge_op, ASTEdge):
                continue
            sem = EdgeSemantics.from_edge(edge_op)

            edge_alias = self.meta.alias_for_step(edge_idx)
            edge_cols_needed = {
                ref.column for clause in edge_clauses
                for ref in [clause.left, clause.right] if ref.alias == edge_alias
            }

            edge_cols = [src_col, dst_col] + [c for c in edge_cols_needed if c in edges_df.columns]
            edges_subset = edges_df[list(set(edge_cols))].copy()

            rename_map = {
                col: f'e{edge_idx}_{col}' for col in edge_cols_needed
                if col in edges_subset.columns and col not in [src_col, dst_col]
            }
            edges_subset = edges_subset.rename(columns=rename_map)

            left_col = f'n{left_node_idx}'
            join_on, result_col = sem.join_cols(src_col, dst_col)
            if sem.is_undirected:
                join1 = paths_df.merge(
                    edges_subset, left_on=left_col, right_on=src_col, how='inner'
                )
                join1[f'n{right_node_idx}'] = join1[dst_col]
                join2 = paths_df.merge(
                    edges_subset, left_on=left_col, right_on=dst_col, how='inner'
                )
                join2[f'n{right_node_idx}'] = join2[src_col]
                paths_df = pd.concat([join1, join2], ignore_index=True)
            else:
                paths_df = paths_df.merge(
                    edges_subset, left_on=left_col, right_on=join_on, how='inner'
                )
                paths_df[f'n{right_node_idx}'] = paths_df[result_col]

            right_allowed = path_state.allowed_nodes.get(right_node_idx, set())
            if right_allowed:
                paths_df = paths_df[paths_df[f'n{right_node_idx}'].isin(list(right_allowed))]

            paths_df = paths_df.drop(columns=[src_col, dst_col], errors='ignore')

        if len(paths_df) == 0:
            for idx in node_indices:
                path_state.allowed_nodes[idx] = set()
            return path_state

        nodes_df = self.inputs.graph._nodes
        if nodes_df is not None:
            for clause in edge_clauses:
                for ref in [clause.left, clause.right]:
                    binding = self.inputs.alias_bindings.get(ref.alias)
                    if binding and binding.kind == "node" and ref.column != node_id_col:
                        step_idx = binding.step_index
                        col_name = f'n{step_idx}_{ref.column}'
                        if col_name not in paths_df.columns and ref.column in nodes_df.columns:
                            node_attr = nodes_df[[node_id_col, ref.column]].rename(
                                columns={node_id_col: f'n{step_idx}', ref.column: col_name}
                            )
                            paths_df = paths_df.merge(node_attr, on=f'n{step_idx}', how='left')

        mask = pd.Series(True, index=paths_df.index)
        for clause in edge_clauses:
            left_binding = self.inputs.alias_bindings[clause.left.alias]
            right_binding = self.inputs.alias_bindings[clause.right.alias]

            if left_binding.kind == "edge":
                left_col_name = f'e{left_binding.step_index}_{clause.left.column}'
            else:
                if clause.left.column == node_id_col or clause.left.column == "id":
                    left_col_name = f'n{left_binding.step_index}'
                else:
                    left_col_name = f'n{left_binding.step_index}_{clause.left.column}'

            if right_binding.kind == "edge":
                right_col_name = f'e{right_binding.step_index}_{clause.right.column}'
            else:
                if clause.right.column == node_id_col or clause.right.column == "id":
                    right_col_name = f'n{right_binding.step_index}'
                else:
                    right_col_name = f'n{right_binding.step_index}_{clause.right.column}'

            if left_col_name not in paths_df.columns or right_col_name not in paths_df.columns:
                continue

            left_vals = paths_df[left_col_name]
            right_vals = paths_df[right_col_name]

            # SQL NULL semantics: any comparison with NULL is NULL (treated as False)
            # We need to check for NULL before comparing, because pandas != returns True for X != NaN
            valid = left_vals.notna() & right_vals.notna()

            if clause.op == "==":
                clause_mask = valid & (left_vals == right_vals)
            elif clause.op == "!=":
                clause_mask = valid & (left_vals != right_vals)
            elif clause.op == "<":
                clause_mask = valid & (left_vals < right_vals)
            elif clause.op == "<=":
                clause_mask = valid & (left_vals <= right_vals)
            elif clause.op == ">":
                clause_mask = valid & (left_vals > right_vals)
            elif clause.op == ">=":
                clause_mask = valid & (left_vals >= right_vals)
            else:
                continue

            mask &= clause_mask.fillna(False)

        # Filter paths
        valid_paths = paths_df[mask]

        # Update allowed nodes based on valid paths
        for node_idx in node_indices:
            col_name = f'n{node_idx}'
            if col_name in valid_paths.columns:
                valid_node_ids = set(valid_paths[col_name].unique())
                current = path_state.allowed_nodes.get(node_idx, set())
                path_state.allowed_nodes[node_idx] = current & valid_node_ids if current else valid_node_ids

        for i, edge_idx in enumerate(edge_indices):
            left_node_idx = node_indices[i]
            right_node_idx = node_indices[i + 1]
            left_col = f'n{left_node_idx}'
            right_col = f'n{right_node_idx}'

            if left_col in valid_paths.columns and right_col in valid_paths.columns:
                valid_pairs = valid_paths[[left_col, right_col]].drop_duplicates()
                edges_df = self.forward_steps[edge_idx]._edges
                if edges_df is not None:
                    edge_op = self.inputs.chain[edge_idx]
                    if not isinstance(edge_op, ASTEdge):
                        continue
                    sem = EdgeSemantics.from_edge(edge_op)

                    if sem.is_undirected:
                        fwd = edges_df.merge(
                            valid_pairs.rename(columns={left_col: src_col, right_col: dst_col}),
                            on=[src_col, dst_col], how='inner'
                        )
                        rev = edges_df.merge(
                            valid_pairs.rename(columns={left_col: dst_col, right_col: src_col}),
                            on=[src_col, dst_col], how='inner'
                        )
                        edges_df = pd.concat([fwd, rev], ignore_index=True).drop_duplicates(
                            subset=[src_col, dst_col]
                        )
                    else:
                        # For directed edges, use endpoint_cols to get proper src/dst mapping
                        start_endpoint, end_endpoint = sem.endpoint_cols(src_col, dst_col)
                        edges_df = edges_df.merge(
                            valid_pairs.rename(columns={left_col: start_endpoint, right_col: end_endpoint}),
                            on=[src_col, dst_col], how='inner'
                        )
                    self.forward_steps[edge_idx]._edges = edges_df

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

        relevant_edge_indices = [idx for idx in edge_indices if start_idx < idx < end_idx]

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
                edges_df = self._filter_multihop_edges_by_endpoints(
                    edges_df, edge_op, left_allowed, right_allowed, sem.is_reverse, sem.is_undirected
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
                    # For directed edges, use endpoint_cols to determine filter columns
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
                new_src_nodes = self._find_multihop_start_nodes(
                    edges_df, edge_op, right_allowed, sem.is_reverse, sem.is_undirected
                )
            else:
                new_src_nodes = sem.start_nodes(edges_df, src_col, dst_col)

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

        Uses vectorized bidirectional reachability propagation:
        1. Forward: find nodes reachable from left_allowed at each hop
        2. Backward: find nodes that can reach right_allowed at each hop
        3. Keep edges connecting forward-reachable to backward-reachable nodes
        """
        src_col = self._source_column
        dst_col = self._destination_column

        if not src_col or not dst_col or not left_allowed or not right_allowed:
            return edges_df

        # Only max_hops needed here - min_hops is enforced at path level, not per-edge
        max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
            edge_op.hops if edge_op.hops is not None else 1
        )

        # Build edge pairs and compute bidirectional reachability
        edge_pairs = _build_edge_pairs(edges_df, src_col, dst_col, is_reverse, is_undirected)
        fwd_df = _bfs_reachability(edge_pairs, left_allowed, max_hops, '__fwd_hop__')
        rev_edge_pairs = edge_pairs.rename(columns={'__from__': '__to__', '__to__': '__from__'})
        bwd_df = _bfs_reachability(rev_edge_pairs, right_allowed, max_hops, '__bwd_hop__')

        # An edge (u, v) is valid if:
        # - u is forward-reachable at hop h_fwd (path length from left_allowed to u)
        # - v is backward-reachable at hop h_bwd (path length from v to right_allowed)
        # - h_fwd + 1 + h_bwd is in [min_hops, max_hops]
        if len(fwd_df) == 0 or len(bwd_df) == 0:
            return edges_df.iloc[:0]

        # Yannakakis: min hop is correct here - edge validity uses shortest path through node
        fwd_df = fwd_df.groupby('__node__')['__fwd_hop__'].min().reset_index()
        bwd_df = bwd_df.groupby('__node__')['__bwd_hop__'].min().reset_index()

        # Join edges with hop distances
        if is_undirected:
            # For undirected, check both directions
            # An edge is valid if it lies on ANY valid path from left_allowed to right_allowed.
            # This means: fwd_hop(u) + 1 + bwd_hop(v) <= max_hops
            # We also need at least one path through the edge to have length >= min_hops.

            # Direction 1: src is fwd, dst is bwd
            edges_annotated1 = edges_df.merge(
                fwd_df, left_on=src_col, right_on='__node__', how='inner'
            ).merge(
                bwd_df, left_on=dst_col, right_on='__node__', how='inner', suffixes=('', '_bwd')
            )
            edges_annotated1['__total_hops__'] = edges_annotated1['__fwd_hop__'] + 1 + edges_annotated1['__bwd_hop__']
            # Keep edges that can be part of a valid path (total <= max_hops)
            # The min_hops constraint is enforced at the path level, not per-edge
            valid1 = edges_annotated1[edges_annotated1['__total_hops__'] <= max_hops]

            # Direction 2: dst is fwd, src is bwd
            edges_annotated2 = edges_df.merge(
                fwd_df, left_on=dst_col, right_on='__node__', how='inner'
            ).merge(
                bwd_df, left_on=src_col, right_on='__node__', how='inner', suffixes=('', '_bwd')
            )
            edges_annotated2['__total_hops__'] = edges_annotated2['__fwd_hop__'] + 1 + edges_annotated2['__bwd_hop__']
            valid2 = edges_annotated2[edges_annotated2['__total_hops__'] <= max_hops]

            # Get original edge columns only
            orig_cols = list(edges_df.columns)
            valid_edges = pd.concat([valid1[orig_cols], valid2[orig_cols]], ignore_index=True).drop_duplicates()
            return valid_edges
        else:
            # Determine which column is "source" (fwd) and which is "dest" (bwd)
            if is_reverse:
                fwd_col, bwd_col = dst_col, src_col
            else:
                fwd_col, bwd_col = src_col, dst_col

            edges_annotated = edges_df.merge(
                fwd_df, left_on=fwd_col, right_on='__node__', how='inner'
            ).merge(
                bwd_df, left_on=bwd_col, right_on='__node__', how='inner', suffixes=('', '_bwd')
            )
            edges_annotated['__total_hops__'] = edges_annotated['__fwd_hop__'] + 1 + edges_annotated['__bwd_hop__']

            # Keep edges that can be part of a valid path (total <= max_hops)
            # The min_hops constraint is enforced at the path level, not per-edge
            valid_edges = edges_annotated[edges_annotated['__total_hops__'] <= max_hops]

            # Return only original columns
            orig_cols = list(edges_df.columns)
            return valid_edges[orig_cols]

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

        Uses vectorized hop-by-hop backward propagation via merge+groupby.
        """
        src_col = self._source_column
        dst_col = self._destination_column

        if not src_col or not dst_col or not right_allowed:
            return set()

        min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
        max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
            edge_op.hops if edge_op.hops is not None else 1
        )

        # Build edge pairs for backward traversal (inverted direction)
        # For forward edges, backward trace goes dst->src, so we invert is_reverse
        edge_pairs = _build_edge_pairs(edges_df, src_col, dst_col, not is_reverse, is_undirected)

        # Vectorized backward BFS: propagate reachability hop by hop
        # Use DataFrame-based tracking throughout (no Python sets internally)
        # Start with right_allowed as target destinations (hop 0 means "at the destination")
        # We trace backward to find nodes that can REACH these destinations
        frontier = pd.DataFrame({'__node__': list(right_allowed)})
        all_visited = frontier.copy()
        valid_starts_frames: List[DataFrameT] = []

        # Collect nodes at each hop distance FROM the destination
        for hop in range(1, max_hops + 1):
            # Join with edges to find nodes one hop back from frontier
            # edge_pairs: __from__ = dst (target), __to__ = src (predecessor)
            # We want nodes (__to__) that can reach frontier nodes (__from__)
            new_frontier = edge_pairs.merge(
                frontier,
                left_on='__from__',
                right_on='__node__',
                how='inner'
            )[['__to__']].drop_duplicates()

            if len(new_frontier) == 0:
                break

            new_frontier = new_frontier.rename(columns={'__to__': '__node__'})

            # Collect valid starts (nodes at hop distance in [min_hops, max_hops])
            # These are nodes that can reach right_allowed in exactly `hop` hops
            if hop >= min_hops:
                valid_starts_frames.append(new_frontier[['__node__']])

            # Anti-join: filter out nodes already visited to avoid infinite loops
            # But still keep nodes for valid_starts even if visited before at different hop
            merged = new_frontier.merge(
                all_visited[['__node__']], on='__node__', how='left', indicator=True
            )
            unvisited = merged[merged['_merge'] == 'left_only'][['__node__']]

            if len(unvisited) == 0:
                break

            frontier = unvisited
            all_visited = pd.concat([all_visited, unvisited], ignore_index=True)

        # Combine all valid starts and convert to set (caller expects set)
        if valid_starts_frames:
            valid_starts_df = pd.concat(valid_starts_frames, ignore_index=True).drop_duplicates()
            return set(valid_starts_df['__node__'].tolist())
        return set()

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
                self._equality_values[alias][col] = series_values(frame[col])

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
                    filtered = self._filter_edges_by_clauses(
                        filtered, left_alias, right_alias, allowed_nodes, sem.is_reverse, sem.is_undirected
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

    def _filter_edges_by_clauses(
        self,
        edges_df: DataFrameT,
        left_alias: str,
        right_alias: str,
        allowed_nodes: Dict[int, Set[Any]],
        is_reverse: bool = False,
        is_undirected: bool = False,
    ) -> DataFrameT:
        """Filter edges using WHERE clauses that connect adjacent aliases.

        For forward edges: left_alias matches src, right_alias matches dst.
        For reverse edges: left_alias matches dst, right_alias matches src.
        For undirected edges: try both orientations, keep edges matching either.
        """
        # Early return for empty edges - no filtering needed
        if len(edges_df) == 0:
            return edges_df

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

        # For undirected edges, we need to try both orientations
        if is_undirected:
            # Orientation 1: src=left, dst=right (forward)
            fwd_df = self._merge_and_filter_edges(
                edges_df, lf, rf, left_alias, right_alias, relevant,
                left_merge_col=self._source_column,
                right_merge_col=self._destination_column
            )
            # Orientation 2: dst=left, src=right (reverse)
            rev_df = self._merge_and_filter_edges(
                edges_df, lf, rf, left_alias, right_alias, relevant,
                left_merge_col=self._destination_column,
                right_merge_col=self._source_column
            )
            # Combine both orientations - keep edges that match either
            if len(fwd_df) == 0 and len(rev_df) == 0:
                return fwd_df  # Empty dataframe with correct schema
            elif len(fwd_df) == 0:
                out_df = rev_df
            elif len(rev_df) == 0:
                out_df = fwd_df
            else:
                from graphistry.Engine import safe_concat
                out_df = safe_concat([fwd_df, rev_df], ignore_index=True, sort=False)
                # Deduplicate by edge columns (src, dst) to avoid double-counting
                out_df = out_df.drop_duplicates(
                    subset=[self._source_column, self._destination_column]
                )
            return out_df

        # For reverse edges, left_alias is reached via dst column, right_alias via src column
        # For forward edges, left_alias is reached via src column, right_alias via dst column
        if is_reverse:
            left_merge_col = self._destination_column
            right_merge_col = self._source_column
        else:
            left_merge_col = self._source_column
            right_merge_col = self._destination_column

        out_df = self._merge_and_filter_edges(
            edges_df, lf, rf, left_alias, right_alias, relevant,
            left_merge_col=left_merge_col,
            right_merge_col=right_merge_col
        )

        return out_df

    def _merge_and_filter_edges(
        self,
        edges_df: DataFrameT,
        lf: DataFrameT,
        rf: DataFrameT,
        left_alias: str,
        right_alias: str,
        relevant: List[WhereComparison],
        left_merge_col: str,
        right_merge_col: str,
    ) -> DataFrameT:
        """Helper to merge edges with alias frames and apply WHERE clauses."""
        out_df = edges_df.merge(
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
                    mask = evaluate_clause(out_df[col_left_name], clause.op, out_df[col_right_name])
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

        sem = EdgeSemantics.from_edge(edge_op)

        # Check if hop labels are usable (filtered start node gives unambiguous labels)
        # For unfiltered starts, all edges have hop_label=1, making them useless for identification
        first_node_step = self.inputs.chain[0] if self.inputs.chain else None
        has_filtered_start = (
            isinstance(first_node_step, ASTNode) and first_node_step.filter_dict
        )

        if edge_label and edge_label in edges_df.columns and has_filtered_start:
            # Use hop labels to identify start/end nodes (accurate when start is filtered)
            hop_col = edges_df[edge_label]
            min_hop = hop_col.min()
            first_hop_edges = edges_df[hop_col == min_hop]

            chain_min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
            valid_endpoint_edges = edges_df[hop_col >= chain_min_hops]

            if sem.is_undirected:
                start_nodes_df = pd.concat([
                    first_hop_edges[[self._source_column]].rename(columns={self._source_column: '__node__'}),
                    first_hop_edges[[self._destination_column]].rename(columns={self._destination_column: '__node__'})
                ], ignore_index=True).drop_duplicates()
                end_nodes_df = pd.concat([
                    valid_endpoint_edges[[self._source_column]].rename(columns={self._source_column: '__node__'}),
                    valid_endpoint_edges[[self._destination_column]].rename(columns={self._destination_column: '__node__'})
                ], ignore_index=True).drop_duplicates()
            else:
                # For directed edges, use endpoint_cols to get proper src/dst mapping
                start_col, end_col = sem.endpoint_cols(self._source_column or '', self._destination_column or '')
                start_nodes_df = first_hop_edges[[start_col]].rename(
                    columns={start_col: '__node__'}
                ).drop_duplicates()
                end_nodes_df = valid_endpoint_edges[[end_col]].rename(
                    columns={end_col: '__node__'}
                ).drop_duplicates()

            start_nodes = set(start_nodes_df['__node__'].tolist())
            end_nodes = set(end_nodes_df['__node__'].tolist())
        else:
            # Fallback: use alias frames directly when hop labels are ambiguous
            # (unfiltered start makes all edges "hop 1" from some start)
            start_nodes = series_values(left_frame[self._node_column])
            end_nodes = series_values(right_frame[self._node_column])

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
                mask = evaluate_clause(pairs_df[left_col], clause.op, pairs_df[actual_right_col])
                pairs_df = pairs_df[mask]

        if len(pairs_df) == 0:
            return edges_df.iloc[:0]

        # Get valid start and end nodes
        valid_starts = set(pairs_df["__start_id__"].tolist())
        valid_ends = set(pairs_df["__end_id__"].tolist())

        # Use vectorized bidirectional reachability to filter edges
        # This reuses the same logic as _filter_multihop_edges_by_endpoints
        return self._filter_multihop_edges_by_endpoints(
            edges_df, edge_op, valid_starts, valid_ends, sem.is_reverse, sem.is_undirected
        )

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
                mask = evaluate_clause(merged[col_left], clause.op, merged[col_right])
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
            isinstance(op, ASTEdge) and not self._is_single_hop(op)
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
            allowed = common_values(lhs[left_col], rhs[right_col])
            self.alias_frames[clause.left.alias] = filter_by_values(
                lhs, left_col, allowed
            )
            self.alias_frames[clause.right.alias] = filter_by_values(
                rhs, right_col, allowed
            )
        elif clause.op == ">":
            right_min = safe_min(rhs[right_col])
            left_max = safe_max(lhs[left_col])
            if right_min is not None:
                self.alias_frames[clause.left.alias] = lhs[lhs[left_col] > right_min]
            if left_max is not None:
                self.alias_frames[clause.right.alias] = rhs[rhs[right_col] < left_max]
        elif clause.op == ">=":
            right_min = safe_min(rhs[right_col])
            left_max = safe_max(lhs[left_col])
            if right_min is not None:
                self.alias_frames[clause.left.alias] = lhs[lhs[left_col] >= right_min]
            if left_max is not None:
                self.alias_frames[clause.right.alias] = rhs[
                    rhs[right_col] <= left_max
                ]
        elif clause.op == "<":
            right_max = safe_max(rhs[right_col])
            left_min = safe_min(lhs[left_col])
            if right_max is not None:
                self.alias_frames[clause.left.alias] = lhs[lhs[left_col] < right_max]
            if left_min is not None:
                self.alias_frames[clause.right.alias] = rhs[
                    rhs[right_col] > left_min
                ]
        elif clause.op == "<=":
            right_max = safe_max(rhs[right_col])
            left_min = safe_min(lhs[left_col])
            if right_max is not None:
                self.alias_frames[clause.left.alias] = lhs[
                    lhs[left_col] <= right_max
                ]
            if left_min is not None:
                self.alias_frames[clause.right.alias] = rhs[
                    rhs[right_col] >= left_min
                ]



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
