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

            # For single-hop edges, filter by allowed dst first
            # For multi-hop, defer dst filtering to _filter_multihop_by_where
            if not is_multihop:
                if self._destination_column and self._destination_column in filtered.columns:
                    allowed_dst = allowed_nodes.get(right_node_idx)
                    if allowed_dst is not None:
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
                        filtered, left_alias, right_alias, allowed_nodes
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

            if self._destination_column and self._destination_column in filtered.columns:
                allowed_dst_actual = self._series_values(filtered[self._destination_column])
                current_dst = allowed_nodes.get(right_node_idx, set())
                allowed_nodes[right_node_idx] = (
                    current_dst & allowed_dst_actual if current_dst else allowed_dst_actual
                )

            if self._edge_column and self._edge_column in filtered.columns:
                allowed_edges[edge_idx] = self._series_values(filtered[self._edge_column])

            if self._source_column and self._source_column in filtered.columns:
                allowed_src = self._series_values(filtered[self._source_column])
                current = allowed_nodes.get(left_node_idx, set())
                allowed_nodes[left_node_idx] = current & allowed_src if current else allowed_src

        return self._PathState(allowed_nodes=allowed_nodes, allowed_edges=allowed_edges)

    def _filter_edges_by_clauses(
        self,
        edges_df: DataFrameT,
        left_alias: str,
        right_alias: str,
        allowed_nodes: Dict[int, Set[Any]],
    ) -> DataFrameT:
        """Filter edges using WHERE clauses that connect adjacent aliases."""

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

        out_df = out_df.merge(
            lf,
            left_on=self._source_column,
            right_on="__left_id__",
            how="inner",
        )
        out_df = out_df.merge(
            rf,
            left_on=self._destination_column,
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
                out_df = out_df.rename(columns={
                    left_col: col_left_name,
                    f"{left_col}__r": col_left_name if f"{left_col}__r" in out_df.columns else col_left_name,
                })
                placeholder = {}
                if right_col in out_df.columns:
                    placeholder[right_col] = col_right_name
                if f"{right_col}__r" in out_df.columns:
                    placeholder[f"{right_col}__r"] = col_right_name
                if placeholder:
                    out_df = out_df.rename(columns=placeholder)
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

        # Identify first-hop and last-hop edges
        hop_col = edges_df[edge_label]
        min_hop = hop_col.min()
        max_hop = hop_col.max()

        first_hop_edges = edges_df[hop_col == min_hop]
        last_hop_edges = edges_df[hop_col == max_hop]

        # Get start nodes (sources of first-hop edges)
        start_nodes = set(first_hop_edges[self._source_column].tolist())
        # Get end nodes (destinations of last-hop edges)
        end_nodes = set(last_hop_edges[self._destination_column].tolist())

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
        pairs_df = lf.merge(rf, on="__cross_key__").drop(columns=["__cross_key__"])

        # Apply WHERE clauses to filter valid (start, end) pairs
        for clause in relevant:
            left_col = clause.left.column if clause.left.alias == left_alias else clause.right.column
            right_col = clause.right.column if clause.right.alias == right_alias else clause.left.column
            if left_col in pairs_df.columns and right_col in pairs_df.columns:
                mask = self._evaluate_clause(pairs_df[left_col], clause.op, pairs_df[right_col])
                pairs_df = pairs_df[mask]

        if len(pairs_df) == 0:
            return edges_df.iloc[:0]

        # Get valid start and end nodes
        valid_starts = set(pairs_df["__start_id__"].tolist())
        valid_ends = set(pairs_df["__end_id__"].tolist())

        # Filter edges: keep edges where:
        # - First hop edges have src in valid_starts
        # - Last hop edges have dst in valid_ends
        # - Intermediate edges are kept if they connect valid paths
        # For simplicity, we filter first/last hop edges and keep all intermediates
        # (path coherence will be enforced by allowed_nodes propagation)

        def filter_row(row):
            hop = row[edge_label]
            if hop == min_hop:
                return row[self._source_column] in valid_starts
            elif hop == max_hop:
                return row[self._destination_column] in valid_ends
            else:
                return True  # Intermediate edges kept for now

        mask = edges_df.apply(filter_row, axis=1)
        return edges_df[mask]

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
