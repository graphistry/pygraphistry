from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Literal, Sequence, List, Optional, Any, Tuple

from graphistry.Engine import Engine, safe_merge
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject
from graphistry.gfql.ref.enumerator import OracleCaps, OracleResult, enumerate_chain
from graphistry.compute.gfql.same_path_types import WhereComparison, PathState
from graphistry.compute.gfql.same_path.chain_meta import ChainMeta
from graphistry.compute.gfql.same_path.edge_semantics import EdgeSemantics
from graphistry.compute.gfql.same_path.df_utils import (
    concat_frames,
    df_cons,
    domain_from_values,
    domain_intersect,
    domain_is_empty,
    domain_union,
    domain_union_all,
    series_values,
)
from graphistry.otel import otel_span, otel_enabled
from graphistry.compute.gfql.same_path.multihop import apply_non_adjacent_where_post_prune
from graphistry.compute.gfql.same_path.where_filter import apply_edge_where_post_prune, filter_edges_by_where
from graphistry.compute.typing import DataFrameT, DomainT

AliasKind = Literal["node", "edge"]
_CUDF_MODE_ENV = "GRAPHISTRY_CUDF_SAME_PATH_MODE"


@dataclass(frozen=True)
class AliasBinding:
    alias: str
    step_index: int
    kind: AliasKind
    ast: ASTObject


@dataclass(frozen=True)
class SamePathExecutorInputs:
    graph: Plottable
    chain: Sequence[ASTObject]
    where: Sequence[WhereComparison]
    engine: Engine
    alias_bindings: Dict[str, AliasBinding]
    column_requirements: Dict[str, Sequence[str]]
    include_paths: bool = False


class DFSamePathExecutor:
    def __init__(self, inputs: SamePathExecutorInputs) -> None:
        self.inputs = inputs
        self.meta = ChainMeta.from_chain(inputs.chain, inputs.alias_bindings)
        self.forward_steps: List[Plottable] = []
        self.alias_frames: Dict[str, DataFrameT] = {}
        self._node_column = inputs.graph._node
        self._edge_column = inputs.graph._edge
        self._source_column = inputs.graph._source
        self._destination_column = inputs.graph._destination

    def _otel_attrs(self) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "gfql.engine": self.inputs.engine.value,
            "gfql.chain_len": len(self.inputs.chain),
            "gfql.where_len": len(self.inputs.where),
            "gfql.include_paths": self.inputs.include_paths,
        }
        nodes, edges = self.inputs.graph._nodes, self.inputs.graph._edges
        if nodes is not None:
            attrs["graphistry.nodes"] = len(nodes)
        if edges is not None:
            attrs["graphistry.edges"] = len(edges)
        return attrs

    def edges_df_for_step(self, edge_idx: int, state: Optional[PathState] = None) -> Optional[DataFrameT]:
        return state.pruned_edges[edge_idx] if state is not None and edge_idx in state.pruned_edges else self.forward_steps[edge_idx]._edges

    def run(self) -> Plottable:
        attrs = self._otel_attrs() if otel_enabled() else None
        with otel_span("gfql.df_executor.run", attrs=attrs):
            self._forward()
            mode = os.environ.get(_CUDF_MODE_ENV, "auto").lower()
            if mode == "oracle":
                return self._unsafe_run_test_only_oracle()
            if mode == "strict" and self.inputs.engine == Engine.CUDF:
                try:  # check cudf presence
                    import cudf  # type: ignore  # noqa: F401
                except Exception:
                    raise RuntimeError(
                        "cuDF engine requested with strict mode but cudf is unavailable")
            return self._run_native()

    def _forward(self) -> None:
        with otel_span("gfql.df_executor.forward", attrs={"gfql.forward_steps": len(self.inputs.chain)}):
            graph = self.inputs.graph
            ops = self.inputs.chain
            self.forward_steps = []
            for idx, op in enumerate(ops):
                is_call = isinstance(op, ASTCall)
                current_g = self.forward_steps[-1] if is_call and self.forward_steps else graph
                prev_nodes = None if is_call or not self.forward_steps else self.forward_steps[-1]._nodes
                g_step = op(
                    g=current_g,
                    prev_node_wavefront=prev_nodes,
                    target_wave_front=None,
                    engine=self.inputs.engine,
                )
                self.forward_steps.append(g_step)
                self._capture_alias_frame(op, g_step, idx)
            self._apply_forward_where_pruning()

    def _capture_alias_frame(self, op: ASTObject, step_result: Plottable, step_index: int) -> None:
        alias = getattr(op, "_name", None)
        if not alias or alias not in self.inputs.alias_bindings:
            return
        binding = self.inputs.alias_bindings[alias]
        frame = step_result._nodes if binding.kind == "node" else step_result._edges
        if frame is None:
            raise ValueError(
                f"Alias '{alias}' did not produce a {'node' if binding.kind == 'node' else 'edge'} frame"
            )
        required_cols = list(self.inputs.column_requirements.get(alias, ()))
        id_col = self._node_column if binding.kind == "node" else self._edge_column
        if id_col and id_col not in required_cols:
            required_cols.append(id_col)
        missing = [col for col in required_cols if col not in frame.columns]
        if missing:
            raise ValueError(f"Alias '{alias}' missing required columns: {', '.join(missing)}")
        alias_frame = frame[required_cols].copy()
        self.alias_frames[alias] = alias_frame

    def _apply_forward_where_pruning(self) -> None:
        if not self.inputs.where:
            return
        with otel_span("gfql.df_executor.forward_where_prune", attrs={"gfql.where_len": len(self.inputs.where)}):
            def _apply_mask(alias: str, frame: DataFrameT, mask: Any) -> bool:
                new_frame = frame[mask]
                if len(new_frame) >= len(frame):
                    return False
                self.alias_frames[alias] = new_frame
                return True
            changed = True
            while changed:
                changed = False
                for clause in self.inputs.where:
                    left_alias, right_alias, left_col, right_col = clause.left.alias, clause.right.alias, clause.left.column, clause.right.column
                    left_frame, right_frame = self.alias_frames.get(left_alias), self.alias_frames.get(right_alias)
                    if (
                        left_frame is None
                        or right_frame is None
                        or left_col not in left_frame.columns
                        or right_col not in right_frame.columns
                    ):
                        continue

                    if clause.op == "==":
                        left_values = series_values(left_frame[left_col])
                        right_values = series_values(right_frame[right_col])
                        common = domain_intersect(left_values, right_values)
                        if len(common) < len(left_values):
                            changed |= _apply_mask(
                                left_alias,
                                left_frame,
                                left_frame[left_col].isin(common),
                            )
                        if len(common) < len(right_values):
                            changed |= _apply_mask(
                                right_alias,
                                right_frame,
                                right_frame[right_col].isin(common),
                            )
                    elif clause.op in {"<", "<=", ">", ">="}:
                        left_vals = left_frame[left_col]
                        right_vals = right_frame[right_col]
                        left_min, left_max = left_vals.min(), left_vals.max()
                        right_min, right_max = right_vals.min(), right_vals.max()
                        masks = {
                            "<": (left_vals < right_max, right_vals > left_min),
                            "<=": (left_vals <= right_max, right_vals >= left_min),
                            ">": (left_vals > right_min, right_vals < left_max),
                            ">=": (left_vals >= right_min, right_vals <= left_max),
                        }
                        left_mask, right_mask = masks[clause.op]
                        changed |= _apply_mask(left_alias, left_frame, left_mask)
                        changed |= _apply_mask(right_alias, right_frame, right_mask)

    def _filter_edges_by_allowed_nodes(self, edges_df: DataFrameT, sem: EdgeSemantics, src_col: Optional[str], dst_col: Optional[str], left_allowed: Optional[DomainT] = None, right_allowed: Optional[DomainT] = None) -> DataFrameT:
        if not src_col or not dst_col:
            return edges_df
        if sem.is_undirected:
            if left_allowed is not None and right_allowed is not None:
                mask = (edges_df[src_col].isin(left_allowed) & edges_df[dst_col].isin(right_allowed)) | (edges_df[dst_col].isin(left_allowed) & edges_df[src_col].isin(right_allowed))
                return edges_df[mask]
            if left_allowed is not None:
                return edges_df[edges_df[src_col].isin(left_allowed) | edges_df[dst_col].isin(left_allowed)]
            if right_allowed is not None:
                return edges_df[edges_df[src_col].isin(right_allowed) | edges_df[dst_col].isin(right_allowed)]
            return edges_df
        start_col, end_col = sem.join_cols(src_col, dst_col)
        if left_allowed is not None:
            edges_df = edges_df[edges_df[start_col].isin(left_allowed)]
        if right_allowed is not None:
            edges_df = edges_df[edges_df[end_col].isin(right_allowed)]
        return edges_df

    def _unsafe_run_test_only_oracle(self) -> Plottable:
        caps = OracleCaps(max_nodes=1000, max_edges=5000, max_length=20, max_partial_rows=1_000_000)
        oracle = enumerate_chain(self.inputs.graph, self.inputs.chain, where=self.inputs.where, include_paths=self.inputs.include_paths, caps=caps)
        nodes_df, edges_df = self._apply_oracle_hop_labels(oracle)
        self._update_alias_frames_from_oracle(oracle.tags)
        return self._materialize_from_oracle(nodes_df, edges_df)

    def _run_native(self) -> Plottable:
        with otel_span("gfql.df_executor.compute_allowed_tags"):
            allowed_tags = self._compute_allowed_tags()
        with otel_span("gfql.df_executor.backward_prune"):
            state = self._backward_prune(allowed_tags)
        with otel_span("gfql.df_executor.post_prune.non_adjacent"):
            state = apply_non_adjacent_where_post_prune(self, state)
        with otel_span("gfql.df_executor.post_prune.edge_where"):
            state = apply_edge_where_post_prune(self, state)
        with otel_span("gfql.df_executor.materialize"):
            return self._materialize_filtered(state)

    _run_gpu = _run_native

    def _update_alias_frames_from_oracle(self, tags: Dict[str, Any]) -> None:
        for alias, binding in self.inputs.alias_bindings.items():
            if alias not in tags or binding.step_index >= len(self.forward_steps):
                continue
            step_result = self.forward_steps[binding.step_index]
            id_col = self._node_column if binding.kind == "node" else self._edge_column
            frame = step_result._nodes if binding.kind == "node" else step_result._edges
            if frame is None or id_col is None:
                continue
            ids = domain_from_values(tags.get(alias), frame)
            if domain_is_empty(ids):
                self.alias_frames[alias] = frame.iloc[0:0].copy()
                continue
            filtered = frame[frame[id_col].isin(ids)].copy()
            self.alias_frames[alias] = filtered

    def _materialize_from_oracle(self, nodes_df: DataFrameT, edges_df: DataFrameT) -> Plottable:
        g = self.inputs.graph
        edge_id, src, dst, node_id = g._edge, g._source, g._destination, g._node
        if node_id and node_id not in nodes_df.columns:
            raise ValueError(f"Oracle nodes missing id column '{node_id}'")
        if dst and dst not in edges_df.columns:
            raise ValueError(f"Oracle edges missing destination column '{dst}'")
        if src and src not in edges_df.columns:
            raise ValueError(f"Oracle edges missing source column '{src}'")
        if edge_id and edge_id not in edges_df.columns:
            if "__enumerator_edge_id__" in edges_df.columns:
                edges_df = edges_df.rename(columns={"__enumerator_edge_id__": edge_id})
            else:
                raise ValueError(f"Oracle edges missing id column '{edge_id}'")
        return g.nodes(nodes_df, node=node_id).edges(edges_df, source=src, destination=dst, edge=edge_id)

    def _compute_allowed_tags(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for alias, binding in self.inputs.alias_bindings.items():
            frame = self.alias_frames.get(alias)
            if frame is None:
                continue
            id_col = self._node_column if binding.kind == "node" else self._edge_column
            if id_col is None or id_col not in frame.columns:
                continue
            out[alias] = series_values(frame[id_col])
        return out

    def _backward_prune(self, allowed_tags: Dict[str, Any]) -> PathState:
        self.meta.validate()  # Raises if chain structure is invalid
        node_indices, edge_indices = self.meta.node_indices, self.meta.edge_indices
        allowed_nodes: Dict[int, DomainT] = {}
        allowed_edges: Dict[int, DomainT] = {}
        pruned_edges: Dict[int, DataFrameT] = {}

        def _update_allowed(idx: int, values: DomainT) -> None:
            current = allowed_nodes.get(idx)
            allowed_nodes[idx] = domain_intersect(current, values) if current is not None else values

        node_col = self._node_column
        for idx in node_indices:
            frame = self.forward_steps[idx]._nodes
            if frame is None or node_col is None:
                continue
            alias = self.meta.alias_for_step(idx)
            allowed = allowed_tags.get(alias) if alias is not None else None
            allowed_nodes[idx] = allowed if allowed is not None else series_values(frame[node_col])

        for edge_idx, left_node_idx, right_node_idx in zip(reversed(edge_indices), reversed(node_indices[:-1]), reversed(node_indices[1:])):
            edge_alias = self.meta.alias_for_step(edge_idx)
            edges_df = self.forward_steps[edge_idx]._edges
            if edges_df is None:
                continue

            filtered = edges_df
            edge_op = self.inputs.chain[edge_idx]
            if not isinstance(edge_op, ASTEdge):
                continue
            sem = EdgeSemantics.from_edge(edge_op)
            if not sem.is_multihop:
                allowed_dst = allowed_nodes.get(right_node_idx)
                if allowed_dst is not None:
                    filtered = self._filter_edges_by_allowed_nodes(
                        filtered,
                        sem,
                        self._source_column,
                        self._destination_column,
                        right_allowed=allowed_dst,
                    )

            left_alias = self.meta.alias_for_step(left_node_idx)
            right_alias = self.meta.alias_for_step(right_node_idx)
            if left_alias and right_alias:
                filtered = filter_edges_by_where(self, filtered, edge_op, left_alias, right_alias, allowed_nodes, sem)

            edge_alias = self.meta.alias_for_step(edge_idx)
            if edge_alias and edge_alias in allowed_tags and self._edge_column and self._edge_column in filtered.columns:
                filtered = filtered[filtered[self._edge_column].isin(allowed_tags[edge_alias])]

            if sem.is_undirected:
                if self._source_column and self._destination_column:
                    all_nodes_in_edges = domain_union(series_values(filtered[self._source_column]), series_values(filtered[self._destination_column]))
                    _update_allowed(right_node_idx, all_nodes_in_edges)
                    _update_allowed(left_node_idx, all_nodes_in_edges)
            else:
                start_col, end_col = sem.join_cols(
                    self._source_column or '', self._destination_column or '')
                if end_col and end_col in filtered.columns:
                    _update_allowed(right_node_idx, series_values(filtered[end_col]))
                if start_col and start_col in filtered.columns:
                    _update_allowed(left_node_idx, series_values(filtered[start_col]))

            if self._edge_column and self._edge_column in filtered.columns:
                allowed_edges[edge_idx] = series_values(
                    filtered[self._edge_column])

            if len(filtered) < len(edges_df):
                pruned_edges[edge_idx] = filtered

        return PathState.from_mutable(allowed_nodes, allowed_edges, pruned_edges)

    def backward_propagate_constraints(self, state: PathState, start_node_idx: int, end_node_idx: int) -> PathState:
        from graphistry.compute.gfql.same_path.multihop import filter_multihop_edges_by_endpoints, find_multihop_start_nodes
        src_col, dst_col, edge_id_col = self._source_column, self._destination_column, self._edge_column
        node_indices = self.meta.node_indices
        edge_indices = self.meta.edge_indices
        if not src_col or not dst_col:
            return state
        relevant_edge_indices = [idx for idx in edge_indices if start_node_idx < idx < end_node_idx]
        local_allowed_nodes: Dict[int, DomainT] = dict(state.allowed_nodes)
        local_allowed_edges: Dict[int, DomainT] = dict(state.allowed_edges)
        pruned_edges: Dict[int, DataFrameT] = dict(state.pruned_edges)

        for edge_idx in reversed(relevant_edge_indices):
            edge_pos = edge_indices.index(edge_idx)
            left_node_idx = node_indices[edge_pos]
            right_node_idx = node_indices[edge_pos + 1]

            edges_df = self.edges_df_for_step(edge_idx, state)
            if edges_df is None:
                continue
            original_len = len(edges_df)
            allowed_edges = local_allowed_edges.get(edge_idx)
            if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
                edges_df = edges_df[edges_df[edge_id_col].isin(allowed_edges)]

            edge_op = self.inputs.chain[edge_idx]
            if not isinstance(edge_op, ASTEdge):
                continue
            sem = EdgeSemantics.from_edge(edge_op)
            left_allowed = local_allowed_nodes.get(left_node_idx)
            right_allowed = local_allowed_nodes.get(right_node_idx)

            if sem.is_multihop:
                edges_df = filter_multihop_edges_by_endpoints(
                    edges_df, left_allowed, right_allowed, sem, src_col, dst_col
                )
            else:
                edges_df = self._filter_edges_by_allowed_nodes(
                    edges_df, sem, src_col, dst_col, left_allowed, right_allowed)

            if edge_id_col and edge_id_col in edges_df.columns:
                new_edge_ids = series_values(edges_df[edge_id_col])
                local_allowed_edges[edge_idx] = domain_intersect(local_allowed_edges[edge_idx], new_edge_ids) if edge_idx in local_allowed_edges else new_edge_ids

            if sem.is_multihop:
                new_src_nodes = find_multihop_start_nodes(
                    edges_df, right_allowed, sem, src_col, dst_col
                )
            else:
                new_src_nodes = sem.start_nodes(edges_df, src_col, dst_col)
            local_allowed_nodes[left_node_idx] = domain_intersect(local_allowed_nodes[left_node_idx], new_src_nodes) if left_node_idx in local_allowed_nodes else new_src_nodes

            if len(edges_df) < original_len:
                pruned_edges[edge_idx] = edges_df

        return PathState.from_mutable(local_allowed_nodes, local_allowed_edges, pruned_edges)

    def _materialize_filtered(self, state: PathState) -> Plottable:
        nodes_df, node_id, edge_id, src, dst = self.inputs.graph._nodes, self._node_column, self._edge_column, self._source_column, self._destination_column
        edge_frames = [edges for idx, op in enumerate(self.inputs.chain) if isinstance(op, ASTEdge) and (edges := self.edges_df_for_step(idx, state)) is not None]
        concatenated_edges = concat_frames(edge_frames)
        edges_df = concatenated_edges if concatenated_edges is not None else self.inputs.graph._edges

        if nodes_df is None or edges_df is None or node_id is None or src is None or dst is None:
            raise ValueError(
                "Graph bindings are incomplete for same-path execution")

        if any(domain_is_empty(node_set) for node_set in state.allowed_nodes.values()):
            return self._materialize_from_oracle(nodes_df.iloc[0:0], edges_df.iloc[0:0])
        allowed_nodes_domain = domain_union_all(list(state.allowed_nodes.values())) if state.allowed_nodes else None
        if any(isinstance(op, ASTEdge) and EdgeSemantics.from_edge(op).is_multihop for op in self.inputs.chain) and src in edges_df.columns and dst in edges_df.columns:
            endpoints = domain_union(series_values(edges_df[src]), series_values(edges_df[dst]))
            allowed_nodes_domain = endpoints if allowed_nodes_domain is None else domain_union(allowed_nodes_domain, endpoints)

        if domain_is_empty(allowed_nodes_domain):
            filtered_nodes = nodes_df.iloc[0:0]
            filtered_edges = edges_df.iloc[0:0]
        else:
            filtered_nodes = nodes_df[nodes_df[node_id].isin(allowed_nodes_domain)]
            filtered_edges = edges_df[edges_df[src].isin(allowed_nodes_domain) & edges_df[dst].isin(allowed_nodes_domain)]

        if edge_id and edge_id in filtered_edges.columns:
            allowed_edges_domain = domain_union_all(list(state.allowed_edges.values()))
            if allowed_edges_domain is not None:
                filtered_edges = filtered_edges[filtered_edges[edge_id].isin(allowed_edges_domain)]
        filtered_nodes = self._merge_label_frames(filtered_nodes, self._collect_label_frames("node"), node_id)
        if edge_id is not None:
            filtered_edges = self._merge_label_frames(filtered_edges, self._collect_label_frames("edge"), edge_id)

        filtered_edges = self._apply_output_slices(filtered_edges, "edge")
        if any(isinstance(op, ASTEdge) and (
            op.output_min_hops is not None or op.output_max_hops is not None) for op in self.inputs.chain) and len(filtered_edges) > 0:
            endpoint_ids_concat = concat_frames([filtered_edges[[src]].rename(
                columns={src: "__node__"}), filtered_edges[[dst]].rename(columns={dst: "__node__"})])
            if endpoint_ids_concat is not None:
                endpoint_ids_df = endpoint_ids_concat.drop_duplicates()
                filtered_nodes = filtered_nodes[filtered_nodes[node_id].isin(
                    endpoint_ids_df["__node__"])]
        else:
            filtered_nodes = self._apply_output_slices(filtered_nodes, "node")

        for alias, binding in self.inputs.alias_bindings.items():
            frame = filtered_nodes if binding.kind == "node" else filtered_edges
            id_col = self._node_column if binding.kind == "node" else self._edge_column
            if id_col is None or id_col not in frame.columns:
                continue
            required_cols = list(self.inputs.column_requirements.get(alias, ()))
            if id_col not in required_cols:
                required_cols.append(id_col)
            subset = frame[[
                c for c in frame.columns if c in required_cols]].copy()
            self.alias_frames[alias] = subset

        return self._materialize_from_oracle(filtered_nodes, filtered_edges)

    @staticmethod
    def _resolve_label_cols(op: ASTEdge) -> Tuple[Optional[str], Optional[str]]:
        node_label = op.label_node_hops
        edge_label = op.label_edge_hops
        if (
            op.output_min_hops is not None
            or op.output_max_hops is not None
            or (op.min_hops is not None and op.min_hops > 0)
        ):
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
    def _merge_label_frames(base_df: DataFrameT, label_frames: Sequence[DataFrameT], id_col: str) -> DataFrameT:
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
                    merged = merged.assign(
                        **{col: merged[col_x].fillna(merged[col_y])})
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
            node_label, edge_label = self._resolve_label_cols(op)
            label_col = node_label if kind == "node" else edge_label
            if label_col is None or label_col not in out_df.columns:
                hop_like = [c for c in out_df.columns if "hop" in c]
                label_col = hop_like[0] if hop_like else None
            if label_col is None or label_col not in out_df.columns:
                continue
            mask = out_df[label_col].notna()
            if op.output_min_hops is not None:
                mask = mask & (out_df[label_col] >= op.output_min_hops)
            if op.output_max_hops is not None:
                mask = mask & (out_df[label_col] <= op.output_max_hops)
            out_df = out_df[mask]
        return out_df

    def _apply_oracle_hop_labels(self, oracle: "OracleResult") -> Tuple[DataFrameT, DataFrameT]:
        nodes_df, edges_df = oracle.nodes, oracle.edges
        node_id, edge_id = self._node_column, self._edge_column
        node_labels, edge_labels = oracle.node_hop_labels or {}, oracle.edge_hop_labels or {}
        node_frames: List[DataFrameT] = []
        edge_frames: List[DataFrameT] = []
        for op in self.inputs.chain:
            if not isinstance(op, ASTEdge):
                continue
            node_label, edge_label = self._resolve_label_cols(op)
            if node_label and node_id and node_id in nodes_df.columns and node_labels:
                node_series = nodes_df[node_id].map(node_labels)
                node_frames.append(df_cons(nodes_df, {node_id: nodes_df[node_id], node_label: node_series}))
            if edge_label and edge_id and edge_id in edges_df.columns and edge_labels:
                edge_series = edges_df[edge_id].map(edge_labels)
                edge_frames.append(df_cons(edges_df, {edge_id: edges_df[edge_id], edge_label: edge_series}))
        if node_id is not None and node_frames:
            nodes_df = self._merge_label_frames(nodes_df, node_frames, node_id)
        if edge_id is not None and edge_frames:
            edges_df = self._merge_label_frames(edges_df, edge_frames, edge_id)
        return nodes_df, edges_df


def build_same_path_inputs(g: Plottable, chain: Sequence[ASTObject], where: Sequence[WhereComparison], engine: Engine, include_paths: bool = False) -> SamePathExecutorInputs:
    bindings = _collect_alias_bindings(chain)
    _validate_where_aliases(bindings, where)
    return SamePathExecutorInputs(graph=g, chain=tuple(chain), where=tuple(where), engine=engine, alias_bindings=bindings, column_requirements=_collect_required_columns(where), include_paths=include_paths)


def execute_same_path_chain(g: Plottable, chain: Sequence[ASTObject], where: Sequence[WhereComparison], engine: Engine, include_paths: bool = False) -> Plottable:
    return DFSamePathExecutor(build_same_path_inputs(g, chain, where, engine, include_paths)).run()


def _collect_alias_bindings(chain: Sequence[ASTObject]) -> Dict[str, AliasBinding]:
    bindings: Dict[str, AliasBinding] = {}
    for idx, step in enumerate(chain):
        alias = getattr(step, "_name", None)
        if not alias or not isinstance(alias, str):
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


def _collect_required_columns(where: Sequence[WhereComparison]) -> Dict[str, Sequence[str]]:
    requirements: Dict[str, set] = {}
    for clause in where:
        requirements.setdefault(clause.left.alias, set()).add(clause.left.column)
        requirements.setdefault(clause.right.alias, set()).add(clause.right.column)
    return {alias: tuple(cols) for alias, cols in requirements.items()}


def _validate_where_aliases(bindings: Dict[str, AliasBinding], where: Sequence[WhereComparison]) -> None:
    if not where:
        return
    referenced = {clause.left.alias for clause in where} | {clause.right.alias for clause in where}
    missing = sorted(alias for alias in referenced if alias not in bindings)
    if missing:
        raise ValueError(f"WHERE references aliases with no node/edge bindings: {', '.join(missing)}")
