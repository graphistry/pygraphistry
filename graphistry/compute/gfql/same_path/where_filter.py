"""WHERE clause filtering for edges in same-path execution.

Contains functions for filtering edges based on WHERE clause comparisons
between adjacent or multi-hop connected aliases.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

from graphistry.Engine import safe_concat
from graphistry.compute.ast import ASTEdge, ASTNode
from graphistry.compute.typing import DataFrameT, DomainT
from .edge_semantics import EdgeSemantics
from .df_utils import (
    evaluate_clause,
    series_values,
    concat_frames,
    domain_intersect,
    domain_is_empty,
)
from .multihop import filter_multihop_edges_by_endpoints

if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import (
        DFSamePathExecutor,
        WhereComparison,
    )


def filter_edges_by_clauses(
    executor: "DFSamePathExecutor",
    edges_df: DataFrameT,
    left_alias: str,
    right_alias: str,
    allowed_nodes: Dict[int, DomainT],
    sem: EdgeSemantics,
) -> DataFrameT:
    if len(edges_df) == 0:
        return edges_df

    relevant = [
        clause
        for clause in executor.inputs.where
        if {clause.left.alias, clause.right.alias} == {left_alias, right_alias}
    ]
    src_col = executor._source_column
    dst_col = executor._destination_column
    node_col = executor._node_column

    if not relevant or not src_col or not dst_col:
        return edges_df

    left_frame = executor.alias_frames.get(left_alias)
    right_frame = executor.alias_frames.get(right_alias)
    if left_frame is None or right_frame is None or node_col is None:
        return edges_df

    left_allowed = allowed_nodes.get(executor.inputs.alias_bindings[left_alias].step_index)
    right_allowed = allowed_nodes.get(executor.inputs.alias_bindings[right_alias].step_index)

    lf = left_frame
    rf = right_frame
    if left_allowed is not None:
        lf = lf[lf[node_col].isin(left_allowed)]
    if right_allowed is not None:
        rf = rf[rf[node_col].isin(right_allowed)]

    left_cols = [
        col for col in executor.inputs.column_requirements.get(left_alias, [])
        if col != node_col
    ]
    right_cols = [
        col for col in executor.inputs.column_requirements.get(right_alias, [])
        if col != node_col
    ]

    lf = lf[[node_col] + left_cols].rename(columns={
        node_col: "__left_id__",
        **{c: f"__L_{c}" for c in left_cols}
    })
    rf = rf[[node_col] + right_cols].rename(columns={
        node_col: "__right_id__",
        **{c: f"__R_{c}" for c in right_cols}
    })

    if sem.is_undirected:
        merge_cols = [(src_col, dst_col), (dst_col, src_col)]
    elif sem.is_reverse:
        merge_cols = [(dst_col, src_col)]
    else:
        merge_cols = [(src_col, dst_col)]

    frames = [
        _merge_and_filter_edges(
            executor, edges_df, lf, rf, left_alias, right_alias, relevant,
            left_merge_col=left_merge_col,
            right_merge_col=right_merge_col,
        )
        for left_merge_col, right_merge_col in merge_cols
    ]
    non_empty = [frame for frame in frames if len(frame) > 0]
    if not non_empty:
        return frames[0]
    if len(non_empty) == 1:
        return non_empty[0]

    out_df = safe_concat(non_empty, ignore_index=True, sort=False)
    return out_df.drop_duplicates(subset=[src_col, dst_col])


def _merge_and_filter_edges(
    executor: "DFSamePathExecutor",
    edges_df: DataFrameT,
    lf: DataFrameT,
    rf: DataFrameT,
    left_alias: str,
    right_alias: str,
    relevant: List["WhereComparison"],
    left_merge_col: str,
    right_merge_col: str,
) -> DataFrameT:
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
    )

    node_col = executor._node_column
    for clause in relevant:
        left_col = clause.left.column if clause.left.alias == left_alias else clause.right.column
        right_col = clause.right.column if clause.right.alias == right_alias else clause.left.column

        if node_col and left_col == node_col:
            col_left = "__left_id__"
        else:
            col_left = f"__L_{left_col}"
        if node_col and right_col == node_col:
            col_right = "__right_id__"
        else:
            col_right = f"__R_{right_col}"

        if col_left in out_df.columns and col_right in out_df.columns:
            mask = evaluate_clause(out_df[col_left], clause.op, out_df[col_right], null_safe=True)
            out_df = out_df[mask]

    return out_df


def filter_multihop_by_where(
    executor: "DFSamePathExecutor",
    edges_df: DataFrameT,
    edge_op: ASTEdge,
    left_alias: str,
    right_alias: str,
    allowed_nodes: Dict[int, DomainT],
) -> DataFrameT:
    relevant = [
        clause
        for clause in executor.inputs.where
        if {clause.left.alias, clause.right.alias} == {left_alias, right_alias}
    ]
    src_col = executor._source_column
    dst_col = executor._destination_column
    node_col = executor._node_column

    if not relevant or not src_col or not dst_col:
        return edges_df

    left_frame = executor.alias_frames.get(left_alias)
    right_frame = executor.alias_frames.get(right_alias)
    if left_frame is None or right_frame is None or node_col is None:
        return edges_df

    node_label, edge_label = executor._resolve_label_cols(edge_op)

    sem = EdgeSemantics.from_edge(edge_op)

    first_node_step = executor.inputs.chain[0] if executor.inputs.chain else None
    has_filtered_start = (
        isinstance(first_node_step, ASTNode) and first_node_step.filter_dict
    )

    if edge_label and edge_label in edges_df.columns and has_filtered_start:
        hop_col = edges_df[edge_label]
        min_hop = hop_col.min()
        first_hop_edges = edges_df[hop_col == min_hop]

        chain_min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
        valid_endpoint_edges = edges_df[hop_col >= chain_min_hops]

        if sem.is_undirected:
            start_concat = concat_frames([
                first_hop_edges[[src_col]].rename(columns={src_col: '__node__'}),
                first_hop_edges[[dst_col]].rename(columns={dst_col: '__node__'})
            ])
            start_nodes_df = start_concat.drop_duplicates() if start_concat is not None else first_hop_edges[[src_col]].iloc[:0].rename(columns={src_col: '__node__'})
            end_concat = concat_frames([
                valid_endpoint_edges[[src_col]].rename(columns={src_col: '__node__'}),
                valid_endpoint_edges[[dst_col]].rename(columns={dst_col: '__node__'})
            ])
            end_nodes_df = end_concat.drop_duplicates() if end_concat is not None else valid_endpoint_edges[[src_col]].iloc[:0].rename(columns={src_col: '__node__'})
        else:
            start_col, end_col = sem.endpoint_cols(src_col, dst_col)
            start_nodes_df = first_hop_edges[[start_col]].rename(
                columns={start_col: '__node__'}
            ).drop_duplicates()
            end_nodes_df = valid_endpoint_edges[[end_col]].rename(
                columns={end_col: '__node__'}
            ).drop_duplicates()

        start_nodes = series_values(start_nodes_df['__node__'])
        end_nodes = series_values(end_nodes_df['__node__'])
    else:
        start_nodes = series_values(left_frame[node_col])
        end_nodes = series_values(right_frame[node_col])

    left_step_idx = executor.inputs.alias_bindings[left_alias].step_index
    right_step_idx = executor.inputs.alias_bindings[right_alias].step_index
    if left_step_idx in allowed_nodes and not domain_is_empty(allowed_nodes[left_step_idx]):
        start_nodes = domain_intersect(start_nodes, allowed_nodes[left_step_idx])
    if right_step_idx in allowed_nodes and not domain_is_empty(allowed_nodes[right_step_idx]):
        end_nodes = domain_intersect(end_nodes, allowed_nodes[right_step_idx])

    if domain_is_empty(start_nodes) or domain_is_empty(end_nodes):
        return edges_df.iloc[:0]  # Empty dataframe

    lf = left_frame[left_frame[node_col].isin(start_nodes)]
    rf = right_frame[right_frame[node_col].isin(end_nodes)]

    left_cols = [
        col for col in executor.inputs.column_requirements.get(left_alias, [])
        if col != node_col
    ]
    right_cols = [
        col for col in executor.inputs.column_requirements.get(right_alias, [])
        if col != node_col
    ]

    lf = lf[[node_col] + left_cols].rename(columns={
        node_col: "__start_id__",
        **{c: f"__L_{c}" for c in left_cols}
    })
    rf = rf[[node_col] + right_cols].rename(columns={
        node_col: "__end_id__",
        **{c: f"__R_{c}" for c in right_cols}
    })

    lf = lf.assign(__cross_key__=1)
    rf = rf.assign(__cross_key__=1)
    pairs_df = lf.merge(rf, on="__cross_key__").drop(columns=["__cross_key__"])

    for clause in relevant:
        left_col = clause.left.column if clause.left.alias == left_alias else clause.right.column
        right_col = clause.right.column if clause.right.alias == right_alias else clause.left.column
        col_left = f"__L_{left_col}"
        col_right = f"__R_{right_col}"
        if col_left in pairs_df.columns and col_right in pairs_df.columns:
            mask = evaluate_clause(pairs_df[col_left], clause.op, pairs_df[col_right], null_safe=True)
            pairs_df = pairs_df[mask]

    if len(pairs_df) == 0:
        return edges_df.iloc[:0]

    valid_starts = series_values(pairs_df["__start_id__"])
    valid_ends = series_values(pairs_df["__end_id__"])

    return filter_multihop_edges_by_endpoints(
        edges_df, edge_op, valid_starts, valid_ends, sem,
        src_col, dst_col
    )
