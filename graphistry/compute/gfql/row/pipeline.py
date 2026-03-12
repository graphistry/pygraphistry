import ast
import re
from functools import lru_cache
import math
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from typing_extensions import Literal

import pandas as pd
from graphistry.Engine import Engine, EngineAbstract, resolve_engine
from graphistry.compute.dataframe_utils import concat_frames
from graphistry.compute.gfql.row.order_expr import (
    extract_temporal_duration_sort_ast,
    is_order_aggregate_alias_ast,
    order_expr_ast_static_supported,
)
from graphistry.compute.gfql.language_defs import (
    GFQL_COMPARISON_BINARY_OPS,
    GFQL_GROUPBY_AGG_METHODS,
)
from graphistry.compute.gfql.row.dispatch import (
    apply_string_predicate_scalar,
    apply_string_predicate_series,
    eval_sequence_fn_scalar,
    eval_sequence_fn_series,
)
from graphistry.compute.gfql.row.entity_props import (
    edge_property_columns,
    entity_keys_series,
    node_property_columns,
)
from graphistry.compute.gfql.row.entity_text import (
    entity_labels_scalar,
    entity_labels_series,
    entity_properties_scalar,
    entity_properties_series,
    entity_property_scalar,
    entity_property_series,
    entity_type_scalar,
    entity_type_series,
    is_entity_text_scalar,
)
from graphistry.compute.gfql.series_str_compat import series_sequence_len, series_str_match
from graphistry.compute.gfql.row.ordering import (
    build_list_sort_columns,
    build_temporal_sort_columns,
    is_null_scalar,
    order_detect_list_series,
    order_detect_temporal_mode,
    validate_order_series_vector_safe,
)
from graphistry.compute.gfql.temporal_text import parse_temporal_sort_duration_components
from graphistry.compute.gfql.temporal_text import (
    DATETIME_CALL_TEXT_RE,
    DATE_CALL_TEXT_RE,
    LOCALDATETIME_CALL_TEXT_RE,
    LOCALTIME_CALL_TEXT_RE,
    TIME_CALL_TEXT_RE,
    resolve_duration_text_property,
)

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable
    from graphistry.compute.gfql.expr_parser import ExprNode


GFQLParseExprFn = Callable[[str], "ExprNode"]
GFQLValidateExprFn = Callable[["ExprNode"], List[str]]
GFQLRuntimeParserBundle = Tuple[GFQLParseExprFn, GFQLValidateExprFn, ModuleType]
_GFQL_MISSING_BOOL_OPERAND = object()


@lru_cache(maxsize=1)
def _gfql_expr_runtime_parser_bundle() -> Optional[GFQLRuntimeParserBundle]:
    try:
        import graphistry.compute.gfql.expr_parser as expr_parser_mod
        from graphistry.compute.gfql.expr_parser import parse_expr, validate_expr_capabilities
        # Ensure parser backend dependency is present (e.g., lark).
        try:
            parse_expr("1 = 1")
        except ImportError:
            return None
        return parse_expr, validate_expr_capabilities, expr_parser_mod
    except Exception:
        return None


ROW_PIPELINE_CALLS = frozenset(
    {
        "rows",
        "where_rows",
        "select",
        "with_",
        "return_",
        "order_by",
        "skip",
        "limit",
        "distinct",
        "unwind",
        "group_by",
    }
)


def is_row_pipeline_call(function: str) -> bool:
    return function in ROW_PIPELINE_CALLS


class RowPipelineMixin:
    _nodes: Any
    _edges: Any
    _node: Any
    _source: Any
    _destination: Any
    _edge: Any

    def bind(self) -> "Plottable":
        raise NotImplementedError("RowPipelineMixin.bind() must be implemented by host plotter")

    _GFQL_ALIAS_PROP_RE = re.compile(r"^(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.(?P<prop>[A-Za-z_][A-Za-z0-9_]*)$")

    @staticmethod
    def _gfql_fresh_col_name(columns: Any, prefix: str) -> str:
        col = prefix
        while col in columns:
            col = f"{col}_x"
        return col

    @staticmethod
    def _gfql_mask_fill(series: Any, mask: Any, value: Any) -> Any:
        out = series.copy()
        has_mask = True
        if hasattr(mask, "any"):
            try:
                has_mask = bool(mask.any())
            except Exception:
                has_mask = True
        if not has_mask:
            return out
        if value is None and hasattr(out, "astype") and not out.__class__.__module__.startswith("cudf"):
            out = out.astype("object")
        out.loc[mask] = value
        return out

    @staticmethod
    def _gfql_table_has_graph_shape(table_df: Any) -> bool:
        cols = {str(col) for col in table_df.columns}
        return (
            "id" in cols
            or "s" in cols
            or "d" in cols
            or "src" in cols
            or "dst" in cols
            or "edge_id" in cols
            or "type" in cols
            or any(col.startswith("label__") for col in cols)
        )

    @staticmethod
    def _gfql_infer_graph_table_kind(table_df: Any) -> Literal["nodes", "edges"]:
        cols = {str(col) for col in table_df.columns}
        if {"s", "d"} <= cols or {"src", "dst"} <= cols or "edge_id" in cols:
            return "edges"
        return "nodes"

    def _gfql_eval_comparison_op(
        self, table_df: Any, left: Any, right: Any, op: str
    ) -> Optional[Any]:
        cmp_fn = GFQL_COMPARISON_BINARY_OPS.get(op)
        if cmp_fn is None:
            return None

        left_is_list = (
            isinstance(left, (list, tuple))
            or (hasattr(left, "astype") and RowPipelineMixin._gfql_series_is_list_like(left))
        )
        right_is_list = (
            isinstance(right, (list, tuple))
            or (hasattr(right, "astype") and RowPipelineMixin._gfql_series_is_list_like(right))
        )
        if op in {"=", "!=", "<>", "<", "<=", ">", ">="} and (left_is_list or right_is_list):
            list_cmp = self._gfql_eval_list_comparison_op(table_df, left, right, op)
            if list_cmp is not None:
                return list_cmp

        temporal_cmp = self._gfql_eval_temporal_comparison_op(table_df, left, right, op)
        if temporal_cmp is not None:
            return temporal_cmp

        left_null_mask = self._gfql_null_mask(table_df, left)
        right_null_mask = self._gfql_null_mask(table_df, right)
        if isinstance(left, float) and math.isnan(left):
            left_null_mask = self._gfql_broadcast_scalar(table_df, False).astype(bool)
        if isinstance(right, float) and math.isnan(right):
            right_null_mask = self._gfql_broadcast_scalar(table_df, False).astype(bool)
        out = cmp_fn(left, right)
        if hasattr(out, "where"):
            out = out.where(~(left_null_mask | right_null_mask), pd.NA)
        else:
            null_mask = left_null_mask | right_null_mask
            if hasattr(null_mask, "where"):
                out = self._gfql_broadcast_scalar(table_df, out).where(~null_mask, pd.NA)
            elif bool(null_mask):
                out = None
        return out

    def _gfql_eval_temporal_comparison_op(
        self,
        table_df: Any,
        left_value: Any,
        right_value: Any,
        op: str,
    ) -> Optional[Any]:
        if op not in {"=", "!=", "<>", "<", "<=", ">", ">="}:
            return None

        left_series = left_value if hasattr(left_value, "astype") else self._gfql_broadcast_scalar(table_df, left_value)
        right_series = right_value if hasattr(right_value, "astype") else self._gfql_broadcast_scalar(table_df, right_value)
        left_mode = order_detect_temporal_mode(left_series)
        right_mode = order_detect_temporal_mode(right_series)
        if left_mode is None or right_mode is None:
            return None

        def _temporal_family(mode: str) -> str:
            if mode in {"date", "date_constructor"}:
                return "date"
            if mode in {"time", "time_constructor"}:
                return "time"
            return "datetime"

        if _temporal_family(left_mode) != _temporal_family(right_mode):
            return None

        left_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_cmp_left_temporal__")
        right_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_cmp_right_temporal__")
        work_df = table_df.reset_index(drop=True).copy()
        work_df = work_df.assign(**{left_col: left_series, right_col: right_series})

        work_df, left_keys = build_temporal_sort_columns(
            work_df,
            left_col,
            "__gfql_cmp_left_key__",
            left_mode,
            null_mask_fn=self._gfql_null_mask,
            fresh_col_name_fn=RowPipelineMixin._gfql_fresh_col_name,
        )
        work_df, right_keys = build_temporal_sort_columns(
            work_df,
            right_col,
            "__gfql_cmp_right_key__",
            right_mode,
            null_mask_fn=self._gfql_null_mask,
            fresh_col_name_fn=RowPipelineMixin._gfql_fresh_col_name,
        )

        left_null_mask = self._gfql_null_mask(work_df, work_df[left_col])
        right_null_mask = self._gfql_null_mask(work_df, work_df[right_col])
        any_null_mask = left_null_mask | right_null_mask

        eq_out = work_df[left_keys[0]] == work_df[right_keys[0]]
        if len(left_keys) > 1:
            eq_out = eq_out & (work_df[left_keys[1]] == work_df[right_keys[1]])

        lt_out = work_df[left_keys[0]] < work_df[right_keys[0]]
        if len(left_keys) > 1:
            lt_out = lt_out | (
                (work_df[left_keys[0]] == work_df[right_keys[0]])
                & (work_df[left_keys[1]] < work_df[right_keys[1]])
            )

        if op == "=":
            out = eq_out
        elif op in {"!=", "<>"}:
            out = ~eq_out
        elif op == "<":
            out = lt_out
        elif op == "<=":
            out = lt_out | eq_out
        elif op == ">":
            out = ~(lt_out | eq_out)
        else:
            out = ~lt_out

        if hasattr(out, "where"):
            out = out.where(~any_null_mask, pd.NA)
        return out

    def _gfql_eval_list_comparison_op(
        self,
        table_df: Any,
        left_value: Any,
        right_value: Any,
        op: str,
    ) -> Optional[Any]:
        left_series = left_value if hasattr(left_value, "astype") else self._gfql_broadcast_scalar(table_df, left_value)
        right_series = right_value if hasattr(right_value, "astype") else self._gfql_broadcast_scalar(table_df, right_value)

        if not RowPipelineMixin._gfql_series_is_list_like(left_series):
            return None
        if not RowPipelineMixin._gfql_series_is_list_like(right_series):
            return None

        if op in {"<", "<=", ">", ">="}:
            left_values = self._gfql_series_to_pylist(left_series)
            right_values = self._gfql_series_to_pylist(right_series)
            out_values: List[Optional[bool]] = []
            for left_item, right_item in zip(left_values, right_values):
                if is_null_scalar(left_item) or is_null_scalar(right_item):
                    out_values.append(None)
                    continue
                try:
                    if op == "<":
                        out_values.append(left_item < right_item)
                    elif op == "<=":
                        out_values.append(left_item <= right_item)
                    elif op == ">":
                        out_values.append(left_item > right_item)
                    else:
                        out_values.append(left_item >= right_item)
                except Exception:
                    out_values.append(None)
            out_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_cmp_py__")
            return table_df.reset_index(drop=True).assign(**{out_col: out_values})[out_col]

        row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_cmp_row__")
        lhs_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_cmp_lhs__")
        rhs_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_cmp_rhs__")
        lhs_len_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_cmp_lhs_len__")
        rhs_len_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_cmp_rhs_len__")
        pos_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_cmp_pos__")
        match_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_cmp_match__")
        unknown_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_cmp_unknown__")

        base = table_df.assign(**{row_col: range(len(table_df)), lhs_col: left_series, rhs_col: right_series})
        left_null = self._gfql_null_mask(base, base[lhs_col])
        right_null = self._gfql_null_mask(base, base[rhs_col])
        null_mask = left_null | right_null

        lhs_len = series_sequence_len(base[lhs_col]).fillna(0).astype("int64")
        rhs_len = series_sequence_len(base[rhs_col]).fillna(0).astype("int64")
        base = base.assign(**{lhs_len_col: lhs_len, rhs_len_col: rhs_len})

        non_null = base.loc[~null_mask, [row_col, lhs_col, rhs_col, lhs_len_col, rhs_len_col]].copy()
        if len(non_null) == 0:
            return self._gfql_broadcast_scalar(table_df, pd.NA)

        lhs_expanded = non_null[[row_col, lhs_col, lhs_len_col]].explode(lhs_col)
        rhs_expanded = non_null[[row_col, rhs_col, rhs_len_col]].explode(rhs_col)

        if len(lhs_expanded) > 0:
            lhs_expanded = lhs_expanded.assign(**{pos_col: lhs_expanded.groupby(row_col, sort=False).cumcount()})
            lhs_expanded = lhs_expanded.loc[lhs_expanded[pos_col] < lhs_expanded[lhs_len_col]]
        if len(rhs_expanded) > 0:
            rhs_expanded = rhs_expanded.assign(**{pos_col: rhs_expanded.groupby(row_col, sort=False).cumcount()})
            rhs_expanded = rhs_expanded.loc[rhs_expanded[pos_col] < rhs_expanded[rhs_len_col]]

        expanded = lhs_expanded.merge(
            rhs_expanded[[row_col, rhs_col, pos_col]],
            on=[row_col, pos_col],
            how="outer",
            sort=False,
        )

        if len(expanded) > 0:
            lhs_elem = expanded[lhs_col]
            rhs_elem = expanded[rhs_col]
            lhs_elem_null = self._gfql_null_mask(expanded, lhs_elem)
            rhs_elem_null = self._gfql_null_mask(expanded, rhs_elem)
            both_null = lhs_elem_null & rhs_elem_null
            elem_equal = lhs_elem == rhs_elem
            if hasattr(elem_equal, "where"):
                elem_equal = elem_equal.where(~(lhs_elem_null | rhs_elem_null), both_null)
            else:
                elem_equal = both_null if bool(lhs_elem_null | rhs_elem_null) else elem_equal
            elem_unknown = (lhs_elem_null | rhs_elem_null) & (~both_null)
            eval_df = expanded.assign(
                **{
                    match_col: elem_equal.astype("int64"),
                    unknown_col: elem_unknown.astype("int64"),
                }
            )
            match_counts = eval_df.groupby(row_col, sort=False)[match_col].sum().reset_index()
            unknown_counts = eval_df.groupby(row_col, sort=False)[unknown_col].sum().reset_index()
        else:
            match_counts = non_null[[row_col]].iloc[0:0].copy()
            match_counts[match_col] = []
            unknown_counts = non_null[[row_col]].iloc[0:0].copy()
            unknown_counts[unknown_col] = []

        summary = base[[row_col, lhs_len_col, rhs_len_col]].merge(match_counts, on=row_col, how="left", sort=False)
        summary = summary.merge(unknown_counts, on=row_col, how="left", sort=False)
        summary = self._gfql_restore_row_order(summary, row_col)
        summary = summary.assign(
            **{
                match_col: summary[match_col].fillna(0).astype("int64"),
                unknown_col: summary[unknown_col].fillna(0).astype("int64"),
            }
        )
        lengths_equal = summary[lhs_len_col] == summary[rhs_len_col]
        known_equal = lengths_equal & (summary[match_col] == summary[lhs_len_col])
        unknown_equal = lengths_equal & (summary[match_col] + summary[unknown_col] == summary[lhs_len_col]) & (~known_equal)

        out = self._gfql_broadcast_scalar(summary, False)
        out = out.where(~unknown_equal, pd.NA)
        out = out.where(~known_equal, True)
        out = out.where(~null_mask, pd.NA)
        if op in {"!=", "<>"}:
            out = (~out.astype("boolean")).where(~out.isna(), pd.NA)
        return out.reset_index(drop=True)

    def _gfql_eval_expr_ast(self, table_df: Any, node: Any) -> Tuple[bool, Any]:
        parser_bundle = _gfql_expr_runtime_parser_bundle()
        if parser_bundle is None:
            return False, None
        _parse_expr, _validate_expr_capabilities, expr_parser_mod = parser_bundle

        Identifier = expr_parser_mod.Identifier
        Literal = expr_parser_mod.Literal
        UnaryOp = expr_parser_mod.UnaryOp
        BinaryOp = expr_parser_mod.BinaryOp
        IsNullOp = expr_parser_mod.IsNullOp
        FunctionCall = expr_parser_mod.FunctionCall
        CaseWhen = expr_parser_mod.CaseWhen
        QuantifierExpr = expr_parser_mod.QuantifierExpr
        ListComprehension = expr_parser_mod.ListComprehension
        ListLiteral = expr_parser_mod.ListLiteral
        MapLiteral = expr_parser_mod.MapLiteral
        PropertyAccessExpr = expr_parser_mod.PropertyAccessExpr
        SubscriptExpr = expr_parser_mod.SubscriptExpr
        SliceExpr = expr_parser_mod.SliceExpr

        if isinstance(node, Identifier):
            txt = node.name
            if txt in table_df.columns:
                return True, table_df[txt]
            return True, self._gfql_resolve_token(table_df, txt)

        if isinstance(node, Literal):
            value = node.value
            if isinstance(value, (list, tuple, dict)):
                return True, self._gfql_broadcast_scalar(table_df, value)
            return True, value

        if isinstance(node, ListLiteral):
            item_values: List[Any] = []
            for item in node.items:
                ok, val = self._gfql_eval_expr_ast(table_df, item)
                if not ok:
                    return False, None
                item_values.append(val)
            if not any(hasattr(val, "astype") for val in item_values):
                return True, list(item_values)

            row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_ast_list_row__")
            ord_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_ast_list_ord__")
            val_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_ast_list_val__")

            base = table_df.assign(**{row_col: range(len(table_df))})[[row_col]]
            value_cols: List[str] = []
            for idx, val in enumerate(item_values):
                if not hasattr(val, "astype"):
                    val = self._gfql_broadcast_scalar(table_df, val)
                col = RowPipelineMixin._gfql_fresh_col_name(base.columns, f"__gfql_ast_item_{idx}__")
                base[col] = val
                value_cols.append(col)

            try:
                melted = base.melt(id_vars=[row_col], value_vars=value_cols, var_name=ord_col, value_name=val_col)
                order_map = {col: idx for idx, col in enumerate(value_cols)}
                if hasattr(melted[ord_col], "map"):
                    melted[ord_col] = melted[ord_col].map(order_map)
                else:
                    melted[ord_col] = melted[ord_col].replace(order_map)
                melted[ord_col] = melted[ord_col].astype("int64")
                melted = melted.sort_values(by=[row_col, ord_col], kind="mergesort")
                grouped = melted.groupby(row_col, sort=False)[val_col].agg(list).reset_index()
                out = self._gfql_restore_row_order(
                    base[[row_col]].merge(grouped, on=row_col, how="left", sort=False),
                    row_col,
                )[val_col]
                return True, out.reset_index(drop=True)
            except Exception:
                row_count = len(table_df)
                materialized_items: List[List[Any]] = []
                for val in item_values:
                    if hasattr(val, "astype"):
                        materialized_items.append(RowPipelineMixin._gfql_series_to_pylist(val))
                    else:
                        materialized_items.append([val] * row_count)
                out_values = [
                    [materialized_items[col_idx][row_idx] for col_idx in range(len(materialized_items))]
                    for row_idx in range(row_count)
                ]
                return True, pd.Series(out_values, dtype="object")

        if isinstance(node, MapLiteral):
            out_map: Dict[str, Any] = {}
            for key, value_node in node.items:
                ok, value = self._gfql_eval_expr_ast(table_df, value_node)
                if not ok:
                    return False, None
                if hasattr(value, "astype"):
                    # Vector map values are deferred to legacy evaluator for now.
                    return False, None
                out_map[str(key)] = value
            return True, out_map

        if isinstance(node, PropertyAccessExpr):
            if isinstance(node.value, Identifier):
                alias_name = node.value.name
                has_bound_graph_table = (
                    (self._node is not None and self._node in table_df.columns)
                    or (self._edge is not None and self._edge in table_df.columns)
                    or RowPipelineMixin._gfql_table_has_graph_shape(table_df)
                )
                if (
                    "." not in alias_name
                    and alias_name in table_df.columns
                    and has_bound_graph_table
                    and RowPipelineMixin._gfql_series_bool_like(table_df[alias_name])
                ):
                    alias_mask = table_df[alias_name]
                    prop_value = (
                        table_df[node.property]
                        if node.property in table_df.columns
                        else self._gfql_broadcast_scalar(table_df, None)
                    )
                    if hasattr(prop_value, "where"):
                        prop_value = self._gfql_mask_fill(prop_value, alias_mask != True, None)  # noqa: E712
                    return True, prop_value
            ok, value = self._gfql_eval_expr_ast(table_df, node.value)
            if not ok:
                return False, None
            return True, self._gfql_eval_property_access_value(
                table_df,
                value,
                node.property,
                f"ast property access .{node.property}",
            )

        if isinstance(node, IsNullOp):
            ok, value = self._gfql_eval_expr_ast(table_df, node.value)
            if not ok:
                return False, None
            out = self._gfql_null_mask(table_df, value)
            if node.negated:
                out = ~out
            return True, out

        if isinstance(node, UnaryOp):
            ok, operand = self._gfql_eval_expr_ast(table_df, node.operand)
            if not ok:
                return False, None
            if node.op == "+":
                return True, operand
            if node.op == "-":
                return True, 0 - operand
            if node.op == "not":
                bool_out = self._gfql_eval_boolean_op(table_df, operand, _GFQL_MISSING_BOOL_OPERAND, "not")
                if bool_out is None:
                    return False, None
                return True, bool_out
            return False, None

        if isinstance(node, BinaryOp):
            op = str(node.op).lower()

            if op in {"or", "and"}:
                prefer_right = (
                    isinstance(node.right, IsNullOp)
                    or (isinstance(node.right, Literal) and isinstance(node.right.value, bool))
                ) and not (
                    isinstance(node.left, IsNullOp)
                    or (isinstance(node.left, Literal) and isinstance(node.left.value, bool))
                )
                first_node = node.right if prefer_right else node.left
                second_node = node.left if prefer_right else node.right
                ok_first, first = self._gfql_eval_expr_ast(table_df, first_node)
                if not ok_first:
                    return False, None
                short = self._gfql_boolean_short_circuit_result(table_df, first, op)
                if short is not None:
                    return True, short
                ok_second, second = self._gfql_eval_expr_ast(table_df, second_node)
                if not ok_second:
                    return False, None
                if prefer_right:
                    left, right = second, first
                else:
                    left, right = first, second
                bool_out = self._gfql_eval_boolean_op(table_df, left, right, op)
                if bool_out is None:
                    return False, None
                return True, bool_out

            ok_l, left = self._gfql_eval_expr_ast(table_df, node.left)
            ok_r, right = self._gfql_eval_expr_ast(table_df, node.right)
            if not (ok_l and ok_r):
                return False, None

            if op == "xor":
                bool_out = self._gfql_eval_boolean_op(table_df, left, right, op)
                if bool_out is None:
                    return False, None
                return True, bool_out

            cmp_out = RowPipelineMixin._gfql_eval_comparison_op(self, table_df, left, right, op)
            if cmp_out is not None:
                return True, cmp_out

            if op == "in":
                return True, self._gfql_eval_in_expr(table_df, left, right, "ast IN")
            if op == "contains":
                return True, self._gfql_eval_string_predicate_expr(table_df, left, right, "contains", "ast CONTAINS")
            if op == "starts_with":
                return True, self._gfql_eval_string_predicate_expr(table_df, left, right, "starts_with", "ast STARTS WITH")
            if op == "ends_with":
                return True, self._gfql_eval_string_predicate_expr(table_df, left, right, "ends_with", "ast ENDS WITH")

            if op == "+":
                if isinstance(left, (list, tuple)) and not isinstance(right, (list, tuple)):
                    right_is_series = hasattr(right, "astype")
                    right_is_list = right_is_series and RowPipelineMixin._gfql_series_is_list_like(right)
                    if right_is_list:
                        out = right
                        for item in reversed(list(left)):
                            out = self._gfql_concat_list_scalar(
                                table_df,
                                out,
                                self._gfql_broadcast_scalar(table_df, item),
                                prepend=True,
                            )
                        return True, out
                    if right_is_series:
                        left_series = self._gfql_broadcast_scalar(table_df, list(left))
                        return True, self._gfql_concat_list_scalar(table_df, left_series, right, prepend=False)
                    return True, list(left) + [right]
                if isinstance(right, (list, tuple)) and not isinstance(left, (list, tuple)):
                    left_is_series = hasattr(left, "astype")
                    left_is_list = left_is_series and RowPipelineMixin._gfql_series_is_list_like(left)
                    if left_is_list:
                        out = left
                        for item in list(right):
                            out = self._gfql_concat_list_scalar(
                                table_df,
                                out,
                                self._gfql_broadcast_scalar(table_df, item),
                                prepend=False,
                            )
                        return True, out
                    if left_is_series:
                        right_series = self._gfql_broadcast_scalar(table_df, list(right))
                        return True, self._gfql_concat_list_scalar(table_df, right_series, left, prepend=True)
                    return True, [left] + list(right)
                if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
                    return True, list(left) + list(right)
                left_is_list = hasattr(left, "astype") and RowPipelineMixin._gfql_series_is_list_like(left)
                right_is_list = hasattr(right, "astype") and RowPipelineMixin._gfql_series_is_list_like(right)
                if left_is_list and not right_is_list:
                    if not hasattr(right, "astype"):
                        right = self._gfql_broadcast_scalar(table_df, right)
                    return True, self._gfql_concat_list_scalar(table_df, left, right, prepend=False)
                if right_is_list and not left_is_list:
                    if not hasattr(left, "astype"):
                        left = self._gfql_broadcast_scalar(table_df, left)
                    return True, self._gfql_concat_list_scalar(table_df, right, left, prepend=True)
                return True, left + right
            if op == "-":
                return True, left - right
            if op == "*":
                return True, left * right
            if op == "/":
                try:
                    return True, left / right
                except ZeroDivisionError:
                    if isinstance(left, bool) or isinstance(right, bool):
                        return False, None
                    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        if right == 0:
                            if isinstance(left, float) or isinstance(right, float):
                                if left == 0:
                                    return True, float("nan")
                                return True, float("inf") if left > 0 else float("-inf")
                    return False, None
            if op == "%":
                if (hasattr(left, "astype") and RowPipelineMixin._gfql_series_bool_like(left)) or (
                    hasattr(right, "astype") and RowPipelineMixin._gfql_series_bool_like(right)
                ):
                    return False, None
                if isinstance(left, bool) or isinstance(right, bool):
                    return False, None
                return True, left % right

            return False, None

        if isinstance(node, FunctionCall):
            fn = str(node.name).lower()
            if fn in {"__node_keys__", "__edge_keys__"}:
                if len(node.args) < 1 or not isinstance(node.args[0], Identifier):
                    return False, None
                alias_names: List[str] = [node.args[0].name]
                for extra_arg in node.args[1:]:
                    if isinstance(extra_arg, Identifier):
                        alias_names.append(extra_arg.name)
                    elif isinstance(extra_arg, Literal) and isinstance(extra_arg.value, str):
                        alias_names.append(extra_arg.value)
                    else:
                        return False, None
                source_alias = alias_names[0]
                if source_alias not in table_df.columns:
                    return False, None
                out = entity_keys_series(
                    table_df,
                    alias_col=source_alias,
                    table=("nodes" if fn == "__node_keys__" else "edges"),
                    excluded=tuple(alias_names),
                )
                null_mask = self._gfql_null_mask(table_df, table_df[source_alias])
                if hasattr(out, "where"):
                    out = self._gfql_mask_fill(out, null_mask, None)
                return True, out
            if fn in {"__node_entity__", "__edge_entity__"}:
                if len(node.args) < 1 or not isinstance(node.args[0], Identifier):
                    return False, None
                entity_alias_names: List[str] = [node.args[0].name]
                for extra_arg in node.args[1:]:
                    if isinstance(extra_arg, Identifier):
                        entity_alias_names.append(extra_arg.name)
                    elif isinstance(extra_arg, Literal) and isinstance(extra_arg.value, str):
                        entity_alias_names.append(extra_arg.value)
                    else:
                        return False, None
                source_alias = entity_alias_names[0]
                if source_alias not in table_df.columns:
                    return False, None
                out = self._gfql_format_entity_series(
                    table_df,
                    alias_col=source_alias,
                    table=("nodes" if fn == "__node_entity__" else "edges"),
                    excluded=tuple(entity_alias_names),
                )
                null_mask = self._gfql_null_mask(table_df, table_df[source_alias])
                if hasattr(out, "where"):
                    out = self._gfql_mask_fill(out, null_mask, None)
                return True, out
            if fn == "labels" and len(node.args) == 1 and isinstance(node.args[0], Identifier):
                alias_name = node.args[0].name
                if (
                    "." not in alias_name
                    and alias_name in table_df.columns
                    and "id" in table_df.columns
                    and RowPipelineMixin._gfql_series_bool_like(table_df[alias_name])
                ):
                    out = self._gfql_format_labels_series(table_df, alias_col=alias_name)
                    null_mask = self._gfql_null_mask(table_df, table_df[alias_name])
                    if hasattr(out, "where"):
                        out = self._gfql_mask_fill(out, null_mask, None)
                    return True, out
            if fn == "type" and len(node.args) == 1 and isinstance(node.args[0], Identifier):
                alias_name = node.args[0].name
                edge_like_cols = {"s", "d", "src", "dst", "edge_id"}
                if (
                    "." not in alias_name
                    and alias_name in table_df.columns
                    and "type" in table_df.columns
                    and any(col in table_df.columns for col in edge_like_cols)
                    and RowPipelineMixin._gfql_series_bool_like(table_df[alias_name])
                ):
                    null_mask = self._gfql_null_mask(table_df, table_df[alias_name])
                    out = table_df["type"]
                    if hasattr(out, "where"):
                        out = self._gfql_mask_fill(out, null_mask, None)
                    return True, out
            if fn == "properties" and len(node.args) == 1 and isinstance(node.args[0], Identifier):
                alias_name = node.args[0].name
                if (
                    "." not in alias_name
                    and alias_name in table_df.columns
                    and RowPipelineMixin._gfql_series_bool_like(table_df[alias_name])
                ):
                    edge_like_cols = {"s", "d", "src", "dst", "edge_id"}
                    table_name = "edges" if any(col in table_df.columns for col in edge_like_cols) else "nodes"
                    entity_text = self._gfql_format_entity_series(
                        table_df,
                        alias_col=alias_name,
                        table=table_name,
                        excluded=(alias_name,),
                    )
                    if not hasattr(entity_text, "str"):
                        return False, None
                    null_mask = self._gfql_null_mask(table_df, table_df[alias_name])
                    props = entity_text.str.extract(r"(\{.*\})", expand=False)
                    props = props.where(props.notna(), "{}")
                    props = self._gfql_mask_fill(props, null_mask, None)
                    return True, props

            if fn == "keys" and len(node.args) == 1:
                arg = node.args[0]
                ok, inner = self._gfql_eval_expr_ast(table_df, arg)
                if not ok:
                    return False, None
                if hasattr(inner, "astype"):
                    null_mask = self._gfql_null_mask(table_df, inner)
                    if hasattr(null_mask, "all") and bool(null_mask.all()):
                        out = self._gfql_broadcast_scalar(table_df, None)
                        if hasattr(out, "where"):
                            out = self._gfql_mask_fill(out, null_mask, None)
                        return True, out
                    return False, None
                if is_null_scalar(inner):
                    return True, None
                if isinstance(inner, dict):
                    return True, list(inner.keys())
                return False, None

            values: List[Any] = []
            for arg in node.args:
                ok, val = self._gfql_eval_expr_ast(table_df, arg)
                if not ok:
                    return False, None
                values.append(val)

            if fn == "__cypher_case_eq__" and len(values) == 2:
                left = values[0]
                right = values[1]
                left_null_mask = self._gfql_null_mask(table_df, left)
                right_null_mask = self._gfql_null_mask(table_df, right)
                any_null_mask = left_null_mask | right_null_mask
                left_is_bool = (
                    RowPipelineMixin._gfql_series_bool_like(left)
                    if hasattr(left, "astype")
                    else isinstance(left, bool)
                )
                right_is_bool = (
                    RowPipelineMixin._gfql_series_bool_like(right)
                    if hasattr(right, "astype")
                    else isinstance(right, bool)
                )
                left_is_numeric = (
                    RowPipelineMixin._gfql_series_numeric_non_bool_like(left)
                    if hasattr(left, "astype")
                    else RowPipelineMixin._gfql_scalar_numeric_non_bool(left)
                )
                right_is_numeric = (
                    RowPipelineMixin._gfql_series_numeric_non_bool_like(right)
                    if hasattr(right, "astype")
                    else RowPipelineMixin._gfql_scalar_numeric_non_bool(right)
                )
                if (left_is_bool and right_is_numeric) or (left_is_numeric and right_is_bool):
                    return True, self._gfql_broadcast_scalar(table_df, False).where(~any_null_mask, False)
                out = left == right
                if hasattr(out, "where"):
                    out = out.where(~any_null_mask, False)
                elif hasattr(any_null_mask, "any") and bool(any_null_mask.any()):
                    out = False
                return True, out

            if fn == "properties" and len(values) == 1:
                inner = values[0]
                if is_null_scalar(inner):
                    return True, None
                if isinstance(inner, Mapping):
                    return True, RowPipelineMixin._gfql_format_cypher_map_scalar(inner)
                return True, self._gfql_eval_entity_graph_fn(table_df, inner, "properties", "ast properties()")

            if fn in {"labels", "type"} and len(values) == 1:
                return True, self._gfql_eval_entity_graph_fn(table_df, values[0], fn, f"ast {fn}()")

            if fn == "range" and len(values) in {2, 3}:
                return True, self._gfql_eval_range_expr(table_df, values, "range(...)")

            if fn == "size" and len(values) == 1:
                inner = values[0]
                if is_null_scalar(inner):
                    return True, None
                if hasattr(inner, "astype"):
                    try:
                        return True, series_sequence_len(inner)
                    except Exception:
                        pass
                try:
                    return True, len(inner)
                except Exception:
                    return False, None

            if fn == "abs" and len(values) == 1:
                inner = values[0]
                if hasattr(inner, "abs"):
                    return True, inner.abs()
                return True, abs(inner)

            if fn == "sqrt" and len(values) == 1:
                inner = values[0]
                if hasattr(inner, "astype"):
                    return True, inner.astype(float) ** 0.5
                if is_null_scalar(inner):
                    return True, None
                return True, float(inner) ** 0.5

            if fn == "tofloat" and len(values) == 1:
                inner = values[0]
                if hasattr(inner, "astype"):
                    null_mask = self._gfql_null_mask(table_df, inner)
                    out = inner.astype(float)
                    return True, out.where(~null_mask, pd.NA)
                if is_null_scalar(inner):
                    return True, None
                return True, float(inner)

            if fn == "tointeger" and len(values) == 1:
                inner = values[0]
                if hasattr(inner, "astype"):
                    null_mask = self._gfql_null_mask(table_df, inner)
                    out = inner.astype(float).fillna(0).astype("int64")
                    return True, out.where(~null_mask, pd.NA)
                if is_null_scalar(inner):
                    return True, None
                return True, int(float(inner))

            if fn == "substring" and len(values) in {2, 3}:
                inner = values[0]
                start = values[1]
                length = values[2] if len(values) == 3 else None
                if any(hasattr(value, "astype") for value in (start, length) if value is not None):
                    return False, None
                if not isinstance(start, int) or isinstance(start, bool):
                    return False, None
                if length is not None and (not isinstance(length, int) or isinstance(length, bool)):
                    return False, None
                stop = None if length is None else start + length
                if hasattr(inner, "str") and hasattr(inner.str, "slice"):
                    return True, inner.str.slice(start, stop)
                if is_null_scalar(inner):
                    return True, None
                if not isinstance(inner, str):
                    return False, None
                return True, inner[start:stop]

            if fn == "toboolean" and len(values) == 1:
                inner = values[0]
                if hasattr(inner, "astype"):
                    null_mask = self._gfql_null_mask(table_df, inner)
                    normalized = inner.astype(str).str.strip().str.lower()
                    true_mask = normalized.isin(["true", "t", "1", "yes"])
                    false_mask = normalized.isin(["false", "f", "0", "no"])
                    unsupported_mask = ~(true_mask | false_mask | null_mask)
                    if hasattr(unsupported_mask, "any") and bool(unsupported_mask.any()):
                        return False, None
                    out = true_mask.where(~false_mask, False)
                    return True, out.where(~null_mask, pd.NA)
                if is_null_scalar(inner):
                    return True, None
                if isinstance(inner, bool):
                    return True, inner
                if isinstance(inner, (int, float)):
                    return True, inner != 0
                txt_inner = str(inner).strip().lower()
                if txt_inner in {"true", "t", "1", "yes"}:
                    return True, True
                if txt_inner in {"false", "f", "0", "no"}:
                    return True, False
                return False, None

            if fn == "tostring" and len(values) == 1:
                inner = values[0]
                if hasattr(inner, "astype"):
                    null_mask = self._gfql_null_mask(table_df, inner)
                    out = inner.astype(str)
                    if hasattr(out, "str"):
                        out = out.str.replace(r"^True$", "true", regex=True)
                        out = out.str.replace(r"^False$", "false", regex=True)
                    return True, self._gfql_mask_fill(out, null_mask, None)
                if is_null_scalar(inner):
                    return True, None
                if isinstance(inner, bool):
                    return True, ("true" if inner else "false")
                return True, str(inner)

            if fn == "coalesce" and len(values) >= 1:
                out = values[0]
                if not hasattr(out, "astype"):
                    out = self._gfql_broadcast_scalar(table_df, out)
                for candidate in values[1:]:
                    if not hasattr(candidate, "astype"):
                        candidate = self._gfql_broadcast_scalar(table_df, candidate)
                    null_mask = self._gfql_null_mask(table_df, out)
                    out = out.where(~null_mask, candidate)
                return True, out

            if fn == "sign" and len(values) == 1:
                inner = values[0]
                if hasattr(inner, "astype"):
                    null_mask = self._gfql_null_mask(table_df, inner)
                    gt = inner > 0
                    lt = inner < 0
                    out = self._gfql_broadcast_scalar(table_df, 0)
                    out = out.where(~gt, 1)
                    out = out.where(~lt, -1)
                    return True, out.where(~null_mask, pd.NA)
                if is_null_scalar(inner):
                    return True, None
                if inner > 0:
                    return True, 1
                if inner < 0:
                    return True, -1
                return True, 0

            if fn in {"head", "tail", "reverse", "nodes", "relationships"} and len(values) == 1:
                inner = values[0]
                if fn in {"nodes", "relationships"}:
                    return True, inner
                if hasattr(inner, "str"):
                    try:
                        return True, eval_sequence_fn_series(inner, fn)
                    except Exception as exc:
                        raise ValueError(
                            f"unsupported row expression: {fn}() requires list/string input"
                        ) from exc
                if is_null_scalar(inner):
                    return True, None
                try:
                    return True, eval_sequence_fn_scalar(inner, fn)
                except Exception as exc:
                    raise ValueError(
                        f"unsupported row expression: {fn}() requires list/string input"
                    ) from exc

            return False, None

        if isinstance(node, CaseWhen):
            ok_cond, cond_value = self._gfql_eval_expr_ast(table_df, node.condition)
            if not ok_cond:
                return False, None
            ok_true, true_value = self._gfql_eval_expr_ast(table_df, node.when_true)
            ok_false, false_value = self._gfql_eval_expr_ast(table_df, node.when_false)
            if not (ok_true and ok_false):
                return False, None
            if not hasattr(true_value, "astype"):
                true_value = self._gfql_broadcast_scalar(table_df, true_value)
            if not hasattr(false_value, "astype"):
                false_value = self._gfql_broadcast_scalar(table_df, false_value)
            cond_mask = self._gfql_bool_mask(table_df, cond_value)
            cond_null = self._gfql_null_mask(table_df, cond_value)
            cond_true = cond_mask & ~cond_null
            return True, true_value.where(cond_true, false_value)

        if isinstance(node, QuantifierExpr):
            source_ok, list_value = self._gfql_eval_expr_ast(table_df, node.source)
            if not source_ok:
                return False, None
            list_series = (
                list_value if hasattr(list_value, "astype") else self._gfql_broadcast_scalar(table_df, list_value)
            )

            row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_q_row_ast__")
            list_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_q_list_ast__")
            total_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_q_total_ast__")

            base = table_df.reset_index(drop=True).copy()
            base = base.assign(**{row_col: range(len(base)), list_col: list_series})
            list_null_mask = self._gfql_null_mask(base, base[list_col])
            try:
                total_series = series_sequence_len(base[list_col])
            except Exception:
                total_series = self._gfql_broadcast_scalar(base, pd.NA)
            if hasattr(total_series, "where"):
                total_series = total_series.where(~list_null_mask, 0).fillna(0)
            base = base.assign(**{total_col: total_series})

            non_null = base.loc[~list_null_mask].copy()
            expanded = non_null.explode(list_col)
            if len(expanded) > 0:
                expanded = expanded.reset_index(drop=True)
                expanded = expanded.assign(
                    **{"__gfql_q_pos_ast__": expanded.groupby(row_col, sort=False).cumcount()}
                )
                expanded = expanded.loc[expanded["__gfql_q_pos_ast__"] < expanded[total_col]]
                expanded = expanded.assign(**{node.var: expanded[list_col]})
                pred_ok, pred_value = self._gfql_eval_expr_ast(expanded, node.predicate)
                if not pred_ok:
                    return False, None
                if not hasattr(pred_value, "astype"):
                    pred_value = self._gfql_broadcast_scalar(expanded, pred_value)
                pred_mask = self._gfql_bool_mask(expanded, pred_value)
                pred_null = self._gfql_null_mask(expanded, pred_value)
                stats = expanded.assign(
                    __gfql_q_true_ast__=(pred_mask & ~pred_null).astype("int64"),
                    __gfql_q_null_ast__=pred_null.astype("int64"),
                )
                grouped = stats.groupby(row_col, sort=False).agg(
                    true_count=("__gfql_q_true_ast__", "sum"),
                    null_count=("__gfql_q_null_ast__", "sum"),
                )
                grouped = grouped.reset_index()
            else:
                grouped = non_null[[row_col]].iloc[0:0].copy()
                grouped["true_count"] = []
                grouped["null_count"] = []

            summary = base[[row_col, total_col]].merge(grouped, on=row_col, how="left", sort=False)
            summary = self._gfql_restore_row_order(summary, row_col)
            summary = summary.assign(
                true_count=summary["true_count"].fillna(0),
                null_count=summary["null_count"].fillna(0),
            )
            total = summary[total_col]
            true_count = summary["true_count"]
            null_count = summary["null_count"]
            false_count = total - true_count - null_count

            fn = str(node.fn).lower()
            if fn == "any":
                out = true_count > 0
                unknown = (true_count == 0) & (null_count > 0)
            elif fn == "all":
                out = false_count == 0
                unknown = (false_count == 0) & (null_count > 0)
            elif fn == "none":
                out = true_count == 0
                unknown = (true_count == 0) & (null_count > 0)
            elif fn == "single":
                out = true_count == 1
                unknown = (true_count <= 1) & (null_count > 0)
            else:
                return False, None
            out = out.where(~unknown, pd.NA)
            out = out.where(~list_null_mask, pd.NA)
            return True, out.reset_index(drop=True)

        if isinstance(node, ListComprehension):
            source_ok, list_value = self._gfql_eval_expr_ast(table_df, node.source)
            if not source_ok:
                return False, None
            list_series = (
                list_value if hasattr(list_value, "astype") else self._gfql_broadcast_scalar(table_df, list_value)
            )

            row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_lc_row_ast__")
            list_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_lc_list_ast__")
            len_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_lc_len_ast__")
            out_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_lc_out_ast__")

            base = table_df.reset_index(drop=True).copy()
            base = base.assign(**{row_col: range(len(base)), list_col: list_series})
            null_mask = self._gfql_null_mask(base, base[list_col])
            try:
                lengths = series_sequence_len(base[list_col])
            except Exception:
                lengths = self._gfql_broadcast_scalar(base, pd.NA)
            if hasattr(lengths, "fillna"):
                lengths = lengths.fillna(0)
            base = base.assign(**{len_col: lengths})

            non_null = base.loc[~null_mask].copy()
            expanded = non_null.explode(list_col)
            if len(expanded) > 0:
                expanded = expanded.reset_index(drop=True)
                expanded = expanded.assign(
                    **{
                        "__gfql_lc_pos_ast__": expanded.groupby(row_col, sort=False).cumcount(),
                        node.var: expanded[list_col],
                    }
                )
                expanded = expanded.loc[expanded["__gfql_lc_pos_ast__"] < expanded[len_col]]

                if node.predicate is not None:
                    pred_ok, pred_value = self._gfql_eval_expr_ast(expanded, node.predicate)
                    if not pred_ok:
                        return False, None
                    if not hasattr(pred_value, "astype"):
                        pred_value = self._gfql_broadcast_scalar(expanded, pred_value)
                    pred_mask = self._gfql_bool_mask(expanded, pred_value)
                    expanded = expanded.loc[pred_mask]

                proj_node = node.projection if node.projection is not None else Identifier(node.var)
                proj_ok, projected = self._gfql_eval_expr_ast(expanded, proj_node)
                if not proj_ok:
                    return False, None
                if not hasattr(projected, "astype"):
                    projected = self._gfql_broadcast_scalar(expanded, projected)
                expanded = expanded.assign(**{out_col: projected})
                grouped = expanded.groupby(row_col, sort=False)[out_col].agg(list).reset_index()
            else:
                grouped = non_null[[row_col]].iloc[0:0].copy()
                grouped[out_col] = []

            merged = base[[row_col, len_col]].merge(grouped, on=row_col, how="left", sort=False)
            merged = self._gfql_restore_row_order(merged, row_col)
            result = (
                merged[out_col].copy()
                if out_col in merged.columns
                else self._gfql_broadcast_scalar(merged, None)
            )

            empty_mask = merged[len_col] == 0
            if hasattr(empty_mask, "any") and bool(empty_mask.any()):
                result = self._gfql_mask_fill_sequence_value(merged, result, empty_mask, [])

            missing_mask = result.isna() if hasattr(result, "isna") else None
            if missing_mask is not None:
                filtered_empty_mask = missing_mask & (~null_mask)
                if hasattr(filtered_empty_mask, "any") and bool(filtered_empty_mask.any()):
                    result = self._gfql_mask_fill_sequence_value(merged, result, filtered_empty_mask, [])

            if hasattr(null_mask, "any") and bool(null_mask.any()):
                result = self._gfql_mask_fill_sequence_value(merged, result, null_mask, None)

            return True, result.reset_index(drop=True)

        if isinstance(node, SubscriptExpr):
            ok_base, base_value = self._gfql_eval_expr_ast(table_df, node.value)
            ok_key, key_value = self._gfql_eval_expr_ast(table_df, node.key)
            if not (ok_base and ok_key):
                return False, None
            if is_null_scalar(base_value) or is_null_scalar(key_value):
                return True, None
            if hasattr(key_value, "iloc"):
                key_constant, key_scalar = RowPipelineMixin._gfql_series_scalar_if_constant(key_value)
                if key_constant:
                    key_value = key_scalar
            if isinstance(key_value, str):
                if (
                    isinstance(node.value, Identifier)
                    and node.value.name in table_df.columns
                    and RowPipelineMixin._gfql_series_bool_like(table_df[node.value.name])
                    and RowPipelineMixin._gfql_table_has_graph_shape(table_df)
                ):
                    entity_value = self._gfql_format_entity_series(
                        table_df,
                        alias_col=node.value.name,
                        table=RowPipelineMixin._gfql_infer_graph_table_kind(table_df),
                    )
                    return True, self._gfql_eval_property_access_value(
                        table_df,
                        entity_value,
                        key_value,
                        "ast subscript",
                    )
                if hasattr(base_value, "astype") and (
                    RowPipelineMixin._gfql_series_is_mapping_like(base_value)
                    or RowPipelineMixin._gfql_series_is_entity_text_like(base_value)
                ):
                    return True, self._gfql_eval_property_access_value(
                        table_df,
                        base_value,
                        key_value,
                        "ast subscript",
                    )
                if isinstance(base_value, Mapping) or is_entity_text_scalar(base_value):
                    return True, self._gfql_eval_property_access_value(
                        table_df,
                        base_value,
                        key_value,
                        "ast subscript",
                    )
            if isinstance(base_value, dict):
                return True, base_value.get(str(key_value))
            if (
                isinstance(key_value, int)
                and not isinstance(key_value, bool)
                and hasattr(base_value, "iloc")
            ):
                return True, self._gfql_eval_dynamic_list_subscript(
                    table_df,
                    base_value,
                    self._gfql_broadcast_scalar(table_df, key_value),
                    "ast subscript",
                )
            if hasattr(key_value, "iloc"):
                return True, self._gfql_eval_dynamic_list_subscript(
                    table_df, base_value, key_value, "ast subscript"
                )
            if hasattr(base_value, "str"):
                return True, base_value.str.get(key_value)
            try:
                return True, base_value[key_value]
            except Exception:
                return False, None

        if isinstance(node, SliceExpr):
            ok_base, base_value = self._gfql_eval_expr_ast(table_df, node.value)
            if not ok_base:
                return False, None
            start_present = node.start is not None
            end_present = node.stop is not None
            start_value = None
            end_value = None
            if node.start is not None:
                ok_start, start_value = self._gfql_eval_expr_ast(table_df, node.start)
                if not ok_start:
                    return False, None
            if node.stop is not None:
                ok_end, end_value = self._gfql_eval_expr_ast(table_df, node.stop)
                if not ok_end:
                    return False, None
            return True, self._gfql_eval_slice_subscript(
                table_df,
                base_value,
                start_value,
                end_value,
                start_present,
                end_present,
                "ast slice",
            )

        return False, None
    @staticmethod
    def _gfql_order_expr_static_supported(expr: str) -> bool:
        txt = expr.strip()
        if txt == "":
            return False

        parser_bundle = _gfql_expr_runtime_parser_bundle()
        if parser_bundle is None:
            return False
        parser, capability_checker, _expr_parser_mod = parser_bundle
        try:
            node = parser(txt)
        except Exception:
            return False
        if is_order_aggregate_alias_ast(node):
            return True
        try:
            capability_errors = capability_checker(node)
        except Exception:
            return False
        if len(capability_errors) > 0:
            return False
        return order_expr_ast_static_supported(node)

    @staticmethod
    def _gfql_all_non_null_match(mask: Any, non_null: Any) -> bool:
        return bool(hasattr(mask, "where") and mask.where(non_null, True).all())

    @staticmethod
    def _gfql_series_to_pylist(values: Any) -> List[Any]:
        if hasattr(values, "to_arrow"):
            try:
                return list(values.to_arrow().to_pylist())
            except Exception:
                pass
        if hasattr(values, "to_pandas"):
            try:
                return list(values.to_pandas().tolist())
            except Exception:
                pass
        if hasattr(values, "tolist"):
            try:
                return list(values.tolist())
            except Exception:
                pass
        return list(values)

    def _gfql_mask_fill_sequence_value(self, table_df: Any, series: Any, mask: Any, value: Any) -> Any:
        out = series.copy()
        has_mask = True
        if hasattr(mask, "any"):
            try:
                has_mask = bool(mask.any())
            except Exception:
                has_mask = True
        if not has_mask:
            return out

        try:
            out.loc[out.index[mask]] = value
            return out
        except Exception:
            mask_values = [
                False if is_null_scalar(flagged) else bool(flagged)
                for flagged in self._gfql_series_to_pylist(mask)
            ]
            filled_values = self._gfql_series_to_pylist(out)
            for idx, flagged in enumerate(mask_values):
                if flagged:
                    filled_values[idx] = list(value) if isinstance(value, list) else value
            fill_col = getattr(out, "name", None)
            if not isinstance(fill_col, str) or fill_col in table_df.columns:
                fill_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_seq_fill__")
            return table_df.reset_index(drop=True).assign(**{fill_col: filled_values})[fill_col]

    @staticmethod
    def _gfql_restore_row_order(table_df: Any, row_col: str) -> Any:
        if row_col not in table_df.columns:
            return table_df
        try:
            return table_df.sort_values(by=[row_col], kind="mergesort").reset_index(drop=True)
        except TypeError:
            return table_df.sort_values(by=[row_col]).reset_index(drop=True)

    @staticmethod
    def _gfql_normalize_zero_offset_suffix(timezone: Any) -> Any:
        if not hasattr(timezone, "where") or not hasattr(timezone, "isin"):
            return timezone
        zero_offset = timezone.isin(["+00:00", "-00:00"])
        return timezone.where(~zero_offset, "Z")

    def _gfql_entity_temporal_text(self, table_df: Any, series: Any, text: Any) -> Any:
        if not hasattr(text, "str"):
            return None
        stripped = text.str.strip()
        non_null = ~self._gfql_null_mask(table_df, series)

        def _quote(values: Any) -> Any:
            return "'" + values + "'"

        date_mask = series_str_match(stripped, DATE_CALL_TEXT_RE.pattern, na=False)
        if self._gfql_all_non_null_match(date_mask, non_null):
            parts = stripped.str.extract(DATE_CALL_TEXT_RE.pattern)
            year = parts["year"].fillna("0").astype("int64").astype(str).str.zfill(4)
            month = parts["month"].fillna("0").astype("int64").astype(str).str.zfill(2)
            day = parts["day"].fillna("0").astype("int64").astype(str).str.zfill(2)
            return _quote(year + "-" + month + "-" + day)

        localtime_mask = series_str_match(stripped, LOCALTIME_CALL_TEXT_RE.pattern, na=False)
        if self._gfql_all_non_null_match(localtime_mask, non_null):
            parts = stripped.str.extract(LOCALTIME_CALL_TEXT_RE.pattern)
            hour = parts["hour"].fillna("0").astype("int64").astype(str).str.zfill(2)
            minute = parts["minute"].fillna("0").astype("int64").astype(str).str.zfill(2)
            second = parts["second"].fillna("")
            nanos = parts["nano"].fillna("").str.zfill(9).str.rstrip("0")
            base = hour + ":" + minute
            has_seconds = (second != "") | (nanos != "")
            second_text = ":" + second.where(second != "", "00").astype(str).str.zfill(2)
            frac = "." + nanos
            base = base + second_text.where(has_seconds, "")
            base = base + frac.where(nanos != "", "")
            return _quote(base)

        time_mask = series_str_match(stripped, TIME_CALL_TEXT_RE.pattern, na=False)
        if self._gfql_all_non_null_match(time_mask, non_null):
            parts = stripped.str.extract(TIME_CALL_TEXT_RE.pattern)
            hour = parts["hour"].fillna("0").astype("int64").astype(str).str.zfill(2)
            minute = parts["minute"].fillna("0").astype("int64").astype(str).str.zfill(2)
            second = parts["second"].fillna("")
            nanos = parts["nano"].fillna("").str.zfill(9).str.rstrip("0")
            timezone = self._gfql_normalize_zero_offset_suffix(parts["tz"].fillna(""))
            base = hour + ":" + minute
            has_seconds = (second != "") | (nanos != "")
            second_text = ":" + second.where(second != "", "00").astype(str).str.zfill(2)
            frac = "." + nanos
            base = base + second_text.where(has_seconds, "")
            base = base + frac.where(nanos != "", "")
            base = base + timezone
            return _quote(base)

        localdatetime_mask = series_str_match(stripped, LOCALDATETIME_CALL_TEXT_RE.pattern, na=False)
        if self._gfql_all_non_null_match(localdatetime_mask, non_null):
            parts = stripped.str.extract(LOCALDATETIME_CALL_TEXT_RE.pattern)
            year = parts["year"].fillna("0").astype("int64").astype(str).str.zfill(4)
            month = parts["month"].fillna("0").astype("int64").astype(str).str.zfill(2)
            day = parts["day"].fillna("0").astype("int64").astype(str).str.zfill(2)
            hour = parts["hour"].fillna("0").astype("int64").astype(str).str.zfill(2)
            minute = parts["minute"].fillna("0").astype("int64").astype(str).str.zfill(2)
            second = parts["second"].fillna("")
            nanos = parts["nano"].fillna("").str.zfill(9).str.rstrip("0")
            base = year + "-" + month + "-" + day + "T" + hour + ":" + minute
            has_seconds = (second != "") | (nanos != "")
            second_text = ":" + second.where(second != "", "00").astype(str).str.zfill(2)
            frac = "." + nanos
            base = base + second_text.where(has_seconds, "")
            base = base + frac.where(nanos != "", "")
            return _quote(base)

        datetime_mask = series_str_match(stripped, DATETIME_CALL_TEXT_RE.pattern, na=False)
        if self._gfql_all_non_null_match(datetime_mask, non_null):
            parts = stripped.str.extract(DATETIME_CALL_TEXT_RE.pattern)
            year = parts["year"].fillna("0").astype("int64").astype(str).str.zfill(4)
            month = parts["month"].fillna("0").astype("int64").astype(str).str.zfill(2)
            day = parts["day"].fillna("0").astype("int64").astype(str).str.zfill(2)
            hour = parts["hour"].fillna("0").astype("int64").astype(str).str.zfill(2)
            minute = parts["minute"].fillna("0").astype("int64").astype(str).str.zfill(2)
            second = parts["second"].fillna("")
            nanos = parts["nano"].fillna("").str.zfill(9).str.rstrip("0")
            timezone = self._gfql_normalize_zero_offset_suffix(parts["tz"].fillna(""))
            base = year + "-" + month + "-" + day + "T" + hour + ":" + minute
            has_seconds = (second != "") | (nanos != "")
            second_text = ":" + second.where(second != "", "00").astype(str).str.zfill(2)
            frac = "." + nanos
            base = base + second_text.where(has_seconds, "")
            base = base + frac.where(nanos != "", "")
            base = base + timezone
            return _quote(base)

        return None

    def _gfql_entity_scalar_text(self, table_df: Any, alias_col: str, series: Any) -> Any:
        text = series.astype(str)
        dtype_txt = str(getattr(series, "dtype", "")).lower()
        if "bool" in dtype_txt and hasattr(text, "str"):
            return text.str.lower()
        if "float" in dtype_txt and hasattr(text, "str"):
            return text.str.replace(r"\.0+$", "", regex=True)
        if any(token in dtype_txt for token in ("int", "double", "decimal")):
            return text
        if hasattr(text, "str"):
            stripped = text.str.strip()
            non_null = ~self._gfql_null_mask(table_df, series)
            list_like = series_str_match(stripped, r"^\[.*\]$", na=False)
            if self._gfql_all_non_null_match(list_like, non_null):
                return stripped
            map_like = series_str_match(stripped, r"^\{.*\}$", na=False)
            if self._gfql_all_non_null_match(map_like, non_null):
                return stripped
            temporal = self._gfql_entity_temporal_text(table_df, series, text)
            if temporal is not None:
                return temporal
            bool_like = series_str_match(text, r"^(True|False)$", na=False)
            num_like = series_str_match(text, r"^-?\d+(?:\.\d+)?$", na=False)
            if self._gfql_all_non_null_match(bool_like, non_null):
                return text.str.lower()
            if self._gfql_all_non_null_match(num_like, non_null):
                return text.str.replace(r"\.0+$", "", regex=True)
            escaped = text.str.replace("\\", "\\\\", regex=False).str.replace("'", "\\'", regex=False)
            out = "'" + escaped + "'"
            out = out.where(~bool_like, text.str.lower())
            out = out.where(~num_like, text.str.replace(r"\.0+$", "", regex=True))
            out = out.where(~list_like, stripped)
            out = out.where(~map_like, stripped)
            return out
        if hasattr(text, "str"):
            escaped = text.str.replace("\\", "\\\\", regex=False).str.replace("'", "\\'", regex=False)
            return "'" + escaped + "'"
        return "'" + text + "'"

    def _gfql_format_entity_series(self, table_df: Any, *, alias_col: str, table: str, excluded: Sequence[str] = ()) -> Any:
        blank = table_df[alias_col].astype(str).where(table_df[alias_col].isna(), "")
        excluded_cols = tuple(dict.fromkeys((alias_col,) + tuple(str(name) for name in excluded)))
        if table == "nodes":
            labels = blank.copy()
            if "labels" in table_df.columns and RowPipelineMixin._gfql_series_is_list_like(table_df["labels"]):
                labels_raw = table_df["labels"].astype(str)
                labels_body = labels_raw.str.replace("[", "", regex=False).str.replace("]", "", regex=False)
                labels_body = labels_body.str.replace("'", "", regex=False).str.replace(", ", ":", regex=False)
                has_list_labels = labels_raw != "[]"
                labels = labels + (blank + ":" + labels_body).where(has_list_labels, "")
            else:
                label_cols = [
                    col
                    for col in table_df.columns
                    if str(col).startswith("label__")
                    and str(col).split("label__", 1)[1] not in {"<NA>", "None", "nan"}
                ]
                for col in label_cols:
                    mask = table_df[col] == True  # noqa: E712
                    labels = labels + (blank + ":" + str(col).split("label__", 1)[1]).where(mask, "")
            if "type" in table_df.columns:
                include = ~self._gfql_null_mask(table_df, table_df["type"])
                labels = labels + (blank + ":" + table_df["type"].astype(str)).where(include, "")
            prop_cols = node_property_columns(table_df, alias_col, excluded_cols)
            left_bracket, right_bracket = "(", ")"
        else:
            if "type" in table_df.columns:
                include = ~self._gfql_null_mask(table_df, table_df["type"])
                labels = (blank + ":" + table_df["type"].astype(str)).where(include, "")
            else:
                labels = blank.copy()
            prop_cols = edge_property_columns(table_df, alias_col, excluded_cols)
            left_bracket, right_bracket = "[", "]"

        prop_text = blank.copy()
        has_props = (table_df[alias_col] == True) & False  # noqa: E712
        for col in prop_cols:
            series = table_df[col]
            include = ~self._gfql_null_mask(table_df, series)
            value_text = self._gfql_entity_scalar_text(table_df, alias_col, series)
            segment = f"{col}: " + value_text
            prefix = (blank + ", ").where(has_props & include, "")
            append = (prefix + segment).where(include, "")
            prop_text = prop_text + append
            has_props = has_props | include

        prop_block = ((blank + " {") + prop_text + "}").where(has_props & (labels != ""), "")
        prop_only = ((blank + "{") + prop_text + "}").where(has_props & (labels == ""), "")
        return left_bracket + labels + prop_block + prop_only + right_bracket

    def _gfql_format_labels_series(self, table_df: Any, *, alias_col: str) -> Any:
        blank = table_df[alias_col].astype(str).where(table_df[alias_col].isna(), "")
        if "labels" in table_df.columns and RowPipelineMixin._gfql_series_is_list_like(table_df["labels"]):
            labels_raw = table_df["labels"].astype(str)
            null_mask = self._gfql_null_mask(table_df, table_df[alias_col])
            if hasattr(labels_raw, "where"):
                return self._gfql_mask_fill(labels_raw, null_mask, None)
            return labels_raw
        labels_text = blank.copy()
        has_labels = (table_df[alias_col] == True) & False  # noqa: E712

        label_cols = [
            col
            for col in table_df.columns
            if str(col).startswith("label__")
            and str(col).split("label__", 1)[1] not in {"<NA>", "None", "nan"}
        ]
        for col in label_cols:
            label_name = str(col).split("label__", 1)[1]
            include = table_df[col] == True  # noqa: E712
            segment = (blank + f"'{label_name}'").where(include, "")
            prefix = (blank + ", ").where(has_labels & include, "")
            labels_text = labels_text + prefix + segment
            has_labels = has_labels | include

        if "type" in table_df.columns:
            include = ~self._gfql_null_mask(table_df, table_df["type"])
            segment = (blank + "'" + table_df["type"].astype(str) + "'").where(include, "")
            prefix = (blank + ", ").where(has_labels & include, "")
            labels_text = labels_text + prefix + segment
            has_labels = has_labels | include

        return ((blank + "[") + labels_text + "]").where(has_labels, "[]")

    @staticmethod
    def _gfql_series_is_entity_text_like(series: Any) -> bool:
        if not hasattr(series, "isna") or not hasattr(series, "astype"):
            return is_entity_text_scalar(series)
        null_mask = series.isna()
        non_null = ~null_mask
        if hasattr(non_null, "any") and not bool(non_null.any()):
            return False
        text = series.astype(str)
        if not hasattr(text, "str"):
            return False
        try:
            actual_string = (series == text).where(non_null, False)
        except Exception:
            actual_string = non_null & False
        entity_like = series_str_match(
            text.str.strip(),
            r"^(?:\((?::[A-Za-z_][A-Za-z0-9_]*)*(?:\s*\{.*\})?\)|\[:[^\]\s]+(?:\s+\{.*\})?\]|\(\)|\[\])$",
            na=False,
        )
        return bool(entity_like.where(~null_mask, True).where(actual_string, False).all())

    def _gfql_eval_entity_graph_fn(
        self,
        table_df: Any,
        value: Any,
        fn: str,
        expr: str,
    ) -> Any:
        if hasattr(value, "astype"):
            null_mask = self._gfql_null_mask(table_df, value)
            if RowPipelineMixin._gfql_series_is_entity_text_like(value):
                if fn == "labels":
                    out = entity_labels_series(value)
                elif fn == "type":
                    out = entity_type_series(value)
                elif fn == "properties":
                    out = entity_properties_series(value)
                else:
                    raise ValueError(f"unsupported row expression graph function: {fn} in {expr!r}")
                if hasattr(out, "where"):
                    out = self._gfql_mask_fill(out, null_mask, None)
                return out
            if hasattr(null_mask, "all") and bool(null_mask.all()):
                out = self._gfql_broadcast_scalar(table_df, None)
                if hasattr(out, "where"):
                    out = self._gfql_mask_fill(out, null_mask, None)
                return out
            raise ValueError(
                f"unsupported row expression: {fn}() requires a graph element, entity value, or null in {expr!r}"
            )

        if is_null_scalar(value):
            return None
        if not is_entity_text_scalar(value):
            raise ValueError(
                f"unsupported row expression: {fn}() requires a graph element, entity value, or null in {expr!r}"
            )
        if fn == "labels":
            return entity_labels_scalar(value)
        if fn == "type":
            return entity_type_scalar(value)
        if fn == "properties":
            return entity_properties_scalar(value)
        raise ValueError(f"unsupported row expression graph function: {fn} in {expr!r}")

    def _gfql_eval_property_access_value(
        self,
        table_df: Any,
        value: Any,
        prop: str,
        expr: str,
    ) -> Any:
        def _coerce_duration_property(value_text: str) -> Any:
            resolved = resolve_duration_text_property(value_text, prop)
            if resolved is None:
                return None
            if re.fullmatch(r"-?\d+", resolved):
                return int(resolved)
            return resolved

        if hasattr(value, "astype"):
            null_mask = self._gfql_null_mask(table_df, value)
            scalar_ok, scalar_value = RowPipelineMixin._gfql_series_scalar_if_constant(value)
            if scalar_ok and isinstance(scalar_value, str):
                duration_value = _coerce_duration_property(scalar_value)
                if duration_value is not None:
                    out = self._gfql_broadcast_scalar(table_df, duration_value)
                    if hasattr(out, "where"):
                        out = self._gfql_mask_fill(out, null_mask, None)
                    return out
            if RowPipelineMixin._gfql_series_is_mapping_like(value):
                out = value.str.get(prop)
                if hasattr(out, "where"):
                    out = self._gfql_mask_fill(out, null_mask, None)
                return out
            if RowPipelineMixin._gfql_series_is_entity_text_like(value):
                out = entity_property_series(value, prop)
                if hasattr(out, "where"):
                    fill_mask = null_mask.to_pandas() if isinstance(out, pd.Series) and hasattr(null_mask, "to_pandas") else null_mask
                    out = self._gfql_mask_fill(out, fill_mask, None)
                return out
            if hasattr(null_mask, "all") and bool(null_mask.all()):
                out = self._gfql_broadcast_scalar(table_df, None)
                if hasattr(out, "where"):
                    out = self._gfql_mask_fill(out, null_mask, None)
                return out
            raise ValueError(
                f"unsupported row expression: property access requires a graph element alias, entity value, or map in {expr!r}"
            )

        if is_null_scalar(value):
            return None
        if isinstance(value, str):
            duration_value = _coerce_duration_property(value)
            if duration_value is not None:
                return duration_value
        if isinstance(value, Mapping):
            return value.get(prop)
        if is_entity_text_scalar(value):
            return entity_property_scalar(value, prop)
        raise ValueError(
            f"unsupported row expression: property access requires a graph element alias, entity value, or map in {expr!r}"
        )

    @staticmethod
    def _gfql_series_is_list_like(series: Any) -> bool:
        if not hasattr(series, "isna") or not hasattr(series, "astype"):
            return False
        null_mask = series.isna()
        non_null = ~null_mask
        if hasattr(non_null, "any") and not bool(non_null.any()):
            return False
        sample = series[non_null].head(16) if hasattr(series, "head") else series
        sample_values = [value for value in RowPipelineMixin._gfql_series_to_pylist(sample) if not is_null_scalar(value)]
        if sample_values and all(isinstance(value, (list, tuple)) for value in sample_values):
            return True
        text = series.astype(str)
        if not hasattr(text, "str"):
            return False
        try:
            actual_string = (series == text).where(non_null, False)
        except Exception:
            actual_string = non_null & False
        list_like = series_str_match(text.str.strip(), r"^(?:\[.*\]|\(.*\))$", na=False)
        return bool(list_like.where(~null_mask, True).where(~actual_string, False).all())

    @staticmethod
    def _gfql_series_is_mapping_like(series: Any) -> bool:
        if not hasattr(series, "isna") or not hasattr(series, "astype"):
            return False
        null_mask = series.isna()
        non_null = ~null_mask
        if hasattr(non_null, "any") and not bool(non_null.any()):
            return False
        text = series.astype(str)
        if not hasattr(text, "str"):
            return False
        try:
            actual_string = (series == text).where(non_null, False)
        except Exception:
            actual_string = non_null & False
        mapping_like = series_str_match(text.str.strip(), r"^\{.*\}$", na=False)
        return bool(mapping_like.where(~null_mask, True).where(~actual_string, False).all())

    @staticmethod
    def _gfql_series_scalar_if_constant(series: Any) -> Tuple[bool, Any]:
        if not hasattr(series, "dropna"):
            return True, series
        non_null = series.dropna()
        if len(non_null) == 0:
            return True, None
        try:
            if hasattr(non_null, "nunique") and int(non_null.nunique(dropna=True)) == 1:
                if hasattr(non_null, "iloc"):
                    return True, non_null.iloc[0]
                values = list(non_null)
                return True, values[0] if len(values) > 0 else None
        except Exception:
            return False, None
        return False, None

    @staticmethod
    def _gfql_series_bool_like(series: Any) -> bool:
        dtype_txt = str(getattr(series, "dtype", "")).lower()
        if dtype_txt in {"bool", "boolean"}:
            return True
        if not hasattr(series, "dropna"):
            return isinstance(series, bool)
        sample = series.dropna().head(128)
        if hasattr(sample, "to_pandas"):
            sample = sample.to_pandas()
        values = sample.tolist() if hasattr(sample, "tolist") else list(sample)
        return len(values) > 0 and all(isinstance(v, bool) for v in values)

    @staticmethod
    def _gfql_scalar_numeric_non_bool(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _gfql_series_int_non_bool_like(series: Any) -> bool:
        dtype_txt = str(getattr(series, "dtype", "")).lower()
        if dtype_txt in {"bool", "boolean"}:
            return False
        if any(token in dtype_txt for token in ("int", "long", "short", "uint")):
            return True
        if any(token in dtype_txt for token in ("float", "double", "decimal")):
            return False
        if not hasattr(series, "isna") or not hasattr(series, "astype"):
            return isinstance(series, int) and not isinstance(series, bool)
        null_mask = series.isna()
        non_null = ~null_mask
        if hasattr(non_null, "any") and not bool(non_null.any()):
            return True
        text = series.astype(str)
        if not hasattr(text, "str"):
            return False
        try:
            actual_string = (series == text).where(non_null, False)
        except Exception:
            actual_string = non_null & False
        int_like = series_str_match(text, r"^[+-]?\d+$", na=False)
        return bool(int_like.where(~null_mask, True).where(~actual_string, False).all())

    @staticmethod
    def _gfql_series_numeric_non_bool_like(series: Any) -> bool:
        dtype_txt = str(getattr(series, "dtype", "")).lower()
        if dtype_txt in {"bool", "boolean"}:
            return False
        if any(token in dtype_txt for token in ("int", "float", "double", "decimal")):
            return True
        if not hasattr(series, "dropna"):
            return RowPipelineMixin._gfql_scalar_numeric_non_bool(series)
        sample = series.dropna().head(128)
        if hasattr(sample, "to_pandas"):
            sample = sample.to_pandas()
        values = sample.tolist() if hasattr(sample, "tolist") else list(sample)
        return len(values) > 0 and all(RowPipelineMixin._gfql_scalar_numeric_non_bool(v) for v in values)

    @staticmethod
    def _gfql_format_cypher_scalar_literal(value: Any) -> str:
        if value is None or is_null_scalar(value):
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
        return str(value)

    @staticmethod
    def _gfql_format_cypher_map_scalar(value: Mapping[str, Any]) -> str:
        parts = [
            f"{key}: {RowPipelineMixin._gfql_format_cypher_scalar_literal(item)}"
            for key, item in value.items()
        ]
        return "{" + ", ".join(parts) + "}"

    def _gfql_eval_string_predicate_expr(
        self,
        table_df: Any,
        left: Any,
        right: Any,
        op_name: str,
        expr: str,
    ) -> Any:
        if hasattr(left, "astype") or hasattr(right, "astype"):
            left_series = left if hasattr(left, "astype") else self._gfql_broadcast_scalar(table_df, left)
            if hasattr(right, "astype"):
                raise ValueError(
                    f"unsupported row expression: {op_name} rhs must be scalar in vectorized mode for {expr!r}"
                )

            left_null = self._gfql_null_mask(table_df, left_series)
            right_null = self._gfql_null_mask(table_df, right)
            any_null = left_null | right_null
            left_txt = left_series.astype(str)
            needle = str(right)

            try:
                out = apply_string_predicate_series(left_txt, needle, op_name)
            except ValueError:
                raise ValueError(f"unsupported row expression predicate op: {op_name} in {expr!r}")
            except Exception as exc:
                raise ValueError(
                    f"unsupported row expression predicate op: {op_name} in {expr!r}"
                ) from exc

            if hasattr(out, "where"):
                return out.where(~any_null, pd.NA)
            return out

        if is_null_scalar(left) or is_null_scalar(right):
            return None
        left_txt = str(left)
        right_txt = str(right)
        try:
            return apply_string_predicate_scalar(left_txt, right_txt, op_name)
        except ValueError as exc:
            raise ValueError(f"unsupported row expression predicate op: {op_name} in {expr!r}") from exc

    def _gfql_broadcast_scalar(self, table_df: Any, value: Any) -> Any:
        tmp_col = "__gfql_tmp_scalar__"
        while tmp_col in table_df.columns:
            tmp_col = f"{tmp_col}_x"

        # Treat list/map literals as scalar row values by explicit broadcasting.
        # Plain `assign(col=[...])` interprets list values as column vectors.
        if isinstance(value, (list, tuple, dict)):
            repeated = [value for _ in range(len(table_df))]
            try:
                return table_df.assign(**{tmp_col: repeated})[tmp_col]
            except Exception:
                if resolve_engine(EngineAbstract.AUTO, table_df) == Engine.CUDF:
                    return pd.Series(repeated, name=tmp_col)
                raise

        return table_df.assign(**{tmp_col: value})[tmp_col]

    def _gfql_truth_masks(self, table_df: Any, value: Any) -> Optional[Tuple[Any, Any, Any]]:
        if not hasattr(value, "astype"):
            if is_null_scalar(value):
                series = self._gfql_broadcast_scalar(table_df, pd.NA)
            elif isinstance(value, bool):
                series = self._gfql_broadcast_scalar(table_df, value)
            else:
                return None
        else:
            series = value
        if not hasattr(series, "isin") or not hasattr(series, "where"):
            return None
        null_mask = self._gfql_null_mask(table_df, series)
        dtype_name = str(getattr(series, "dtype", "")).lower()
        if any(token in dtype_name for token in ("int", "float", "double", "long", "short", "uint")):
            return None
        valid_mask = series.isin([True, False]) | null_mask
        if hasattr(valid_mask, "all") and not bool(valid_mask.all()):
            return None
        true_mask = series.isin([True]) & ~null_mask
        false_mask = series.isin([False]) & ~null_mask
        return true_mask, false_mask, null_mask.astype(bool)

    def _gfql_series_from_truth_masks(
        self,
        table_df: Any,
        true_mask: Any,
        false_mask: Any,
        null_mask: Any,
    ) -> Any:
        out = self._gfql_broadcast_scalar(table_df, False)
        if hasattr(out, "astype") and resolve_engine(EngineAbstract.AUTO, table_df) == Engine.PANDAS:
            out = out.astype("object")
        if hasattr(out, "where"):
            out = out.where(~true_mask, True)
            out = out.where(~false_mask, False)
            out = out.where(~null_mask, pd.NA)
            return out
        return out

    def _gfql_boolean_short_circuit_result(self, table_df: Any, value: Any, op: str) -> Optional[Any]:
        if op not in {"or", "and"}:
            return None
        masks = self._gfql_truth_masks(table_df, value)
        if masks is None:
            return None
        true_mask, false_mask, null_mask = masks
        if op == "or":
            if hasattr(false_mask, "any") and bool(false_mask.any()):
                return None
            if hasattr(null_mask, "any") and bool(null_mask.any()):
                return None
            return self._gfql_series_from_truth_masks(table_df, true_mask, false_mask, null_mask)
        if hasattr(true_mask, "any") and bool(true_mask.any()):
            return None
        if hasattr(null_mask, "any") and bool(null_mask.any()):
            return None
        return self._gfql_series_from_truth_masks(table_df, true_mask, false_mask, null_mask)

    def _gfql_eval_boolean_op(self, table_df: Any, left: Any, right: Any, op: str) -> Optional[Any]:
        if op == "not":
            masks = self._gfql_truth_masks(table_df, left)
            if masks is None:
                return None
            true_mask, false_mask, null_mask = masks
            return self._gfql_series_from_truth_masks(table_df, false_mask, true_mask, null_mask)

        if right is _GFQL_MISSING_BOOL_OPERAND:
            return None

        left_masks = self._gfql_truth_masks(table_df, left)
        right_masks = self._gfql_truth_masks(table_df, right)
        if left_masks is None or right_masks is None:
            return None
        left_true, left_false, _left_null = left_masks
        right_true, right_false, _right_null = right_masks

        if op == "and":
            true_mask = left_true & right_true
            false_mask = left_false | right_false
        elif op == "or":
            true_mask = left_true | right_true
            false_mask = left_false & right_false
        elif op == "xor":
            true_mask = (left_true & right_false) | (left_false & right_true)
            false_mask = (left_true & right_true) | (left_false & right_false)
        else:
            return None

        null_mask = ~(true_mask | false_mask)
        return self._gfql_series_from_truth_masks(table_df, true_mask, false_mask, null_mask)

    def _gfql_bool_mask(self, table_df: Any, value: Any) -> Any:
        if hasattr(value, "astype"):
            mask = value
            # Avoid pandas object-dtype fillna() downcast FutureWarning while
            # keeping NA -> False semantics in a vectorized backend-agnostic way.
            if hasattr(mask, "isna") and hasattr(mask, "where"):
                mask = mask.where(~mask.isna(), False)
            elif hasattr(mask, "fillna"):
                mask = mask.fillna(False)
            return mask.astype(bool)
        return self._gfql_broadcast_scalar(table_df, bool(value)).astype(bool)

    def _gfql_null_mask(self, table_df: Any, value: Any) -> Any:
        if hasattr(value, "isna"):
            return value.isna()
        try:
            marker = pd.isna(value)
        except Exception:
            marker = is_null_scalar(value)
        if isinstance(marker, bool):
            return self._gfql_broadcast_scalar(table_df, marker).astype(bool)
        return self._gfql_broadcast_scalar(
            table_df,
            is_null_scalar(value),
        ).astype(bool)

    def _gfql_range_arg_series(
        self,
        table_df: Any,
        value: Any,
        *,
        label: str,
        expr: str,
    ) -> Any:
        series = value if hasattr(value, "astype") else self._gfql_broadcast_scalar(table_df, value)
        null_mask = self._gfql_null_mask(table_df, series)
        if hasattr(null_mask, "any") and bool(null_mask.any()):
            raise ValueError(f"unsupported row expression: range() {label} must be an integer in {expr!r}")

        dtype_txt = str(getattr(series, "dtype", "")).lower()
        if dtype_txt in {"bool", "boolean"}:
            raise ValueError(f"unsupported row expression: range() {label} must be an integer in {expr!r}")
        if any(token in dtype_txt for token in ("int", "long", "short", "uint")):
            return series.astype("int64") if hasattr(series, "astype") else series
        if any(token in dtype_txt for token in ("float", "double", "decimal")):
            raise ValueError(f"unsupported row expression: range() {label} must be an integer in {expr!r}")

        if not hasattr(series, "dropna"):
            if isinstance(series, int) and not isinstance(series, bool):
                return self._gfql_broadcast_scalar(table_df, series)
            raise ValueError(f"unsupported row expression: range() {label} must be an integer in {expr!r}")

        if not RowPipelineMixin._gfql_series_int_non_bool_like(series):
            raise ValueError(f"unsupported row expression: range() {label} must be an integer in {expr!r}")
        return series.astype("int64") if hasattr(series, "astype") else series

    def _gfql_eval_range_expr(self, table_df: Any, values: Sequence[Any], expr: str) -> Any:
        if len(values) not in {2, 3}:
            raise ValueError(f"unsupported row expression: range() expects 2 or 3 arguments in {expr!r}")

        start_series = self._gfql_range_arg_series(table_df, values[0], label="start", expr=expr)
        stop_series = self._gfql_range_arg_series(table_df, values[1], label="stop", expr=expr)
        step_series = (
            self._gfql_range_arg_series(table_df, values[2], label="step", expr=expr)
            if len(values) == 3
            else self._gfql_broadcast_scalar(table_df, 1).astype("int64")
        )

        zero_step_mask = step_series == 0
        if hasattr(zero_step_mask, "any") and bool(zero_step_mask.any()):
            raise ValueError(f"unsupported row expression: range() step must be non-zero in {expr!r}")

        row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_range_row__")
        start_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_range_start__")
        stop_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_range_stop__")
        step_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_range_step__")
        len_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_range_len__")
        pos_list_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_range_pos_list__")
        pos_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_range_pos__")
        out_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_range_out__")

        base = table_df.reset_index(drop=True).copy()
        base = base.assign(
            **{
                row_col: range(len(base)),
                start_col: start_series,
                stop_col: stop_series,
                step_col: step_series,
            }
        )

        positive_mask = base[step_col] > 0
        negative_mask = base[step_col] < 0
        keep_positive = positive_mask & (base[stop_col] >= base[start_col])
        keep_negative = negative_mask & (base[stop_col] <= base[start_col])

        zero_lengths = (base[start_col] * 0).astype("int64")
        positive_lengths = (((base[stop_col] - base[start_col]) // base[step_col]) + 1).where(keep_positive, 0)
        negative_lengths = (((base[start_col] - base[stop_col]) // (-base[step_col])) + 1).where(keep_negative, 0)
        lengths = zero_lengths + positive_lengths.astype("int64") + negative_lengths.astype("int64")
        base = base.assign(**{len_col: lengths})

        if len(base) == 0:
            return base[len_col]

        max_len = int(base[len_col].max()) if len(base) > 0 else 0
        result: Any = self._gfql_broadcast_scalar(base, [])
        if max_len <= 0:
            return result.reset_index(drop=True)

        non_empty = base.loc[base[len_col] > 0, [row_col, start_col, step_col, len_col]].copy()
        pos_template = self._gfql_broadcast_scalar(non_empty, list(range(max_len)))
        expanded = non_empty.assign(**{pos_list_col: pos_template}).explode(pos_list_col)
        if len(expanded) == 0:
            return result.reset_index(drop=True)

        expanded = expanded.reset_index(drop=True)
        expanded = expanded.rename(columns={pos_list_col: pos_col})
        expanded[pos_col] = expanded[pos_col].astype("int64")
        expanded = expanded.loc[expanded[pos_col] < expanded[len_col]]
        expanded = expanded.assign(**{out_col: expanded[start_col] + expanded[pos_col] * expanded[step_col]})

        grouped = expanded.groupby(row_col, sort=False)[out_col].agg(list).reset_index()
        merged = base[[row_col]].merge(grouped, on=row_col, how="left", sort=False)
        merged = self._gfql_restore_row_order(merged, row_col)
        result = merged[out_col].copy()
        missing_mask = result.isna() if hasattr(result, "isna") else None
        if missing_mask is not None and bool(missing_mask.any()):
            result = self._gfql_mask_fill_sequence_value(merged, result, missing_mask, [])
        if hasattr(result, "reset_index"):
            return result.reset_index(drop=True)
        return result

    def _gfql_resolve_token(self, table_df: Any, token: str) -> Any:
        txt = token.strip()
        if txt in table_df.columns:
            return table_df[txt]
        if txt == "__gfql_edge_index_0__" and self._edge is not None and self._edge in table_df.columns:
            return table_df[self._edge]
        prop_match = RowPipelineMixin._GFQL_ALIAS_PROP_RE.fullmatch(txt)
        if prop_match is not None:
            alias = prop_match.group("alias")
            prop = prop_match.group("prop")
            if (
                alias in table_df.columns
                and hasattr(table_df[alias], "str")
                and RowPipelineMixin._gfql_series_is_mapping_like(table_df[alias])
            ):
                try:
                    return table_df[alias].str.get(prop)
                except Exception:
                    pass
            if prop in table_df.columns:
                return table_df[prop]
            if alias in table_df.columns:
                if not RowPipelineMixin._gfql_table_has_graph_shape(table_df):
                    raise ValueError(
                        f"unsupported row expression: property access requires a graph element alias in {token!r}"
                    )
                return self._gfql_broadcast_scalar(table_df, pd.NA)
        raise ValueError(f"unsupported token in row expression: {token!r}")

    def _gfql_eval_dynamic_list_subscript(
        self,
        table_df: Any,
        base_value: Any,
        key_value: Any,
        expr: str,
    ) -> Any:
        if not hasattr(base_value, "iloc"):
            base_value = self._gfql_broadcast_scalar(table_df, base_value)

        base_dtype = str(getattr(base_value, "dtype", "")).lower()
        if base_dtype != "object":
            raise ValueError(
                f"unsupported row expression: dynamic subscript requires list-like base in {expr!r}"
            )

        normalized_values: List[Any] = []
        saw_string_parse = False
        parse_failed = False
        for value in RowPipelineMixin._gfql_series_to_pylist(base_value):
            if is_null_scalar(value):
                normalized_values.append(None)
                continue
            if isinstance(value, (list, tuple)):
                normalized_values.append(list(value))
                continue
            if isinstance(value, str):
                try:
                    parsed_value = ast.literal_eval(value)
                except Exception:
                    parsed_value = None
                if isinstance(parsed_value, (list, tuple)):
                    normalized_values.append(list(parsed_value))
                    saw_string_parse = True
                    continue
            parse_failed = True
            break
        if parse_failed:
            raise ValueError(
                f"unsupported row expression: dynamic subscript requires list-like base in {expr!r}"
            )
        if saw_string_parse:
                host_index = getattr(base_value, "index", None)
                if host_index is not None and hasattr(host_index, "to_pandas"):
                    try:
                        host_index = host_index.to_pandas()
                    except Exception:
                        host_index = None
                base_value = pd.Series(normalized_values, index=host_index, dtype="object")
        if not RowPipelineMixin._gfql_series_is_list_like(base_value):
            raise ValueError(
                f"unsupported row expression: dynamic subscript requires list-like base in {expr!r}"
            )

        key_dtype = str(getattr(key_value, "dtype", "")).lower()
        key_is_int_like = "int" in key_dtype and "bool" not in key_dtype
        if not key_is_int_like:
            if key_dtype != "object":
                raise ValueError(
                    f"unsupported row expression: dynamic subscript keys must be integer typed in {expr!r}"
                )
            if not RowPipelineMixin._gfql_series_int_non_bool_like(key_value):
                raise ValueError(
                    f"unsupported row expression: dynamic subscript keys must be integer typed in {expr!r}"
                )

        row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_dynsub_row__")
        base_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_dynsub_base__")
        key_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_dynsub_key__")
        pos_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_dynsub_pos__")

        if isinstance(base_value, pd.Series) or isinstance(key_value, pd.Series):
            base_assign = base_value.to_pandas() if hasattr(base_value, "to_pandas") else base_value
            key_assign = key_value.to_pandas() if hasattr(key_value, "to_pandas") else key_value
            if hasattr(base_assign, "reset_index"):
                base_assign = base_assign.reset_index(drop=True)
            if hasattr(key_assign, "reset_index"):
                key_assign = key_assign.reset_index(drop=True)
            base = pd.DataFrame({row_col: range(len(table_df))})
            base = base.assign(
                **{
                    base_col: base_assign,
                    key_col: key_assign,
                }
            )[[row_col, base_col, key_col]]
        else:
            base = table_df.assign(
                **{
                    row_col: range(len(table_df)),
                    base_col: base_value,
                    key_col: key_value,
                }
            )[[row_col, base_col, key_col]]

        expanded = base[[row_col, base_col]].explode(base_col)
        expanded = expanded.assign(
            **{pos_col: expanded.groupby(row_col, sort=False).cumcount()}
        )[[row_col, pos_col, base_col]]
        if key_dtype == "object" and hasattr(expanded[pos_col], "astype"):
            expanded[pos_col] = expanded[pos_col].astype("object")

        merged = base[[row_col, key_col]].merge(
            expanded,
            left_on=[row_col, key_col],
            right_on=[row_col, pos_col],
            how="left",
            sort=False,
        )
        return merged[base_col].reset_index(drop=True)

    def _gfql_eval_slice_subscript(
        self,
        table_df: Any,
        base_value: Any,
        start_value: Any,
        end_value: Any,
        start_present: bool,
        end_present: bool,
        expr: str,
    ) -> Any:
        if not hasattr(base_value, "astype"):
            base_value = self._gfql_broadcast_scalar(table_df, base_value)

        if hasattr(start_value, "astype"):
            start_ok, start_scalar = RowPipelineMixin._gfql_series_scalar_if_constant(start_value)
            if not start_ok:
                raise ValueError(f"unsupported row expression: dynamic slice start is not supported in {expr!r}")
            start_value = start_scalar
        if hasattr(end_value, "astype"):
            end_ok, end_scalar = RowPipelineMixin._gfql_series_scalar_if_constant(end_value)
            if not end_ok:
                raise ValueError(f"unsupported row expression: dynamic slice end is not supported in {expr!r}")
            end_value = end_scalar

        if (start_present and is_null_scalar(start_value)) or (
            end_present and is_null_scalar(end_value)
        ):
            return self._gfql_broadcast_scalar(table_df, None)

        def _coerce_bound(v: Any, label: str) -> Optional[int]:
            if is_null_scalar(v):
                return None
            if isinstance(v, bool):
                raise ValueError(f"unsupported row expression: {label} bound must be integer/null in {expr!r}")
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                if not float(v).is_integer():
                    raise ValueError(f"unsupported row expression: {label} bound must be integer/null in {expr!r}")
                return int(v)
            if isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
                return int(v.strip())
            raise ValueError(f"unsupported row expression: {label} bound must be integer/null in {expr!r}")

        start_i = _coerce_bound(start_value, "slice start")
        end_i = _coerce_bound(end_value, "slice end")

        if RowPipelineMixin._gfql_series_is_list_like(base_value):
            row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_slice_row__")
            list_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_slice_list__")
            len_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_slice_len__")
            start_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_slice_start__")
            stop_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_slice_stop__")
            pos_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_slice_pos__")
            out_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_slice_out__")

            base = table_df.assign(**{row_col: range(len(table_df)), list_col: base_value})
            null_mask = self._gfql_null_mask(base, base[list_col])
            try:
                lengths = series_sequence_len(base[list_col])
            except Exception as exc:
                raise ValueError(f"unsupported row expression: slice subscript requires list/string base in {expr!r}") from exc
            if hasattr(lengths, "fillna"):
                lengths = lengths.fillna(0)
            lengths = lengths.astype("int64")

            if start_i is None:
                start_norm = lengths * 0
            elif start_i >= 0:
                start_norm = lengths * 0 + start_i
            else:
                start_norm = lengths + start_i

            if end_i is None:
                stop_norm = lengths
            elif end_i >= 0:
                stop_norm = lengths * 0 + end_i
            else:
                stop_norm = lengths + end_i

            start_norm = start_norm.clip(lower=0, upper=lengths)
            stop_norm = stop_norm.clip(lower=0, upper=lengths)
            base = base.assign(**{len_col: lengths, start_col: start_norm, stop_col: stop_norm})

            non_null = base.loc[~null_mask, [row_col, list_col, len_col, start_col, stop_col]].copy()
            expanded = non_null[[row_col, list_col]].explode(list_col)
            if len(expanded) > 0:
                expanded = expanded.reset_index(drop=True)
                expanded = expanded.assign(
                    **{pos_col: expanded.groupby(row_col, sort=False).cumcount()}
                )
                bounds = non_null[[row_col, len_col, start_col, stop_col]]
                expanded = expanded.merge(bounds, on=row_col, how="left", sort=False)
                keep_mask = (
                    (expanded[pos_col] < expanded[len_col])
                    & (expanded[pos_col] >= expanded[start_col])
                    & (expanded[pos_col] < expanded[stop_col])
                )
                expanded = expanded.loc[keep_mask, [row_col, list_col]].rename(columns={list_col: out_col})
                grouped = expanded.groupby(row_col, sort=False)[out_col].agg(list).reset_index()
            else:
                grouped = non_null[[row_col]].iloc[0:0].copy()
                grouped[out_col] = []

            merged = base[[row_col, start_col, stop_col]].merge(grouped, on=row_col, how="left", sort=False)
            merged = self._gfql_restore_row_order(merged, row_col)
            result = (
                merged[out_col].copy()
                if out_col in merged.columns
                else self._gfql_broadcast_scalar(merged, None)
            )
            empty_mask = merged[start_col] >= merged[stop_col]
            if hasattr(empty_mask, "any") and bool(empty_mask.any()):
                result = self._gfql_mask_fill_sequence_value(merged, result, empty_mask, [])
            if hasattr(null_mask, "any") and bool(null_mask.any()):
                result = self._gfql_mask_fill_sequence_value(merged, result, null_mask, None)
            return result.reset_index(drop=True)

        if not hasattr(base_value, "str"):
            raise ValueError(f"unsupported row expression: slice subscript requires list/string base in {expr!r}")
        try:
            return base_value.str.slice(start=start_i, stop=end_i)
        except Exception as exc:
            raise ValueError(
                f"unsupported row expression: slice subscript failed for {expr!r}: {exc}"
            ) from exc

    def _gfql_concat_list_scalar(
        self,
        table_df: Any,
        list_series: Any,
        scalar_series: Any,
        prepend: bool = False,
    ) -> Any:
        use_pandas_fallback = (
            isinstance(list_series, pd.Series)
            or isinstance(scalar_series, pd.Series)
            or (
                resolve_engine(EngineAbstract.AUTO, table_df) == Engine.CUDF
                and RowPipelineMixin._gfql_series_bool_like(scalar_series)
            )
        )
        if use_pandas_fallback:
            list_values = self._gfql_series_to_pylist(list_series)
            scalar_values = self._gfql_series_to_pylist(scalar_series)
            out_values: List[Any] = []
            for list_value, scalar_value in zip(list_values, scalar_values):
                if is_null_scalar(list_value):
                    out_values.append(None)
                    continue
                seq = list(list_value) if isinstance(list_value, (list, tuple)) else [list_value]
                scalar_item = None if is_null_scalar(scalar_value) else scalar_value
                out_values.append(([scalar_item] + seq) if prepend else (seq + [scalar_item]))
            return pd.Series(out_values)

        row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_add_row__")
        list_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_add_list__")
        val_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_add_val__")
        pos_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_add_pos__")
        len_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_add_len__")

        base = table_df.assign(**{row_col: range(len(table_df)), list_col: list_series, val_col: scalar_series})
        null_mask = self._gfql_null_mask(base, base[list_col])
        try:
            lengths = series_sequence_len(base[list_col]).fillna(0)
        except Exception as exc:
            raise ValueError("unsupported row expression: list concatenation requires list/string accessor support") from exc
        base = base.assign(**{len_col: lengths.astype("int64")})

        non_null = base.loc[~null_mask, [row_col, list_col, val_col, len_col]]
        expanded = non_null[[row_col, list_col, len_col]].explode(list_col)
        if len(expanded) > 0:
            expanded = expanded.assign(
                **{pos_col: expanded.groupby(row_col, sort=False).cumcount()}
            )
            expanded = expanded.loc[expanded[pos_col] < expanded[len_col]]
            expanded = expanded[[row_col, pos_col, list_col]].rename(columns={list_col: val_col})
        else:
            expanded = non_null[[row_col]].iloc[0:0].copy()
            expanded[pos_col] = []
            expanded[val_col] = []

        append_rows = non_null[[row_col, val_col, len_col]].copy()
        append_rows = append_rows.assign(**{pos_col: 0 if prepend else append_rows[len_col]})
        append_rows = append_rows[[row_col, pos_col, val_col]]

        if prepend:
            if len(expanded) > 0:
                expanded = expanded.assign(**{pos_col: expanded[pos_col] + 1})
            combined = concat_frames([append_rows, expanded])
        else:
            combined = concat_frames([expanded, append_rows])
        if combined is None:
            combined = non_null[[row_col]].iloc[0:0].copy()
            combined[pos_col] = []
            combined[val_col] = []
        if len(combined) > 0:
            combined = combined.sort_values(by=[row_col, pos_col], kind="mergesort")
            grouped = combined.groupby(row_col, sort=False)[val_col].agg(list).reset_index()
        else:
            grouped = non_null[[row_col]].iloc[0:0].copy()
            grouped[val_col] = []

        out = self._gfql_restore_row_order(
            base[[row_col, len_col]].merge(grouped, on=row_col, how="left", sort=False),
            row_col,
        )[val_col]
        empty_mask = base[len_col] == 0
        if hasattr(empty_mask, "any") and bool(empty_mask.any()):
            out.loc[out.index[empty_mask]] = self._gfql_broadcast_scalar(
                base.loc[empty_mask],
                [],
            )
        out = self._gfql_mask_fill(out, null_mask, None)
        return out.reset_index(drop=True)

    def _gfql_eval_in_expr(
        self,
        table_df: Any,
        left_value: Any,
        right_value: Any,
        expr: str,
    ) -> Any:
        left_series = left_value if hasattr(left_value, "astype") else self._gfql_broadcast_scalar(table_df, left_value)
        right_series = right_value if hasattr(right_value, "astype") else self._gfql_broadcast_scalar(table_df, right_value)

        try:
            rhs_len = series_sequence_len(right_series).fillna(0).astype("int64")
        except Exception as exc:
            raise ValueError(f"unsupported row expression: IN rhs must be list-like in {expr!r}") from exc

        row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_in_row__")
        rhs_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_in_rhs__")
        lhs_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_in_lhs__")
        len_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_in_len__")
        pos_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_in_pos__")

        base = table_df.assign(**{row_col: range(len(table_df)), lhs_col: left_series, rhs_col: right_series})
        rhs_null = self._gfql_null_mask(base, base[rhs_col])
        base = base.assign(**{len_col: rhs_len})

        non_null = base.loc[~rhs_null, [row_col, lhs_col, rhs_col, len_col]]
        expanded = non_null[[row_col, lhs_col, rhs_col, len_col]].explode(rhs_col)
        if len(expanded) > 0:
            expanded = expanded.assign(
                **{pos_col: expanded.groupby(row_col, sort=False).cumcount()}
            )
            expanded = expanded.loc[expanded[pos_col] < expanded[len_col]]

        if len(expanded) == 0:
            true_counts = non_null[[row_col]].iloc[0:0].copy()
            true_counts["__gfql_in_true__"] = []
            unknown_counts = non_null[[row_col]].iloc[0:0].copy()
            unknown_counts["__gfql_in_unknown__"] = []
        else:
            lhs = expanded[lhs_col]
            rhs = expanded[rhs_col]
            lhs_null = self._gfql_null_mask(expanded, lhs)
            rhs_null_elem = self._gfql_null_mask(expanded, rhs)
            equal = lhs == rhs
            equal = equal.where(~(lhs_null | rhs_null_elem), False)
            unknown = lhs_null | rhs_null_elem

            eval_df = expanded.assign(
                __gfql_in_true__=equal.astype("int64"),
                __gfql_in_unknown__=unknown.astype("int64"),
            )
            true_counts = eval_df.groupby(row_col, sort=False)["__gfql_in_true__"].sum().reset_index()
            unknown_counts = eval_df.groupby(row_col, sort=False)["__gfql_in_unknown__"].sum().reset_index()

        summary = base[[row_col, len_col]].merge(true_counts, on=row_col, how="left", sort=False)
        summary = summary.merge(unknown_counts, on=row_col, how="left", sort=False)
        summary = self._gfql_restore_row_order(summary, row_col)
        summary = summary.assign(
            __gfql_in_true__=summary["__gfql_in_true__"].fillna(0),
            __gfql_in_unknown__=summary["__gfql_in_unknown__"].fillna(0),
        )

        out = summary["__gfql_in_true__"] > 0
        unknown_mask = (summary["__gfql_in_true__"] == 0) & (summary["__gfql_in_unknown__"] > 0)
        out = out.where(~unknown_mask, pd.NA)
        out = out.where(~rhs_null, pd.NA)
        return out.reset_index(drop=True)

    def _gfql_eval_string_expr(self, table_df: Any, expr: str) -> Any:
        txt = expr.strip()
        parser_bundle = _gfql_expr_runtime_parser_bundle()
        if parser_bundle is None:
            raise ValueError(
                f"unsupported row expression: parser backend unavailable in {expr!r}"
            )

        parser, capability_checker, _expr_parser_mod = parser_bundle
        try:
            ast_node = parser(txt)
            capability_errors = capability_checker(ast_node)
        except Exception as exc:
            raise ValueError(f"unsupported row expression: parser validation failed in {expr!r}") from exc

        if len(capability_errors) > 0:
            raise ValueError(
                f"unsupported row expression: {', '.join(capability_errors)} in {expr!r}"
            )

        try:
            ast_ok, ast_value = self._gfql_eval_expr_ast(table_df, ast_node)
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"unsupported row expression: AST evaluator unsupported in {expr!r}") from exc

        if ast_ok:
            return ast_value

        raise ValueError(f"unsupported row expression: AST evaluator unsupported in {expr!r}")

    def _gfql_row_table(self, table_df: Any) -> "Plottable":
        """Return a plottable that treats ``table_df`` as the active row table."""
        out = self.bind()
        table_df = table_df.reset_index(drop=True)
        out._nodes = table_df
        if self._edges is not None:
            out._edges = self._edges.iloc[0:0].copy()
        else:
            out._edges = table_df.iloc[0:0].copy()
        out._source = None
        out._destination = None
        out._edge = self._edge if self._edge is not None and self._edge in table_df.columns else None
        if out._node is not None and out._node not in table_df.columns:
            out._node = None
        return out

    def _gfql_get_active_table(self) -> Any:
        if self._nodes is not None:
            return self._nodes
        if self._edges is not None:
            return self._edges
        return pd.DataFrame()

    @staticmethod
    def _gfql_coerce_non_negative_int(value: Any, op_name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{op_name} expects a non-negative integer, got bool")
        if isinstance(value, int):
            out = value
        elif isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{op_name} expects an integer, got {value!r}")
            out = int(value)
        elif isinstance(value, str):
            txt = value.strip()
            if txt.startswith("-"):
                out = int(txt)
            elif txt.isdigit():
                out = int(txt)
            else:
                raise ValueError(f"{op_name} expects an integer, got {value!r}")
        else:
            raise ValueError(f"{op_name} expects an integer, got {type(value).__name__}")
        if out < 0:
            raise ValueError(f"{op_name} must be non-negative, got {out}")
        return out

    def rows(
        self, table: str = "nodes", source: Optional[str] = None
    ) -> "Plottable":
        if table not in {"nodes", "edges"}:
            raise ValueError(
                f"rows(table=...) must be one of 'nodes' or 'edges', got {table!r}"
            )

        table_df = self._nodes if table == "nodes" else self._edges
        if table_df is None:
            if self._nodes is not None:
                table_df = self._nodes.iloc[0:0].copy()
            elif self._edges is not None:
                table_df = self._edges.iloc[0:0].copy()
            else:
                table_df = pd.DataFrame()
        else:
            table_df = table_df.copy()

        if source is not None:
            if source not in table_df.columns:
                raise ValueError(f"rows(source=...) alias column not found: {source!r}")
            mask = table_df[source]
            if hasattr(mask, "isna") and hasattr(mask, "where"):
                mask = mask.where(~mask.isna(), False)
            elif hasattr(mask, "fillna"):
                mask = mask.fillna(False)
            table_df = table_df.loc[mask.astype(bool)]

        return self._gfql_row_table(table_df)

    def select(self, items: List[Any]) -> "Plottable":
        table_df = self._gfql_get_active_table()
        if items is None:
            raise ValueError(
                "select(items=...) requires entries of form (alias, expr) or shorthand 'col'"
            )

        projected: Dict[str, Any] = {}
        cudf_row_table = resolve_engine(EngineAbstract.AUTO, table_df) == Engine.CUDF

        def _project_scalar(value: Any) -> Any:
            if cudf_row_table and isinstance(value, (list, tuple, dict)):
                return pd.Series([value for _ in range(len(table_df))], dtype="object")
            return self._gfql_broadcast_scalar(table_df, value)

        for item in items:
            if isinstance(item, str):
                alias_raw, expr = item, item
            else:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError(
                        "select expects entries of form (alias, expr) or shorthand 'col', "
                        f"got {item!r}"
                    )
                alias_raw, expr = item
            alias = str(alias_raw)
            if alias == "":
                raise ValueError("select alias must be non-empty")
            if isinstance(expr, str):
                if expr in table_df.columns:
                    value = table_df[expr]
                else:
                    value = self._gfql_eval_string_expr(table_df, expr)
                if not hasattr(value, "astype"):
                    value = _project_scalar(value)
                projected[alias] = value
            else:
                projected[alias] = _project_scalar(expr)

        out_table_df = table_df
        if resolve_engine(EngineAbstract.AUTO, table_df) == Engine.CUDF and any(
            isinstance(value, pd.Series) for value in projected.values()
        ):
            out_table_df = table_df.to_pandas()
            projected = {
                alias: (value.to_pandas() if hasattr(value, "to_pandas") else value)
                for alias, value in projected.items()
            }

        out_df = out_table_df.assign(**projected)[list(projected.keys())]
        return self._gfql_row_table(out_df)

    def with_(self, items: List[Any]) -> "Plottable":
        """Python-safe alias for Cypher WITH-style row projection."""
        return self.select(items)

    def where_rows(
        self,
        filter_dict: Optional[Dict[str, Any]] = None,
        expr: Optional[str] = None,
    ) -> "Plottable":
        """Filter active row table with vectorized dict predicates and/or expression."""
        table_df = self._gfql_get_active_table()
        out_df = table_df

        if filter_dict is None:
            filter_dict = {}
        if not isinstance(filter_dict, dict):
            raise ValueError("where_rows(filter_dict=...) must be a dict when provided")
        if filter_dict:
            from graphistry.compute.filter_by_dict import filter_by_dict

            out_df = filter_by_dict(out_df, filter_dict)

        if expr is not None:
            if not isinstance(expr, str) or expr.strip() == "":
                raise ValueError("where_rows(expr=...) must be a non-empty string")
            expr_value = self._gfql_eval_string_expr(out_df, expr)
            mask = self._gfql_bool_mask(out_df, expr_value)
            out_df = out_df.loc[mask]

        return self._gfql_row_table(out_df)

    def return_(self, items: List[Any]) -> "Plottable":
        return self.select(items)

    def order_by(self, keys: List[Any]) -> "Plottable":
        table_df = self._gfql_get_active_table()
        row_order_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_sort_row_order__")
        if keys is None:
            raise ValueError(
                "order_by(keys=...) requires a list/tuple of (expr, direction)"
            )

        sort_cols: List[str] = []
        ascending: List[bool] = []
        work_df = table_df.assign(**{row_order_col: range(len(table_df))})
        tmp_idx = 0
        for key_item in keys:
            if not isinstance(key_item, (list, tuple)) or len(key_item) != 2:
                raise ValueError(
                    f"order_by expects (expr, direction) pairs, got {key_item!r}"
                )
            expr, direction = key_item
            if not isinstance(expr, str):
                raise ValueError(
                    "order_by currently supports string expressions, got "
                    f"{type(expr).__name__}"
                )
            if expr in work_df.columns:
                sort_col = expr
            else:
                parsed_order_node = None
                parser_bundle = _gfql_expr_runtime_parser_bundle()
                if parser_bundle is not None:
                    parser, _capability_checker, _expr_parser_mod = parser_bundle
                    try:
                        parsed_order_node = parser(expr)
                    except Exception:
                        parsed_order_node = None
                if not RowPipelineMixin._gfql_order_expr_static_supported(expr):
                    raise ValueError(
                        "unsupported order_by expression in vectorized mode; "
                        f"use column/scalar arithmetic comparisons only: {expr!r}"
                    )
                temporal_duration_expr = (
                    extract_temporal_duration_sort_ast(parsed_order_node)
                    if parsed_order_node is not None
                    else None
                )
                if temporal_duration_expr is not None:
                    base_node, duration_text, duration_sign = temporal_duration_expr
                    duration_components = parse_temporal_sort_duration_components(duration_text)
                    if duration_components is None:
                        raise ValueError(
                            "unsupported order_by temporal duration expression in vectorized mode; "
                            f"unsupported duration literal: {expr!r}"
                        )
                    month_shift, duration_shift = duration_components
                    ok_base, base_value = self._gfql_eval_expr_ast(work_df, base_node)
                    if not ok_base or not hasattr(base_value, "astype"):
                        raise ValueError(
                            "unsupported order_by temporal duration expression in vectorized mode; "
                            f"base expression must resolve to a vectorized temporal series: {expr!r}"
                        )
                    sort_col = f"__gfql_sort_{tmp_idx}__"
                    tmp_idx += 1
                    work_df = work_df.assign(**{sort_col: base_value})
                    direction_is_asc = str(direction).lower() != "desc"
                    series = work_df[sort_col]
                    temporal_mode = order_detect_temporal_mode(series)
                    if temporal_mode is None:
                        raise ValueError(
                            "unsupported order_by temporal duration expression in vectorized mode; "
                            f"base expression is not a supported temporal series: {expr!r}"
                        )
                    if month_shift != 0 and temporal_mode in {"time", "time_constructor"}:
                        raise ValueError(
                            "unsupported order_by temporal duration expression in vectorized mode; "
                            f"time values do not support year/month duration offsets: {expr!r}"
                        )
                    key_prefix = f"__gfql_sort_temporal_{tmp_idx}__"
                    tmp_idx += 1
                    work_df, temporal_key_cols = build_temporal_sort_columns(
                        work_df,
                        sort_col,
                        key_prefix,
                        temporal_mode,
                        month_shift=duration_sign * month_shift,
                        nanosecond_shift=duration_sign * duration_shift,
                        null_mask_fn=self._gfql_null_mask,
                        fresh_col_name_fn=RowPipelineMixin._gfql_fresh_col_name,
                    )
                    sort_cols.extend(temporal_key_cols)
                    ascending.extend([direction_is_asc] * len(temporal_key_cols))
                    continue
                sort_col = f"__gfql_sort_{tmp_idx}__"
                tmp_idx += 1
                work_df = work_df.assign(**{sort_col: self._gfql_eval_string_expr(work_df, expr)})
            direction_is_asc = str(direction).lower() != "desc"
            series = work_df[sort_col]
            list_candidate = order_detect_list_series(series)
            if list_candidate:
                top_null_mask = self._gfql_null_mask(work_df, series)
                if hasattr(top_null_mask, "any") and bool(top_null_mask.any()):
                    list_candidate = False
            if list_candidate:
                key_prefix = f"__gfql_sort_list_{tmp_idx}__"
                tmp_idx += 1
                work_df, list_key_cols = build_list_sort_columns(
                    work_df,
                    sort_col,
                    key_prefix,
                    null_mask_fn=self._gfql_null_mask,
                    broadcast_scalar_fn=self._gfql_broadcast_scalar,
                    fresh_col_name_fn=RowPipelineMixin._gfql_fresh_col_name,
                )
                sort_cols.extend(list_key_cols)
                ascending.extend([direction_is_asc] * len(list_key_cols))
                continue
            temporal_mode = order_detect_temporal_mode(series)
            if temporal_mode is not None:
                key_prefix = f"__gfql_sort_temporal_{tmp_idx}__"
                tmp_idx += 1
                work_df, temporal_key_cols = build_temporal_sort_columns(
                    work_df,
                    sort_col,
                    key_prefix,
                    temporal_mode,
                    null_mask_fn=self._gfql_null_mask,
                    fresh_col_name_fn=RowPipelineMixin._gfql_fresh_col_name,
                )
                sort_cols.extend(temporal_key_cols)
                ascending.extend([direction_is_asc] * len(temporal_key_cols))
                continue
            validate_order_series_vector_safe(series, expr)
            sort_cols.append(sort_col)
            ascending.append(direction_is_asc)

        if sort_cols:
            try:
                effective_sort_cols = sort_cols + [row_order_col]
                effective_ascending = ascending + [True]
                out_df = work_df.sort_values(by=effective_sort_cols, ascending=effective_ascending)
            except Exception as exc:
                raise ValueError(
                    "unsupported order_by for vectorized execution; "
                    f"cannot sort key set {sort_cols!r} with expression set "
                    f"{[k[0] for k in keys]!r}: {exc}"
                ) from exc
            drop_cols = [c for c in effective_sort_cols if isinstance(c, str) and c.startswith("__gfql_sort_")]
            if drop_cols:
                out_df = out_df.drop(columns=drop_cols)
        else:
            out_df = work_df.drop(columns=[row_order_col])
        return self._gfql_row_table(out_df)

    def skip(self, value: Any) -> "Plottable":
        table_df = self._gfql_get_active_table()
        skip_count = self._gfql_coerce_non_negative_int(value, "skip")
        return self._gfql_row_table(table_df.iloc[skip_count:])

    def limit(self, value: Any) -> "Plottable":
        table_df = self._gfql_get_active_table()
        limit_count = self._gfql_coerce_non_negative_int(value, "limit")
        return self._gfql_row_table(table_df.iloc[:limit_count])

    def distinct(self) -> "Plottable":
        table_df = self._gfql_get_active_table()
        try:
            out_df = table_df.drop_duplicates()
        except Exception:
            # Fallback for unhashable list/map cells: dedupe by string-normalized
            # object-like columns while preserving original row payload.
            work_df = table_df
            object_cols = [col for col in table_df.columns if str(table_df[col].dtype) == "object"]
            if object_cols:
                work_df = table_df.assign(
                    **{col: table_df[col].astype(str) for col in object_cols}
                )
            mask = ~work_df.duplicated(keep="first")
            out_df = table_df.loc[mask]
        return self._gfql_row_table(out_df)

    def unwind(self, expr: Any, as_: str = "value") -> "Plottable":
        """Vectorized UNWIND for column or literal list expressions."""
        table_df = self._gfql_get_active_table()
        if not isinstance(as_, str) or as_.strip() == "":
            raise ValueError("unwind(as_=...) must be a non-empty string")
        as_clean = as_.strip()

        if isinstance(expr, str):
            if expr in table_df.columns:
                explode_src = table_df[expr]
            else:
                explode_src = self._gfql_eval_string_expr(table_df, expr)
        elif isinstance(expr, (list, tuple)):
            explode_src = self._gfql_broadcast_scalar(table_df, list(expr))
        else:
            raise ValueError(
                "unwind(expr=...) currently supports a row expression or list/tuple literal"
            )

        if not hasattr(explode_src, "astype"):
            explode_src = self._gfql_broadcast_scalar(table_df, explode_src)

        tmp_col = "__gfql_unwind_values__"
        list_like = self._gfql_series_is_list_like(explode_src)
        out_df = table_df.assign(**{tmp_col: explode_src})
        row_col = None
        len_col = None
        pos_col = None
        null_mask = self._gfql_null_mask(out_df, out_df[tmp_col])
        if not list_like and hasattr(null_mask, "all") and bool(null_mask.all()):
            empty_df = out_df.iloc[0:0].drop(columns=[tmp_col]).assign(**{as_clean: out_df.iloc[0:0][tmp_col]})
            return self._gfql_row_table(empty_df)
        if list_like:
            row_col = self._gfql_fresh_col_name(out_df.columns, "__gfql_unwind_row__")
            len_col = self._gfql_fresh_col_name(out_df.columns, "__gfql_unwind_len__")
            pos_col = self._gfql_fresh_col_name(out_df.columns, "__gfql_unwind_pos__")
            lengths = series_sequence_len(explode_src)
            if hasattr(lengths, "fillna"):
                lengths = lengths.fillna(0)
            out_df = out_df.assign(**{row_col: range(len(out_df)), len_col: lengths})
        if not hasattr(out_df, "explode"):
            raise ValueError("unwind requires dataframe explode() support")
        out_df = out_df.explode(tmp_col)
        if list_like and row_col is not None and len_col is not None and pos_col is not None:
            out_df = out_df.assign(**{pos_col: out_df.groupby(row_col, sort=False).cumcount()})
            out_df = out_df.loc[out_df[pos_col] < out_df[len_col]]
            out_df = out_df.drop(columns=[row_col, len_col, pos_col])
        out_df = out_df.assign(**{as_clean: out_df[tmp_col]})
        out_df = out_df.drop(columns=[tmp_col])
        return self._gfql_row_table(out_df)

    def group_by(
        self,
        keys: Sequence[str],
        aggregations: Sequence[Sequence[Any]],
    ) -> "Plottable":
        """Vectorized grouped aggregations for row-table pipelines."""
        table_df = self._gfql_get_active_table()
        key_cols = [str(k) for k in keys]
        if not key_cols:
            raise ValueError("group_by(keys=...) requires at least one key column")
        for key in key_cols:
            if key not in table_df.columns:
                raise ValueError(f"group_by key column not found: {key!r}")

        group_order_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_group_order__")
        table_df = table_df.assign(**{group_order_col: range(len(table_df))})

        def _make_grouped(df: Any) -> Any:
            def _build_grouped(group_df: Any) -> Any:
                try:
                    grouped_local = group_df.groupby(key_cols, sort=False, dropna=False)
                except TypeError:
                    grouped_local = group_df.groupby(key_cols, sort=False)
                # Trigger factorization now so unhashable key payloads fail fast
                # and can be normalized to string keys in the outer fallback.
                grouped_local.size()
                return grouped_local

            try:
                return _build_grouped(df)
            except TypeError:
                key_object_cols = [
                    col for col in key_cols if col in df.columns and str(df[col].dtype) == "object"
                ]
                work_df = df
                if key_object_cols:
                    work_df = df.assign(**{col: df[col].astype(str) for col in key_object_cols})
                return _build_grouped(work_df)

        grouped = _make_grouped(table_df)

        base = grouped.size().reset_index(name="__gfql_group_size__")
        group_order_df = grouped[group_order_col].min().reset_index(name=group_order_col)
        base = base.merge(group_order_df, on=key_cols, how="left", sort=False)
        base = base.sort_values(by=[group_order_col]).reset_index(drop=True)
        out_df = base[key_cols + [group_order_col]].copy()

        for agg in aggregations:
            if not isinstance(agg, (list, tuple)) or len(agg) not in {2, 3}:
                raise ValueError(
                    "group_by aggregations must be tuples/lists of "
                    "(alias, func) or (alias, func, expr)"
                )
            alias = str(agg[0])
            func = str(agg[1]).lower()
            expr = agg[2] if len(agg) == 3 else None

            if func == "count" and (expr is None or expr == "*"):
                grouped = _make_grouped(table_df)
                agg_df = grouped.size().reset_index(name=alias)
            else:
                if not isinstance(expr, str):
                    raise ValueError(
                        f"group_by aggregation {alias!r} requires string expr column"
                    )
                expr_col = expr
                if expr_col not in table_df.columns:
                    expr_values = self._gfql_eval_string_expr(table_df, expr_col)
                    if not hasattr(expr_values, "astype"):
                        expr_values = self._gfql_broadcast_scalar(table_df, expr_values)
                    tmp_col = "__gfql_group_expr__"
                    while tmp_col in table_df.columns:
                        tmp_col = f"{tmp_col}_x"
                    table_df = table_df.assign(**{tmp_col: expr_values})
                    expr_col = tmp_col
                grouped = _make_grouped(table_df)
                if func in {"collect", "collect_distinct"}:
                    # collect() ignores null entries; compute collection on
                    # non-null rows and merge against full key space below.
                    grouped_df = grouped.obj
                    non_null_df = grouped_df.loc[
                        ~grouped_df[expr_col].isna(), key_cols + [expr_col]
                    ]
                    if len(non_null_df) == 0:
                        agg_df = out_df[key_cols].iloc[0:0].copy()
                        agg_df[alias] = self._gfql_broadcast_scalar(agg_df, None)
                    else:
                        if func == "collect_distinct":
                            try:
                                non_null_df = non_null_df.drop_duplicates(
                                    subset=key_cols + [expr_col],
                                    keep="first",
                                )
                            except TypeError:
                                norm_col = "__gfql_collect_distinct_norm__"
                                while norm_col in non_null_df.columns:
                                    norm_col = f"{norm_col}_x"
                                norm_df = non_null_df.assign(**{norm_col: non_null_df[expr_col].astype(str)})
                                norm_df = norm_df.drop_duplicates(
                                    subset=key_cols + [norm_col],
                                    keep="first",
                                )
                                non_null_df = norm_df.drop(columns=[norm_col])
                        agg_df = (
                            _make_grouped(non_null_df)[expr_col]
                            .agg(list)
                            .reset_index(name=alias)
                        )
                else:
                    method_name = GFQL_GROUPBY_AGG_METHODS.get(func)
                    if method_name is None:
                        raise ValueError(f"unsupported group_by aggregation function: {func!r}")
                    agg_series = grouped[expr_col]
                    agg_df = getattr(agg_series, method_name)().reset_index(name=alias)

            out_df = out_df.merge(agg_df, on=key_cols, how="left", sort=False)
            if func in {"collect", "collect_distinct"}:
                null_mask = out_df[alias].isna()
                has_nulls = True
                if hasattr(null_mask, "any"):
                    try:
                        has_nulls = bool(null_mask.any())
                    except Exception:
                        has_nulls = True
                if has_nulls:
                    try:
                        empty_lists = self._gfql_broadcast_scalar(out_df, [])
                        out_df[alias] = out_df[alias].where(~null_mask, empty_lists)
                    except Exception:
                        filled_values = [
                            [] if is_null_scalar(value) else value
                            for value in self._gfql_series_to_pylist(out_df[alias])
                        ]
                        out_df = out_df.drop(columns=[alias]).assign(**{alias: filled_values})

        out_df = out_df.sort_values(by=[group_order_col]).reset_index(drop=True)
        out_df = out_df.drop(columns=[group_order_col])
        return self._gfql_row_table(out_df)


class _RowPipelineAdapter(RowPipelineMixin):
    """Adapter for row-pipeline calls without requiring global ComputeMixin inheritance."""

    def __init__(self, g: "Plottable") -> None:
        self._g = g
        self._nodes = getattr(g, "_nodes", None)
        self._edges = getattr(g, "_edges", None)
        self._node = getattr(g, "_node", None)
        self._source = getattr(g, "_source", None)
        self._destination = getattr(g, "_destination", None)
        self._edge = getattr(g, "_edge", None)

    def bind(self) -> "Plottable":
        return self._g.bind()


def execute_row_pipeline_call(
    g: "Plottable", function: str, params: Dict[str, Any]
) -> "Plottable":
    if function not in ROW_PIPELINE_CALLS:
        raise ValueError(f"not a row-pipeline call: {function!r}")
    adapter = _RowPipelineAdapter(g)
    method = getattr(adapter, function)
    out = method(**params)
    return out
