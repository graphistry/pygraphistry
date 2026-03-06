import datetime
import math
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence, Tuple

import pandas as pd

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable


@lru_cache(maxsize=1)
def _gfql_expr_runtime_parser_bundle() -> Any:
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


class _RowPipelineContext(Protocol):
    _nodes: Any
    _edges: Any
    _node: Any
    _source: Any
    _destination: Any
    _edge: Any

    def bind(self) -> "Plottable":
        ...

    def _gfql_row_table(self, table_df: Any) -> "Plottable":
        ...

    def _gfql_get_active_table(self) -> Any:
        ...

    @staticmethod
    def _gfql_coerce_non_negative_int(value: Any, op_name: str) -> int:
        ...

    def select(self, items: List[Any]) -> "Plottable":
        ...

    def _gfql_broadcast_scalar(self, table_df: Any, value: Any) -> Any:
        ...

    def _gfql_bool_mask(self, table_df: Any, value: Any) -> Any:
        ...

    def _gfql_null_mask(self, table_df: Any, value: Any) -> Any:
        ...

    def _gfql_resolve_token(self, table_df: Any, token: str) -> Any:
        ...

    def _gfql_eval_string_expr(self, table_df: Any, expr: str) -> Any:
        ...

    def _gfql_eval_dynamic_list_subscript(
        self, table_df: Any, base_value: Any, key_value: Any, expr: str
    ) -> Any:
        ...

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
        ...

    def _gfql_concat_list_scalar(
        self, table_df: Any, list_series: Any, scalar_series: Any, prepend: bool = False
    ) -> Any:
        ...

    def _gfql_eval_in_expr(self, table_df: Any, left_expr: str, right_expr: str, expr: str) -> Any:
        ...

    def _gfql_eval_expr_ast(self, table_df: Any, node: Any) -> Tuple[bool, Any]:
        ...

    def _gfql_eval_string_predicate_expr(
        self,
        table_df: Any,
        left: Any,
        right: Any,
        op_name: str,
        expr: str,
    ) -> Any:
        ...

    def _gfql_build_list_sort_columns(
        self, work_df: Any, sort_col: str, key_prefix: str
    ) -> Tuple[Any, List[str]]:
        ...

    def _gfql_build_temporal_sort_columns(
        self, work_df: Any, sort_col: str, key_prefix: str, mode: str
    ) -> Tuple[Any, List[str]]:
        ...


class RowPipelineMixin:
    _GFQL_BIN_OP_RE = re.compile(
        r"^\s*(?P<left>[^()]+?)\s*(?P<op><=|>=|<>|!=|=|<|>|\+|-|\*|/|%)\s*(?P<right>[^()]+?)\s*$"
    )
    _GFQL_ALIAS_PROP_RE = re.compile(r"^(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.(?P<prop>[A-Za-z_][A-Za-z0-9_]*)$")
    _GFQL_IS_NULL_RE = re.compile(r"^(?P<value>.+?)\s+IS\s+NULL$", re.IGNORECASE)
    _GFQL_IS_NOT_NULL_RE = re.compile(r"^(?P<value>.+?)\s+IS\s+NOT\s+NULL$", re.IGNORECASE)
    _GFQL_LABEL_PRED_RE = re.compile(
        r"^(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<label>[A-Za-z_][A-Za-z0-9_]*)$"
    )
    _GFQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _GFQL_QUANTIFIER_RE = re.compile(r"^(?P<fn>ANY|ALL|NONE|SINGLE)\s*\((?P<body>.*)\)$", re.IGNORECASE | re.DOTALL)
    _GFQL_SIZE_RE = re.compile(r"(?is)^size\s*\((?P<inner>.+)\)$")
    _GFQL_ABS_RE = re.compile(r"(?is)^abs\s*\((?P<inner>.+)\)$")
    _GFQL_SIGN_RE = re.compile(r"(?is)^sign\s*\((?P<inner>.+)\)$")
    _GFQL_TOBOOLEAN_RE = re.compile(r"(?is)^toBoolean\s*\((?P<inner>.+)\)$")
    _GFQL_TOSTRING_RE = re.compile(r"(?is)^toString\s*\((?P<inner>.+)\)$")
    _GFQL_COALESCE_RE = re.compile(r"(?is)^coalesce\s*\((?P<inner>.+)\)$")
    _GFQL_HEAD_RE = re.compile(r"(?is)^head\s*\((?P<inner>.+)\)$")
    _GFQL_TAIL_RE = re.compile(r"(?is)^tail\s*\((?P<inner>.+)\)$")
    _GFQL_REVERSE_RE = re.compile(r"(?is)^reverse\s*\((?P<inner>.+)\)$")
    _GFQL_NODES_RE = re.compile(r"(?is)^nodes\s*\((?P<inner>.+)\)$")
    _GFQL_RELATIONSHIPS_RE = re.compile(r"(?is)^relationships\s*\((?P<inner>.+)\)$")
    _GFQL_RAND_RE = re.compile(r"(?is)^rand\s*\(\s*\)\s*$")
    _GFQL_ORDER_SAFE_FUNCS = {
        "abs",
        "tostring",
        "toboolean",
        "coalesce",
        "size",
        "sign",
        "reverse",
        "head",
        "tail",
        "rand",
        "count",
        "sum",
        "min",
        "max",
        "avg",
        "mean",
        "collect",
    }

    @staticmethod
    def _gfql_fresh_col_name(columns: Any, prefix: str) -> str:
        col = prefix
        while col in columns:
            col = f"{col}_x"
        return col

    def _gfql_eval_expr_ast(self: _RowPipelineContext, table_df: Any, node: Any) -> Tuple[bool, Any]:
        try:
            import graphistry.compute.gfql.expr_parser as expr_parser_mod
        except Exception:
            return False, None

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
        SubscriptExpr = expr_parser_mod.SubscriptExpr
        SliceExpr = expr_parser_mod.SliceExpr

        if isinstance(node, Identifier):
            txt = node.name
            if txt in table_df.columns:
                return True, table_df[txt]
            try:
                return True, self._gfql_resolve_token(table_df, txt)
            except Exception:
                return False, None

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

            melted = base.melt(id_vars=[row_col], value_vars=value_cols, var_name=ord_col, value_name=val_col)
            order_map = {col: idx for idx, col in enumerate(value_cols)}
            if hasattr(melted[ord_col], "map"):
                melted[ord_col] = melted[ord_col].map(order_map)
            else:
                melted[ord_col] = melted[ord_col].replace(order_map)
            melted[ord_col] = melted[ord_col].astype("int64")
            melted = melted.sort_values(by=[row_col, ord_col], kind="mergesort")
            grouped = melted.groupby(row_col, sort=False)[val_col].agg(list).reset_index()
            out = base[[row_col]].merge(grouped, on=row_col, how="left", sort=False)[val_col]
            return True, out.reset_index(drop=True)

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
                return True, ~self._gfql_bool_mask(table_df, operand)
            return False, None

        if isinstance(node, BinaryOp):
            ok_l, left = self._gfql_eval_expr_ast(table_df, node.left)
            ok_r, right = self._gfql_eval_expr_ast(table_df, node.right)
            if not (ok_l and ok_r):
                return False, None
            op = str(node.op).lower()

            if op == "or":
                return True, self._gfql_bool_mask(table_df, left) | self._gfql_bool_mask(table_df, right)
            if op == "and":
                return True, self._gfql_bool_mask(table_df, left) & self._gfql_bool_mask(table_df, right)

            if op in {"=", "!=", "<>", "<", "<=", ">", ">="}:
                left_null_mask = self._gfql_null_mask(table_df, left)
                right_null_mask = self._gfql_null_mask(table_df, right)
                any_null_mask = left_null_mask | right_null_mask
                if op == "=":
                    out = left == right
                elif op in {"!=", "<>"}:
                    out = left != right
                elif op == "<":
                    out = left < right
                elif op == "<=":
                    out = left <= right
                elif op == ">":
                    out = left > right
                else:
                    out = left >= right
                if hasattr(out, "where"):
                    out = out.where(~any_null_mask, pd.NA)
                return True, out

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
                    return True, list(left) + [right]
                if isinstance(right, (list, tuple)) and not isinstance(left, (list, tuple)):
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
                return True, left / right
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
            values: List[Any] = []
            for arg in node.args:
                ok, val = self._gfql_eval_expr_ast(table_df, arg)
                if not ok:
                    return False, None
                values.append(val)

            if fn == "size" and len(values) == 1:
                inner = values[0]
                if hasattr(inner, "str") and hasattr(inner.str, "len"):
                    return True, inner.str.len()
                try:
                    return True, len(inner)
                except Exception:
                    return False, None

            if fn == "abs" and len(values) == 1:
                inner = values[0]
                if hasattr(inner, "abs"):
                    return True, inner.abs()
                return True, abs(inner)

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
                if RowPipelineMixin._gfql_is_null_scalar(inner):
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
                    return True, out.where(~null_mask, None)
                if RowPipelineMixin._gfql_is_null_scalar(inner):
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
                if RowPipelineMixin._gfql_is_null_scalar(inner):
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
                    return True, RowPipelineMixin._gfql_eval_sequence_fn_series(inner, fn, f"ast {fn}")
                return True, RowPipelineMixin._gfql_eval_sequence_fn_scalar(inner, fn, f"ast {fn}")

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
            if hasattr(base[list_col], "str") and hasattr(base[list_col].str, "len"):
                total_series = base[list_col].str.len()
            else:
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
                unknown = (true_count == 0) & (null_count > 0)
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
            if hasattr(base[list_col], "str") and hasattr(base[list_col].str, "len"):
                lengths = base[list_col].str.len()
            else:
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
            result = (
                merged[out_col].copy()
                if out_col in merged.columns
                else self._gfql_broadcast_scalar(merged, None)
            )

            empty_mask = merged[len_col] == 0
            if hasattr(empty_mask, "any") and bool(empty_mask.any()):
                empty_idx = result.index[empty_mask]
                empty_vals: Any = pd.Series([[] for _ in range(len(empty_idx))], index=empty_idx, dtype="object")
                result.loc[empty_idx] = empty_vals

            if hasattr(null_mask, "any") and bool(null_mask.any()):
                result.loc[result.index[null_mask]] = None

            return True, result.reset_index(drop=True)

        if isinstance(node, SubscriptExpr):
            ok_base, base_value = self._gfql_eval_expr_ast(table_df, node.value)
            ok_key, key_value = self._gfql_eval_expr_ast(table_df, node.key)
            if not (ok_base and ok_key):
                return False, None
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
    _GFQL_LIST_NUMERIC_TEXT_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
    _GFQL_TIME_TEXT_RE = re.compile(
        r"^(?P<h>\d{2}):(?P<m>\d{2})"
        r"(?::(?P<s>\d{2})(?:\.(?P<f>\d{1,9}))?)?"
        r"(?:(?P<off_sign>[+-])(?P<off_h>\d{2}):(?P<off_m>\d{2}))?$"
    )
    _GFQL_DATETIME_TEXT_RE = re.compile(
        r"^(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})T"
        r"(?P<h>\d{2}):(?P<m>\d{2})"
        r"(?::(?P<s>\d{2})(?:\.(?P<f>\d{1,9}))?)?"
        r"(?:(?P<off_sign>[+-])(?P<off_h>\d{2}):(?P<off_m>\d{2}))?$"
    )

    @staticmethod
    def _gfql_is_null_scalar(value: Any) -> bool:
        if value is None:
            return True
        try:
            marker = pd.isna(value)
        except Exception:
            return False
        return bool(marker) if isinstance(marker, bool) else False

    @staticmethod
    def _gfql_is_nan_scalar(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        try:
            return math.isnan(value)
        except Exception:
            return False

    @staticmethod
    def _gfql_order_expr_static_supported(expr: str) -> bool:
        txt = expr.strip()
        if txt == "":
            return False
        if re.search(r"[\[\]{}]", txt):
            return False
        if re.search(r"(?i)\b(?:ANY|ALL|NONE|SINGLE)\s*\(", txt):
            return False
        func_calls = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", txt)
        if any(fn.lower() not in RowPipelineMixin._GFQL_ORDER_SAFE_FUNCS for fn in func_calls):
            return False
        if re.fullmatch(r"[A-Za-z0-9_.'\"+\-*/%<>=!(),\s]+", txt) is None:
            return False
        return True

    @staticmethod
    def _gfql_order_value_family(value: Any) -> Optional[str]:
        if RowPipelineMixin._gfql_is_null_scalar(value) or RowPipelineMixin._gfql_is_nan_scalar(value):
            return None
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, str):
            return "str"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, (datetime.datetime, datetime.date, datetime.time, pd.Timestamp)):
            return "datetime"
        type_name = type(value).__name__.lower()
        if "datetime64" in type_name or "timedelta64" in type_name:
            return "datetime"
        if isinstance(value, (list, tuple, dict, set)):
            return "unsupported"
        return "unsupported"

    @staticmethod
    def _gfql_validate_order_series_vector_safe(series: Any, expr: str) -> None:
        dtype_txt = str(getattr(series, "dtype", "")).lower()
        if dtype_txt != "object":
            return
        non_null = series.dropna()
        sample = non_null.head(128)
        if hasattr(sample, "to_pandas"):
            sample = sample.to_pandas()
        if hasattr(sample, "tolist"):
            values = sample.tolist()
        else:
            values = list(sample)
        families = {
            fam
            for fam in (RowPipelineMixin._gfql_order_value_family(v) for v in values)
            if fam is not None
        }
        if len(families) == 0:
            return
        if "unsupported" in families or len(families) > 1:
            fams = ", ".join(sorted(families))
            raise ValueError(
                "unsupported order_by expression for vectorized execution; "
                f"mixed/dynamic value families ({fams}) in {expr!r}"
            )

    @staticmethod
    def _gfql_order_sample_values(series: Any) -> List[Any]:
        sample = series.dropna().head(128)
        if hasattr(sample, "to_pandas"):
            sample = sample.to_pandas()
        if hasattr(sample, "tolist"):
            return list(sample.tolist())
        return list(sample)

    @staticmethod
    def _gfql_order_detect_list_series(series: Any) -> bool:
        sample_values = RowPipelineMixin._gfql_order_sample_values(series)
        return len(sample_values) > 0 and all(isinstance(v, (list, tuple)) for v in sample_values)

    @staticmethod
    def _gfql_order_detect_temporal_mode(series: Any) -> Optional[str]:
        sample_values = RowPipelineMixin._gfql_order_sample_values(series)
        if len(sample_values) == 0:
            return None
        if not all(isinstance(v, str) for v in sample_values):
            return None
        if all(RowPipelineMixin._GFQL_DATETIME_TEXT_RE.fullmatch(v) is not None for v in sample_values):
            return "datetime"
        if all(RowPipelineMixin._GFQL_TIME_TEXT_RE.fullmatch(v) is not None for v in sample_values):
            return "time"
        return None

    def _gfql_build_list_sort_columns(
        self: _RowPipelineContext,
        work_df: Any,
        sort_col: str,
        key_prefix: str,
    ) -> Tuple[Any, List[str]]:
        row_col = f"{key_prefix}__row"
        while row_col in work_df.columns:
            row_col = f"{row_col}_x"
        list_col = f"{key_prefix}__list"
        while list_col in work_df.columns:
            list_col = f"{list_col}_x"
        len_col = f"{key_prefix}__len"
        while len_col in work_df.columns:
            len_col = f"{len_col}_x"
        pos_col = f"{key_prefix}__pos"
        while pos_col in work_df.columns:
            pos_col = f"{pos_col}_x"
        tok_col = f"{key_prefix}__tok"
        while tok_col in work_df.columns:
            tok_col = f"{tok_col}_x"

        base = work_df.assign(**{row_col: range(len(work_df)), list_col: work_df[sort_col]})[[row_col, list_col]]
        if not hasattr(base[list_col], "str") or not hasattr(base[list_col].str, "len"):
            raise ValueError("order_by list sorting requires string/list accessor support")
        lengths = base[list_col].str.len()
        base = base.assign(**{len_col: lengths})
        expanded = base[[row_col, list_col, len_col]].explode(list_col)
        if len(expanded) > 0:
            expanded = expanded.assign(
                **{pos_col: expanded.groupby(row_col, sort=False).cumcount()}
            )
            keep = self._gfql_null_mask(expanded, expanded[len_col]) | (
                expanded[pos_col] < expanded[len_col]
            )
            expanded = expanded.loc[keep]

        if len(expanded) == 0:
            key_cols = [f"{key_prefix}_0"]
            key_frame = base[[row_col]].assign(**{key_cols[0]: ""})
        else:
            value = expanded[list_col]
            value_str = value.astype(str)
            null_mask = self._gfql_null_mask(expanded, value)
            lower_str = value_str.str.lower() if hasattr(value_str, "str") else value_str
            bool_mask = (~null_mask) & lower_str.isin(["true", "false"])
            num_mask = (~null_mask) & (~bool_mask) & value_str.str.fullmatch(
                RowPipelineMixin._GFQL_LIST_NUMERIC_TEXT_RE.pattern
            )
            str_mask = (~null_mask) & (~bool_mask) & (~num_mask)

            num_values = value_str.where(num_mask, None).astype("float64")
            num_rank = num_values.rank(method="dense")
            str_values = value_str.where(str_mask, None)
            str_rank = str_values.rank(method="dense")

            token = self._gfql_broadcast_scalar(expanded, "9:000000000000")
            if hasattr(str_mask, "any") and bool(str_mask.any()):
                str_token = "5:" + str_rank.fillna(0).astype("int64").astype(str).str.zfill(12)
                token = token.where(~str_mask, str_token)
            if hasattr(num_mask, "any") and bool(num_mask.any()):
                num_token = "7:" + num_rank.fillna(0).astype("int64").astype(str).str.zfill(12)
                token = token.where(~num_mask, num_token)
            if hasattr(bool_mask, "any") and bool(bool_mask.any()):
                bool_token = "6:" + lower_str.where(bool_mask, "false").replace(
                    {"false": "0", "true": "1"}
                )
                token = token.where(~bool_mask, bool_token)

            expanded = expanded.assign(**{tok_col: token})
            key_wide = expanded.pivot(index=row_col, columns=pos_col, values=tok_col).sort_index(axis=1)
            key_wide = key_wide.reset_index()
            rename_map: Dict[Any, str] = {}
            for col in key_wide.columns:
                if col == row_col:
                    continue
                rename_map[col] = f"{key_prefix}_{int(col)}"
            key_wide = key_wide.rename(columns=rename_map)
            key_cols = [col for col in key_wide.columns if col != row_col]
            key_frame = base[[row_col]].merge(key_wide, on=row_col, how="left", sort=False)
            for col in key_cols:
                key_frame[col] = key_frame[col].fillna("")

        merged = work_df.assign(**{row_col: range(len(work_df))}).merge(
            key_frame[[row_col] + key_cols],
            on=row_col,
            how="left",
            sort=False,
        )
        merged = merged.drop(columns=[row_col])
        return merged, key_cols

    def _gfql_build_temporal_sort_columns(
        self: _RowPipelineContext,
        work_df: Any,
        sort_col: str,
        key_prefix: str,
        mode: str,
    ) -> Tuple[Any, List[str]]:
        value = work_df[sort_col]
        text = value.astype(str)
        null_mask = self._gfql_null_mask(work_df, value)
        if mode == "datetime":
            parts = text.str.extract(RowPipelineMixin._GFQL_DATETIME_TEXT_RE)
        else:
            parts = text.str.extract(RowPipelineMixin._GFQL_TIME_TEXT_RE)

        hour = parts["h"].fillna("0").astype("int64")
        minute = parts["m"].fillna("0").astype("int64")
        second = parts["s"].fillna("0").astype("int64")
        frac = parts["f"].fillna("").str.pad(9, side="right", fillchar="0").replace("", "0")
        nanos = frac.astype("int64")
        off_sign = parts["off_sign"].fillna("+")
        off_hours = parts["off_h"].fillna("0").astype("int64")
        off_minutes = parts["off_m"].fillna("0").astype("int64")
        sign_mult = off_sign.eq("-").astype("int64")
        sign_mult = sign_mult.where(sign_mult == 0, -1)
        sign_mult = sign_mult.where(sign_mult != 0, 1)
        offset_total_minutes = sign_mult * (off_hours * 60 + off_minutes)
        time_nanos = (
            (hour * 3600 + minute * 60 + second) * 1_000_000_000
            + nanos
            - offset_total_minutes * 60 * 1_000_000_000
        )

        if mode == "time":
            key_col = f"{key_prefix}_time_ns"
            while key_col in work_df.columns:
                key_col = f"{key_col}_x"
            out = work_df.assign(
                **{key_col: time_nanos.where(~null_mask, 9_223_372_036_854_775_000)}
            )
            return out, [key_col]

        year = parts["y"].fillna("0").astype("int64")
        month = parts["mo"].fillna("1").astype("int64")
        day = parts["d"].fillna("1").astype("int64")
        a = (14 - month) // 12
        y2 = year + 4800 - a
        m2 = month + 12 * a - 3
        julian_day = (
            day
            + ((153 * m2 + 2) // 5)
            + (365 * y2)
            + (y2 // 4)
            - (y2 // 100)
            + (y2 // 400)
            - 32045
        )
        day_nanos = 86_400 * 1_000_000_000
        day_adjust = time_nanos // day_nanos
        nanos_of_day = time_nanos - (day_adjust * day_nanos)
        day_key = julian_day + day_adjust
        day_col = f"{key_prefix}_day"
        while day_col in work_df.columns:
            day_col = f"{day_col}_x"
        nanos_col = f"{key_prefix}_ns"
        while nanos_col in work_df.columns:
            nanos_col = f"{nanos_col}_x"
        out = work_df.assign(
            **{
                day_col: day_key.where(~null_mask, 9_223_372_036_854_775_000),
                nanos_col: nanos_of_day.where(~null_mask, day_nanos + 1),
            }
        )
        return out, [day_col, nanos_col]

    @staticmethod
    def _gfql_series_is_list_like(series: Any) -> bool:
        if not hasattr(series, "dropna"):
            return False
        sample = series.dropna().head(128)
        if hasattr(sample, "to_pandas"):
            sample = sample.to_pandas()
        values = sample.tolist() if hasattr(sample, "tolist") else list(sample)
        return len(values) > 0 and all(isinstance(v, (list, tuple)) for v in values)

    @staticmethod
    def _gfql_series_scalar_if_constant(series: Any) -> Tuple[bool, Any]:
        if not hasattr(series, "dropna"):
            return True, series
        non_null = series.dropna()
        if len(non_null) == 0:
            return True, None
        sample = non_null.head(128)
        if hasattr(sample, "to_pandas"):
            sample = sample.to_pandas()
        values = sample.tolist() if hasattr(sample, "tolist") else list(sample)
        if len(values) == 0:
            return True, None
        first = values[0]
        if all(v == first for v in values):
            return True, first
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
    def _gfql_eval_sequence_fn_series(series_value: Any, fn_name: str, expr: str) -> Any:
        try:
            if fn_name == "head":
                return series_value.str.get(0)
            if fn_name == "tail":
                return series_value.str.slice(start=1)
            if fn_name == "reverse":
                return series_value.str[::-1]
        except Exception as exc:
            raise ValueError(
                f"unsupported row expression: {fn_name}() requires list/string input in {expr!r}"
            ) from exc
        raise ValueError(f"unsupported row expression function: {fn_name} in {expr!r}")

    @staticmethod
    def _gfql_eval_sequence_fn_scalar(value: Any, fn_name: str, expr: str) -> Any:
        if RowPipelineMixin._gfql_is_null_scalar(value):
            return None
        try:
            if fn_name == "head":
                return value[0] if len(value) > 0 else None
            if fn_name == "tail":
                return value[1:]
            if fn_name == "reverse":
                if isinstance(value, str):
                    return value[::-1]
                if isinstance(value, (list, tuple)):
                    return list(reversed(value))
            raise ValueError(f"unsupported row expression function: {fn_name} in {expr!r}")
        except Exception as exc:
            raise ValueError(
                f"unsupported row expression: {fn_name}() requires list/string input in {expr!r}"
            ) from exc

    def _gfql_eval_string_predicate_expr(
        self: _RowPipelineContext,
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

            if op_name == "contains":
                out = left_txt.str.contains(needle, regex=False)
            elif op_name == "starts_with":
                out = left_txt.str.startswith(needle)
            elif op_name == "ends_with":
                out = left_txt.str.endswith(needle)
            else:
                raise ValueError(f"unsupported row expression predicate op: {op_name} in {expr!r}")

            if hasattr(out, "where"):
                return out.where(~any_null, pd.NA)
            return out

        if RowPipelineMixin._gfql_is_null_scalar(left) or RowPipelineMixin._gfql_is_null_scalar(right):
            return None
        left_txt = str(left)
        right_txt = str(right)
        if op_name == "contains":
            return right_txt in left_txt
        if op_name == "starts_with":
            return left_txt.startswith(right_txt)
        if op_name == "ends_with":
            return left_txt.endswith(right_txt)
        raise ValueError(f"unsupported row expression predicate op: {op_name} in {expr!r}")

    def _gfql_broadcast_scalar(self: _RowPipelineContext, table_df: Any, value: Any) -> Any:
        tmp_col = "__gfql_tmp_scalar__"
        while tmp_col in table_df.columns:
            tmp_col = f"{tmp_col}_x"

        # Treat list/map literals as scalar row values by explicit broadcasting.
        # Plain `assign(col=[...])` interprets list values as column vectors.
        if isinstance(value, (list, tuple, dict)):
            return table_df.assign(**{tmp_col: [value for _ in range(len(table_df))]})[tmp_col]

        return table_df.assign(**{tmp_col: value})[tmp_col]

    def _gfql_bool_mask(self: _RowPipelineContext, table_df: Any, value: Any) -> Any:
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

    def _gfql_null_mask(self: _RowPipelineContext, table_df: Any, value: Any) -> Any:
        if hasattr(value, "isna"):
            return value.isna()
        try:
            marker = pd.isna(value)
        except Exception:
            marker = RowPipelineMixin._gfql_is_null_scalar(value)
        if isinstance(marker, bool):
            return self._gfql_broadcast_scalar(table_df, marker).astype(bool)
        return self._gfql_broadcast_scalar(
            table_df,
            RowPipelineMixin._gfql_is_null_scalar(value),
        ).astype(bool)

    def _gfql_resolve_token(self: _RowPipelineContext, table_df: Any, token: str) -> Any:
        txt = token.strip()
        if txt in table_df.columns:
            return table_df[txt]
        prop_match = RowPipelineMixin._GFQL_ALIAS_PROP_RE.fullmatch(txt)
        if prop_match is not None:
            alias = prop_match.group("alias")
            prop = prop_match.group("prop")
            if alias in table_df.columns and hasattr(table_df[alias], "str"):
                try:
                    return table_df[alias].str.get(prop)
                except Exception:
                    pass
            if prop in table_df.columns:
                return table_df[prop]
        raise ValueError(f"unsupported token in row expression: {token!r}")

    def _gfql_eval_dynamic_list_subscript(
        self: _RowPipelineContext,
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

        non_null_base = base_value.dropna()
        sample = non_null_base.head(128)
        if hasattr(sample, "to_pandas"):
            sample = sample.to_pandas()
        sample_values = sample.tolist() if hasattr(sample, "tolist") else list(sample)
        if any(not isinstance(v, (list, tuple)) for v in sample_values):
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
            non_null_keys = key_value.dropna()
            key_sample = non_null_keys.head(128)
            if hasattr(key_sample, "to_pandas"):
                key_sample = key_sample.to_pandas()
            key_values = key_sample.tolist() if hasattr(key_sample, "tolist") else list(key_sample)
            if any((not isinstance(v, int)) or isinstance(v, bool) for v in key_values):
                raise ValueError(
                    f"unsupported row expression: dynamic subscript keys must be integer typed in {expr!r}"
                )

        row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_dynsub_row__")
        base_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_dynsub_base__")
        key_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_dynsub_key__")
        pos_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_dynsub_pos__")

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
        self: _RowPipelineContext,
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
        if not hasattr(base_value, "str"):
            raise ValueError(f"unsupported row expression: slice subscript requires list/string base in {expr!r}")

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

        if (start_present and RowPipelineMixin._gfql_is_null_scalar(start_value)) or (
            end_present and RowPipelineMixin._gfql_is_null_scalar(end_value)
        ):
            return self._gfql_broadcast_scalar(table_df, None)

        def _coerce_bound(v: Any, label: str) -> Optional[int]:
            if RowPipelineMixin._gfql_is_null_scalar(v):
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

        try:
            return base_value.str.slice(start=start_i, stop=end_i)
        except Exception as exc:
            raise ValueError(
                f"unsupported row expression: slice subscript failed for {expr!r}: {exc}"
            ) from exc

    def _gfql_concat_list_scalar(
        self: _RowPipelineContext,
        table_df: Any,
        list_series: Any,
        scalar_series: Any,
        prepend: bool = False,
    ) -> Any:
        row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_add_row__")
        list_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_add_list__")
        val_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_add_val__")
        pos_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_add_pos__")
        len_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_list_add_len__")

        base = table_df.assign(**{row_col: range(len(table_df)), list_col: list_series, val_col: scalar_series})
        null_mask = self._gfql_null_mask(base, base[list_col])
        if hasattr(base[list_col], "str") and hasattr(base[list_col].str, "len"):
            lengths = base[list_col].str.len().fillna(0)
        else:
            raise ValueError("unsupported row expression: list concatenation requires list/string accessor support")
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
            combined = pd.concat([append_rows, expanded], ignore_index=True, sort=False)
        else:
            combined = pd.concat([expanded, append_rows], ignore_index=True, sort=False)
        if len(combined) > 0:
            combined = combined.sort_values(by=[row_col, pos_col], kind="mergesort")
            grouped = combined.groupby(row_col, sort=False)[val_col].agg(list).reset_index()
        else:
            grouped = non_null[[row_col]].iloc[0:0].copy()
            grouped[val_col] = []

        out = base[[row_col, len_col]].merge(grouped, on=row_col, how="left", sort=False)[val_col]
        empty_mask = base[len_col] == 0
        if hasattr(empty_mask, "any") and bool(empty_mask.any()):
            idx = out.index[empty_mask]
            empty_vals: Any = pd.Series([[] for _ in range(len(idx))], index=idx, dtype="object")
            out.loc[idx] = empty_vals
        out = out.where(~null_mask, None)
        return out.reset_index(drop=True)

    def _gfql_eval_in_expr(
        self: _RowPipelineContext,
        table_df: Any,
        left_value: Any,
        right_value: Any,
        expr: str,
    ) -> Any:
        left_series = left_value if hasattr(left_value, "astype") else self._gfql_broadcast_scalar(table_df, left_value)
        right_series = right_value if hasattr(right_value, "astype") else self._gfql_broadcast_scalar(table_df, right_value)

        if not hasattr(right_series, "str") or not hasattr(right_series.str, "len"):
            raise ValueError(f"unsupported row expression: IN rhs must be list-like in {expr!r}")

        row_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_in_row__")
        rhs_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_in_rhs__")
        lhs_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_in_lhs__")
        len_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_in_len__")
        pos_col = RowPipelineMixin._gfql_fresh_col_name(table_df.columns, "__gfql_in_pos__")

        base = table_df.assign(**{row_col: range(len(table_df)), lhs_col: left_series, rhs_col: right_series})
        rhs_null = self._gfql_null_mask(base, base[rhs_col])
        rhs_len = base[rhs_col].str.len().fillna(0).astype("int64")
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
        summary = summary.assign(
            __gfql_in_true__=summary["__gfql_in_true__"].fillna(0),
            __gfql_in_unknown__=summary["__gfql_in_unknown__"].fillna(0),
        )

        out = summary["__gfql_in_true__"] > 0
        unknown_mask = (summary["__gfql_in_true__"] == 0) & (summary["__gfql_in_unknown__"] > 0)
        out = out.where(~unknown_mask, pd.NA)
        out = out.where(~rhs_null, pd.NA)
        return out.reset_index(drop=True)

    def _gfql_eval_string_expr(self: _RowPipelineContext, table_df: Any, expr: str) -> Any:
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
            raise ValueError(f"unsupported row expression: parser validation failed in {expr!r}")

        try:
            ast_ok, ast_value = self._gfql_eval_expr_ast(table_df, ast_node)
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"unsupported row expression: AST evaluator unsupported in {expr!r}") from exc

        if ast_ok:
            return ast_value

        raise ValueError(f"unsupported row expression: AST evaluator unsupported in {expr!r}")

    def _gfql_row_table(self: _RowPipelineContext, table_df: Any) -> "Plottable":
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
        out._edge = None
        if out._node is not None and out._node not in table_df.columns:
            out._node = None
        return out

    def _gfql_get_active_table(self: _RowPipelineContext) -> Any:
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
        self: _RowPipelineContext, table: str = "nodes", source: Optional[str] = None
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
            if hasattr(mask, "fillna"):
                mask = mask.fillna(False)
            table_df = table_df.loc[mask.astype(bool)]

        return self._gfql_row_table(table_df)

    def select(self: _RowPipelineContext, items: List[Any]) -> "Plottable":
        table_df = self._gfql_get_active_table()
        if items is None:
            raise ValueError(
                "select(items=...) requires entries of form (alias, expr) or shorthand 'col'"
            )

        projected: Dict[str, Any] = {}
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
                value = self._gfql_eval_string_expr(table_df, expr)
                if not hasattr(value, "astype"):
                    value = self._gfql_broadcast_scalar(table_df, value)
                projected[alias] = value
            else:
                projected[alias] = self._gfql_broadcast_scalar(table_df, expr)

        out_df = table_df.assign(**projected)[list(projected.keys())]
        return self._gfql_row_table(out_df)

    def with_(self: _RowPipelineContext, items: List[Any]) -> "Plottable":
        """Python-safe alias for Cypher WITH-style row projection."""
        return self.select(items)

    def where_rows(
        self: _RowPipelineContext,
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

    def return_(self: _RowPipelineContext, items: List[Any]) -> "Plottable":
        return self.select(items)

    def order_by(self: _RowPipelineContext, keys: List[Any]) -> "Plottable":
        table_df = self._gfql_get_active_table()
        if keys is None:
            raise ValueError(
                "order_by(keys=...) requires a list/tuple of (expr, direction)"
            )

        sort_cols: List[str] = []
        ascending: List[bool] = []
        work_df = table_df
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
                if not RowPipelineMixin._gfql_order_expr_static_supported(expr):
                    raise ValueError(
                        "unsupported order_by expression in vectorized mode; "
                        f"use column/scalar arithmetic comparisons only: {expr!r}"
                    )
                sort_col = f"__gfql_sort_{tmp_idx}__"
                tmp_idx += 1
                work_df = work_df.assign(**{sort_col: self._gfql_eval_string_expr(work_df, expr)})
            direction_is_asc = str(direction).lower() != "desc"
            series = work_df[sort_col]
            list_candidate = RowPipelineMixin._gfql_order_detect_list_series(series)
            if list_candidate:
                top_null_mask = self._gfql_null_mask(work_df, series)
                if hasattr(top_null_mask, "any") and bool(top_null_mask.any()):
                    list_candidate = False
            if list_candidate:
                key_prefix = f"__gfql_sort_list_{tmp_idx}__"
                tmp_idx += 1
                work_df, list_key_cols = self._gfql_build_list_sort_columns(
                    work_df, sort_col, key_prefix
                )
                sort_cols.extend(list_key_cols)
                ascending.extend([direction_is_asc] * len(list_key_cols))
                continue
            temporal_mode = RowPipelineMixin._gfql_order_detect_temporal_mode(series)
            if temporal_mode is not None:
                key_prefix = f"__gfql_sort_temporal_{tmp_idx}__"
                tmp_idx += 1
                work_df, temporal_key_cols = self._gfql_build_temporal_sort_columns(
                    work_df, sort_col, key_prefix, temporal_mode
                )
                sort_cols.extend(temporal_key_cols)
                ascending.extend([direction_is_asc] * len(temporal_key_cols))
                continue
            RowPipelineMixin._gfql_validate_order_series_vector_safe(series, expr)
            sort_cols.append(sort_col)
            ascending.append(direction_is_asc)

        if sort_cols:
            try:
                out_df = work_df.sort_values(by=sort_cols, ascending=ascending)
            except Exception as exc:
                raise ValueError(
                    "unsupported order_by for vectorized execution; "
                    f"cannot sort key set {sort_cols!r} with expression set "
                    f"{[k[0] for k in keys]!r}: {exc}"
                ) from exc
            drop_cols = [c for c in sort_cols if isinstance(c, str) and c.startswith("__gfql_sort_")]
            if drop_cols:
                out_df = out_df.drop(columns=drop_cols)
        else:
            out_df = work_df
        return self._gfql_row_table(out_df)

    def skip(self: _RowPipelineContext, value: Any) -> "Plottable":
        table_df = self._gfql_get_active_table()
        skip_count = self._gfql_coerce_non_negative_int(value, "skip")
        return self._gfql_row_table(table_df.iloc[skip_count:])

    def limit(self: _RowPipelineContext, value: Any) -> "Plottable":
        table_df = self._gfql_get_active_table()
        limit_count = self._gfql_coerce_non_negative_int(value, "limit")
        return self._gfql_row_table(table_df.iloc[:limit_count])

    def distinct(self: _RowPipelineContext) -> "Plottable":
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

    def unwind(self: _RowPipelineContext, expr: Any, as_: str = "value") -> "Plottable":
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
        out_df = table_df.assign(**{tmp_col: explode_src})
        if not hasattr(out_df, "explode"):
            raise ValueError("unwind requires dataframe explode() support")
        out_df = out_df.explode(tmp_col)
        out_df = out_df.assign(**{as_clean: out_df[tmp_col]})
        out_df = out_df.drop(columns=[tmp_col])
        return self._gfql_row_table(out_df)

    def group_by(
        self: _RowPipelineContext,
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
        out_df = base[key_cols].copy()

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
                if func == "count":
                    agg_df = grouped[expr_col].count().reset_index(name=alias)
                elif func == "count_distinct":
                    agg_df = grouped[expr_col].nunique().reset_index(name=alias)
                elif func == "sum":
                    agg_df = grouped[expr_col].sum().reset_index(name=alias)
                elif func == "min":
                    agg_df = grouped[expr_col].min().reset_index(name=alias)
                elif func == "max":
                    agg_df = grouped[expr_col].max().reset_index(name=alias)
                elif func in {"avg", "mean"}:
                    agg_df = grouped[expr_col].mean().reset_index(name=alias)
                elif func == "collect":
                    # collect() ignores null entries; compute collection on
                    # non-null rows and merge against full key space below.
                    grouped_df = grouped.obj
                    non_null_df = grouped_df.loc[
                        ~grouped_df[expr_col].isna(), key_cols + [expr_col]
                    ]
                    agg_df = (
                        _make_grouped(non_null_df)[expr_col]
                        .agg(list)
                        .reset_index(name=alias)
                    )
                else:
                    raise ValueError(f"unsupported group_by aggregation function: {func!r}")

            out_df = out_df.merge(agg_df, on=key_cols, how="left")
            if func == "collect":
                out_df[alias] = out_df[alias].where(~out_df[alias].isna(), [[] for _ in range(len(out_df))])

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
