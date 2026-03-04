import ast
import datetime
import math
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence, Tuple

import pandas as pd


if TYPE_CHECKING:
    from graphistry.Plottable import Plottable


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

    @staticmethod
    def _gfql_strip_outer_parens(expr: str) -> str:
        ...

    @staticmethod
    def _gfql_split_top_level_keyword(expr: str, keyword: str) -> Optional[Tuple[str, str]]:
        ...

    @staticmethod
    def _gfql_parse_literal_token(token: str) -> Tuple[bool, Any]:
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
    _GFQL_TOBOOLEAN_RE = re.compile(r"(?is)^toBoolean\s*\((?P<inner>.+)\)$")
    _GFQL_TOSTRING_RE = re.compile(r"(?is)^toString\s*\((?P<inner>.+)\)$")
    _GFQL_COALESCE_RE = re.compile(r"(?is)^coalesce\s*\((?P<inner>.+)\)$")

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
        # Keep order_by deterministic and vector-safe for this cycle by
        # rejecting function-call forms up front.
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", txt):
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
    def _gfql_strip_outer_parens(expr: str) -> str:
        txt = expr.strip()
        while txt.startswith("(") and txt.endswith(")"):
            depth = 0
            in_single = False
            in_double = False
            balanced = True
            for idx, ch in enumerate(txt):
                if ch == "'" and not in_double:
                    in_single = not in_single
                    continue
                if ch == '"' and not in_single:
                    in_double = not in_double
                    continue
                if in_single or in_double:
                    continue
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and idx < len(txt) - 1:
                        balanced = False
                        break
            if not balanced or depth != 0:
                break
            txt = txt[1:-1].strip()
        return txt

    @staticmethod
    def _gfql_split_top_level_keyword(expr: str, keyword: str) -> Optional[Tuple[str, str]]:
        txt = expr
        upper = txt.upper()
        needle = keyword.upper()
        depth_paren = 0
        depth_bracket = 0
        depth_brace = 0
        in_single = False
        in_double = False
        idx = 0
        while idx < len(txt):
            ch = txt[idx]
            if ch == "'" and not in_double:
                in_single = not in_single
                idx += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                idx += 1
                continue
            if in_single or in_double:
                idx += 1
                continue
            if ch == "(":
                depth_paren += 1
                idx += 1
                continue
            if ch == ")":
                depth_paren = max(0, depth_paren - 1)
                idx += 1
                continue
            if ch == "[":
                depth_bracket += 1
                idx += 1
                continue
            if ch == "]":
                depth_bracket = max(0, depth_bracket - 1)
                idx += 1
                continue
            if ch == "{":
                depth_brace += 1
                idx += 1
                continue
            if ch == "}":
                depth_brace = max(0, depth_brace - 1)
                idx += 1
                continue
            if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0 and upper.startswith(needle, idx):
                left_ok = idx == 0 or not (upper[idx - 1].isalnum() or upper[idx - 1] == "_")
                right_idx = idx + len(needle)
                right_ok = right_idx >= len(upper) or not (upper[right_idx].isalnum() or upper[right_idx] == "_")
                if left_ok and right_ok:
                    left = txt[:idx].strip()
                    right = txt[right_idx:].strip()
                    if left and right:
                        return left, right
            idx += 1
        return None

    @staticmethod
    def _gfql_split_top_level_operator(
        expr: str, operators: Sequence[str]
    ) -> Optional[Tuple[str, str, str]]:
        txt = expr
        depth_paren = 0
        depth_bracket = 0
        depth_brace = 0
        in_single = False
        in_double = False
        idx = 0
        while idx < len(txt):
            ch = txt[idx]
            if ch == "'" and not in_double:
                in_single = not in_single
                idx += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                idx += 1
                continue
            if in_single or in_double:
                idx += 1
                continue
            if ch == "(":
                depth_paren += 1
                idx += 1
                continue
            if ch == ")":
                depth_paren = max(0, depth_paren - 1)
                idx += 1
                continue
            if ch == "[":
                depth_bracket += 1
                idx += 1
                continue
            if ch == "]":
                depth_bracket = max(0, depth_bracket - 1)
                idx += 1
                continue
            if ch == "{":
                depth_brace += 1
                idx += 1
                continue
            if ch == "}":
                depth_brace = max(0, depth_brace - 1)
                idx += 1
                continue
            if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
                for op in operators:
                    if not txt.startswith(op, idx):
                        continue

                    # Prevent treating unary +/- as binary operators.
                    if op in {"+", "-"}:
                        prev_idx = idx - 1
                        while prev_idx >= 0 and txt[prev_idx].isspace():
                            prev_idx -= 1
                        if prev_idx < 0 or txt[prev_idx] in {"(", "[", "{", ",", "+", "-", "*", "/", "%", "=", "<", ">"}:
                            continue

                    left = txt[:idx].strip()
                    right = txt[idx + len(op) :].strip()
                    if left and right:
                        return left, op, right
            idx += 1
        return None

    @staticmethod
    def _gfql_split_top_level_commas(expr: str) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        depth_paren = 0
        depth_bracket = 0
        depth_brace = 0
        in_single = False
        in_double = False
        escaped = False

        for ch in expr:
            if in_single or in_double:
                current.append(ch)
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if in_single and ch == "'":
                    in_single = False
                elif in_double and ch == '"':
                    in_double = False
                continue

            if ch == "'":
                in_single = True
                current.append(ch)
                continue
            if ch == '"':
                in_double = True
                current.append(ch)
                continue

            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket = max(0, depth_bracket - 1)
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace = max(0, depth_brace - 1)

            if ch == "," and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(ch)

        part = "".join(current).strip()
        if part:
            parts.append(part)
        return parts

    @staticmethod
    def _gfql_parse_quantifier_expr(expr: str) -> Optional[Tuple[str, str, str, str]]:
        match = RowPipelineMixin._GFQL_QUANTIFIER_RE.fullmatch(expr.strip())
        if match is None:
            return None

        fn = match.group("fn").lower()
        body = match.group("body").strip()
        in_split = RowPipelineMixin._gfql_split_top_level_keyword(body, "IN")
        if in_split is None:
            return None
        var = in_split[0].strip()
        if RowPipelineMixin._GFQL_IDENT_RE.fullmatch(var) is None:
            return None
        where_split = RowPipelineMixin._gfql_split_top_level_keyword(in_split[1], "WHERE")
        if where_split is None:
            return None
        list_expr = where_split[0].strip()
        predicate_expr = where_split[1].strip()
        if list_expr == "" or predicate_expr == "":
            return None
        return fn, var, list_expr, predicate_expr

    @staticmethod
    def _gfql_replace_identifier(expr: str, identifier: str, replacement: str) -> str:
        return re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
            replacement,
            expr,
        )

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
            if hasattr(mask, "fillna"):
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

    @staticmethod
    def _gfql_parse_literal_token(token: str) -> Tuple[bool, Any]:
        txt = token.strip()
        if txt == "":
            return False, None
        low = txt.lower()
        if low == "null":
            return True, None
        if low == "true":
            return True, True
        if low == "false":
            return True, False
        if len(txt) >= 2 and txt[0] == txt[-1] and txt[0] in {"'", '"'}:
            return True, txt[1:-1]
        if re.fullmatch(r"-?\d+", txt):
            return True, int(txt)
        if re.fullmatch(r"-?\d+\.\d+", txt):
            return True, float(txt)
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = ast.literal_eval(txt)
            except Exception:
                normalized = re.sub(r"(?i)\bnull\b", "None", txt)
                normalized = re.sub(r"(?i)\btrue\b", "True", normalized)
                normalized = re.sub(r"(?i)\bfalse\b", "False", normalized)
                try:
                    parsed = ast.literal_eval(normalized)
                except Exception:
                    return False, None
            if isinstance(parsed, (list, tuple)):
                return True, list(parsed)
        if txt.startswith("{") and txt.endswith("}"):
            try:
                parsed = ast.literal_eval(txt)
            except Exception:
                normalized = re.sub(r"(?i)\bnull\b", "None", txt)
                normalized = re.sub(r"(?i)\btrue\b", "True", normalized)
                normalized = re.sub(r"(?i)\bfalse\b", "False", normalized)
                try:
                    parsed = ast.literal_eval(normalized)
                except Exception:
                    return False, None
            if isinstance(parsed, dict):
                return True, parsed
        return False, None

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
        is_lit, lit = self._gfql_parse_literal_token(txt)
        if is_lit:
            return lit
        raise ValueError(f"unsupported token in row expression: {token!r}")

    def _gfql_eval_quantifier_expr(
        self: _RowPipelineContext,
        table_df: Any,
        fn: str,
        var: str,
        list_expr: str,
        predicate_expr: str,
    ) -> Any:
        list_value = self._gfql_eval_string_expr(table_df, list_expr)
        list_series = list_value if hasattr(list_value, "astype") else self._gfql_broadcast_scalar(table_df, list_value)

        row_col = "__gfql_q_row__"
        while row_col in table_df.columns:
            row_col = f"{row_col}_x"
        list_col = "__gfql_q_list__"
        while list_col in table_df.columns:
            list_col = f"{list_col}_x"
        total_col = "__gfql_q_total__"
        while total_col in table_df.columns:
            total_col = f"{total_col}_x"
        var_col = f"__gfql_q_{var}__"
        while var_col in table_df.columns:
            var_col = f"{var_col}_x"

        base = table_df.assign(
            **{
                row_col: range(len(table_df)),
                list_col: list_series,
            }
        )
        list_null_mask = self._gfql_null_mask(base, base[list_col])
        if hasattr(base[list_col], "str") and hasattr(base[list_col].str, "len"):
            total_series = base[list_col].str.len()
        else:
            total_series = self._gfql_broadcast_scalar(base, pd.NA)
        if hasattr(total_series, "where"):
            total_series = total_series.where(~total_series.isna(), 1)
            total_series = total_series.where(~list_null_mask, pd.NA)
        base = base.assign(**{total_col: total_series})

        expanded = base[[row_col, list_col, total_col]].explode(list_col)
        expanded = expanded.assign(**{var_col: expanded[list_col]})
        rewritten_predicate = RowPipelineMixin._gfql_replace_identifier(
            predicate_expr, var, var_col
        )
        predicate_value = self._gfql_eval_string_expr(expanded, rewritten_predicate)
        if not hasattr(predicate_value, "astype"):
            predicate_value = self._gfql_broadcast_scalar(expanded, predicate_value)

        predicate_null = self._gfql_null_mask(expanded, predicate_value)
        predicate_bool = self._gfql_bool_mask(expanded, predicate_value)
        predicate_true = predicate_bool & ~predicate_null
        predicate_false = (~predicate_bool) & ~predicate_null

        agg = (
            expanded.assign(
                __gfql_q_true=predicate_true.astype("int64"),
                __gfql_q_false=predicate_false.astype("int64"),
                __gfql_q_null=predicate_null.astype("int64"),
            )
            .groupby(row_col, sort=False)[["__gfql_q_true", "__gfql_q_false", "__gfql_q_null"]]
            .sum()
            .reset_index()
        )
        summary = base[[row_col, total_col]].merge(agg, on=row_col, how="left")
        summary = summary.assign(
            __gfql_q_true=summary["__gfql_q_true"].fillna(0),
            __gfql_q_false=summary["__gfql_q_false"].fillna(0),
            __gfql_q_null=summary["__gfql_q_null"].fillna(0),
        )
        empty_mask = summary[total_col] == 0
        summary.loc[empty_mask, "__gfql_q_true"] = 0
        summary.loc[empty_mask, "__gfql_q_false"] = 0
        summary.loc[empty_mask, "__gfql_q_null"] = 0

        true_count = summary["__gfql_q_true"]
        false_count = summary["__gfql_q_false"]
        null_count = summary["__gfql_q_null"]
        total_count = summary[total_col]
        has_null_list = self._gfql_null_mask(summary, total_count)

        if fn == "any":
            out = true_count > 0
            unknown = (true_count == 0) & (null_count > 0)
        elif fn == "all":
            out = false_count == 0
            unknown = (false_count == 0) & (null_count > 0) & (total_count > 0)
        elif fn == "none":
            out = true_count == 0
            unknown = (true_count == 0) & (null_count > 0) & (total_count > 0)
        elif fn == "single":
            out = true_count == 1
            unknown = (true_count <= 1) & (null_count > 0) & (total_count > 0)
        else:
            raise ValueError(f"unsupported quantifier function: {fn!r}")

        out = out.where(~unknown, pd.NA)
        out = out.where(~has_null_list, pd.NA)
        return out.reset_index(drop=True)

    def _gfql_eval_string_expr(self: _RowPipelineContext, table_df: Any, expr: str) -> Any:
        txt = self._gfql_strip_outer_parens(expr.strip())
        if txt in table_df.columns:
            return table_df[txt]

        label_match = RowPipelineMixin._GFQL_LABEL_PRED_RE.fullmatch(txt)
        if label_match is not None:
            label_col = f"label__{label_match.group('label')}"
            if label_col in table_df.columns:
                return self._gfql_bool_mask(table_df, table_df[label_col])

        is_lit, lit = self._gfql_parse_literal_token(txt)
        if is_lit:
            return lit

        quant_parts = RowPipelineMixin._gfql_parse_quantifier_expr(txt)
        if quant_parts is not None:
            return RowPipelineMixin._gfql_eval_quantifier_expr(self, table_df, *quant_parts)

        size_match = RowPipelineMixin._GFQL_SIZE_RE.fullmatch(txt)
        if size_match is not None:
            inner = self._gfql_eval_string_expr(table_df, size_match.group("inner"))
            if hasattr(inner, "str") and hasattr(inner.str, "len"):
                return inner.str.len()
            try:
                return len(inner)
            except Exception as exc:
                raise ValueError(
                    f"unsupported row expression: size() requires list/string/map value in {expr!r}"
                ) from exc

        abs_match = RowPipelineMixin._GFQL_ABS_RE.fullmatch(txt)
        if abs_match is not None:
            inner = self._gfql_eval_string_expr(table_df, abs_match.group("inner"))
            if hasattr(inner, "abs"):
                return inner.abs()
            return abs(inner)

        to_boolean_match = RowPipelineMixin._GFQL_TOBOOLEAN_RE.fullmatch(txt)
        if to_boolean_match is not None:
            inner = self._gfql_eval_string_expr(table_df, to_boolean_match.group("inner"))
            if hasattr(inner, "astype"):
                null_mask = self._gfql_null_mask(table_df, inner)
                normalized = inner.astype(str).str.strip().str.lower()
                true_mask = normalized.isin(["true", "t", "1", "yes"])
                false_mask = normalized.isin(["false", "f", "0", "no"])
                unsupported_mask = ~(true_mask | false_mask | null_mask)
                if hasattr(unsupported_mask, "any") and bool(unsupported_mask.any()):
                    raise ValueError(f"unsupported row expression: toBoolean() invalid input in {expr!r}")
                out = true_mask.where(~false_mask, False)
                return out.where(~null_mask, pd.NA)
            if RowPipelineMixin._gfql_is_null_scalar(inner):
                return None
            if isinstance(inner, bool):
                return inner
            if isinstance(inner, (int, float)):
                return inner != 0
            txt_inner = str(inner).strip().lower()
            if txt_inner in {"true", "t", "1", "yes"}:
                return True
            if txt_inner in {"false", "f", "0", "no"}:
                return False
            raise ValueError(f"unsupported row expression: toBoolean() invalid input in {expr!r}")

        to_string_match = RowPipelineMixin._GFQL_TOSTRING_RE.fullmatch(txt)
        if to_string_match is not None:
            inner = self._gfql_eval_string_expr(table_df, to_string_match.group("inner"))
            if hasattr(inner, "astype"):
                null_mask = self._gfql_null_mask(table_df, inner)
                out = inner.astype(str)
                if hasattr(out, "str"):
                    out = out.str.replace(r"^True$", "true", regex=True)
                    out = out.str.replace(r"^False$", "false", regex=True)
                return out.where(~null_mask, None)
            if RowPipelineMixin._gfql_is_null_scalar(inner):
                return None
            if isinstance(inner, bool):
                return "true" if inner else "false"
            return str(inner)

        coalesce_match = RowPipelineMixin._GFQL_COALESCE_RE.fullmatch(txt)
        if coalesce_match is not None:
            arg_text = coalesce_match.group("inner").strip()
            arg_parts = RowPipelineMixin._gfql_split_top_level_commas(arg_text)
            if len(arg_parts) == 0:
                raise ValueError(f"unsupported row expression: coalesce() requires arguments in {expr!r}")

            values = [self._gfql_eval_string_expr(table_df, part) for part in arg_parts]
            out = values[0]
            if not hasattr(out, "astype"):
                out = self._gfql_broadcast_scalar(table_df, out)
            for candidate in values[1:]:
                if not hasattr(candidate, "astype"):
                    candidate = self._gfql_broadcast_scalar(table_df, candidate)
                null_mask = self._gfql_null_mask(table_df, out)
                out = out.where(~null_mask, candidate)
            return out

        if txt.startswith("-"):
            inner_txt = txt[1:].strip()
            if inner_txt:
                return 0 - self._gfql_eval_string_expr(table_df, inner_txt)
        if txt.startswith("+"):
            inner_txt = txt[1:].strip()
            if inner_txt:
                return self._gfql_eval_string_expr(table_df, inner_txt)

        subscript_match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_.]*)\s*\[\s*([^\]]+)\s*\]",
            txt,
        )
        if subscript_match is not None:
            base_value = self._gfql_eval_string_expr(table_df, subscript_match.group(1))
            key_value = self._gfql_eval_string_expr(table_df, subscript_match.group(2))
            if hasattr(key_value, "iloc"):
                raise ValueError(
                    f"unsupported row expression: dynamic subscript keys in {expr!r}"
                )
            if hasattr(base_value, "str"):
                return base_value.str.get(key_value)
            try:
                return base_value[key_value]
            except Exception as exc:
                raise ValueError(
                    f"unsupported row expression: subscript failed for {expr!r}: {exc}"
                ) from exc

        is_not_null_match = RowPipelineMixin._GFQL_IS_NOT_NULL_RE.fullmatch(txt)
        if is_not_null_match is not None:
            base = self._gfql_eval_string_expr(table_df, is_not_null_match.group("value"))
            return ~self._gfql_null_mask(table_df, base)

        is_null_match = RowPipelineMixin._GFQL_IS_NULL_RE.fullmatch(txt)
        if is_null_match is not None:
            base = self._gfql_eval_string_expr(table_df, is_null_match.group("value"))
            return self._gfql_null_mask(table_df, base)

        and_split = self._gfql_split_top_level_keyword(txt, "AND")
        if and_split is not None:
            left = self._gfql_bool_mask(table_df, self._gfql_eval_string_expr(table_df, and_split[0]))
            right = self._gfql_bool_mask(table_df, self._gfql_eval_string_expr(table_df, and_split[1]))
            return left & right

        or_split = self._gfql_split_top_level_keyword(txt, "OR")
        if or_split is not None:
            left = self._gfql_bool_mask(table_df, self._gfql_eval_string_expr(table_df, or_split[0]))
            right = self._gfql_bool_mask(table_df, self._gfql_eval_string_expr(table_df, or_split[1]))
            return left | right

        if txt.upper().startswith("NOT "):
            inner = txt[4:].strip()
            inner_mask = self._gfql_bool_mask(table_df, self._gfql_eval_string_expr(table_df, inner))
            return ~inner_mask

        # Comparison operators have lower precedence than arithmetic.
        comp_split = RowPipelineMixin._gfql_split_top_level_operator(
            txt, ["<=", ">=", "<>", "!=", "=", "<", ">"]
        )
        if comp_split is not None:
            left_txt, op, right_txt = comp_split
            left = self._gfql_eval_string_expr(table_df, left_txt)
            right = self._gfql_eval_string_expr(table_df, right_txt)
            left_null_mask = self._gfql_null_mask(table_df, left)
            right_null_mask = self._gfql_null_mask(table_df, right)
            any_null_mask = left_null_mask | right_null_mask
            if op == "=":
                out = left == right
                return out.where(~any_null_mask, pd.NA) if hasattr(out, "where") else out
            if op in {"<>", "!="}:
                out = left != right
                return out.where(~any_null_mask, pd.NA) if hasattr(out, "where") else out
            if op == "<":
                out = left < right
                return out.where(~any_null_mask, pd.NA) if hasattr(out, "where") else out
            if op == "<=":
                out = left <= right
                return out.where(~any_null_mask, pd.NA) if hasattr(out, "where") else out
            if op == ">":
                out = left > right
                return out.where(~any_null_mask, pd.NA) if hasattr(out, "where") else out
            if op == ">=":
                out = left >= right
                return out.where(~any_null_mask, pd.NA) if hasattr(out, "where") else out

        add_split = RowPipelineMixin._gfql_split_top_level_operator(txt, ["+", "-"])
        if add_split is not None:
            left_txt, op, right_txt = add_split
            left = self._gfql_eval_string_expr(table_df, left_txt)
            right = self._gfql_eval_string_expr(table_df, right_txt)
            if op == "+":
                return left + right
            if op == "-":
                return left - right

        mul_split = RowPipelineMixin._gfql_split_top_level_operator(txt, ["*", "/", "%"])
        if mul_split is not None:
            left_txt, op, right_txt = mul_split
            left = self._gfql_eval_string_expr(table_df, left_txt)
            right = self._gfql_eval_string_expr(table_df, right_txt)
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "%":
                return left % right

        m = RowPipelineMixin._GFQL_BIN_OP_RE.match(txt)
        if m:
            left = self._gfql_resolve_token(table_df, m.group("left"))
            right = self._gfql_resolve_token(table_df, m.group("right"))
            op = m.group("op")
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "%":
                return left % right
            if op == "=":
                return left == right
            if op in {"<>", "!="}:
                return left != right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right

        try:
            return self._gfql_resolve_token(table_df, txt)
        except Exception:
            pass

        raise ValueError(f"unsupported row expression: {expr!r}")

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
            raise ValueError("select(items=...) requires a list/tuple of (alias, expr)")

        projected: Dict[str, Any] = {}
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"select expects (alias, expr) pairs, got {item!r}")
            alias_raw, expr = item
            alias = str(alias_raw)
            if isinstance(expr, str):
                projected[alias] = self._gfql_eval_string_expr(table_df, expr)
            else:
                projected[alias] = self._gfql_broadcast_scalar(table_df, expr)

        out_df = table_df.assign(**projected)[list(projected.keys())]
        return self._gfql_row_table(out_df)

    def with_(self: _RowPipelineContext, items: List[Any]) -> "Plottable":
        """Python-safe alias for Cypher WITH-style row projection."""
        return self.select(items)

    def where_rows(
        self: _RowPipelineContext, filter_dict: Optional[Dict[str, Any]] = None
    ) -> "Plottable":
        """Filter active row table with vectorized column/predicate checks."""
        table_df = self._gfql_get_active_table()
        if filter_dict is None:
            filter_dict = {}
        from graphistry.compute.filter_by_dict import filter_by_dict
        out_df = filter_by_dict(table_df, filter_dict)
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
            if not RowPipelineMixin._gfql_order_expr_static_supported(expr):
                raise ValueError(
                    "unsupported order_by expression in vectorized mode; "
                    f"use column/scalar arithmetic comparisons only: {expr!r}"
                )
            if expr in work_df.columns:
                sort_col = expr
            else:
                sort_col = f"__gfql_sort_{tmp_idx}__"
                tmp_idx += 1
                work_df = work_df.assign(**{sort_col: self._gfql_eval_string_expr(work_df, expr)})
            RowPipelineMixin._gfql_validate_order_series_vector_safe(work_df[sort_col], expr)
            sort_cols.append(sort_col)
            ascending.append(str(direction).lower() != "desc")

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

        try:
            grouped = table_df.groupby(key_cols, sort=False, dropna=False)
        except TypeError:
            grouped = table_df.groupby(key_cols, sort=False)

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
                agg_df = grouped.size().reset_index(name=alias)
            else:
                if not isinstance(expr, str):
                    raise ValueError(
                        f"group_by aggregation {alias!r} requires string expr column"
                    )
                if expr not in table_df.columns:
                    raise ValueError(
                        f"group_by aggregation column not found: {expr!r}"
                    )
                if func == "count":
                    agg_df = grouped[expr].count().reset_index(name=alias)
                elif func == "count_distinct":
                    agg_df = grouped[expr].nunique().reset_index(name=alias)
                elif func == "sum":
                    agg_df = grouped[expr].sum().reset_index(name=alias)
                elif func == "min":
                    agg_df = grouped[expr].min().reset_index(name=alias)
                elif func == "max":
                    agg_df = grouped[expr].max().reset_index(name=alias)
                elif func in {"avg", "mean"}:
                    agg_df = grouped[expr].mean().reset_index(name=alias)
                elif func == "collect":
                    # collect() ignores null entries; compute collection on
                    # non-null rows and merge against full key space below.
                    non_null_df = table_df.loc[~table_df[expr].isna(), key_cols + [expr]]
                    agg_df = non_null_df.groupby(key_cols, sort=False, dropna=False)[expr].agg(list).reset_index(name=alias)
                else:
                    raise ValueError(f"unsupported group_by aggregation function: {func!r}")

            out_df = out_df.merge(agg_df, on=key_cols, how="left")
            if func == "collect":
                out_df[alias] = out_df[alias].where(~out_df[alias].isna(), [[] for _ in range(len(out_df))])

        return self._gfql_row_table(out_df)
