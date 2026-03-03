import ast
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
    def _gfql_cypher_sort_key(value: Any) -> Any:
        if RowPipelineMixin._gfql_is_nan_scalar(value):
            return (8, 0)
        if RowPipelineMixin._gfql_is_null_scalar(value):
            return (9, 0)
        if isinstance(value, dict):
            items = tuple((str(k), RowPipelineMixin._gfql_cypher_sort_key(v)) for k, v in sorted(value.items(), key=lambda kv: str(kv[0])))
            return (0, items)
        if isinstance(value, str):
            if value.startswith("<(") and value.endswith(")>"):
                return (4, value)
            if value.startswith("(") and value.endswith(")"):
                return (1, value)
            if value.startswith("[") and value.endswith("]"):
                return (2, value)
            return (5, value)
        if isinstance(value, (list, tuple)):
            nested = tuple(RowPipelineMixin._gfql_cypher_sort_key(v) for v in value)
            return (3, nested)
        if isinstance(value, bool):
            return (6, int(value))
        if isinstance(value, (int, float)):
            return (7, float(value))
        return (10, repr(value))

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
        depth = 0
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
                depth += 1
                idx += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                idx += 1
                continue
            if depth == 0 and upper.startswith(needle, idx):
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
        depth = 0
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
                depth += 1
                idx += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                idx += 1
                continue
            if depth == 0:
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
        return self._gfql_broadcast_scalar(table_df, pd.isna(value)).astype(bool)

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
                return False, None
            if isinstance(parsed, (list, tuple)):
                return True, list(parsed)
        if txt.startswith("{") and txt.endswith("}"):
            try:
                parsed = ast.literal_eval(txt)
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
            prop = prop_match.group("prop")
            if prop in table_df.columns:
                return table_df[prop]
        is_lit, lit = self._gfql_parse_literal_token(txt)
        if is_lit:
            return lit
        raise ValueError(f"unsupported token in row expression: {token!r}")

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
            if op == "=":
                if RowPipelineMixin._gfql_is_null_scalar(right):
                    return self._gfql_null_mask(table_df, left)
                return left == right
            if op in {"<>", "!="}:
                if RowPipelineMixin._gfql_is_null_scalar(right):
                    return ~self._gfql_null_mask(table_df, left)
                return left != right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right

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
            if expr in work_df.columns:
                sort_col = expr
            else:
                sort_col = f"__gfql_sort_{tmp_idx}__"
                tmp_idx += 1
                work_df = work_df.assign(**{sort_col: self._gfql_eval_string_expr(work_df, expr)})
            sort_cols.append(sort_col)
            ascending.append(str(direction).lower() != "desc")

        if sort_cols:
            try:
                out_df = work_df.sort_values(by=sort_cols, ascending=ascending)
            except Exception:
                key_df = work_df
                key_cols: List[str] = []
                for idx, col in enumerate(sort_cols):
                    key_col = f"__gfql_sort_key_{idx}__"
                    key_df = key_df.assign(**{key_col: key_df[col].map(RowPipelineMixin._gfql_cypher_sort_key)})
                    key_cols.append(key_col)
                out_df = key_df.sort_values(by=key_cols, ascending=ascending).drop(columns=key_cols)
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
        return self._gfql_row_table(table_df.drop_duplicates())

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
                else:
                    raise ValueError(f"unsupported group_by aggregation function: {func!r}")

            out_df = out_df.merge(agg_df, on=key_cols, how="left")

        return self._gfql_row_table(out_df)
