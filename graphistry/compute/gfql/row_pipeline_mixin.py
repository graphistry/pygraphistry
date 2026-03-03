from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence

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


class RowPipelineMixin:
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
                if expr not in table_df.columns:
                    raise ValueError(f"select expression column not found: {expr!r}")
                projected[alias] = table_df[expr]
            else:
                projected[alias] = expr

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
        for key_item in keys:
            if not isinstance(key_item, (list, tuple)) or len(key_item) != 2:
                raise ValueError(
                    f"order_by expects (expr, direction) pairs, got {key_item!r}"
                )
            expr, direction = key_item
            if not isinstance(expr, str):
                raise ValueError(
                    "order_by currently supports string column expressions, got "
                    f"{type(expr).__name__}"
                )
            if expr not in table_df.columns:
                raise ValueError(f"order_by column not found: {expr!r}")
            sort_cols.append(expr)
            ascending.append(str(direction).lower() != "desc")

        if sort_cols:
            out_df = table_df.sort_values(by=sort_cols, ascending=ascending)
        else:
            out_df = table_df
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
            if expr not in table_df.columns:
                raise ValueError(f"unwind expression column not found: {expr!r}")
            explode_src = table_df[expr]
        elif isinstance(expr, (list, tuple)):
            explode_src = [list(expr)] * len(table_df)
        else:
            raise ValueError(
                "unwind(expr=...) currently supports a column name or list/tuple literal"
            )

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
