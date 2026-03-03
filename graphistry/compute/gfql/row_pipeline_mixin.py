from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

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
