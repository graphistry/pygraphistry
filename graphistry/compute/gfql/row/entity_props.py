from __future__ import annotations

from typing import Literal, Sequence, cast

import pandas as pd

from graphistry.compute.dataframe_utils import df_cons as template_df_cons
from graphistry.compute.typing import DataFrameT, SeriesT


_NODE_INTERNAL_COLS = frozenset({"id", "labels", "type"})
_EDGE_INTERNAL_COLS = frozenset({"s", "d", "src", "dst", "edge_id", "type", "__gfql_edge_index_0__", "undirected"})


def _fresh_col_name(columns: Sequence[object], prefix: str) -> str:
    existing = {str(col) for col in columns}
    candidate = prefix
    counter = 0
    while candidate in existing:
        counter += 1
        candidate = f"{prefix}{counter}"
    return candidate


def _include_numeric_id_as_property(df: DataFrameT) -> bool:
    if "id" not in df.columns:
        return False
    try:
        return bool(pd.api.types.is_numeric_dtype(df["id"]))
    except Exception:
        return False


def _list_series_from_python_rows(df: DataFrameT, col_name: str, row_count: int) -> SeriesT:
    return cast(SeriesT, template_df_cons(df, {col_name: [[] for _ in range(row_count)]})[col_name])


def node_property_columns(df: DataFrameT, alias_col: str, excluded: Sequence[str]) -> list[str]:
    include_id = _include_numeric_id_as_property(df)
    return [
        str(col)
        for col in df.columns
        if str(col) != alias_col
        and str(col) not in excluded
        and not str(col).startswith("__")
        and not str(col).startswith("label__")
        and (str(col) not in _NODE_INTERNAL_COLS or (include_id and str(col) == "id"))
    ]


def edge_property_columns(df: DataFrameT, alias_col: str, excluded: Sequence[str]) -> list[str]:
    return [
        str(col)
        for col in df.columns
        if str(col) != alias_col
        and str(col) not in excluded
        and not str(col).startswith("__")
        and str(col) not in _EDGE_INTERNAL_COLS
    ]


def entity_property_columns(
    df: DataFrameT,
    *,
    alias_col: str,
    table: Literal["nodes", "edges"],
    excluded: Sequence[str],
) -> list[str]:
    if table == "nodes":
        return node_property_columns(df, alias_col, excluded)
    return edge_property_columns(df, alias_col, excluded)


def entity_keys_series(
    df: DataFrameT,
    *,
    alias_col: str,
    table: Literal["nodes", "edges"],
    excluded: Sequence[str],
) -> SeriesT:
    property_cols = entity_property_columns(
        df,
        alias_col=alias_col,
        table=table,
        excluded=excluded,
    )
    if len(property_cols) == 0:
        empty_col = _fresh_col_name(df.columns, "__gfql_entity_key_empty__")
        return _list_series_from_python_rows(df, empty_col, len(df))

    row_col = _fresh_col_name(df.columns, "__gfql_entity_key_row__")
    key_col = _fresh_col_name(df.columns, "__gfql_entity_key_name__")
    present_col = _fresh_col_name(df.columns, "__gfql_entity_key_present__")

    presence = cast(DataFrameT, df[property_cols].notna().copy())
    presence[row_col] = range(len(df))

    melted = cast(
        DataFrameT,
        presence.melt(
            id_vars=[row_col],
            value_vars=property_cols,
            var_name=key_col,
            value_name=present_col,
        ),
    )
    non_null = cast(DataFrameT, melted.loc[melted[present_col]])
    grouped = cast(
        DataFrameT,
        non_null.groupby(row_col, sort=False)[key_col].agg(list).reset_index(),
    )

    base_rows = cast(DataFrameT, template_df_cons(df, {row_col: range(len(df))}))
    merged = cast(DataFrameT, base_rows.merge(grouped, on=row_col, how="left", sort=False))
    out = cast(SeriesT, merged[key_col])
    missing_mask = cast(SeriesT, out.isna())
    if bool(missing_mask.any()):
        fill_col = _fresh_col_name(df.columns, "__gfql_entity_key_fill__")
        filled = _list_series_from_python_rows(df, fill_col, len(merged))
        out = cast(SeriesT, out.where(~missing_mask, filled))
    return out.reset_index(drop=True)
