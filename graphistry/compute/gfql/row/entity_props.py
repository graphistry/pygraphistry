from __future__ import annotations

from typing import Literal, Sequence, cast

import pandas as pd

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


def node_property_columns(df: DataFrameT, alias_col: str, excluded: Sequence[str]) -> list[str]:
    return [
        str(col)
        for col in df.columns
        if str(col) != alias_col
        and str(col) not in excluded
        and not str(col).startswith("__")
        and not str(col).startswith("label__")
        and str(col) not in _NODE_INTERNAL_COLS
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
        return cast(SeriesT, pd.Series([[] for _ in range(len(df))], index=df.index, dtype="object"))

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

    base_rows = cast(DataFrameT, pd.DataFrame({row_col: range(len(df))}))
    merged = cast(DataFrameT, base_rows.merge(grouped, on=row_col, how="left", sort=False))
    out = cast(SeriesT, merged[key_col])
    missing_mask = cast(SeriesT, out.isna())
    if bool(missing_mask.any()):
        empty_index = out.index[missing_mask]
        out.loc[empty_index] = pd.Series([[] for _ in range(len(empty_index))], index=empty_index, dtype="object")
    return out.reset_index(drop=True)
