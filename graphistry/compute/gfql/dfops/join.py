"""Join-related DataFrame operations for GFQL runtime execution paths."""

from typing import Dict, List, Sequence, cast

from graphistry.Engine import Engine
from graphistry.compute.typing import DataFrameT


def binding_join_columns(frame: DataFrameT) -> List[str]:
    return [column for column in frame.columns if isinstance(column, str) and "." in column]


def joined_hidden_scalar_columns(frame: DataFrameT) -> DataFrameT:
    hidden_suffixes: Dict[str, List[str]] = {}
    for column in frame.columns:
        if not isinstance(column, str) or "." not in column:
            continue
        _, suffix = column.split(".", 1)
        if suffix.startswith("__cypher_reentry_") or suffix.startswith("__gfql_hidden_"):
            hidden_suffixes.setdefault(suffix, []).append(column)
    out = frame
    for suffix, columns in hidden_suffixes.items():
        if suffix in out.columns:
            continue
        series = out[columns[0]]
        for column in columns[1:]:
            if hasattr(series, "combine_first"):
                series = series.combine_first(out[column])
        out = out.assign(**{suffix: series})
    return out


def joined_alias_columns(frame: DataFrameT) -> DataFrameT:
    alias_candidates: Dict[str, str] = {}
    for column in frame.columns:
        if not isinstance(column, str) or "." not in column:
            continue
        alias, suffix = column.split(".", 1)
        if alias in frame.columns:
            continue
        if suffix == alias:
            alias_candidates.setdefault(alias, column)
        elif suffix == "id" and alias not in alias_candidates:
            alias_candidates[alias] = column
    out = frame
    for alias, source_column in alias_candidates.items():
        out = out.assign(**{alias: out[source_column]})
    return out


def connected_inner_join_rows(
    joined_rows: DataFrameT,
    pattern_rows: DataFrameT,
    *,
    join_cols: Sequence[str],
    keep_cols: Sequence[str],
    engine: Engine,
) -> DataFrameT:
    """Inner-join connected MATCH row payloads.

    cuDF path: avoid full-row merge on dotted payload columns by joining only
    compact key/index frames, then gathering payload rows by position.
    """
    join_cols_list = list(join_cols)
    keep_cols_list = list(keep_cols)
    rhs = cast(DataFrameT, pattern_rows[keep_cols_list])
    if engine != Engine.CUDF:
        return cast(DataFrameT, joined_rows.merge(rhs, on=join_cols_list, how="inner"))

    lhs_row_id = "__gfql_connected_lhs_row_id__"
    rhs_row_id = "__gfql_connected_rhs_row_id__"
    lhs = cast(DataFrameT, joined_rows.reset_index(drop=True))
    rhs = cast(DataFrameT, rhs.reset_index(drop=True))
    lhs_with_idx = cast(DataFrameT, lhs.reset_index().rename(columns={"index": lhs_row_id}))
    rhs_with_idx = cast(DataFrameT, rhs.reset_index().rename(columns={"index": rhs_row_id}))
    lhs_keys = cast(DataFrameT, lhs_with_idx[[lhs_row_id] + join_cols_list])
    rhs_keys = cast(DataFrameT, rhs_with_idx[[rhs_row_id] + join_cols_list])
    row_pairs = cast(DataFrameT, lhs_keys.merge(rhs_keys, on=join_cols_list, how="inner"))
    rhs_payload_cols = [column for column in keep_cols_list if column not in join_cols_list]
    if len(row_pairs) == 0:
        out = cast(DataFrameT, lhs.head(0))
        for column in rhs_payload_cols:
            out = out.assign(**{column: rhs[column].head(0)})
        return out

    lhs_taken = cast(DataFrameT, lhs.take(row_pairs[lhs_row_id]))
    if not rhs_payload_cols:
        return cast(DataFrameT, lhs_taken.reset_index(drop=True))
    rhs_payload = cast(DataFrameT, rhs[rhs_payload_cols].take(row_pairs[rhs_row_id]).reset_index(drop=True))
    lhs_taken = cast(DataFrameT, lhs_taken.reset_index(drop=True))
    for column in rhs_payload_cols:
        lhs_taken = lhs_taken.assign(**{column: rhs_payload[column]})
    return lhs_taken
