"""Join-related engine-polymorphic DataFrame operations."""

import operator
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from graphistry.Engine import Engine, POLARS_ENGINES
from graphistry.compute.gfql.cypher.reentry.naming import REENTRY_HIDDEN_COLUMN_PREFIX
from graphistry.compute.gfql.identifiers import HIDDEN_ALIAS_COLUMN_PREFIX
from graphistry.compute.typing import DataFrameT, DomainT


def binding_join_columns(frame: DataFrameT) -> List[str]:
    return [column for column in frame.columns if isinstance(column, str) and "." in column]


def joined_hidden_scalar_columns(frame: DataFrameT) -> DataFrameT:
    hidden_suffixes: Dict[str, List[str]] = {}
    for column in frame.columns:
        if not isinstance(column, str) or "." not in column:
            continue
        _, suffix = column.split(".", 1)
        if suffix.startswith(REENTRY_HIDDEN_COLUMN_PREFIX) or suffix.startswith(HIDDEN_ALIAS_COLUMN_PREFIX):
            hidden_suffixes.setdefault(suffix, []).append(column)
    out = frame
    is_polars = "polars" in type(frame).__module__
    for suffix, columns in hidden_suffixes.items():
        if suffix in out.columns:
            continue
        if is_polars:
            import polars as pl
            expr = pl.coalesce([pl.col(column) for column in columns]).alias(suffix)
            out = out.with_columns(expr)
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
    is_polars = "polars" in type(frame).__module__
    if is_polars and alias_candidates:
        import polars as pl
        return cast(DataFrameT, out.with_columns([
            pl.col(source_column).alias(alias)
            for alias, source_column in alias_candidates.items()
        ]))
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
    if engine in POLARS_ENGINES:
        return cast(DataFrameT, joined_rows.join(rhs, on=join_cols_list, how="inner"))
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


def _row_notna_all(frame: DataFrameT, cols: Sequence[str]) -> Any:
    if len(cols) == 0:
        return True
    # cuDF 25.02 can segfault on DataFrame.notna().all(axis=1); reduce masks
    # column-by-column instead of using row-wise DataFrame reductions.
    mask = frame[cols[0]].notna()
    for col in cols[1:]:
        mask = mask & frame[col].notna()
    return mask


def project_node_attrs(
    frame: DataFrameT,
    node_col: str,
    cols: Sequence[str],
    *,
    id_label: str,
    prefix: str = "",
    labels: Optional[Sequence[str]] = None,
    node_domain: Optional[DomainT] = None,
    dedupe: bool = False,
    drop_nulls: bool = False,
) -> DataFrameT:
    df = frame[frame[node_col].isin(node_domain)] if node_domain is not None else frame
    data_cols = [node_col] + [col for col in cols if col != node_col]
    rename_map = {node_col: id_label}
    if labels is not None:
        rename_map.update(
            {col: label for col, label in zip(cols, labels) if col != node_col}
        )
    else:
        rename_map.update({col: f"{prefix}{col}" for col in cols if col != node_col})
    df = df[data_cols].rename(columns=rename_map)
    if labels is not None and node_col in cols:
        df[labels[cols.index(node_col)]] = df[id_label]
    if drop_nulls and labels is not None:
        label_cols = [label for label in labels if label in df.columns]
        if label_cols:
            df = df[_row_notna_all(df, label_cols)]
    return df.drop_duplicates() if dedupe else df


def ineq_eval_pairs(
    left_pairs: DataFrameT,
    right_pairs: DataFrameT,
    op: str,
    *,
    group_cols: Optional[Sequence[str]] = None,
    left_value: str = "__value__",
    right_value: str = "__value__",
) -> Tuple[DataFrameT, DataFrameT]:
    group_cols = list(group_cols) if group_cols is not None else ["__mid__"]
    if op in {"<", "<="}:
        right_bound = (
            right_pairs.groupby(group_cols)[right_value]
            .max()
            .reset_index()
            .rename(columns={right_value: "__right_bound__"})
        )
        left_bound = (
            left_pairs.groupby(group_cols)[left_value]
            .min()
            .reset_index()
            .rename(columns={left_value: "__left_bound__"})
        )
        left_eval = left_pairs.merge(right_bound, on=group_cols, how="inner")
        right_eval = right_pairs.merge(left_bound, on=group_cols, how="inner")
        left_cmp = operator.lt if op == "<" else operator.le
        right_cmp = operator.gt if op == "<" else operator.ge
    else:
        right_bound = (
            right_pairs.groupby(group_cols)[right_value]
            .min()
            .reset_index()
            .rename(columns={right_value: "__right_bound__"})
        )
        left_bound = (
            left_pairs.groupby(group_cols)[left_value]
            .max()
            .reset_index()
            .rename(columns={left_value: "__left_bound__"})
        )
        left_eval = left_pairs.merge(right_bound, on=group_cols, how="inner")
        right_eval = right_pairs.merge(left_bound, on=group_cols, how="inner")
        left_cmp = operator.gt if op == ">" else operator.ge
        right_cmp = operator.lt if op == ">" else operator.le
    left_eval = left_eval[left_cmp(left_eval[left_value], left_eval["__right_bound__"])]
    right_eval = right_eval[
        right_cmp(right_eval[right_value], right_eval["__left_bound__"])
    ]
    return left_eval, right_eval


def semijoin_eval_pairs(
    left_pairs: DataFrameT,
    right_pairs: DataFrameT,
    op: str,
    *,
    left_value: str = "__value__",
    right_value: str = "__value__",
    left_unique_col: str = "__left_unique__",
    right_unique_col: str = "__right_unique__",
    left_only_col: str = "__left_only__",
    right_only_col: str = "__right_only__",
    left_keep: Optional[Sequence[str]] = None,
    right_keep: Optional[Sequence[str]] = None,
) -> Tuple[Optional[DataFrameT], Optional[DataFrameT], Optional[DataFrameT]]:
    mid_values = None
    if op == "==":
        left_mid = left_pairs[["__mid__", left_value]].drop_duplicates().rename(
            columns={left_value: "__value__"}
        )
        right_mid = right_pairs[["__mid__", right_value]].drop_duplicates().rename(
            columns={right_value: "__value__"}
        )
        mid_values = left_mid.merge(right_mid, on=["__mid__", "__value__"], how="inner")
        if len(mid_values) == 0:
            return None, None, mid_values
        left_eval = left_pairs.merge(
            mid_values.rename(columns={"__value__": left_value}),
            on=["__mid__", left_value],
            how="inner",
        )
        right_eval = right_pairs.merge(
            mid_values.rename(columns={"__value__": right_value}),
            on=["__mid__", right_value],
            how="inner",
        )
    elif op == "!=":

        def _uniq(df: DataFrameT, col: str, name: str) -> DataFrameT:
            counts = df[["__mid__", col]].drop_duplicates().groupby("__mid__").size().reset_index()
            return counts.rename(columns={0: name, "size": name})

        left_unique = _uniq(left_pairs, left_value, left_unique_col)
        right_unique = _uniq(right_pairs, right_value, right_unique_col)
        right_only = (
            right_pairs[["__mid__", right_value]]
            .drop_duplicates()
            .merge(
                right_unique[right_unique[right_unique_col] == 1],
                on="__mid__",
                how="inner",
            )[["__mid__", right_value]]
            .rename(columns={right_value: right_only_col})
        )
        left_only = (
            left_pairs[["__mid__", left_value]]
            .drop_duplicates()
            .merge(left_unique[left_unique[left_unique_col] == 1], on="__mid__", how="inner")[
                ["__mid__", left_value]
            ]
            .rename(columns={left_value: left_only_col})
        )
        left_eval = left_pairs.merge(right_unique, on="__mid__", how="inner").merge(
            right_only, on="__mid__", how="left"
        )
        left_eval = left_eval[
            (left_eval[right_unique_col] > 1)
            | left_eval[right_only_col].isna()
            | (left_eval[right_only_col] != left_eval[left_value])
        ]
        right_eval = right_pairs.merge(left_unique, on="__mid__", how="inner").merge(
            left_only, on="__mid__", how="left"
        )
        right_eval = right_eval[
            (right_eval[left_unique_col] > 1)
            | right_eval[left_only_col].isna()
            | (right_eval[left_only_col] != right_eval[right_value])
        ]
    else:
        left_eval, right_eval = ineq_eval_pairs(
            left_pairs,
            right_pairs,
            op,
            left_value=left_value,
            right_value=right_value,
        )
    if left_eval is None or right_eval is None:
        return None, None, mid_values
    if left_keep:
        left_eval = left_eval[list(left_keep)]
    if right_keep:
        right_eval = right_eval[list(right_keep)]
    return left_eval, right_eval, mid_values


def estimate_inner_join_rows(
    left: DataFrameT,
    right: DataFrameT,
    *,
    left_on: str,
    right_on: str,
    engine: Engine,
) -> int:
    """Rows an inner join WOULD emit, without materializing it.

    Sum over matched keys of (left count * right count) — a group-by on each side
    instead of the join itself, so a planner can cost-gate a hop before paying for
    it.
    """
    if len(left) == 0 or len(right) == 0:
        return 0
    left_n, right_n = "__gfql_join_left_n__", "__gfql_join_right_n__"
    if engine in POLARS_ENGINES:
        import polars as pl

        left_counts = left.group_by(left_on).len().rename({"len": left_n})  # type: ignore[operator]
        right_counts = right.group_by(right_on).len().rename({"len": right_n})  # type: ignore[operator]
        value = (
            left_counts.join(right_counts, left_on=left_on, right_on=right_on, how="inner")
            .select((pl.col(left_n) * pl.col(right_n)).sum())
            .item()
        )
        return 0 if value is None else int(value)

    left_counts = left.groupby(left_on, sort=False).size().reset_index()
    left_counts.columns = [left_on, left_n]
    right_counts = right.groupby(right_on, sort=False).size().reset_index()
    right_counts.columns = [right_on, right_n]
    counts = left_counts.merge(
        right_counts, left_on=left_on, right_on=right_on, how="inner", sort=False,
    )
    if len(counts) == 0:
        return 0
    return int((counts[left_n] * counts[right_n]).sum())


def path_ordered_expand_join(
    state: DataFrameT,
    step: DataFrameT,
    *,
    current_col: str,
    from_col: str,
    to_col: str,
    path_order_col: str,
    tiebreak_cols: Sequence[str],
    alias: Optional[str],
    engine: Engine,
) -> DataFrameT:
    """Expand a path-bag by one step, preserving (path, tiebreak...) row order.

    ``state[current_col]`` joins ``step[from_col]``; ``step[to_col]`` becomes the new
    ``current_col``. A synthetic ``path_order_col`` pins the incoming row order so the
    result is deterministic under any engine's join, and the bookkeeping columns are
    dropped on the way out. ``alias``, when given, also exposes the new current value
    under that name.
    """
    drop_after = [from_col, path_order_col, *tiebreak_cols]
    if engine in POLARS_ENGINES:
        import polars as pl

        joined = (
            state.with_row_index(path_order_col)  # type: ignore[operator]
            .join(step, left_on=current_col, right_on=from_col, how="inner")
            .sort([path_order_col, *tiebreak_cols])
            .drop(current_col)
            .rename({to_col: current_col})
        )
        if isinstance(alias, str):
            joined = joined.with_columns(pl.col(current_col).alias(alias))
        return cast(
            DataFrameT,
            joined.drop([col for col in drop_after if col in joined.columns]),
        )

    import numpy as np

    out = state.assign(**{path_order_col: np.arange(len(state))}).merge(
        step, left_on=current_col, right_on=from_col, how="inner", sort=False,
    )
    if len(out):
        sort_cols = [path_order_col, *tiebreak_cols]
        out = (
            out.sort_values(sort_cols, kind="stable") if engine == Engine.PANDAS
            else out.sort_values(sort_cols)
        )
    out = out.drop(columns=[current_col]).rename(columns={to_col: current_col})
    if isinstance(alias, str):
        out = out.assign(**{alias: out[current_col]})
    return cast(
        DataFrameT,
        out.drop(columns=[col for col in drop_after if col in out.columns]),
    )


def semijoin_by_column(
    frame: DataFrameT,
    keys: DataFrameT,
    *,
    left_on: str,
    right_on: str,
    engine: Engine,
) -> DataFrameT:
    """Rows of ``frame`` whose ``left_on`` value appears in ``keys[right_on]``."""
    if engine in POLARS_ENGINES:
        return cast(
            DataFrameT,
            frame.join(  # type: ignore[call-arg]
                keys.select(right_on).unique(),  # type: ignore[operator]
                left_on=left_on,
                right_on=right_on,
                how="semi",  # type: ignore[arg-type]
            ),
        )
    return cast(DataFrameT, frame[frame[left_on].isin(keys[right_on])])
