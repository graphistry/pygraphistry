from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, cast

import pandas as pd

from graphistry.compute.dataframe_utils import df_cons as template_df_cons
from graphistry.compute.gfql.row.prefilter import AliasPrefilters
from graphistry.utils.json import JSONVal

if TYPE_CHECKING:
    from typing import Protocol
    from graphistry.Plottable import Plottable
    from graphistry.compute.typing import DataFrameT

    class RowPipelineCtx(Protocol):
        """Structural contract the row-pipeline frame ops need from their host graph.

        Satisfied by ``RowPipelineMixin`` / ``_RowPipelineAdapter`` (pipeline.py). Replaces the
        former ``ctx: Any`` so the attribute/method access is type-checked instead of duck-typed.
        Type-check-only (annotations are strings under ``from __future__ import annotations``) —
        zero runtime effect, no runtime Protocol import."""
        _nodes: Optional["DataFrameT"]
        _edges: Optional["DataFrameT"]
        _edge: Optional[str]
        _node: Optional[str]
        # Back-reference to the graph an adapter wraps; None on a graph that IS one.
        _g: Optional["Plottable"]
        _gfql_rows_base_graph: Optional["Plottable"]
        _gfql_start_nodes: Optional["DataFrameT"]
        _gfql_rows_edge_aliases: Optional[Iterable[str]]
        def bind(self) -> "Plottable": ...
        def _gfql_binding_ops_row_table(
            self,
            binding_ops: List[Dict[str, JSONVal]],
            alias_prefilters: Optional[AliasPrefilters] = None,
            attach_prop_aliases: Optional[List[str]] = None,
        ) -> "Plottable": ...
        def _gfql_bindings_row_table(self, alias_endpoints: Any) -> "Plottable": ...


from graphistry.Engine import is_polars_df as _is_polars


def _empty_like(df: Any) -> Any:
    """Zero-row copy preserving schema, for pandas/cuDF and polars frames."""
    if _is_polars(df):
        return df.clear()
    return df.iloc[0:0].copy()


def _alias_true_mask(table_df: Any, source: str) -> Any:
    """Boolean row mask of an alias-marker column with NULL→False (pandas/cuDF; the
    polars equivalent expr is ``pl.col(source).fill_null(False).cast(pl.Boolean)``).
    Shared by ``rows``/``count_table`` so the null handling can't diverge."""
    mask = table_df[source]
    if hasattr(mask, "isna") and hasattr(mask, "where"):
        mask = mask.where(~mask.isna(), False)
    elif hasattr(mask, "fillna"):
        mask = mask.fillna(False)
    return mask.astype(bool)


def _restore_alias_shadowed_user_column(
    ctx: RowPipelineCtx, table_df: "DataFrameT", table: Optional[str], source: str
) -> "DataFrameT":
    """An alias named like a user column (``MATCH (name:P) RETURN name.name``) has that
    column overwritten by the alias marker upstream, so ``source.source`` read back the
    marker. Re-key the user's values from the base frame and expose them under the
    dotted name the projection resolves first, keeping the boolean marker intact for
    every other property read. No-op when the alias shadows nothing or rows cannot be
    re-keyed (marker stays, as before)."""
    base_graph = ctx._gfql_rows_base_graph if ctx._gfql_rows_base_graph is not None else ctx._g
    base_frame = None if base_graph is None else (
        base_graph._nodes if table == "nodes" else base_graph._edges
    )
    if base_frame is None or source not in base_frame.columns:
        return table_df
    key = base_graph._node if table == "nodes" else base_graph._edge  # type: ignore[union-attr]
    if _is_polars(table_df):
        if (
            key is not None and key != source
            and key in table_df.columns and key in base_frame.columns
            and base_frame[key].n_unique() == len(base_frame)
        ):
            orig_cols = list(table_df.columns)
            return table_df.drop(source).join(
                base_frame.select([key, source]), on=key, how="left"
            ).select(orig_cols)
        return table_df
    dotted = f"{source}.{source}"
    base_index = getattr(base_frame, "index", None)
    if base_index is not None and bool(base_index.is_unique):
        # guarded .loc proves index-subset alignment (cuDF Index.isin disagrees with pandas)
        try:
            restored = base_frame[source].loc[table_df.index]
        except (KeyError, IndexError, TypeError):
            restored = None
        if restored is not None and len(restored) == len(table_df):
            out = table_df.copy()
            out[dotted] = restored
            return out
    if (
        key is not None and key != source
        and key in table_df.columns and key in base_frame.columns
        and bool(base_frame[key].is_unique)
    ):
        renamed = base_frame[[key, source]].rename(columns={source: dotted})
        return table_df.merge(renamed, on=key, how="left")
    return table_df


def row_table(ctx: RowPipelineCtx, table_df: Any) -> "Plottable":
    """Return a plottable that treats ``table_df`` as the active row table."""
    from graphistry.compute.gfql.index.handoff import clear_handoff, read_handoff

    handoff = read_handoff(ctx)
    out = ctx.bind()
    clear_handoff(out)  # internal plumbing must never escape on a user-visible result
    # polars has no row index, so reset_index is both unnecessary and absent.
    if not _is_polars(table_df):
        table_df = table_df.reset_index(drop=True)
    out._nodes = table_df
    if ctx._edges is not None:
        out._edges = _empty_like(ctx._edges)
    else:
        out._edges = _empty_like(table_df)
    if handoff is not None and handoff.state is not None and out._edges is not None:
        # The canonical traversal we skipped is what would have added these alias
        # marker columns to the (zero-row) edge frame; synthesize them for parity.
        indexed_edge_aliases = handoff.edge_aliases
        if indexed_edge_aliases:
            missing = [
                alias
                for alias in indexed_edge_aliases
                if alias not in out._edges.columns
            ]
            if _is_polars(out._edges):
                # polars' canonical traversal APPENDS alias flag columns; pandas'
                # puts them first. Match each engine's own canonical column order.
                import polars as pl

                if missing:
                    out._edges = out._edges.with_columns(
                        [pl.lit(True).alias(alias) for alias in missing]
                    )
            else:
                original_cols = [
                    col
                    for col in out._edges.columns
                    if col not in indexed_edge_aliases
                ]
                out._edges = out._edges.assign(
                    **{alias: True for alias in missing}
                )[list(indexed_edge_aliases) + original_cols]
    out._source = None
    out._destination = None
    out._edge = ctx._edge if ctx._edge is not None and ctx._edge in table_df.columns else None
    if out._node is not None and out._node not in table_df.columns:
        out._node = None
    base_graph = ctx._gfql_rows_base_graph
    if base_graph is None:
        base_graph = ctx._g  # adapter-only back-reference
    if base_graph is not None:
        out._gfql_rows_base_graph = base_graph
    if ctx._gfql_start_nodes is not None:
        out._gfql_start_nodes = ctx._gfql_start_nodes
    if ctx._gfql_rows_edge_aliases is not None:
        out._gfql_rows_edge_aliases = ctx._gfql_rows_edge_aliases
    return cast("Plottable", out)


def empty_frame(
    ctx: RowPipelineCtx,
    template_df: Optional[Any] = None,
    columns: Optional[Sequence[str]] = None,
) -> Any:
    if template_df is None:
        if ctx._nodes is not None:
            template_df = ctx._nodes
        elif ctx._edges is not None:
            template_df = ctx._edges
        else:
            base_graph = ctx._gfql_rows_base_graph
            if base_graph is None:
                base_graph = ctx._g
            if base_graph is not None:
                template_df = base_graph._nodes
                if template_df is None:
                    template_df = base_graph._edges

    if template_df is not None:
        if columns is None:
            return _empty_like(template_df)
        if _is_polars(template_df):
            import polars as pl
            return pl.DataFrame(schema={str(col): pl.Object for col in columns})
        return template_df_cons(template_df, {str(col): [] for col in columns})

    if columns is None:
        return pd.DataFrame()
    return pd.DataFrame({str(col): pd.Series(dtype="object") for col in columns})


def get_active_table(ctx: RowPipelineCtx) -> Any:
    if ctx._nodes is not None:
        return ctx._nodes
    if ctx._edges is not None:
        return ctx._edges
    return empty_frame(ctx)


def coerce_non_negative_int(value: Any, op_name: str) -> int:
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
    ctx: RowPipelineCtx,
    table: Optional[str] = None,
    source: Optional[str] = None,
    alias_endpoints: Optional[Dict[str, str]] = None,
    binding_ops: Optional[List[Dict[str, JSONVal]]] = None,
    alias_prefilters: Optional[AliasPrefilters] = None,
    attach_prop_aliases: Optional[List[str]] = None,
) -> "Plottable":
    if binding_ops is not None:
        return ctx._gfql_binding_ops_row_table(
            binding_ops,
            alias_prefilters=alias_prefilters,
            attach_prop_aliases=attach_prop_aliases,
        )
    if alias_endpoints is not None:
        return cast("Plottable", ctx._gfql_bindings_row_table(alias_endpoints))

    if table not in {"nodes", "edges"}:
        raise ValueError(
            f"rows(table=...) must be one of 'nodes' or 'edges', got {table!r}"
        )

    table_df = ctx._nodes if table == "nodes" else ctx._edges
    if table_df is None:
        if ctx._nodes is not None:
            table_df = _empty_like(ctx._nodes)
        elif ctx._edges is not None:
            table_df = _empty_like(ctx._edges)
        else:
            table_df = empty_frame(ctx)
    elif not _is_polars(table_df):
        table_df = table_df.copy()

    if source is not None:
        if source not in table_df.columns:
            raise ValueError(f"rows(source=...) alias column not found: {source!r}")
        if _is_polars(table_df):
            import polars as pl
            # returns straight out of the guarded branch instead of rebinding ``table_df``:
            # the polars frame would otherwise widen the variable's type and break the pandas
            # ``.loc`` branch below (``is_polars_df`` is a TypeGuard, so it does not narrow the
            # negative branch back to pandas). Same call, same argument, same result.
            return row_table(ctx, _restore_alias_shadowed_user_column(
                ctx, table_df.filter(pl.col(source).fill_null(False).cast(pl.Boolean)), table, source))
        # unreachable for polars (returned above), but the guard on the ``.copy()`` branch
        # further up leaves the polars arm in this variable's type: TypeGuard narrows only
        # the positive branch, so ``not _is_polars(...)`` cannot narrow back to pandas.
        table_df = table_df.loc[_alias_true_mask(table_df, source)]  # type: ignore[union-attr]
        table_df = _restore_alias_shadowed_user_column(ctx, table_df, table, source)

    return row_table(ctx, table_df)


def count_table(
    ctx: RowPipelineCtx,
    table: str = "nodes",
    source: Optional[str] = None,
    alias: str = "count(*)",
) -> "Plottable":
    """Count matched rows and set a one-row ``{alias: n}`` result table.

    Fast path for a lone ``count(*)``: reads the height of the active node/edge
    table (or the truthy count of the ``source`` alias-mask column) with a single
    reduction, never materializing/copying the whole frame the way ``rows`` +
    ``group_by`` would. Engine-polymorphic across pandas/cuDF/polars (eager or
    lazy). See ``graphistry.compute.ast.count_table`` and the Cypher lowering
    short-circuit.
    """
    if table not in {"nodes", "edges"}:
        raise ValueError(
            f"count_table(table=...) must be one of 'nodes' or 'edges', got {table!r}"
        )
    table_df = ctx._nodes if table == "nodes" else ctx._edges

    if table_df is None:
        # Keep the 0-count result in the pipeline's engine (mirror empty_frame's
        # template discovery) — a pandas frame inside a polars pipeline would
        # break the engine-consistency the executor asserts.
        other_df = ctx._edges if table == "nodes" else ctx._nodes
        if other_df is not None:
            if _is_polars(other_df):
                import polars as pl
                return row_table(ctx, pl.DataFrame({alias: [0]}))
            return row_table(ctx, template_df_cons(other_df, {alias: [0]}))
        return row_table(ctx, pd.DataFrame({alias: [0]}))

    if _is_polars(table_df):
        import polars as pl
        if source is not None:
            # LazyFrame lacks .columns without a resolve; collect_schema is lazy-safe.
            cols = table_df.collect_schema().names()
            if source not in cols:
                raise ValueError(
                    f"count_table(source=...) alias column not found: {source!r}"
                )
            count_expr = pl.col(source).fill_null(False).cast(pl.Boolean).sum()
        else:
            count_expr = pl.len()
        res = table_df.select(count_expr.alias(alias))
        # eager DataFrame.select -> DataFrame (no collect); LazyFrame.select -> LazyFrame.
        if hasattr(res, "collect"):
            res = res.collect()
        n = int(res.item())
        return row_table(ctx, pl.DataFrame({alias: [n]}))

    # pandas / cuDF (API-compatible)
    if source is not None:
        if source not in table_df.columns:
            raise ValueError(
                f"count_table(source=...) alias column not found: {source!r}"
            )
        n = int(_alias_true_mask(table_df, source).sum())
    else:
        n = int(len(table_df))
    return row_table(ctx, template_df_cons(table_df, {alias: [n]}))


def drop_cols(ctx: RowPipelineCtx, cols: Sequence[str]) -> "Plottable":
    """Drop named columns from the active row table, ignoring any that don't exist."""
    table_df = get_active_table(ctx)
    to_drop = [c for c in cols if c in table_df.columns]
    if to_drop:
        if _is_polars(table_df):
            table_df = table_df.drop(to_drop)
        else:
            table_df = table_df.drop(columns=to_drop)
    return row_table(ctx, table_df)


def skip(ctx: RowPipelineCtx, value: Any) -> "Plottable":
    table_df = get_active_table(ctx)
    skip_count = coerce_non_negative_int(value, "skip")
    if _is_polars(table_df):
        return row_table(ctx, table_df.slice(skip_count))
    return row_table(ctx, table_df.iloc[skip_count:])


def limit(ctx: RowPipelineCtx, value: Any) -> "Plottable":
    table_df = get_active_table(ctx)
    limit_count = coerce_non_negative_int(value, "limit")
    if _is_polars(table_df):
        return row_table(ctx, table_df.head(limit_count))
    return row_table(ctx, table_df.iloc[:limit_count])


def distinct(ctx: RowPipelineCtx) -> "Plottable":
    table_df = get_active_table(ctx)
    if _is_polars(table_df):
        # maintain_order matches pandas drop_duplicates(keep='first') semantics.
        return row_table(ctx, table_df.unique(maintain_order=True))
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
    return row_table(ctx, out_df)
