from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, cast

import pandas as pd

from graphistry.compute.dataframe_utils import df_cons as template_df_cons

if TYPE_CHECKING:
    from typing import Protocol
    from graphistry.Plottable import Plottable

    class RowPipelineCtx(Protocol):
        """Structural contract the row-pipeline frame ops need from their host graph.

        Satisfied by ``RowPipelineMixin`` / ``_RowPipelineAdapter`` (pipeline.py). Replaces the
        former ``ctx: Any`` so the attribute/method access is type-checked instead of duck-typed.
        Type-check-only (annotations are strings under ``from __future__ import annotations``) —
        zero runtime effect, no runtime Protocol import."""
        _nodes: Any
        _edges: Any
        _edge: Any
        def bind(self) -> "Plottable": ...
        def _gfql_binding_ops_row_table(
            self,
            binding_ops: Any,
            alias_prefilters: Optional[Dict[str, Any]] = None,
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


def row_table(ctx: RowPipelineCtx, table_df: Any) -> "Plottable":
    """Return a plottable that treats ``table_df`` as the active row table."""
    out = ctx.bind()
    # polars has no row index, so reset_index is both unnecessary and absent.
    if not _is_polars(table_df):
        table_df = table_df.reset_index(drop=True)
    out._nodes = table_df
    if ctx._edges is not None:
        out._edges = _empty_like(ctx._edges)
    else:
        out._edges = _empty_like(table_df)
    out._source = None
    out._destination = None
    out._edge = ctx._edge if ctx._edge is not None and ctx._edge in table_df.columns else None
    if out._node is not None and out._node not in table_df.columns:
        out._node = None
    base_graph = getattr(ctx, "_gfql_rows_base_graph", None)
    if base_graph is None:
        base_graph = getattr(ctx, "_g", None)
    if base_graph is not None:
        setattr(out, "_gfql_rows_base_graph", base_graph)
    start_nodes = getattr(ctx, "_gfql_start_nodes", None)
    if start_nodes is not None:
        setattr(out, "_gfql_start_nodes", start_nodes)
    edge_aliases = getattr(ctx, "_gfql_rows_edge_aliases", None)
    if edge_aliases is not None:
        setattr(out, "_gfql_rows_edge_aliases", edge_aliases)
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
            base_graph = getattr(ctx, "_gfql_rows_base_graph", None)
            if base_graph is None:
                base_graph = getattr(ctx, "_g", None)
            if base_graph is not None:
                template_df = getattr(base_graph, "_nodes", None)
                if template_df is None:
                    template_df = getattr(base_graph, "_edges", None)

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
    table: str = "nodes",
    source: Optional[str] = None,
    alias_endpoints: Optional[Dict[str, str]] = None,
    binding_ops: Optional[List[Dict[str, Any]]] = None,
    alias_prefilters: Optional[Dict[str, Any]] = None,
) -> "Plottable":
    if binding_ops is not None:
        return cast(
            "Plottable",
            ctx._gfql_binding_ops_row_table(binding_ops, alias_prefilters=alias_prefilters),
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
            table_df = table_df.filter(pl.col(source).fill_null(False).cast(pl.Boolean))
        else:
            table_df = table_df.loc[_alias_true_mask(table_df, source)]

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
