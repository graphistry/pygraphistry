"""Native polars lowering for the ``search_any`` cross-column search row op
(viz-filter L2; semantics in gfql/search_any.py). Module-per-family, split from
row_pipeline.py (the expression/projection lowering core) — degrees.py precedent."""
from __future__ import annotations

from typing import TYPE_CHECKING, Collection, FrozenSet, List, Mapping, Optional, Sequence

from graphistry.Plottable import Plottable

from .row_pipeline import _active_table, _rewrap

if TYPE_CHECKING:
    import polars as pl
    import pandas as pd


def pandas_categorical_columns(frame: Optional["pd.DataFrame"]) -> FrozenSet[str]:
    """Names of pandas categorical columns before Polars conversion erases dtype kind."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        return frozenset()
    return frozenset(
        col for col in frame.columns
        if isinstance(col, str) and isinstance(frame[col].dtype, pd.CategoricalDtype)
    )


def _is_categorical_dtype(dtype: "pl.DataType") -> bool:
    import polars as pl

    return dtype == pl.Categorical or isinstance(dtype, pl.Enum)


def auto_search_columns(
    schema: "Mapping[str, pl.DataType]", pool_cols: Sequence[str], term: str,
    *, categorical_columns: Collection[str] = (),
) -> Optional[List[str]]:
    """The pandas kernel's dtype auto-gate: string columns always, int AND float columns
    iff the term is a numeric literal; bool/nested never (see search_any_polars).

    ``None`` declines (NIE), which is NOT the same as ``[]`` -- reserved for a dtype the pandas
    kernel searches but this engine cannot render, where skipping the column would answer
    all-False against a pandas match.
    """
    import polars as pl
    from graphistry.compute.gfql.search_any import is_numeric_term
    numeric_ok = is_numeric_term(term)
    chosen = []
    for real in pool_cols:
        if real in categorical_columns:
            continue
        dt = schema[real]
        if dt == pl.String:
            chosen.append(real)
        elif numeric_ok and dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                                   pl.Float32, pl.Float64):
            chosen.append(real)
        elif numeric_ok and (isinstance(dt, pl.Datetime) or dt == pl.Date):
            chosen.append(real)
    return chosen


def search_match_expr(schema: "Mapping[str, pl.DataType]", chosen: Sequence[str], term: str,
                      *, case_sensitive: bool, regex: bool,
                      categorical_columns: Collection[str] = ()) -> "Optional[pl.Expr]":
    """OR-across-columns match expr over already-chosen columns; None declines (NIE).

    Single lowering shared by the ``search_any`` row op and the ``search_any`` alias
    prefilter so the dtype/regex decline gates cannot drift between them.
    """
    import polars as pl
    from .predicates import _regex_rust_incompatible
    if regex and _regex_rust_incompatible(term):
        return None
    # Explicit columns= reaches beyond the auto gate: only dtypes whose canonical
    # toString provably matches the pandas kernel are searched natively — ints render
    # identically; Boolean is canonicalized below (polars 'true' vs pandas 'True' was
    # a SILENT divergence under caseSensitive — wave-2 W2-3). Float no longer relies on
    # repr: it is RENDERED as the inspector displays it (wysiwyg.py), which also removes
    # the old exponent-regime divergence. Temporal/categorical/nested decline honestly.
    _stringify_ok = {
        pl.String, pl.Boolean,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }
    for real in chosen:
        dtype = schema[real]
        if real in categorical_columns and (
            dtype == pl.Date or isinstance(dtype, pl.Datetime)
        ):
            raise NotImplementedError(
                "Polars searchAny explicit datetime categorical cannot guarantee "
                "pandas categorical astype(str) parity across timezone and null values; "
                "use engine='pandas' or engine='cudf'"
            )
        renderable = (
            dtype in _stringify_ok or dtype == pl.Date
            or isinstance(dtype, pl.Datetime) or _is_categorical_dtype(dtype)
        )
        if not renderable:
            return None
    exprs = []
    for real in chosen:
        dt = schema[real]
        if dt == pl.Boolean:
            # pandas categorical.astype(str) renders booleans with title case.
            # Keep null cells null so they cannot match either spelling.
            base = (
                pl.when(pl.col(real).is_null()).then(pl.lit(None, dtype=pl.String))
                .when(pl.col(real)).then(pl.lit("True"))
                .otherwise(pl.lit("False"))
            )
        elif real in categorical_columns or _is_categorical_dtype(dt):
            base = pl.col(real).cast(pl.String)
        elif dt == pl.String:
            base = pl.col(real)
        elif dt in (pl.Float32, pl.Float64):
            # native on purpose: a device must not change the answer
            from graphistry.compute.gfql.wysiwyg import float_render_expr_polars
            base = float_render_expr_polars(pl.col(real), dt)
        elif isinstance(dt, pl.Datetime) or dt == pl.Date:
            # native on purpose: a device must not change the answer
            from graphistry.compute.gfql.wysiwyg import datetime_render_expr_polars
            base = datetime_render_expr_polars(pl.col(real), dt)
        else:
            base = pl.col(real).cast(pl.String)
        if regex:
            pat = term if case_sensitive else f"(?i){term}"
            exprs.append(base.str.contains(pat, literal=False))
        elif case_sensitive:
            exprs.append(base.str.contains(term, literal=True))
        else:
            exprs.append(base.str.to_lowercase().str.contains(term.lower(), literal=True))
    return pl.any_horizontal(exprs)


def search_any_polars(
    g: Plottable,
    alias: str,
    term: str,
    out_col: str,
    case_sensitive: bool = False,
    regex: bool = False,
    columns: Optional[Sequence[str]] = None,
) -> Optional[Plottable]:
    """Native polars ``search_any`` (viz-filter L2): OR-across-columns marker, same
    dtype gate as the pandas kernel (string cols always; int cols iff numeric-literal
    term; float/date/bool auto-gated out). Regex path applies the same Rust-regex
    decline gate as Contains; literal default folds via lowercase (never regex).
    None declines (honest NIE)."""
    import polars as pl
    left = _active_table(g)
    if left is None:
        return None
    prefix = f"{alias}."
    prefixed = [c for c in left.columns if c.startswith(prefix)]
    if prefixed:
        pool = {c[len(prefix):]: c for c in prefixed}
    else:
        pool = {c: c for c in left.columns
                if not c.startswith("__gfql_") and c != alias}
    schema = dict(left.schema)
    base_graph = g._gfql_rows_base_graph if g._gfql_rows_base_graph is not None else g
    if alias in (g._gfql_rows_edge_aliases or ()):
        source_categories = base_graph._gfql_search_any_categorical_columns["edges"]
    else:
        source_categories = base_graph._gfql_search_any_categorical_columns["nodes"]
    categorical_columns = [
        actual for source, actual in pool.items()
        if source in source_categories or actual in source_categories
    ]
    if columns is not None:
        if any(c not in pool for c in columns):
            # same validation error as the pandas row pipeline (there is no pandas
            # fallback behind this dispatch — a generic NIE here would misreport a
            # user input error as an engine gap; wave-1 I2)
            from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
            raise GFQLValidationError(
                ErrorCode.E108,
                "searchAny columns= includes a column absent from the searched table",
                field="columns",
                value=list(columns),
                suggestion="List only columns present on the searched entity.",
                language="cypher",
            )
        chosen = [pool[c] for c in columns]
    else:
        auto = auto_search_columns(
            schema, list(pool.values()), term, categorical_columns=categorical_columns,
        )
        if auto is None:
            return None
        chosen = auto
    if len(left) == 0 or not chosen:
        marked = left.with_columns(
            pl.lit(False).alias(out_col) if len(left) else pl.lit(None).cast(pl.Boolean).alias(out_col))
        return _rewrap(g, marked)
    match = search_match_expr(
        schema, chosen, term, case_sensitive=case_sensitive, regex=regex,
        categorical_columns=categorical_columns,
    )
    if match is None:
        return None
    marked = left.with_columns(match.fill_null(False).alias(out_col))
    return _rewrap(g, marked)
