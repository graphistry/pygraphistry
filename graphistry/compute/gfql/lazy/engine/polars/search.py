"""Native polars lowering for the ``search_any`` cross-column search row op
(viz-filter L2; semantics in gfql/search_any.py). Module-per-family, split from
row_pipeline.py (the expression/projection lowering core) — degrees.py precedent."""
from __future__ import annotations

from typing import Optional, Sequence

from graphistry.Plottable import Plottable

from .row_pipeline import _active_table, _rewrap


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
    from graphistry.compute.gfql.search_any import is_numeric_term
    from .predicates import _regex_rust_incompatible
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
        numeric_ok = is_numeric_term(term)
        chosen = []
        for real in pool.values():
            dt = schema[real]
            if dt == pl.String:
                chosen.append(real)
            elif numeric_ok and dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                       pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                chosen.append(real)
    if len(left) == 0 or not chosen:
        marked = left.with_columns(
            pl.lit(False).alias(out_col) if len(left) else pl.lit(None).cast(pl.Boolean).alias(out_col))
        return _rewrap(g, marked)
    if regex and _regex_rust_incompatible(term):
        return None
    # Explicit columns= reaches beyond the auto gate: only dtypes whose canonical
    # toString provably matches the pandas kernel are searched natively — floats/ints
    # render identically; Boolean is canonicalized below (polars 'true' vs pandas
    # 'True' was a SILENT divergence under caseSensitive — wave-2 W2-3); everything
    # else (temporal, categorical, nested) declines honestly.
    _stringify_ok = {
        pl.String, pl.Boolean, pl.Float32, pl.Float64,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    }
    if any(schema[real] not in _stringify_ok for real in chosen):
        return None
    exprs = []
    for real in chosen:
        dt = schema[real]
        if dt == pl.String:
            base = pl.col(real)
        elif dt == pl.Boolean:
            # null cells must STAY null (never match) — bare when/otherwise would
            # send null conditions to the 'False' branch
            base = (
                pl.when(pl.col(real).is_null()).then(pl.lit(None, dtype=pl.String))
                .when(pl.col(real)).then(pl.lit("True"))
                .otherwise(pl.lit("False"))
            )
        else:
            base = pl.col(real).cast(pl.String)
        if regex:
            pat = term if case_sensitive else f"(?i){term}"
            exprs.append(base.str.contains(pat, literal=False))
        elif case_sensitive:
            exprs.append(base.str.contains(term, literal=True))
        else:
            exprs.append(base.str.to_lowercase().str.contains(term.lower(), literal=True))
    marked = left.with_columns(pl.any_horizontal(exprs).fill_null(False).alias(out_col))
    return _rewrap(g, marked)
