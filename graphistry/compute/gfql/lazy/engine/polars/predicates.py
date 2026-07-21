"""Vectorized polars filter_by_dict for the native polars GFQL engine.

Common comparison/membership/string/null predicates lower to native polars expressions.
NO-CHEATING contract: no pandas bridge — a predicate with no native lowering raises
NotImplementedError (bridging one column would misrepresent pandas semantics as polars and
break columnar/GPU assumptions; use engine='pandas'). All filtering is one vectorized
``df.filter(expr)`` — no per-row work, no Python materialization.
"""
from __future__ import annotations

import operator
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.predicates.str import Contains, Endswith, Fullmatch, Match, Startswith
from graphistry.compute.filter_by_dict import resolve_filter_column
from .dtypes import is_numeric as _dtype_numeric, is_stringlike as _dtype_stringlike

if TYPE_CHECKING:
    import datetime
    import polars as pl

# Comparison-predicate RHS: genuinely dynamic (Cypher properties are dynamically typed) — a
# python scalar or a GFQL/py temporal matched structurally by type(val).__name__
# (DateValue/TemporalValue/…, never imported here).
CmpValue = Union[int, float, str, bool, "datetime.date", "datetime.time", Any]

# Python-`re` features Rust regex (polars) rejects or evaluates differently: lookaround
# ((?=/(?!/(?<=/(?<!) and backreferences (\\1, (?P=name), \\k<name>) — ComputeError or silent
# difference, so such patterns decline (NIE), never guess; pandas evaluates them with Python `re`.
_REGEX_RUST_INCOMPAT = re.compile(r"\(\?<?[=!]|\\[1-9]|\(\?P=|\\k<")


def _regex_rust_incompatible(pat: str) -> bool:
    return _REGEX_RUST_INCOMPAT.search(pat) is not None


def _homogeneous_scalar_category(opts: List[Any]) -> Optional[str]:
    """The single value-category (num/str/bool) of a literal list, else None (mixed/empty).
    polars ``is_in`` raises on a cross-type list, so a non-homogeneous IN must decline (NIE)."""
    def _cat(o: Any) -> Optional[str]:
        if isinstance(o, bool):
            return "bool"
        if isinstance(o, (int, float)):
            return "num"
        if isinstance(o, str):
            return "str"
        return None
    cats = {_cat(o) for o in opts}
    return next(iter(cats)) if (len(cats) == 1 and None not in cats) else None


# Comparison callables predicates declare (op = staticmethod(operator.gt) etc.); pl.Expr
# implements Python rich comparison so op(lhs, rhs) builds exactly `lhs > rhs`/... . Ops outside
# this whitelist have no proven lowering and fall through to the decline paths.
_CMP_OPS = frozenset({operator.gt, operator.lt, operator.ge, operator.le, operator.eq, operator.ne})


def _cmp_expr(
    col_expr: "pl.Expr",
    op: Callable[[Any, Any], Any],
    val: CmpValue,
    dtype: "Optional[pl.DataType]" = None,
) -> "Optional[pl.Expr]":
    import datetime as _dt

    # Native temporal SAFE SUBSET: DateValue (Cypher date('YYYY-MM-DD') / p.gt(date(...))) vs a
    # NAIVE pl.Datetime column. pandas oracle: s.dt.date compared to the tz-free python date
    # (DateValue._parsed); col_expr.dt.date() <op> pl.lit(date) is the exact equivalent — same
    # calendar-date truncation, no tz on either side, parity provable. REQUIRE dtype (from the
    # frame schema) to prove naive Datetime, else fall through to the decline. All else declines:
    # tz-tagged DateTimeValue (always carries tz; pandas tz_localize/convert), TimeValue,
    # tz-aware columns, raw python datetimes — each risks a silent tz/semantic mismatch.
    if type(val).__name__ == "DateValue":
        import polars as pl
        d = getattr(val, "_parsed", None)
        if (
            isinstance(dtype, pl.Datetime) and dtype.time_zone is None
            and isinstance(d, _dt.date) and not isinstance(d, _dt.datetime)
        ):
            if op in _CMP_OPS:
                return op(col_expr.dt.date(), pl.lit(d))
        # naive-Datetime parity unprovable here -> fall through to the decline below.

    # decline (NIE): remaining temporal values (datetime/time/TemporalValue etc.) have no SAFE
    # polars-literal comparison — returning `col > TemporalValue` would be a non-None broken expr
    # erroring at df.filter (or silently misordering). Tracked feature gap; numeric/string
    # vals unaffected.
    if isinstance(val, (_dt.date, _dt.datetime, _dt.time)) or type(val).__name__ in (
        "TemporalValue", "Timestamp", "Timedelta", "datetime64",
        "DateTimeValue", "TimeValue", "DateValue",
    ):
        return None
    # Narrow residual: no IEEE NaN mask here (unlike the WHERE/row-pipeline _nan_guard) — on a
    # GENUINE polars NaN, `col > x` keeps the row (NaN = largest) where pandas drops it.
    # Unreachable on standard ingestion: from_pandas/df_to_engine convert NaN→null (nan_to_null)
    # and filter_by_dict runs on INGESTED columns (no in-query float math — that's the WHERE
    # path). Only a natively-built polars frame with raw NaN diverges; documented, not guarded,
    # to keep the lowering simple. (Mirrors the documented integer 0/0 column-compare residual.)
    if op in _CMP_OPS:
        return op(col_expr, val)
    return None


def _inline_regex_flag_prefix(case: bool, flags: int) -> str:
    """Translate pandas-style `re` flags int + case bool to a Rust-regex inline prefix like
    ``(?im)`` (empty when nothing applies), keeping the polars regex lowering faithful to pandas."""
    letters = ""
    if not case or (flags & re.IGNORECASE):
        letters += "i"
    if flags & re.MULTILINE:
        letters += "m"
    if flags & re.DOTALL:
        letters += "s"
    if flags & re.VERBOSE:
        letters += "x"
    return f"(?{letters})" if letters else ""


def predicate_to_expr(col: str, pred: ASTPredicate, dtype: "Optional[pl.DataType]" = None) -> "Optional[pl.Expr]":
    """Lower an ASTPredicate to a polars boolean expression, or None if unsupported. ``dtype`` =
    schema dtype of ``col`` (None if unknown), letting the temporal lowering prove a NAIVE
    Datetime column before lowering a DateValue comparison (else decline)."""
    import polars as pl

    c = pl.col(col)
    name = type(pred).__name__

    op = getattr(pred, "op", None)
    if op is not None and hasattr(pred, "val"):
        expr = _cmp_expr(c, op, pred.val, dtype)
        if expr is not None:
            return expr

    if name == "Between" and hasattr(pred, "lower") and hasattr(pred, "upper"):
        lo, hi = pred.lower, pred.upper
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if getattr(pred, "inclusive", True):
                return (c >= lo) & (c <= hi)
            return (c > lo) & (c < hi)
        # Temporal Between: pandas evaluates as GE/LE (inclusive) or GT/LT (exclusive)
        # sub-predicates, and for DateValue bounds each is the date-truncated compare _cmp_expr
        # already lowers with proven parity — composing the two endpoint compares adds NO new
        # proof obligation. Both endpoints must be DateValue over a naive Datetime or _cmp_expr
        # returns None -> honest NIE (tz-aware DateTimeValue, TimeValue, raw datetime, mixed
        # bounds, non-Datetime dtype all decline this way — never a silent mismatch).
        inclusive = getattr(pred, "inclusive", True)
        lo_expr = _cmp_expr(c, operator.ge if inclusive else operator.gt, lo, dtype)
        hi_expr = _cmp_expr(c, operator.le if inclusive else operator.lt, hi, dtype)
        if lo_expr is not None and hi_expr is not None:
            return lo_expr & hi_expr

    if name == "IsIn" and hasattr(pred, "options"):
        opts = list(pred.options)
        # Only non-empty single-category literal lists are parity-safe: is_in raises ComputeError
        # on a cross-type list ([1, 'a'] over Int), so mixed/empty declines (NIE) — matching the
        # row-pipeline _lower_in discipline.
        if opts and _homogeneous_scalar_category(opts) is not None:
            return c.is_in(opts)
        return None

    if name == "AllOf" and hasattr(pred, "predicates"):
        # Conjunction (n.val > 20 AND n.val < 90 folds to AllOf[GT, LT]): lower each child and
        # AND them; if ANY child can't lower, the whole predicate can't (caller NIEs).
        child_exprs = [predicate_to_expr(col, p, dtype) for p in pred.predicates]
        if child_exprs and all(e is not None for e in child_exprs):
            lowered: "List[pl.Expr]" = [e for e in child_exprs if e is not None]
            combined = lowered[0]
            for e in lowered[1:]:
                combined = combined & e
            return combined
        return None

    if name in ("IsNull", "IsNA"):
        return c.is_null()
    if name in ("NotNull", "NotNA"):
        return c.is_not_null()

    if name == "IsLeapYear":
        # pandas s.dt.is_leap_year and polars expr.dt.is_leap_year() are the identical Boolean
        # over the calendar year (Gregorian rule incl. 1900-not-leap / 2000-leap) — parity
        # provable. Require naive Datetime or Date from the schema (a tz derives the year in
        # LOCAL time; year-boundary equality unproven); else decline (NIE). Mirrors the
        # naive-Datetime guard used for DateValue.
        if (isinstance(dtype, pl.Datetime) and dtype.time_zone is None) or dtype == pl.Date:
            return c.dt.is_leap_year()
        return None

    if name in ("IsMonthStart", "IsMonthEnd", "IsQuarterStart",
                "IsQuarterEnd", "IsYearStart", "IsYearEnd"):
        # Temporal boundary predicates: polars has no is_month_start/... BOOLEAN accessor
        # (month_start()/month_end() ROLL to a Datetime), but the pandas oracle with freq=None is
        # a pure calendar-field test (is_month_start = day==1; is_month_end = day==days_in_month;
        # quarter/year add a month-set / month==N). polars dt.day()/dt.month()/dt.days_in_month()
        # extract the SAME fields (both proleptic Gregorian, days_in_month leap-aware), so each
        # compose is BIT-IDENTICAL on non-null rows — a proven derivation, not a guess. pandas
        # returns False for NaT; polars comparison yields null -> fill_null(False) restores
        # parity. Require naive Datetime or Date (tz shifts wall-clock fields), like IsLeapYear;
        # else honest NIE.
        if (isinstance(dtype, pl.Datetime) and dtype.time_zone is None) or dtype == pl.Date:
            day = c.dt.day()
            month = c.dt.month()
            if name == "IsMonthStart":
                expr = day == 1
            elif name == "IsMonthEnd":
                expr = day == c.dt.days_in_month()
            elif name == "IsQuarterStart":
                expr = (day == 1) & month.is_in([1, 4, 7, 10])
            elif name == "IsQuarterEnd":
                expr = (day == c.dt.days_in_month()) & month.is_in([3, 6, 9, 12])
            elif name == "IsYearStart":
                expr = (day == 1) & (month == 1)
            else:  # IsYearEnd
                expr = (day == c.dt.days_in_month()) & (month == 12)
            return expr.fill_null(False)
        return None

    if name == "Contains" and hasattr(pred, "pat") and isinstance(pred.pat, str):
        case = getattr(pred, "case", True)
        regex = getattr(pred, "regex", True)
        flags = getattr(pred, "flags", 0)
        if not regex:
            # Literal substring must NOT be regex-interpreted (metachars like ./*/( over-match).
            # polars has no literal+case flag, so lowercase both sides for the case-insensitive
            # literal (matches pandas).
            if case:
                return c.str.contains(pred.pat, literal=True)
            return c.str.to_lowercase().str.contains(pred.pat.lower(), literal=True)
        # Regex: mirror pandas case/flags via a Rust inline flag prefix. decline (NIE):
        # Python-re-only features (lookaround, backreferences) — no silent wrong answer.
        if _regex_rust_incompatible(pred.pat):
            return None
        prefix = _inline_regex_flag_prefix(case, flags)
        return c.str.contains(f"{prefix}{pred.pat}", literal=False)

    if isinstance(pred, (Match, Fullmatch)):
        # Regex predicates: Match = start-anchored (``re.match`` semantics), Fullmatch =
        # fully anchored (``^..$``) — the target for Cypher's ``=~`` (full/anchored, Java
        # ``Matcher.matches()``). Wrap the user pattern in a non-capturing group so a
        # top-level alternation (``a|b``) anchors as a whole. ``case``/``flags`` → inline prefix.
        # Same Rust-regex gate as Contains: lookaround/backrefs raised a non-NIE
        # ComputeError at collect (dgx-repro'd) — decline honestly instead.
        # isinstance narrowing (not name-string dispatch): pat/case/flags statically typed.
        if _regex_rust_incompatible(pred.pat):
            return None
        prefix = _inline_regex_flag_prefix(pred.case, pred.flags)
        body = f"^(?:{pred.pat})$" if isinstance(pred, Fullmatch) else f"^(?:{pred.pat})"
        return c.str.contains(f"{prefix}{body}", literal=False)

    if name in ("Startswith", "Endswith") and hasattr(pred, "pat") and isinstance(pred.pat, str):
        if getattr(pred, "case", True):
            return c.str.starts_with(pred.pat) if name == "Startswith" else c.str.ends_with(pred.pat)
        # Case-insensitive: anchored (?i) regex on the escaped literal — matches the pandas
        # boundary predicate's lowercase-both-sides semantics for a single str pat.
        anchored = f"(?i)^{re.escape(pred.pat)}" if name == "Startswith" else f"(?i){re.escape(pred.pat)}$"
        return c.str.contains(anchored, literal=False)

    if name in ("Startswith", "Endswith") and hasattr(pred, "pat") and isinstance(pred.pat, (tuple, list)):
        # pandas boundary predicates accept a tuple of prefixes/suffixes (match if ANY) — OR-fold.
        case = getattr(pred, "case", True)
        parts = []
        for p in pred.pat:
            if not isinstance(p, str):
                return None
            if case:
                parts.append(c.str.starts_with(p) if name == "Startswith" else c.str.ends_with(p))
            else:
                anc = f"(?i)^{re.escape(p)}" if name == "Startswith" else f"(?i){re.escape(p)}$"
                parts.append(c.str.contains(anc, literal=False))
        if not parts:
            return pl.lit(False)
        expr = parts[0]
        for p in parts[1:]:
            expr = expr | p
        return expr

    return None


def _is_membership(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset))


def _is_cross_type_predicate(df: "pl.DataFrame", col: str, pred: ASTPredicate) -> bool:
    """True iff the predicate compares a numeric column to a string value (or vice versa):
    polars raises `cannot compare string with numeric type` (an uncatchable Rust panic when
    nested); pandas/cypher return a value/null. Recurses into AllOf (fold of x>a AND x<b) and
    Between (lower+upper) — a cross-type compare hidden inside those would otherwise panic."""
    name = type(pred).__name__
    if name == "AllOf" and hasattr(pred, "predicates"):
        return any(_is_cross_type_predicate(df, col, p) for p in pred.predicates)
    dtype = df.schema.get(col)
    col_num = _dtype_numeric(dtype)
    col_str = _dtype_stringlike(dtype)

    def _mismatch(v: Any) -> bool:
        v_str = isinstance(v, str)
        v_num = isinstance(v, (int, float)) and not isinstance(v, bool)
        return (col_num and v_str) or (col_str and v_num)

    if name == "Between":
        return any(_mismatch(getattr(pred, b, None)) for b in ("lower", "upper"))
    val = getattr(pred, "val", None)
    if val is None or getattr(pred, "op", None) is None:
        return False
    return _mismatch(val)


def filter_by_dict_polars(df: "pl.DataFrame", filter_dict: "Optional[Dict[str, Any]]") -> "pl.DataFrame":
    """Return rows of polars ``df`` matching all entries in ``filter_dict`` via one filter."""
    combined = filter_expr_by_dict_polars(df, filter_dict)
    if combined is None:
        return df
    return df.filter(combined)


def filter_expr_by_dict_polars(df: "pl.DataFrame", filter_dict: "Optional[Dict[str, Any]]") -> "Optional[pl.Expr]":
    """Build the combined boolean ``pl.Expr`` filter_by_dict_polars would apply, or None
    for an empty/absent filter dict. ``df`` supplies the schema for column/dtype
    resolution only — callers may apply the expr to a LazyFrame over the same schema
    (the fused connected-join lane), with identical semantics incl. the same typed
    error/NIE contract for unsupported shapes."""
    import polars as pl

    if not filter_dict:
        return None

    exprs: "List[pl.Expr]" = []
    for col, val in filter_dict.items():
        resolved_col, resolved_val = resolve_filter_column(df, col, val)
        if isinstance(resolved_val, ASTPredicate):
            if _is_cross_type_predicate(df, resolved_col, resolved_val):
                # numeric-vs-string comparison -> polars ComputeError; decline (NIE).
                raise NotImplementedError(
                    f"polars engine does not yet natively support a numeric-vs-string "
                    f"comparison on column {resolved_col!r}; use engine='pandas' for this "
                    f"query (no pandas fallback; parity-or-error by design)"
                )
            # String predicate on a non-`String` column: pandas/cuDF raise a clean
            # GFQLSchemaError (E302) at schema validation, but polars would build `.str.<op>` and
            # raise an OPAQUE `InvalidOperationError` ("expected String type, got: cat") at collect
            # on a Categorical/Enum/numeric column. Raise the SAME clean, typed error so all three
            # engines agree (categorical is treated as non-string here, exactly as filter_by_dict).
            if isinstance(resolved_val, (Contains, Startswith, Endswith, Match)):
                _col_dtype = df.schema.get(resolved_col)
                if _col_dtype is not None and _col_dtype != pl.String:
                    from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError
                    raise GFQLSchemaError(
                        ErrorCode.E302,
                        f'Type mismatch: string predicate used on non-string column "{resolved_col}"',
                        field=col,
                        value=f"{resolved_val.__class__.__name__}(...)",
                        column_type=str(_col_dtype),
                        suggestion='Use numeric predicates like gt() or lt() for numeric columns',
                    )
            expr = predicate_to_expr(resolved_col, resolved_val, df.schema.get(resolved_col))
            if expr is None:
                # decline (NIE): no native lowering for this predicate; no pandas bridge.
                raise NotImplementedError(
                    f"polars engine does not yet natively support the "
                    f"{type(resolved_val).__name__} predicate on column "
                    f"{resolved_col!r}; use engine='pandas' for this query "
                    f"(no pandas fallback; parity-or-error by design)"
                )
            exprs.append(expr)
        elif _is_membership(resolved_val):
            exprs.append(pl.col(resolved_col).is_in(list(resolved_val)))
        elif isinstance(df.schema.get(resolved_col), pl.List):
            if resolved_col == "labels":
                # MATCH (n:Label) = scalar match on the RESERVED `labels` List column ->
                # list.contains (empty for a non-existent label, matching pandas). A plain ==
                # would cast the List to String and crash.
                exprs.append(pl.col(resolved_col).list.contains(resolved_val))
            else:
                # decline (NIE): a user List property vs scalar is NOT membership — pandas
                # compares the whole list (always unequal to a scalar); contains-membership
                # would be a silent wrong answer.
                raise NotImplementedError(
                    f"polars engine does not yet natively support comparing the List "
                    f"column {resolved_col!r} to a scalar; use engine='pandas' "
                    f"(no pandas fallback; parity-or-error by design)"
                )
        else:
            exprs.append(pl.col(resolved_col) == resolved_val)

    if not exprs:
        return None
    combined = exprs[0]
    for e in exprs[1:]:
        combined = combined & e
    return combined
