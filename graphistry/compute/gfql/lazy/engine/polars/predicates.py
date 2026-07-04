"""Vectorized polars filter_by_dict for the native polars GFQL engine.

Predicates are lowered to native polars expressions (no pandas round-trip) for
the common comparison / membership / string / null cases. A predicate with no
native lowering raises ``NotImplementedError`` (NO-CHEATING: no pandas bridge —
silently evaluating one column via pandas would misrepresent pandas behavior as
polars and break the columnar/GPU assumptions; use ``engine='pandas'``). All
filtering is a single vectorized ``df.filter(expr)`` — no per-row work, no Python
materialization.
"""
import operator
import re
from typing import Any, List, Optional

from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.filter_by_dict import resolve_filter_column
from .dtypes import is_numeric as _dtype_numeric, is_stringlike as _dtype_stringlike

# Python-`re` features the Rust regex engine (polars) rejects or evaluates differently:
# lookaround ((?=/(?!/(?<=/(?<!) and backreferences (\\1, (?P=name), \\k<name>). polars would
# ComputeError (lookaround) or silently differ, so a pattern using them must decline (NIE),
# never guess. The pandas engine evaluates these with Python `re`.
_REGEX_RUST_INCOMPAT = re.compile(r"\(\?<?[=!]|\\[1-9]|\(\?P=|\\k<")


def _regex_rust_incompatible(pat: str) -> bool:
    return _REGEX_RUST_INCOMPAT.search(pat) is not None


def _homogeneous_scalar_category(opts: list) -> Optional[str]:
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


def _cmp_expr(col_expr, op, val, dtype=None):
    import datetime as _dt

    # NATIVE TEMPORAL (SAFE SUBSET — NO CHEATING): lower a GFQL ``DateValue`` (Cypher
    # ``date('YYYY-MM-DD')`` or ``p.gt(date(...))``) compared against a NAIVE ``pl.Datetime``
    # column. The pandas oracle truncates the column to its calendar date (``s.dt.date``) and
    # compares to a tz-free python ``date`` (``DateValue._parsed``); ``col_expr.dt.date() <op>
    # pl.lit(date)`` is the exact native equivalent — identical calendar-date truncation, no
    # timezone on EITHER side, so parity is provable. We REQUIRE ``dtype`` (threaded from the
    # frame schema) to confirm the column is a naive ``Datetime``; without that proof we fall
    # through to the decline below. Everything else stays declined: tz-tagged ``DateTimeValue``
    # (always carries a tz; pandas does tz_localize/convert), ``TimeValue``, tz-aware columns,
    # and raw python datetimes — lowering any of those risks a silent tz/semantic mismatch.
    if type(val).__name__ == "DateValue":
        import polars as pl
        d = getattr(val, "_parsed", None)
        if (
            isinstance(dtype, pl.Datetime) and dtype.time_zone is None
            and isinstance(d, _dt.date) and not isinstance(d, _dt.datetime)
        ):
            date_col = col_expr.dt.date()
            lit = pl.lit(d)
            if op is operator.gt:
                return date_col > lit
            if op is operator.lt:
                return date_col < lit
            if op is operator.ge:
                return date_col >= lit
            if op is operator.le:
                return date_col <= lit
            if op is operator.eq:
                return date_col == lit
            if op is operator.ne:
                return date_col != lit
        # naive-Datetime parity unprovable here -> fall through to the decline below.

    # Remaining temporal values (datetime/datetime-tagged/time or the GFQL TemporalValue) have
    # no SAFE polars-literal comparison here — DECLINE (return None → honest NotImplementedError)
    # rather than build `col > TemporalValue`, a non-None broken expr that errors at
    # ``df.filter`` (or silently misorders). Remaining temporal-comparison lowering is a tracked
    # feature gap; numeric/string vals are unaffected.
    if isinstance(val, (_dt.date, _dt.datetime, _dt.time)) or type(val).__name__ in (
        "TemporalValue", "Timestamp", "Timedelta", "datetime64",
        "DateTimeValue", "TimeValue", "DateValue",
    ):
        return None
    # NOTE (narrow residual): these comparisons do NOT apply the IEEE NaN mask that the
    # WHERE/row-pipeline lowering does (``_nan_guard``). On a GENUINE polars NaN (not
    # null), ``col > x`` keeps the NaN row (polars treats NaN as largest) where pandas
    # drops it. This is unreachable on the standard ingestion path — ``from_pandas`` /
    # ``df_to_engine`` convert NaN→null (``nan_to_null``), and filter_by_dict runs on
    # INGESTED property columns (no in-query float math, which is the WHERE path). Only a
    # natively-constructed polars frame carrying raw NaN would diverge; documented rather
    # than guarded to keep the predicate lowering simple. (Mirrors the documented integer
    # ``0/0`` column-compare residual.)
    if op is operator.gt:
        return col_expr > val
    if op is operator.lt:
        return col_expr < val
    if op is operator.ge:
        return col_expr >= val
    if op is operator.le:
        return col_expr <= val
    if op is operator.eq:
        return col_expr == val
    if op is operator.ne:
        return col_expr != val
    return None


def _inline_regex_flag_prefix(case: bool, flags: int) -> str:
    """Translate a pandas-style ``re`` ``flags`` int + ``case`` bool into a Rust-regex
    inline flag prefix like ``(?im)`` (polars' regex engine honors inline flags). Empty
    when nothing applies. Keeps the polars regex lowering faithful to the pandas one."""
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


def predicate_to_expr(col: str, pred: ASTPredicate, dtype=None):
    """Lower an ASTPredicate to a polars boolean expression, or None if unsupported.

    ``dtype`` is the polars dtype of ``col`` (from the frame schema), threaded so the
    temporal lowering can confirm a NAIVE ``Datetime`` column before lowering a ``DateValue``
    comparison (else it declines). ``None`` when the dtype is unknown."""
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
        # NATIVE TEMPORAL Between (SAFE SUBSET — NO CHEATING): the GFQL temporal Between
        # evaluates in pandas as GE/LE (inclusive) or GT/LT (exclusive) sub-predicates on the
        # SAME bound; for a DateValue bound each is the exact date-truncated compare that
        # _cmp_expr already lowers with PROVEN parity (col.dt.date() <op> pl.lit(date)). Compose
        # the two proven endpoint compares so Between inherits that parity with NO new proof
        # obligation. Both endpoints must be DateValue over a NAIVE Datetime column or _cmp_expr
        # returns None -> we fall through to honest NIE (tz-aware DateTimeValue, TimeValue, raw
        # datetime, mixed bounds, non-Datetime dtype all decline this way — never a silent mismatch).
        inclusive = getattr(pred, "inclusive", True)
        lo_expr = _cmp_expr(c, operator.ge if inclusive else operator.gt, lo, dtype)
        hi_expr = _cmp_expr(c, operator.le if inclusive else operator.lt, hi, dtype)
        if lo_expr is not None and hi_expr is not None:
            return lo_expr & hi_expr

    if name == "IsIn" and hasattr(pred, "options"):
        opts = list(pred.options)
        # Only a NON-EMPTY, single-category literal list is parity-safe: polars is_in raises
        # (ComputeError) on a cross-type list (e.g. [1, 'a'] over an Int column), so mixed/empty
        # declines (NIE) — matching the row-pipeline _lower_in discipline.
        if opts and _homogeneous_scalar_category(opts) is not None:
            return c.is_in(opts)
        return None

    if name == "AllOf" and hasattr(pred, "predicates"):
        # Conjunction (e.g. ``n.val > 20 AND n.val < 90`` folds to AllOf[GT, LT]).
        # Lower each child natively and AND them; if ANY child can't lower, the
        # whole predicate can't (caller raises NIE — no pandas bridge).
        child_exprs = [predicate_to_expr(col, p, dtype) for p in pred.predicates]
        if child_exprs and all(e is not None for e in child_exprs):
            combined = child_exprs[0]
            for e in child_exprs[1:]:
                combined = combined & e
            return combined
        return None

    if name in ("IsNull", "IsNA"):
        return c.is_null()
    if name in ("NotNull", "NotNA"):
        return c.is_not_null()

    if name == "IsLeapYear":
        # NATIVE TEMPORAL (SAFE — NO CHEATING): pandas ``s.dt.is_leap_year`` is a Boolean over
        # each value's calendar year; polars ``expr.dt.is_leap_year()`` is documented as the
        # identical Boolean over the same year, for Date and Datetime columns — so parity is
        # provable (Gregorian rule incl. the 1900-not-leap / 2000-leap century cases). REQUIRE a
        # temporal dtype from the frame schema: a NAIVE ``Datetime`` (a tz would have pandas/polars
        # derive the year in LOCAL time, whose equality at year boundaries we have NOT proven) or
        # ``Date``. Anything else (tz-aware Datetime, non-temporal column, unknown dtype) declines
        # -> honest NotImplementedError. Mirrors the naive-Datetime guard used for ``DateValue``.
        if (isinstance(dtype, pl.Datetime) and dtype.time_zone is None) or dtype == pl.Date:
            return c.dt.is_leap_year()
        return None

    if name in ("IsMonthStart", "IsMonthEnd", "IsQuarterStart",
                "IsQuarterEnd", "IsYearStart", "IsYearEnd"):
        # NATIVE TEMPORAL boundary (PROVABLE parity — NO CHEATING): polars has NO is_month_start/.../
        # is_year_end BOOLEAN accessor (month_start()/month_end() ROLL to a Datetime), but the pandas
        # oracle with freq=None is a pure CALENDAR-FIELD test: is_month_start = day==1; is_month_end =
        # day==days_in_month; quarter/year add a month-set / month==N. polars dt.day()/dt.month()/
        # dt.days_in_month() extract the SAME fields from a naive Datetime/Date (both proleptic
        # Gregorian, days_in_month leap-aware), so each compose is BIT-IDENTICAL to the oracle on
        # non-null rows — a proven derivation, not a guess. pandas returns False for NaT; a polars
        # comparison yields null -> fill_null(False) restores parity. Require naive Datetime or Date
        # (a tz shifts the wall-clock fields), exactly like IsLeapYear above; else honest NIE.
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
            # Literal substring: must NOT be regex-interpreted, or a metacharacter
            # (``.``/``*``/``(`` …) over-matches. polars has no literal+case flag, so
            # fold both sides for the case-insensitive literal (matches pandas' result).
            if case:
                return c.str.contains(pred.pat, literal=True)
            return c.str.to_lowercase().str.contains(pred.pat.lower(), literal=True)
        # Regex: mirror pandas' ``case``/``flags`` via a Rust-regex inline flag prefix.
        # Decline (NIE) patterns using Python-re-only features the Rust engine rejects /
        # evaluates differently (lookaround, backreferences) — no silent wrong answer.
        if _regex_rust_incompatible(pred.pat):
            return None
        prefix = _inline_regex_flag_prefix(case, flags)
        return c.str.contains(f"{prefix}{pred.pat}", literal=False)

    if name in ("Startswith", "Endswith") and hasattr(pred, "pat") and isinstance(pred.pat, str):
        if getattr(pred, "case", True):
            return c.str.starts_with(pred.pat) if name == "Startswith" else c.str.ends_with(pred.pat)
        # Case-insensitive: anchored regex on the escaped literal (the literal pat
        # is treated literally; (?i) makes it case-insensitive). Matches the pandas
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

    if name in ("Match", "Fullmatch") and hasattr(pred, "pat") and isinstance(pred.pat, str):
        # pandas str.match = regex anchored at START; str.fullmatch = anchored BOTH ends.
        # Decline when custom regex flags (beyond case) are set, to avoid a flag-semantics gap.
        if getattr(pred, "flags", 0):
            return None
        prefix = "" if getattr(pred, "case", True) else "(?i)"
        body = f"(?:{pred.pat})"
        anchored = f"{prefix}^{body}" if name == "Match" else f"{prefix}^{body}$"
        return c.str.contains(anchored, literal=False)

    return None


def _is_membership(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset))


def _is_cross_type_predicate(df, col: str, pred: ASTPredicate) -> bool:
    """True if a comparison predicate compares a numeric column to a string value
    (or vice versa) — polars raises ``cannot compare string with numeric type`` (an
    uncatchable Rust panic when nested) ; pandas/cypher return a value/null. Recurses
    into ``AllOf`` (the fold of ``x>a AND x<b``) and ``Between`` (lower+upper), since a
    cross-type comparison hidden inside those is otherwise lowered and panics."""
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


def filter_by_dict_polars(df, filter_dict: Optional[dict]):
    """Return rows of polars ``df`` matching all entries in ``filter_dict`` via one filter."""
    import polars as pl

    if not filter_dict:
        return df

    exprs: List[Any] = []
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
            expr = predicate_to_expr(resolved_col, resolved_val, df.schema.get(resolved_col))
            if expr is None:
                # NO-CHEATING: no native lowering for this predicate, and we will
                # NOT bridge through pandas (evaluating one column via pandas would
                # present pandas semantics as polars). Decline honestly.
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
                # Cypher label membership: ``MATCH (n:Label)`` lowers to a scalar match
                # on the RESERVED ``labels`` List column → ``list.contains`` (Label ∈
                # node's labels), empty for a non-existent label, matching pandas. A
                # plain ``==`` would cast the List to String and crash.
                exprs.append(pl.col(resolved_col).list.contains(resolved_val))
            else:
                # A user List-valued property compared to a scalar is NOT membership —
                # pandas compares the whole list (always unequal to a scalar). Don't
                # silently apply contains-membership (wrong answer); decline natively.
                raise NotImplementedError(
                    f"polars engine does not yet natively support comparing the List "
                    f"column {resolved_col!r} to a scalar; use engine='pandas' "
                    f"(no pandas fallback; parity-or-error by design)"
                )
        else:
            exprs.append(pl.col(resolved_col) == resolved_val)

    if not exprs:
        return df
    combined = exprs[0]
    for e in exprs[1:]:
        combined = combined & e
    return df.filter(combined)
