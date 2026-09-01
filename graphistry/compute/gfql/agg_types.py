"""ONE definition of which GFQL aggregates accept which column types, shared by every engine.

WHY ONE MODULE: the aggregate kernels are written three times (pandas/cuDF row pipeline, native
polars row pipeline, OLAP three-hop fast path). Each was inheriting its host dataframe library's
opinion about non-numeric input, so the SAME query returned a value on one engine and raised on
another -- ``avg(<string column>)`` raised ``GFQLTypeError`` on pandas but silently returned
``null`` on polars, and ``sum(<string column>)`` returned the string CONCATENATION on pandas but
leaked a raw ``polars.exceptions.InvalidOperationError`` on polars. Both directions are wrong,
so "match the other engine" was not available: the contract had to be pinned to Cypher.

THE CONTRACT (openCypher / Neo4j, verified against two independent implementations):

  ``avg(input)`` / ``sum(input)``  accept ``INTEGER | FLOAT | DURATION`` (and ``null``) ONLY.
      Neo4j declares exactly that signature for both functions
      (neo4j/docs-cypher ``modules/ROOT/pages/functions/aggregating.adoc``: "Returns the average
      of a set of ``INTEGER``, ``FLOAT`` or ``DURATION`` values", ``input : INTEGER | FLOAT |
      DURATION``; same for ``sum()``), and enforces it at runtime -- Neo4j 5.26.26 answers
      ``RETURN avg(r.s)`` over strings with "AVG(...) can only handle numerical values, duration,
      or null." and ``RETURN sum(date(...))`` with "Type mismatch: expected Float, Integer or
      Duration but was Date". Kuzu 0.11.3 rejects the same two at bind time ("Function AVG did
      not receive correct arguments: Actual: (STRING)").
      => a non-numeric column is a TYPE ERROR, never a null and never a concatenation.

  ``min(input)`` / ``max(input)`` / ``count(input)`` / ``collect(input)`` accept ``ANY``
      (same doc: ``input : ANY``; openCypher TCK ``Aggregation2`` scenarios 7-12 cover
      ``min()``/``max()`` over strings, lists and mixed values).
      => these must NOT raise on strings/categoricals.

  Nulls are excluded from every aggregate; ``sum`` over an empty-or-all-null set is ``0`` and
      ``avg`` over one is ``null`` (Neo4j "Considerations": "``sum(null)`` returns ``0``",
      "``avg(null)`` returns ``null``"; confirmed live on 5.26.26).
      => an all-null column carries no type evidence and must NOT be rejected.

DELIBERATE GFQL EXTENSION, not an oversight: ``sum``/``avg`` over BOOLEAN is a type error in
Neo4j ("expected Float, Integer or Duration but was Boolean") but is accepted here on every
engine, because summing an indicator column is idiomatic in the dataframe surface GFQL also
serves and both engines already agreed on it. It is a strict SUPERSET -- no Cypher-valid query
changes meaning -- so the only cost is documenting it, which the aggregates docs now do.

THE BOOLEAN RETURN-TYPE CONTRACT (adopted 2026-07-28; values AND dtypes, on every engine)::

    sum(BOOLEAN)   -> INTEGER (int64)   count of true, nulls skipped; 0 over zero non-null
    avg(BOOLEAN)   -> FLOAT   (float64) true_count / non_null_count; NULL over zero non-null
    min(BOOLEAN)   -> BOOLEAN           ordering false < true; NULL over zero non-null
    max(BOOLEAN)   -> BOOLEAN           ordering false < true; NULL over zero non-null
    count(BOOLEAN) -> INTEGER (int64)   non-null count

``min``/``max`` are stated as ORDERING, not as a logical fold. ``min == AND`` / ``max == OR`` is a
DERIVATION from ``false < true`` and it gets the empty case backwards: the conventional identity of
AND over zero elements is ``true`` and of OR over zero elements is ``false``, but every engine here
answers NULL -- the same answer ``ORDER BY`` already gives, and the same answer ``min``/``max`` give
over any other empty input.

``sum -> 0`` over zero rows is CONFORMANCE, not a compromise: Cypher's ``sum()`` returns 0 where
SQL's returns NULL, and Cypher's ``avg()`` returns null; the engines here already match Cypher on
both. Pinning the DTYPES is what was still missing -- polars answered ``sum(BOOLEAN)`` and every
``count()`` with ``UInt32`` while pandas/cuDF answered ``int64``, so the values agreed and the
return types did not.

Each engine classifies its OWN dtypes (a pandas dtype and a polars ``DataType`` are not
comparable) and then funnels into the one raiser below, so the diagnostic text, the error class
and the ``ErrorCode`` cannot drift between engines.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Final, FrozenSet, List, Mapping, NoReturn, Optional, Union

from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

if TYPE_CHECKING:
    import polars as pl

    from graphistry.compute.gfql.cypher.ast import CypherScalar
    from graphistry.compute.typing import SeriesT


#: Aggregates Cypher restricts to ``INTEGER | FLOAT | DURATION`` (``mean`` spells ``avg``).
GFQL_NUMERIC_ONLY_AGGREGATIONS: Final[FrozenSet[str]] = frozenset({"sum", "avg", "mean"})

#: What an aggregate answers for an EMPTY group, when it has an answer other than ``null``.
CypherEmptyGroupValue = Union[int, List["CypherScalar"]]

#: Output column -> its empty-group value, for the columns that HAVE one.
CypherEmptyGroupFills = Mapping[str, CypherEmptyGroupValue]

#: Aggregates whose empty-group answer is ``0``.
CYPHER_ZERO_EMPTY_GROUP_AGGREGATIONS: Final[FrozenSet[str]] = frozenset(
    {"count", "count_distinct", "sum"}
)

#: Aggregates whose empty-group answer is ``[]``.
CYPHER_EMPTY_LIST_EMPTY_GROUP_AGGREGATIONS: Final[FrozenSet[str]] = frozenset(
    {"collect", "collect_distinct"}
)

#: Aggregates whose Cypher return type is INTEGER for EVERY input type.
CYPHER_INTEGER_RESULT_AGGREGATIONS: Final[FrozenSet[str]] = frozenset(
    {"count", "count_distinct"}
)


def agg_result_is_integer(func: str, input_is_boolean: bool) -> bool:
    """True when this aggregate's return type is INTEGER (int64) on this input.

    ``count``/``count_distinct`` are INTEGER over ANY input. ``sum`` is INTEGER over BOOLEAN --
    the documented extension counts the true values, so its result is a count, not a boolean.
    ``avg`` stays FLOAT and ``min``/``max`` stay BOOLEAN, so neither is retyped here.
    """
    if func in CYPHER_INTEGER_RESULT_AGGREGATIONS:
        return True
    return func == "sum" and input_is_boolean


def polars_agg_result_cast(func: str, input_dtype: "Optional[pl.DataType]") -> "Optional[pl.DataType]":
    """The dtype polars' own aggregate kernel does NOT produce, or ``None`` when it conforms.

    Polars answers EVERY ``count()`` with ``UInt32`` and ``sum()`` over ``Boolean`` with ``UInt32``,
    where pandas and cuDF answer ``int64`` -- the values agree and the return types do not, which is
    the divergence class the aggregate type contract exists to close. Every OTHER numeric input
    already sums to ``Int64``/``Float64``/``Duration`` on polars, so the ``sum`` half of this cast
    can only fire on a boolean column; the ``count`` half is input-independent on both sides.
    """
    import polars as pl

    is_boolean = input_dtype is not None and input_dtype == pl.Boolean
    return pl.Int64 if agg_result_is_integer(func, is_boolean) else None


def polars_conform_agg_dtype(expr: "pl.Expr", func: str, input_dtype: "Optional[pl.DataType]",
                             alias: str) -> "pl.Expr":
    """Land a polars aggregate on its CONTRACT dtype rather than on its kernel dtype."""
    import polars as pl

    target = polars_agg_result_cast(func, input_dtype)
    if target is None:
        return expr.alias(alias)
    if func == "sum" and input_dtype == pl.Boolean:
        expr = expr.fill_null(0)
    return expr.cast(target).alias(alias)  # hygiene-ok: explicit-cast -- polars dtype conversion


def polars_all_null_agg_literal(func: str, alias: str) -> "pl.Expr":
    """Cypher's all-null answer as a TYPED literal: a bare ``pl.lit(0)`` is ``Int32``, which
    neither pandas nor cuDF ever produces for a ``sum``."""
    import polars as pl

    value = numeric_agg_all_null_value(func)
    return pl.lit(value, dtype=pl.Int64 if value is not None else None).alias(alias)


def _describe_agg_input(column: str, alias: Optional[str]) -> str:
    """How to point the user at the offending value in THEIR query text.

    The cypher lowering materializes ``avg(n.score)``'s argument into an internal
    ``__cypher_agg__`` column before the group-by, so naming the raw column would hand the user
    a name that appears nowhere in their query. When the column is one of those temporaries, name
    the aggregate's output alias instead -- that one they wrote (``... AS score_avg``).
    """
    if alias and column.startswith("__") and column.endswith("__"):
        return f"the argument of {alias!r}"
    return f"column {column!r}"


def raise_non_numeric_aggregation(
    func: str, column: str, dtype: str, alias: Optional[str] = None
) -> NoReturn:
    """The single diagnostic for "this aggregate needs numbers and this input has none".

    Names the OPERATION and the INPUT (and the offending type): an aggregate's output alias is
    usually not its input's column name, so "type error" alone leaves the user grepping.
    """
    target = _describe_agg_input(column, alias)
    raise GFQLTypeError(
        ErrorCode.E302,
        f"Aggregation {func}() requires numeric or duration values, "
        f"but {target} has type {dtype}",
        field=column,
        value=dtype,
        suggestion=(
            f"Cypher restricts {func}() to INTEGER/FLOAT/DURATION; "
            f"use count()/collect()/min()/max() over {target}, or cast it to a number"
        ),
    )


def numeric_agg_all_null_value(func: str) -> Optional[int]:
    """The Cypher answer for a numeric-only aggregate over an ALL-NULL input: ``0`` / ``null``.

    Callers apply this BEFORE the dtype check, because an all-null column carries no type
    evidence and so can never be a type error. It also has to bypass the host kernels entirely:
    pandas answers an all-null column with ``0``/``NaN`` when it is ``object`` but with ``''``,
    ``NaT`` or a ``TypeError`` once it is typed (``string``/``category``/``datetime64``), and
    polars raises for both ``str`` and ``null`` dtypes. One substitution, one answer.
    """
    return 0 if func == "sum" else None


#: Integer widths a pandas/cuDF aggregate may land on that the INTEGER contract widens to int64.
_NARROW_INTEGER_DTYPES: Final[FrozenSet[str]] = frozenset(
    {"int8", "int16", "int32", "uint8", "uint16", "uint32"}
)


def pandas_conform_agg_dtype(result: "SeriesT", func: str, input_is_boolean: bool) -> "SeriesT":
    """Widen a pandas/cuDF aggregate whose kernel answered narrower than the INTEGER contract.

    cuDF's grouped ``nunique`` answers ``int32`` where pandas answers ``int64`` -- the same value
    behind a different return type, on an aggregate Cypher declares INTEGER. Only the narrow
    integer widths are eligible: ``int64``/``Int64`` are already the contract, and a float, boolean
    or object result must never be retyped by this.
    """
    if not agg_result_is_integer(func, input_is_boolean):
        return result
    if str(getattr(result, "dtype", "")).lower() not in _NARROW_INTEGER_DTYPES:
        return result
    return result.astype("int64")  # hygiene-ok: explicit-cast -- dataframe dtype conversion


def pandas_agg_kernel_null_fill(func: str, series: "SeriesT") -> Optional[int]:
    """The value a pandas/cuDF aggregate kernel's NULL answer must be repaired to, else ``None``.

    Cypher's ``sum()`` never returns null -- 0 is its zero-row answer -- but cuDF's grouped ``sum``
    over a group with no non-null values answers ``<NA>``, on boolean AND on ``Int64``/``float64``,
    where pandas answers 0. The two engines therefore disagreed on a VALUE, not merely a dtype, on
    exactly the all-null row. Applied to the kernel's OUTPUT, so it repairs the per-group answer
    that :func:`numeric_agg_all_null_value` (a whole-column pre-substitution) cannot see.
    """
    if func != "sum":
        return None
    if str(getattr(series, "dtype", "")).lower() == "object":
        return None  # untyped kernel answer; the object-bool retype already owns this column
    return 0


def pandas_dtype_is_numeric_for_agg(series: "SeriesT") -> bool:
    """True when the pandas/cuDF dtype ITSELF proves the column is a valid sum/avg input.

    The cheap, O(1), hot-path answer: a column that passes here needs no data inspection at all
    -- no null scan, no value sampling -- so an ordinary numeric aggregate pays a string check
    and nothing else. Everything that fails here (object, string, categorical, temporal) is
    already headed for a slow or erroring path, which is where the O(n) questions get asked.
    ``timedelta`` is Cypher's ``DURATION``; ``bool`` is the documented GFQL extension.
    """
    dtype_txt = str(getattr(series, "dtype", "")).lower()
    if "interval" in dtype_txt or "datetime" in dtype_txt or "period" in dtype_txt:
        return False
    if "timedelta" in dtype_txt or "duration" in dtype_txt:
        return True
    if dtype_txt in {"bool", "boolean"}:
        return True
    return any(token in dtype_txt for token in ("int", "float", "double", "decimal"))


def pandas_non_numeric_agg_dtype(series: "SeriesT") -> Optional[str]:
    """Dtype label if this pandas/cuDF column must be REJECTED by ``sum``/``avg``, else ``None``.

    Deliberately a DENY list keyed on positively-identified non-numeric types (string dtype,
    categorical, datetime/date) rather than an allow list of numerics: pandas ``object`` columns
    routinely carry numbers through the cypher property path, and demanding positive numeric
    proof would start rejecting queries that compute correctly today. ``timedelta64`` is Cypher's
    ``DURATION`` and is allowed. All-null columns are the callers' job
    (:func:`numeric_agg_all_null_value`), applied before this.
    """
    dtype = getattr(series, "dtype", None)
    dtype_txt = str(dtype).lower()
    if "datetime" in dtype_txt or dtype_txt in {"date32[day][pyarrow]", "date64[ms][pyarrow]"}:
        return str(dtype)
    if "category" in dtype_txt:
        return str(dtype)
    # Prefix match, not a fixed set: pandas spells its string dtype differently across versions and
    # storages -- `object` on pandas 2, `str` by default on pandas 3, plus `string`,
    # `string[pyarrow]`, `string[python]`, `str[pyarrow]`. A missed spelling here fails OPEN
    # (delegates to the kernel, restoring the concatenation), so the check is deliberately broad.
    if dtype_txt.startswith("str") or dtype_txt.startswith("large_string"):
        return str(dtype)
    if dtype_txt == "object" and _object_series_is_str_like(series):
        return "object (strings)"
    return None


def pandas_object_series_is_bool_like(series: "SeriesT") -> bool:
    """Object-dtype column whose sampled non-null values are all booleans.

    ``sum()`` over BOOLEAN is the documented GFQL extension above, but pandas'
    OBJECT-dtype groupby reduction hands a single-row group back as the raw bool,
    mixing ``2`` and ``False`` in one output column. Callers use this probe to
    retype that aggregate to a numeric column. Bounded head sample, same tradeoff
    as :func:`_object_series_is_str_like`; cuDF cannot hold an object-of-bools
    column (its object dtype is strings), so this never fires there.
    """
    if str(getattr(series, "dtype", "")).lower() != "object" or not hasattr(series, "dropna"):
        return False
    import numpy as np

    from graphistry.Engine import series_to_pylist
    values = series_to_pylist(series.dropna().head(128))
    return len(values) > 0 and all(isinstance(v, (bool, np.bool_)) for v in values)


def _object_series_is_str_like(series: "SeriesT") -> bool:
    """Every non-null value in a bounded head sample is a ``str`` (and there is at least one).

    Bounded so the check costs the same on a 10-row and a 10M-row frame, and empty-after-dropna
    returns False so all-null columns fall through to the ``sum(null) == 0`` contract.
    """
    if not hasattr(series, "dropna"):
        return isinstance(series, str)
    from graphistry.Engine import series_to_pylist
    values = series_to_pylist(series.dropna().head(128))
    return len(values) > 0 and all(isinstance(v, str) for v in values)


def polars_non_numeric_agg_dtype(dtype: "Optional[pl.DataType]") -> Optional[str]:
    """Dtype label if this polars column must be REJECTED by ``sum``/``avg``, else ``None``.

    Mirrors :func:`pandas_non_numeric_agg_dtype` as a DENY list over the same type families so
    the two engines cannot classify one column differently: String/Categorical/Enum, temporal
    Date/Datetime/Time, and the composite dtypes (List/Array/Struct/Binary/Object). ``Duration``
    is Cypher's ``DURATION`` and is allowed; ``Null`` never reaches here because an all-null
    column is short-circuited by :func:`numeric_agg_all_null_value` first.
    """
    import polars as pl

    from .lazy.engine.polars.dtypes import is_stringlike

    if dtype is None:
        return None
    if is_stringlike(dtype):
        return str(dtype)
    for name in ("Date", "Datetime", "Time", "List", "Array", "Struct", "Binary", "Object"):
        candidate = getattr(pl, name, None)
        if candidate is not None and (dtype == candidate or isinstance(dtype, candidate)):
            return str(dtype)
    return None
