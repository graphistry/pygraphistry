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
serves and both engines already agreed on it. It is recorded here so the divergence is a choice
with a reason rather than an accident.

Each engine classifies its OWN dtypes (a pandas dtype and a polars ``DataType`` are not
comparable) and then funnels into the one raiser below, so the diagnostic text, the error class
and the ``ErrorCode`` cannot drift between engines.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Final, FrozenSet, NoReturn, Optional

from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

if TYPE_CHECKING:
    import polars as pl

    from graphistry.compute.typing import SeriesT


#: Aggregates Cypher restricts to ``INTEGER | FLOAT | DURATION``. ``mean`` is GFQL's internal
#: spelling of ``avg`` (see ``GFQL_GROUPBY_AGG_METHODS``), so both names must gate.
GFQL_NUMERIC_ONLY_AGGREGATIONS: Final[FrozenSet[str]] = frozenset({"sum", "avg", "mean"})


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
