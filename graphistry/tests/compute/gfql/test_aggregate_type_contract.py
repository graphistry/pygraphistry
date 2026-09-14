"""The Cypher type contract for GFQL aggregates, enforced identically on every engine.

WHY THIS FILE EXISTS: the aggregate kernels are written three times (pandas/cuDF row pipeline,
native polars row pipeline, OLAP three-hop fast path) and each had inherited its host library's
opinion about non-numeric input, so the SAME query answered differently per engine:

  ``avg(<string column>)``  pandas raised, polars returned a silent ``null``
  ``sum(<string column>)``  pandas returned the CONCATENATION ('abac'),
                            polars leaked a raw ``polars.exceptions.InvalidOperationError``

Both directions are wrong, so the fix could not be "match the other engine" -- the contract is
pinned to Cypher (see ``graphistry/compute/gfql/agg_types.py`` for the Neo4j 5.26.26 and Kuzu
0.11.3 receipts). This module is the executable form of that contract: a positive lane (what MUST
compute), a negative lane (what MUST raise, with which typed error), and a differential matrix
over aggregate x dtype that fails on ANY pandas-vs-other-engine disagreement.

ENGINE COVERAGE: parametrized over pandas / polars / cudf / polars-gpu. The GPU engines SKIP with
an explicit reason when their stack is absent -- run the GPU lane on a GPU box
(``graphistry/test-rapids-official:26.02-gfql-polars`` with ``--gpus all``) to close them; a
skipped GPU param states its own boundary in the pytest report rather than passing quietly.
Deliberately does NOT use ``available_nonpandas_engines()``: that helper SHRINKS the parametrization
silently when a stack is missing, so a lane can vanish without any signal.

THE BOOLEAN LANE at the bottom of this file pins the documented ``sum``/``avg``-over-BOOLEAN
extension as VALUES AND RETURN TYPES. Values alone were already uniform; the return types were not
(polars answered ``sum``/``count`` with ``UInt32``, pandas and cuDF with ``int64``), and cuDF
answered ``sum`` over a group with no non-null values with NULL where Cypher says 0 -- a VALUE
divergence that only an exercised GPU arm could find.
"""
import datetime

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

pl = pytest.importorskip("polars")


ALL_ENGINES = ["pandas", "polars", "cudf", "polars-gpu"]


def _require_engine(engine: str) -> None:
    """Skip with a NAMED reason so an absent GPU stack is visible in the report, not silent."""
    if engine == "cudf":
        pytest.importorskip("cudf", reason="cudf engine lane requires a GPU box (--gpus all)")
    if engine == "polars-gpu":
        pytest.importorskip("cudf", reason="polars-gpu lane requires a GPU box (--gpus all)")
        import importlib.util
        if importlib.util.find_spec("cudf_polars") is None:
            pytest.skip("polars-gpu lane requires cudf_polars (RAPIDS 26.02+ image)")


# Every column here is exercised by BOTH the positive and the negative lane; which side a column
# lands on is exactly the contract under test.
_STR = ["a", "b", "a", "c", "b", "d"]
# Column names are deliberately multi-letter and distinct from the pattern aliases used below
# (a / b / e) and from the edge endpoint columns: a node column named `b` shadows the `(b)` alias
# in `MATCH (a)-[e]->(b)` and silently aggregates the wrong values.
_NUMERIC_COLS = {
    "int_col": [1, 2, 3, 4, 5, 6],
    "float_col": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    "bool_col": [True, False, True, True, False, False],
    "dur_col": pd.to_timedelta([1, 2, 3, 4, 5, 6], unit="D"),
    "nullint_col": [1, None, 3, None, 5, 6],
}
_NON_NUMERIC_COLS = {
    "str_col": _STR,
    "nullstr_col": ["a", None, "a", None, "b", None],
    "cat_col": pd.Categorical(_STR),
    "date_col": pd.to_datetime([f"2020-01-0{i}" for i in range(1, 7)]),
}
_ALL_NULL_COLS = {"allnull_col": [None] * 6}


def _graph():
    data = {"id": list(range(6)), "grp": ["x", "x", "x", "y", "y", "y"]}
    data.update(_NUMERIC_COLS)
    data.update(_NON_NUMERIC_COLS)
    data.update(_ALL_NULL_COLS)
    nodes = pd.DataFrame(data)
    edges = pd.DataFrame({"src": [0, 1, 2, 3], "dst": [1, 2, 3, 4]})
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def _cells(df):
    """Engine-neutral, dtype-neutral cell values: [(col, value), ...] rows, sorted.

    Values are normalized (numpy/polars scalars -> python, NaN/NaT -> None, temporal -> iso,
    list-likes -> list) so the matrix compares SEMANTICS. Numeric repr width (int64 vs uint32) is
    a separate, already-decided concern and must not masquerade as a contract violation here.
    """
    import numpy as np

    def norm(v):
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, (pd.Timestamp, datetime.date, datetime.datetime, np.datetime64)):
            return None if pd.isna(v) else str(pd.Timestamp(v))
        if isinstance(v, (pd.Timedelta, datetime.timedelta, np.timedelta64)):
            # nanoseconds, NOT str(): pandas renders a Timedelta "4 days 00:00:00" and polars
            # hands back a python timedelta whose str() is "4 days, 0:00:00" -- a repr gap that
            # would read as a value divergence.
            return None if pd.isna(v) else int(pd.Timedelta(v).value)
        if isinstance(v, str):
            return v
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            return [norm(x) for x in list(v)]
        if v is None:
            return None
        if isinstance(v, bool):
            return bool(v)   # NOT folded into the numeric tag below: True must never equal 1
        if isinstance(v, (int, float)):
            # One tag for int and float: whether an aggregate lands as int64 or float64 is the
            # separate, already-decided nullable-merge dtype question (#1796 BU1), and letting it
            # fail here would bury the type-contract signal this matrix exists to carry.
            return None if v != v else ("num", round(float(v), 9))
        return None if pd.isna(v) else v

    if df is None:
        return None
    if "polars" in type(df).__module__:
        cols, records = df.columns, df.to_dicts()
    else:
        if hasattr(df, "to_pandas"):  # cudf
            df = df.to_pandas()
        cols, records = list(df.columns), [row for _, row in df.iterrows()]
    return sorted(tuple((c, norm(row[c])) for c in cols) for row in records)


def _run(g, query, engine):
    """('ok', cells) | ('raise', ExcTypeName). NOT a try/except that hides failures: the class
    name IS the assertion payload, so an engine swapping a value for an error still fails."""
    try:
        return ("ok", _cells(g.gfql(query, engine=engine)._nodes))
    except Exception as exc:  # noqa: BLE001 - the exception TYPE is the thing under test
        return ("raise", type(exc).__module__.split(".")[0] + "." + type(exc).__name__)


# --------------------------------------------------------------------------------------
# NEGATIVE lane: what MUST raise, and with which typed error
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("func", ["sum", "avg"])
@pytest.mark.parametrize("col", sorted(_NON_NUMERIC_COLS))
@pytest.mark.parametrize("grouped", [True, False])
def test_numeric_only_aggregate_over_non_numeric_column_raises(engine, func, col, grouped):
    """Neo4j: "SUM(...) can only handle numerical values, duration, or null." -- a GFQLTypeError,
    never a null (polars' old answer for avg) and never a concatenation (pandas' old answer for
    sum), on every engine and both the grouped and the whole-table shape."""
    _require_engine(engine)
    g = _graph()
    query = (f"MATCH (n) RETURN n.grp AS grp, {func}(n.{col}) AS agg_out ORDER BY grp" if grouped
             else f"MATCH (n) RETURN {func}(n.{col}) AS agg_out")
    with pytest.raises(GFQLTypeError) as excinfo:
        g.gfql(query, engine=engine)
    assert excinfo.value.code == ErrorCode.E302
    message = str(excinfo.value)
    assert f"{func}()" in message, message           # names the OPERATION
    assert "agg_out" in message, message             # names the user's alias
    assert "numeric or duration" in message, message


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_no_raw_polars_exception_reaches_the_gfql_surface(engine):
    """``sum(<string column>)`` used to surface ``polars.exceptions.InvalidOperationError``
    verbatim. A third-party exception class is never the GFQL surface -- the pandas side has
    always been wrapped by execute_call, and the native polars path must match."""
    _require_engine(engine)
    with pytest.raises(GFQLTypeError) as excinfo:
        _graph().gfql("MATCH (n) RETURN sum(n.str_col) AS agg_out", engine=engine)
    assert "polars" not in type(excinfo.value).__module__


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_native_polars_row_op_wraps_any_polars_error(engine, monkeypatch):
    """The choke point itself, independent of the dtype guard above: ANY polars error raised
    inside a native row op is rewrapped as GFQLTypeError(E303) -- the same code/message shape
    execute_call gives the pandas surface -- instead of escaping as polars.exceptions.*.

    Injected rather than provoked on purpose: the dtype guard now prevents the one polars error
    this surface was known to raise, so a "find a query that still breaks polars" test would be
    testing today's gap list rather than the invariant. The invariant is that the NEXT such error
    is wrapped too."""
    _require_engine(engine)
    import graphistry.compute.gfql.lazy.engine.polars.chain as polars_chain

    def boom(*_args, **_kwargs):
        raise pl.exceptions.ComputeError("injected polars failure")

    monkeypatch.setattr(polars_chain, "_try_native_row_op", boom)
    with pytest.raises(GFQLTypeError) as excinfo:
        _graph().gfql("MATCH (n) RETURN n.grp AS grp, count(n.int_col) AS agg_out", engine=engine)
    assert excinfo.value.code == ErrorCode.E303
    assert "injected polars failure" in str(excinfo.value)
    assert "polars" not in type(excinfo.value).__module__


def test_polars_error_base_is_resolvable():
    """``_polars_error_types()`` returning () would make the wrap above match nothing and fail
    open -- silently restoring the raw-exception leak."""
    from graphistry.compute.gfql.lazy.engine.polars.chain import _polars_error_types
    types = _polars_error_types()
    assert types, "polars error base not found -- the except clause would match nothing"
    assert issubclass(pl.exceptions.InvalidOperationError, types[0])


# --------------------------------------------------------------------------------------
# POSITIVE lane: what MUST still compute
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("func", ["sum", "avg"])
@pytest.mark.parametrize("col", sorted(_NUMERIC_COLS))
def test_numeric_only_aggregate_over_numeric_column_computes(engine, func, col):
    """The guard must not become a blanket rejection: INTEGER / FLOAT / DURATION all aggregate,
    and so does BOOLEAN -- a deliberate GFQL extension over Cypher (see agg_types.py)."""
    _require_engine(engine)
    g = _graph()
    got = _run(g, f"MATCH (n) RETURN n.grp AS grp, {func}(n.{col}) AS agg_out ORDER BY grp", engine)
    assert got[0] == "ok", got
    assert len(got[1]) == 2


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("func", ["min", "max", "count", "collect"])
@pytest.mark.parametrize("col", sorted(_NON_NUMERIC_COLS))
def test_any_typed_aggregate_accepts_non_numeric_columns(engine, func, col):
    """Cypher declares min/max/count/collect over ``ANY`` (openCypher TCK Aggregation2 [7]-[12]
    covers min/max over strings, lists and mixed values). Tightening sum/avg must not tighten
    these -- and pandas' grouped min/max over a CATEGORICAL, which used to raise, now answers."""
    _require_engine(engine)
    g = _graph()
    got = _run(g, f"MATCH (n) RETURN n.grp AS grp, {func}(n.{col}) AS agg_out ORDER BY grp", engine)
    assert got[0] == "ok", got


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_min_max_over_categorical_returns_lexicographic_values(engine):
    """Concrete values, not just "did not raise": pandas raised here before, so a regression
    could otherwise hide behind an empty/None result."""
    _require_engine(engine)
    g = _graph()
    got = _run(g, "MATCH (n) RETURN n.grp AS grp, min(n.cat_col) AS lo, max(n.cat_col) AS hi ORDER BY grp",
               engine)
    assert got == ("ok", [(("grp", "x"), ("lo", "a"), ("hi", "b")),
                          (("grp", "y"), ("lo", "b"), ("hi", "d"))]), got


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_categorical_is_decategorized_before_aggregating_on_polars(engine):
    """The RESULT DTYPE, not just the values: a categorical must reach the aggregate as a plain
    string column. Pinned as a dtype because the failure it prevents is version-dependent and
    therefore invisible to a value assertion on a modern polars -- polars 1.35.2 (the RAPIDS
    26.02 image) PANICS in the rust core on a grouped min/max over a Categorical
    (`categorical.rs: not implemented`), and a pyo3 PanicException is not even a polars
    exception, so no python-side wrapper can turn it into a GFQL error."""
    _require_engine(engine)
    out = _graph().gfql(
        "MATCH (n) RETURN n.grp AS grp, min(n.cat_col) AS lo ORDER BY grp", engine=engine)
    assert str(out._nodes.schema["lo"]) == "String", out._nodes.schema


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_polars_native_null_dtype_column_follows_the_all_null_contract(engine):
    """A polars-NATIVE all-null column carries dtype ``Null`` (a pandas all-None object column
    arrives typed ``String`` instead), and polars refuses `sum`/`mean` on it outright. Distinct
    input, same cypher answer: 0 / null."""
    _require_engine(engine)
    nodes = pl.DataFrame({"id": [0, 1, 2, 3], "grp": ["x", "x", "y", "y"],
                          "nul": pl.Series("nul", [None] * 4, dtype=pl.Null)})
    edges = pl.DataFrame({"src": [0, 1], "dst": [1, 2]})
    assert nodes.schema["nul"] == pl.Null
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    query = "MATCH (n) RETURN n.grp AS grp, sum(n.nul) AS s, avg(n.nul) AS a ORDER BY grp"
    got = _run(g, query, engine)
    assert got == ("ok", [(("grp", "x"), ("s", ("num", 0.0)), ("a", None)),
                          (("grp", "y"), ("s", ("num", 0.0)), ("a", None))]), got
    # The substituted 0 is a LITERAL, not a kernel answer, so it carries whatever dtype the
    # literal was built with -- a bare `pl.lit(0)` is Int32, a width pandas/cuDF never produce.
    assert _dtype_kind(g.gfql(query, engine=engine)._nodes, "s") == "int64"


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_all_null_column_sums_to_zero_and_averages_to_null(engine):
    """Neo4j: "``sum(null)`` returns ``0``", "``avg(null)`` returns ``null``" -- verified live on
    5.26.26. An all-null column carries NO type evidence, so it is never a type error; it also
    cannot be delegated to the host kernels, which answer it 0 / '' / NaT / TypeError depending
    on dtype (pandas) or raise for both str and null dtypes (polars)."""
    _require_engine(engine)
    g = _graph()
    query = ("MATCH (n) RETURN n.grp AS grp, sum(n.allnull_col) AS s, avg(n.allnull_col) AS a "
             "ORDER BY grp")
    got = _run(g, query, engine)
    assert got == ("ok", [(("grp", "x"), ("s", ("num", 0.0)), ("a", None)),
                          (("grp", "y"), ("s", ("num", 0.0)), ("a", None))]), got
    # Same substituted-literal dtype question as the polars-native Null column above, reached by
    # the OTHER branch: this column arrives typed (an all-None pandas object column lands String).
    assert _dtype_kind(g.gfql(query, engine=engine)._nodes, "s") == "int64"


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_empty_result_is_not_treated_as_all_null(engine):
    """A 0-row frame is "no data", not "all null": the all-null substitution must NOT fire, or
    the 0-row schema fill loses the source dtype (an untyped None gives object, which upcasts
    sum/avg and escapes via UNION ALL -- the failure mode
    test_connected_join_empty_edge_aggregate_keeps_numeric_dtype guards)."""
    _require_engine(engine)
    g = _graph()
    got = _run(g, "MATCH (n) WHERE n.int_col > 9999 RETURN sum(n.int_col) AS s", engine)
    assert got[0] == "ok", got


# --------------------------------------------------------------------------------------
# DIFFERENTIAL matrix: aggregate x dtype, pandas oracle vs every other engine
# --------------------------------------------------------------------------------------

_MATRIX_COLS = sorted({**_NUMERIC_COLS, **_NON_NUMERIC_COLS, **_ALL_NULL_COLS})
_MATRIX_AGGS = ["count", "sum", "avg", "min", "max", "collect"]


@pytest.mark.parametrize("engine", ["polars", "cudf", "polars-gpu"])
@pytest.mark.parametrize("func", _MATRIX_AGGS)
@pytest.mark.parametrize("col", _MATRIX_COLS)
def test_aggregate_dtype_matrix_matches_pandas_oracle(engine, func, col):
    """The whole aggregate x dtype cross-product must agree with pandas on BOTH axes an engine
    can disagree on: the VALUE when it computes, and the ERROR CLASS when it raises. This is the
    gate that would have caught the original two divergences -- and the eight sibling cells the
    sweep turned up alongside them (sum over date/categorical/all-null, avg over
    nullable-string/categorical, min/max over categorical)."""
    _require_engine(engine)
    g = _graph()
    query = f"MATCH (n) RETURN n.grp AS grp, {func}(n.{col}) AS agg_out ORDER BY grp"
    oracle = _run(g, query, "pandas")
    got = _run(g, query, engine)
    assert got == oracle, f"{func}(n.{col}) on {engine}: {got} != pandas {oracle}"


# --------------------------------------------------------------------------------------
# The OLAP single-hop grouped fast path -- a SEPARATE aggregate implementation
# --------------------------------------------------------------------------------------

def _fast_path_graph():
    """The shape ``_execute_single_hop_grouped_aggregate_fast_path`` actually accepts: labelled
    endpoints + a labelled edge + a direct grouped RETURN. Verified engaged by the spy below --
    a fast-path test written on a shape the fast path declines tests nothing."""
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 10, 11],
        "node_type": ["Person", "Person", "Person", "City", "City"],
        "age": [20, 30, 40, None, None],
        "nick": ["ann", "bob", "cat", None, None],
        "city": [None, None, None, "NYC", "LA"],
    })
    edges = pd.DataFrame({"s": [0, 1, 2], "d": [10, 10, 11], "rel": ["LIVES_IN"] * 3})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


# NOTE the placeholders are <AGG>/<ARG> and substitution is str.replace, NOT str.format: the
# query text contains cypher property maps ({node_type:'Person'}), which .format() would try to
# interpolate -- it raises KeyError('node_type') before the query is ever run.
_FAST_PATH_QUERY = (
    "MATCH (p {node_type:'Person'})-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
    "RETURN c.city AS city, <AGG>(<ARG>) AS agg_out ORDER BY city"
)


def _fast_path_query(agg: str, arg: str) -> str:
    return _FAST_PATH_QUERY.replace("<AGG>", agg).replace("<ARG>", arg)


def _run_watching_fast_path(g, query, engine):
    """Run, and report whether the fast path was ENTERED (not merely whether it returned rows).

    Entered-ness is recorded before the call so a fast path that raises still counts -- the
    negative lane needs exactly that, and a post-hoc `is not None` check would record nothing.
    """
    import graphistry.compute.gfql_unified as gu
    entered = []
    original = gu._execute_single_hop_grouped_aggregate_fast_path

    def spy(*args, **kwargs):
        entered.append(True)
        return original(*args, **kwargs)

    gu._execute_single_hop_grouped_aggregate_fast_path = spy
    try:
        result = _run(g, query, engine)   # BEFORE reading `entered`: a tuple literal would
        return bool(entered), result      # evaluate bool(entered) first and always report False
    finally:
        gu._execute_single_hop_grouped_aggregate_fast_path = original


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_fast_path_is_actually_engaged_by_this_shape(engine):
    """Canary for the two tests below: if the fast path stops accepting this shape they would
    silently start testing the ordinary row pipeline instead, and pass for the wrong reason."""
    _require_engine(engine)
    entered, got = _run_watching_fast_path(
        _fast_path_graph(), _fast_path_query("avg", "p.age"), engine)
    assert entered, "fast path not engaged -- the fast-path lane below would be testing nothing"
    assert got[0] == "ok", got


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("func", ["sum", "avg"])
def test_fast_path_rejects_non_numeric_aggregates(engine, func):
    """The fast path reimplements the aggregates, so without its own guard whether an ill-typed
    aggregate is caught would depend on whether the query happened to match the fast-path shape.
    Both engine branches of the fast path are covered (pandas/cuDF and polars)."""
    _require_engine(engine)
    entered, got = _run_watching_fast_path(
        _fast_path_graph(), _fast_path_query(func, "p.nick"), engine)
    assert entered, "fast path not engaged -- this test would not be exercising its guard"
    assert got == ("raise", "graphistry.GFQLTypeError"), got


# (func, arg) pairs the fast path ACCEPTS. A SOLE min()/max() aggregate does not compile at all
# ("Cypher row lowering cannot ..."), and aggregating the group key itself declines the fast path
# and then fails in the row pipeline -- both pre-existing, both unrelated to types, so the lane is
# pinned to the pairs that actually reach the code under test (min/max still get fast-path
# coverage via the multi-aggregate RETURN in the pinned-value test below).
_FAST_PATH_CASES = [("sum", "p.age"), ("avg", "p.age"), ("count", "p.age"),
                    ("sum", "p.nick"), ("avg", "p.nick"), ("count", "p.nick")]


@pytest.mark.parametrize("engine", ["polars", "cudf", "polars-gpu"])
@pytest.mark.parametrize("func,arg", _FAST_PATH_CASES)
def test_fast_path_matches_pandas_oracle(engine, func, arg):
    """The fast path has an engine split of its own (a polars branch and a pandas/cuDF branch),
    so its two aggregate implementations get the same differential treatment as the row
    pipeline's -- over a numeric, a string and the group-key column."""
    _require_engine(engine)
    g = _fast_path_graph()
    query = _fast_path_query(func, arg)
    oracle_entered, oracle = _run_watching_fast_path(g, query, "pandas")
    entered, got = _run_watching_fast_path(g, query, engine)
    assert oracle_entered and entered, "fast path not engaged on both engines"
    assert got == oracle, f"{func}({arg}) on {engine}: {got} != pandas {oracle}"


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_fast_path_numeric_aggregates_keep_their_values(engine):
    """Pinned values, so the guard cannot be "passed" by an implementation that stopped
    computing: the fast path must still answer the numeric aggregates it was written for."""
    _require_engine(engine)
    entered, got = _run_watching_fast_path(
        _fast_path_graph(),
        "MATCH (p {node_type:'Person'})-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "RETURN c.city AS city, sum(p.age) AS s, avg(p.age) AS a, max(p.nick) AS m ORDER BY city",
        engine)
    assert entered
    assert got == ("ok", [(("city", "LA"), ("s", ("num", 40.0)), ("a", ("num", 40.0)), ("m", "cat")),
                          (("city", "NYC"), ("s", ("num", 50.0)), ("a", ("num", 25.0)), ("m", "bob"))]), got


# --------------------------------------------------------------------------------------
# The classifiers themselves
# --------------------------------------------------------------------------------------

def test_pandas_fast_numeric_gate_admits_exactly_the_kernel_safe_dtypes():
    """The O(1) hot-path gate. A FALSE POSITIVE here is the dangerous direction -- it would send
    a non-numeric column straight to the host kernel, restoring the concatenation."""
    from graphistry.compute.gfql.agg_types import pandas_dtype_is_numeric_for_agg as numeric
    assert numeric(pd.Series([1, 2]))
    assert numeric(pd.Series([1.5]))
    assert numeric(pd.Series([1], dtype="Int64"))
    assert numeric(pd.Series([True]))
    assert numeric(pd.Series([1], dtype="boolean"))
    assert numeric(pd.to_timedelta([1], unit="D").to_series())
    assert not numeric(pd.Series(["a"]))
    assert not numeric(pd.Series(["a"], dtype="object"))
    assert not numeric(pd.Series(["a"], dtype="string"))
    assert not numeric(pd.Series(pd.Categorical(["a"])))
    assert not numeric(pd.Series(pd.to_datetime(["2020-01-01"])))
    assert not numeric(pd.Series([None, None]))                       # object: no type evidence
    assert not numeric(pd.Series(pd.period_range("2020", periods=1, freq="D")))
    assert not numeric(pd.Series(pd.interval_range(0, 2, periods=1)))


def test_pandas_classifier_rejects_strings_and_admits_numbers_and_durations():
    from graphistry.compute.gfql.agg_types import pandas_non_numeric_agg_dtype as reject
    assert reject(pd.Series([1, 2, 3])) is None
    assert reject(pd.Series([1.0, 2.0])) is None
    assert reject(pd.Series([True, False])) is None
    assert reject(pd.to_timedelta([1, 2], unit="D").to_series()) is None
    assert reject(pd.Series([1, 2], dtype="object")) is None      # numbers boxed in object
    # The LABEL is the dtype's own repr and pandas changes it across versions (a bare
    # pd.Series(["a"]) is `object` on pandas 2 and `str` on pandas 3), so assert the VERDICT and
    # that the label carries the dtype -- pinning the exact text tests pandas, not this contract.
    assert reject(pd.Series(["a", "b"], dtype="object")) == "object (strings)"
    assert reject(pd.Series(["a", "b"])) is not None
    assert reject(pd.Series(["a"], dtype="string")) == "string"
    # every string spelling pandas has used across versions/storages must reject; a missed one
    # fails OPEN (delegates to the kernel and restores the concatenation)
    for spelling in ["object", "str", "string", "string[python]", "string[pyarrow]"]:
        try:
            series = pd.Series(["a", "b"], dtype=spelling)
        except (TypeError, ValueError):
            continue   # dtype not available in this pandas/pyarrow build
        assert reject(series) is not None, spelling
    assert reject(pd.Series(pd.Categorical(["a"]))) is not None
    assert reject(pd.Series(pd.to_datetime(["2020-01-01"]))) is not None


def test_polars_classifier_rejects_strings_and_admits_numbers_and_durations():
    from graphistry.compute.gfql.agg_types import polars_non_numeric_agg_dtype as reject
    assert reject(pl.Int64) is None
    assert reject(pl.Float64) is None
    assert reject(pl.Boolean) is None
    assert reject(pl.Duration) is None
    assert reject(pl.Null) is None          # all-null: no type evidence, never a type error
    assert reject(None) is None
    assert reject(pl.String) == "String"
    assert reject(pl.Categorical) is not None
    assert reject(pl.Date) is not None
    assert reject(pl.Datetime) is not None
    assert reject(pl.List(pl.Int64)) is not None


def test_all_null_substitution_values_follow_cypher():
    from graphistry.compute.gfql.agg_types import numeric_agg_all_null_value
    assert numeric_agg_all_null_value("sum") == 0
    assert numeric_agg_all_null_value("avg") is None
    assert numeric_agg_all_null_value("mean") is None


def test_empty_group_aggregation_sets_cover_both_distinct_spellings():
    """Cypher's non-null empty-group answers: count/sum -> 0, collect -> []. A set holding
    only the plain spelling would leave the ``*_distinct`` output unfilled (NULL, wrong)."""
    from graphistry.compute.gfql.agg_types import (
        CYPHER_EMPTY_LIST_EMPTY_GROUP_AGGREGATIONS,
        CYPHER_ZERO_EMPTY_GROUP_AGGREGATIONS,
    )
    assert CYPHER_ZERO_EMPTY_GROUP_AGGREGATIONS == {"count", "count_distinct", "sum"}
    assert CYPHER_EMPTY_LIST_EMPTY_GROUP_AGGREGATIONS == {"collect", "collect_distinct"}
    assert not (
        CYPHER_ZERO_EMPTY_GROUP_AGGREGATIONS & CYPHER_EMPTY_LIST_EMPTY_GROUP_AGGREGATIONS
    )


# --------------------------------------------------------------------------------------
# The BOOLEAN extension: sum/avg over BOOLEAN, pinned as VALUES **and** RETURN TYPES
# --------------------------------------------------------------------------------------

#: The contract (agg_types.py): sum -> INTEGER, avg -> FLOAT, min/max -> BOOLEAN, count -> INTEGER.
_BOOL_RESULT_DTYPES = {"s": "int64", "a": "float64", "mn": "bool", "mx": "bool", "c": "int64"}

_BOOL_AGGS = ("sum(n.flag) AS s, avg(n.flag) AS a, min(n.flag) AS mn, "
              "max(n.flag) AS mx, count(n.flag) AS c")

#: The four rows the return-type gap and the min/max-as-AND/OR misreading both surface on.
#: `all_null` is the discriminating one: a logical fold answers min->true / max->false there
#: (the conventional AND/OR empty identities), and every engine answers NULL.
_BOOL_ROWS = {
    "mixed":     ([True, False, True], {"s": 2, "a": 2.0 / 3.0, "mn": False, "mx": True, "c": 3}),
    "with_null": ([True, None, False], {"s": 1, "a": 0.5, "mn": False, "mx": True, "c": 2}),
    "all_null":  ([None, None], {"s": 0, "a": None, "mn": None, "mx": None, "c": 0}),
    "all_true":  ([True, True], {"s": 2, "a": 1.0, "mn": True, "mx": True, "c": 2}),
    "all_false": ([False, False], {"s": 0, "a": 0.0, "mn": False, "mx": False, "c": 2}),
}


def _bool_graph(values):
    """Nullable ``boolean``, not numpy ``bool``: the contract's null rows are unrepresentable in a
    numpy bool column, and the nullable dtype is what survives the trip to polars and cuDF."""
    nodes = pd.DataFrame({"id": list(range(len(values))), "grp": ["x"] * len(values),
                          "flag": pd.array(values, dtype="boolean")})
    edges = pd.DataFrame({"src": [0], "dst": [0]})
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def _dtype_kind(df, col):
    """Engine-neutral dtype label: ``int64`` / ``float64`` / ``bool``, else the raw spelling.

    Collapses ONLY the nullability spelling -- pandas ``Int64``/``boolean``, polars
    ``Int64``/``Boolean``, cuDF ``int64``/``bool`` all name the same contract type, and which of
    them a column lands on is the separate nullable-merge axis (#1796 BU1). WIDTH and SIGNEDNESS
    are deliberately NOT collapsed, so polars' ``UInt32`` reports as ``UInt32`` and fails.
    """
    raw = str(df.schema[col]) if "polars" in type(df).__module__ else str(df[col].dtype)
    lowered = raw.lower()
    if lowered == "int64":
        return "int64"
    if lowered == "float64":
        return "float64"
    if lowered in {"bool", "boolean"}:
        return "bool"
    return raw


def _scalar(df, col):
    """The single aggregate row's value, normalized to a python scalar with NULL as ``None``.

    Per-value, NOT ``df.where(df.notna(), None)``: py3.13 pandas renders a missing value as ``nan``
    where 3.12 gave ``None``, and this lane is ABOUT null behaviour, so the null test is explicit.
    """
    if "polars" in type(df).__module__:
        value = df.to_dicts()[0][col]
    else:
        value = df[col].iloc[0]
        if value is pd.NA:
            value = None
    if hasattr(value, "item") and not isinstance(value, bool):
        value = value.item()
    if isinstance(value, float) and value != value:
        value = None
    return value


def _rows(df):
    """Records with NULL as ``None`` on every engine, normalized PER VALUE.

    Not ``df.where(df.notna(), None)``: that reshapes the frame and, on py3.13 pandas, renders a
    missing value as ``nan`` rather than ``None`` -- the exact distinction this lane tests.
    """
    if "polars" in type(df).__module__:
        records = df.to_dicts()
    else:
        if hasattr(df, "to_pandas"):   # cudf
            df = df.to_pandas()
        records = df.to_dict("records")
    out = []
    for record in records:
        row = {}
        for key, value in record.items():
            if value is pd.NA:
                value = None
            if hasattr(value, "item") and not isinstance(value, bool):
                value = value.item()
            row[key] = None if isinstance(value, float) and value != value else value
        out.append(row)
    return out


def _assert_bool_contract(df, expected):
    """VALUES and DTYPES together: values alone already agreed across engines before this lane."""
    for col, want in expected.items():
        got = _scalar(df, col)
        if want is None:
            assert got is None, f"{col}: expected NULL, got {got!r}"
        elif isinstance(want, bool):
            # `is` on the identity, not `==`: True == 1 would let an integer min/max pass.
            assert isinstance(got, bool) and got == want, f"{col}: expected {want!r}, got {got!r}"
        elif isinstance(want, float):
            assert abs(got - want) < 1e-9, f"{col}: expected {want!r}, got {got!r}"
        else:
            assert got == want and not isinstance(got, bool), f"{col}: expected {want!r}, got {got!r}"
        assert _dtype_kind(df, col) == _BOOL_RESULT_DTYPES[col], (
            f"{col}: dtype {_dtype_kind(df, col)!r} != contract {_BOOL_RESULT_DTYPES[col]!r}")


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("row", sorted(_BOOL_ROWS))
@pytest.mark.parametrize("grouped", [True, False])
def test_boolean_aggregate_values_and_return_types(engine, row, grouped, request):
    """sum -> INTEGER(int64), avg -> FLOAT(float64), min/max -> BOOLEAN, count -> INTEGER(int64),
    identically on every engine. Polars used to answer sum/count with ``UInt32``: the same value
    behind a different return type, which is exactly the cross-engine divergence class the
    aggregate type contract exists to close."""
    _require_engine(engine)
    values, expected = _BOOL_ROWS[row]
    query = (f"MATCH (n) RETURN n.grp AS grp, {_BOOL_AGGS} ORDER BY grp" if grouped
             else f"MATCH (n) RETURN {_BOOL_AGGS}")
    out = _bool_graph(values).gfql(query, engine=engine)._nodes
    assert len(out) == 1, out
    _assert_bool_contract(out, expected)


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_boolean_min_max_are_an_ordering_not_a_logical_fold(engine):
    """``min == AND`` / ``max == OR`` is a DERIVATION from ``false < true``, not a definition, and
    it predicts the wrong answer on the empty fold: AND over zero elements is conventionally
    ``true`` and OR over zero elements ``false``, while every engine answers NULL. Pinned as the
    ordering instead -- the same one ``ORDER BY`` gives booleans."""
    _require_engine(engine)
    out = _bool_graph([None, None]).gfql(
        f"MATCH (n) RETURN {_BOOL_AGGS}", engine=engine)._nodes
    assert _scalar(out, "mn") is None, "min over no non-null values must be NULL, not the AND identity true"
    assert _scalar(out, "mx") is None, "max over no non-null values must be NULL, not the OR identity false"
    ordered = _bool_graph([True, False, True]).gfql(
        f"MATCH (n) RETURN {_BOOL_AGGS}", engine=engine)._nodes
    assert _scalar(ordered, "mn") is False and _scalar(ordered, "mx") is True


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_boolean_sum_over_zero_rows_is_cypher_zero_not_sql_null(engine):
    """Cypher's ``sum()`` returns **0** over zero rows where SQL's returns NULL, and its ``avg()``
    returns null; both engines already matched Cypher, so this is conformance rather than a
    compromise. The 0-row shape reaches it by a DIFFERENT route than the all-null column above --
    the ungrouped-aggregate identity row, not the aggregate kernel."""
    _require_engine(engine)
    out = _bool_graph([True, False, True]).gfql(
        f"MATCH (n) WHERE n.id > 9999 RETURN {_BOOL_AGGS}", engine=engine)._nodes
    assert len(out) == 1, out
    assert _scalar(out, "s") == 0 and _scalar(out, "c") == 0
    assert _dtype_kind(out, "s") == "int64" and _dtype_kind(out, "c") == "int64"
    for col in ("a", "mn", "mx"):
        assert _scalar(out, col) is None, col
    # The identity row carries no type evidence for the NULL columns (`avg`/`min`/`max` land on
    # pandas `object` / polars `Null`), so their dtypes are NOT asserted here. That gap is not
    # boolean-specific -- `avg` over an empty INTEGER column loses its dtype the same way -- and
    # is registered on the per-engine semantics matrix rather than pinned to today's behaviour.


@pytest.mark.parametrize("engine", ["polars", "cudf", "polars-gpu"])
@pytest.mark.parametrize("row", sorted(_BOOL_ROWS))
def test_boolean_aggregate_return_types_match_the_pandas_oracle(engine, row):
    """The differential form of the lane above: every engine's boolean aggregate must land on the
    SAME dtype kind as pandas, so a future engine-local dtype drift fails here even if someone
    edits the contract table."""
    _require_engine(engine)
    values, _ = _BOOL_ROWS[row]
    query = f"MATCH (n) RETURN n.grp AS grp, {_BOOL_AGGS} ORDER BY grp"
    oracle = _bool_graph(values).gfql(query, engine="pandas")._nodes
    got = _bool_graph(values).gfql(query, engine=engine)._nodes
    for col in _BOOL_RESULT_DTYPES:
        assert _dtype_kind(got, col) == _dtype_kind(oracle, col), col


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_count_returns_integer_on_every_input_type(engine):
    """``count()`` is INTEGER in Cypher for ANY input. Polars answered it ``UInt32`` for every
    dtype while pandas/cuDF answered ``int64``, so aligning it is wider than the boolean rule --
    but it is the same divergence and the same direction."""
    _require_engine(engine)
    g = _graph()
    out = g.gfql(
        "MATCH (n) RETURN n.grp AS grp, count(n.int_col) AS ci, count(n.str_col) AS cs, "
        "count(n.bool_col) AS cb, count(DISTINCT n.str_col) AS cd, count(*) AS ca ORDER BY grp",
        engine=engine)._nodes
    for col in ("ci", "cs", "cb", "cd", "ca"):
        assert _dtype_kind(out, col) == "int64", f"{col}: {_dtype_kind(out, col)}"


def _bool_fast_path_graph():
    """Fast-path shape with TWO cities, the second of which has only null flags -- the group whose
    ``sum`` cuDF answers NULL and Cypher answers 0. A single-city graph never produces that group,
    so it cannot exercise the repair."""
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 3, 10, 11],
        "node_type": ["Person"] * 4 + ["City"] * 2,
        "age": [20, 30, 40, 50, None, None],
        "flag": pd.array([True, False, None, None, None, None], dtype="boolean"),
        "allnull": [None] * 6,
        "city": [None] * 4 + ["LA", "NYC"],
    })
    edges = pd.DataFrame({"s": [0, 1, 2, 3], "d": [11, 11, 10, 10], "rel": ["LIVES_IN"] * 4})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


_BOOL_FAST_PATH_HEAD = (
    "MATCH (p {node_type:'Person'})-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
    "RETURN c.city AS city, ")


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_fast_path_boolean_aggregate_follows_the_same_contract(engine):
    """The OLAP fast path reimplements the aggregates on both engine branches, so without its own
    conformance the SAME boolean query would answer with a different return type depending on
    whether it happened to match the fast-path shape. The LA group has no non-null flag: its
    ``sum`` is Cypher's 0, which cuDF's kernel answers NULL."""
    _require_engine(engine)
    g = _bool_fast_path_graph()
    entered, _ = _run_watching_fast_path(g, _fast_path_query("sum", "p.flag"), engine)
    assert entered, "fast path not engaged -- this lane would not be exercising its aggregates"
    out = g.gfql(
        _BOOL_FAST_PATH_HEAD + "sum(p.flag) AS s, avg(p.flag) AS a, count(p.flag) AS c "
        "ORDER BY city", engine=engine)._nodes
    rows = _rows(out)
    assert rows[0]["city"] == "LA" and rows[1]["city"] == "NYC"
    assert rows[0]["s"] == 0 and rows[0]["a"] is None and rows[0]["c"] == 0
    assert rows[1]["s"] == 1 and abs(rows[1]["a"] - 0.5) < 1e-9 and rows[1]["c"] == 2
    assert _dtype_kind(out, "s") == "int64" and _dtype_kind(out, "a") == "float64"
    assert _dtype_kind(out, "c") == "int64"


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_fast_path_eager_polars_twin_conforms_when_the_fused_lane_declines(engine):
    """The fast path has THREE polars aggregate formulations -- a fused lazy lane, a
    ``value_counts`` plan for a low-cardinality pure ``count(*)``, and the eager twin the other two
    decline to. All three must land on the same return types, or which one a query happens to
    route to becomes observable. An all-null aggregate input declines the fused lane, so this
    query reaches the eager twin with the other aggregates still on it."""
    _require_engine(engine)
    out = _bool_fast_path_graph().gfql(
        _BOOL_FAST_PATH_HEAD + "sum(p.allnull) AS z, count(*) AS n, count(p.age) AS ca, "
        "sum(p.flag) AS s ORDER BY city", engine=engine)._nodes
    for col in ("z", "n", "ca", "s"):
        assert _dtype_kind(out, col) == "int64", f"{col}: {_dtype_kind(out, col)}"


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_fast_path_pure_count_star_returns_integer(engine):
    """A single-key pure ``count(*)`` over statically-bounded-low inputs takes a ``value_counts``
    formulation that skips the aggregate expressions entirely, so it needs its own conformance --
    it answered ``UInt32`` while the ``group_by`` formulation beside it answered ``int64``, which
    made the two lanes value-identical but NOT type-identical."""
    _require_engine(engine)
    out = _bool_fast_path_graph().gfql(
        _BOOL_FAST_PATH_HEAD + "count(*) AS n ORDER BY city", engine=engine)._nodes
    assert _dtype_kind(out, "n") == "int64", _dtype_kind(out, "n")
    assert [row["n"] for row in _rows(out)] == [2, 2]


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_fast_path_count_star_beside_another_aggregate_returns_integer(engine):
    """The ``value_counts`` plan above serves a PURE ``count(*)`` only, so a ``count(*)`` sharing
    its RETURN with another aggregate reaches the fused lane's own ``pl.len()`` instead -- a
    fourth count formulation, and one the pure-count test cannot reach."""
    _require_engine(engine)
    out = _bool_fast_path_graph().gfql(
        _BOOL_FAST_PATH_HEAD + "count(*) AS n, sum(p.age) AS s, count(p.flag) AS cf "
        "ORDER BY city", engine=engine)._nodes
    for col in ("n", "cf"):
        assert _dtype_kind(out, col) == "int64", f"{col}: {_dtype_kind(out, col)}"
    assert [row["n"] for row in _rows(out)] == [2, 2]


def test_boolean_result_contract_is_the_one_in_agg_types():
    """The table above is a RESTATEMENT of the shipped contract, so a change to one that is not a
    change to the other is a drift this catches rather than a silent disagreement."""
    from graphistry.compute.gfql.agg_types import agg_result_is_integer
    for func, alias in [("sum", "s"), ("count", "c")]:
        assert agg_result_is_integer(func, True), func
        assert _BOOL_RESULT_DTYPES[alias] == "int64"
    for func in ("avg", "mean", "min", "max"):
        assert not agg_result_is_integer(func, True), func
    assert agg_result_is_integer("count_distinct", False)
    # sum is INTEGER only BECAUSE the input is boolean -- a numeric sum keeps its own width
    assert not agg_result_is_integer("sum", False)


def test_polars_agg_result_cast_fires_only_where_polars_misses_the_contract():
    """The cast is scoped, not blanket: widening a FLOAT sum or a DURATION sum to Int64 would be a
    silent wrong answer, so the helper must decline everything except the two INTEGER cells."""
    from graphistry.compute.gfql.agg_types import polars_agg_result_cast as cast_to
    assert cast_to("sum", pl.Boolean) == pl.Int64
    assert cast_to("count", pl.Boolean) == pl.Int64
    assert cast_to("count", pl.String) == pl.Int64          # count is INTEGER over ANY input
    assert cast_to("count", None) == pl.Int64               # count(*) has no input column
    assert cast_to("count_distinct", pl.Float64) == pl.Int64
    assert cast_to("sum", pl.Int64) is None                 # polars already sums ints to Int64
    assert cast_to("sum", pl.Int8) is None
    assert cast_to("sum", pl.Float64) is None               # FLOAT sum must stay FLOAT
    assert cast_to("sum", pl.Duration) is None              # DURATION sum must stay DURATION
    assert cast_to("sum", None) is None
    assert cast_to("avg", pl.Boolean) is None
    assert cast_to("mean", pl.Boolean) is None
    assert cast_to("min", pl.Boolean) is None
    assert cast_to("max", pl.Boolean) is None
    assert cast_to("collect", pl.Boolean) is None


def test_polars_boolean_sum_null_fill_is_scoped_to_exact_boundary():
    """Only Boolean ``sum`` repairs a null aggregate result to Cypher's zero."""
    from graphistry.compute.gfql.agg_types import polars_conform_agg_dtype as conform

    source = pl.DataFrame({"one": [1]})

    boolean_sum = source.select(conform(pl.lit(None), "sum", pl.Boolean, "out"))
    assert boolean_sum.schema["out"] == pl.Int64
    assert boolean_sum["out"][0] == 0

    boolean_count = source.select(conform(pl.lit(None), "count", pl.Boolean, "out"))
    assert boolean_count.schema["out"] == pl.Int64
    assert boolean_count["out"][0] is None

    integer_sum = source.select(conform(pl.lit(None), "sum", pl.Int64, "out"))
    assert integer_sum.schema["out"] == pl.Null
    assert integer_sum["out"][0] is None


def test_polars_all_null_literal_is_a_typed_integer_zero():
    """A bare ``pl.lit(0)`` is ``Int32`` -- a width neither pandas nor cuDF ever produces, so the
    all-null substitution would reintroduce the very dtype divergence the cast above removes."""
    from graphistry.compute.gfql.agg_types import polars_all_null_agg_literal
    frame = pl.DataFrame({"x": [1]}).select(polars_all_null_agg_literal("sum", "s"),
                                            polars_all_null_agg_literal("avg", "a"))
    assert frame.schema["s"] == pl.Int64, frame.schema
    assert frame["s"][0] == 0
    assert frame["a"][0] is None


def test_pandas_agg_kernel_null_fill_repairs_only_sum():
    """Cypher's ``sum()`` never answers null, so a null kernel answer is a bug to repair; ``avg``
    and ``min``/``max`` DO answer null and must not be filled, or an all-null group would silently
    report 0 for an average."""
    from graphistry.compute.gfql.agg_types import pandas_agg_kernel_null_fill as fill
    assert fill("sum", pd.Series([1, 2], dtype="Int64")) == 0
    assert fill("sum", pd.Series([True], dtype="boolean")) == 0
    assert fill("sum", pd.Series([1.0])) == 0
    assert fill("avg", pd.Series([1, 2])) is None
    assert fill("mean", pd.Series([1, 2])) is None
    assert fill("min", pd.Series([True])) is None
    assert fill("max", pd.Series([True])) is None
    assert fill("count", pd.Series([1])) is None
    assert fill("collect", pd.Series([1])) is None
    # object columns keep the existing bool-retype route rather than gaining a second one
    assert fill("sum", pd.Series([True, None], dtype="object")) is None


def test_numeric_only_aggregation_set_covers_both_spellings():
    """``avg`` is the cypher name and ``mean`` GFQL's internal one (GFQL_GROUPBY_AGG_METHODS maps
    avg -> mean); a set holding only one of them would leave the other unguarded."""
    from graphistry.compute.gfql.agg_types import GFQL_NUMERIC_ONLY_AGGREGATIONS
    from graphistry.compute.gfql.language_defs import GFQL_GROUPBY_AGG_METHODS
    assert GFQL_NUMERIC_ONLY_AGGREGATIONS == {"sum", "avg", "mean"}
    assert set(GFQL_GROUPBY_AGG_METHODS) - GFQL_NUMERIC_ONLY_AGGREGATIONS == {
        "count", "count_distinct", "min", "max"
    }
