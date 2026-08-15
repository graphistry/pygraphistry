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
    got = _run(g, "MATCH (n) RETURN n.grp AS grp, sum(n.nul) AS s, avg(n.nul) AS a ORDER BY grp",
               engine)
    assert got == ("ok", [(("grp", "x"), ("s", ("num", 0.0)), ("a", None)),
                          (("grp", "y"), ("s", ("num", 0.0)), ("a", None))]), got


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_all_null_column_sums_to_zero_and_averages_to_null(engine):
    """Neo4j: "``sum(null)`` returns ``0``", "``avg(null)`` returns ``null``" -- verified live on
    5.26.26. An all-null column carries NO type evidence, so it is never a type error; it also
    cannot be delegated to the host kernels, which answer it 0 / '' / NaT / TypeError depending
    on dtype (pandas) or raise for both str and null dtypes (polars)."""
    _require_engine(engine)
    g = _graph()
    got = _run(g, "MATCH (n) RETURN n.grp AS grp, sum(n.allnull_col) AS s, avg(n.allnull_col) AS a ORDER BY grp", engine)
    assert got == ("ok", [(("grp", "x"), ("s", ("num", 0.0)), ("a", None)),
                          (("grp", "y"), ("s", ("num", 0.0)), ("a", None))]), got


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


def test_numeric_only_aggregation_set_covers_both_spellings():
    """``avg`` is the cypher name and ``mean`` GFQL's internal one (GFQL_GROUPBY_AGG_METHODS maps
    avg -> mean); a set holding only one of them would leave the other unguarded."""
    from graphistry.compute.gfql.agg_types import GFQL_NUMERIC_ONLY_AGGREGATIONS
    from graphistry.compute.gfql.language_defs import GFQL_GROUPBY_AGG_METHODS
    assert GFQL_NUMERIC_ONLY_AGGREGATIONS == {"sum", "avg", "mean"}
    assert set(GFQL_GROUPBY_AGG_METHODS) - GFQL_NUMERIC_ONLY_AGGREGATIONS == {
        "count", "count_distinct", "min", "max"
    }
