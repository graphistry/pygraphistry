"""Round-007 ungrouped-aggregate identity-row pins (#1909).

openCypher: an aggregate with NO grouping keys always yields exactly one row, so
"filter, then count what survived" returns a zero -- never an empty frame. The
identities are count -> 0, sum -> 0, collect -> [], min/max/avg -> null. Terminal
SKIP/LIMIT then pages that one row like any other row.

Every expected value is HAND-COMPUTED on the fixture below; engine agreement is
not evidence. Fixture (6 nodes / 6 edges):

  id    n1     n2    n3      n4     n5     n6
  name  Alice  Bob   Carol   Dave   Eve    Frank
  age   30     30    40      null   40     30
  city  NY     NY    SF      SF     null   null
  score 10     20    null    20     30     null

  edges  n1->n2  n1->n3  n2->n3  n3->n4  n4->n5  n1->n4   (w 1,2,3,null,5,1)

Derived facts used by the oracles: 6 nodes, 6 edges; city groups NY={n1,n2},
SF={n3,n4}, null={n5,n6} -- three groups of exactly 2; no node is named 'Zed',
so every 'Zed' predicate empties the stream.
"""
import math

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]

NODES = pd.DataFrame({
    "id": ["n1", "n2", "n3", "n4", "n5", "n6"],
    "name": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"],
    "age": [30.0, 30.0, 40.0, None, 40.0, 30.0],
    "city": ["NY", "NY", "SF", "SF", None, None],
    "score": [10.0, 20.0, None, 20.0, 30.0, None],
})
EDGES = pd.DataFrame({
    "s": ["n1", "n1", "n2", "n3", "n4", "n1"],
    "d": ["n2", "n3", "n3", "n4", "n5", "n4"],
    "eid": ["e1", "e2", "e3", "e4", "e5", "e6"],
    "w": [1.0, 2.0, 3.0, None, 5.0, 1.0],
})


def _graph(engine: str):
    if engine == "polars":
        return (graphistry.nodes(pl.from_pandas(NODES), "id")
                .edges(pl.from_pandas(EDGES), "s", "d").bind(edge="eid"))
    return graphistry.nodes(NODES, "id").edges(EDGES, "s", "d").bind(edge="eid")


def _run(query: str, engine: str) -> pd.DataFrame:
    out = _graph(engine).gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (list, tuple)) or type(value).__name__ == "ndarray":
        return [_scalar(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if hasattr(value, "item"):
        try:
            return _scalar(value.item())
        except (ValueError, AttributeError):
            pass
    return value


def _records(df: pd.DataFrame):
    return [{key: _scalar(value) for key, value in record.items()}
            for record in df.to_dict("records")]


# ===========================================================================
# 1. Multi-stage MATCH -> WITH -> RETURN: a later stage that empties the
#    stream must NOT swallow the identity row (#1909 item 1)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    # post-WITH WHERE empties the carried column stream
    ("MATCH (m) WITH m.name AS n WHERE n = 'Zed' RETURN count(*) AS c", [{"c": 0}]),
    # HAVING-style filter on an aggregate: every city group has c=2, so c>99 empties
    ("MATCH (m) WITH m.city AS city, count(*) AS c WHERE c > 99 RETURN count(*) AS n",
     [{"n": 0}]),
    # LIMIT 0 mid-pipeline
    ("MATCH (m) WITH m.name AS n ORDER BY n ASC LIMIT 0 RETURN count(*) AS c", [{"c": 0}]),
    # SKIP past the end mid-pipeline (6 nodes)
    ("MATCH (m) WITH m.name AS n ORDER BY n ASC SKIP 99 RETURN count(*) AS c", [{"c": 0}]),
    # MATCH-level WHERE empties, WITH carries nothing through
    ("MATCH (m) WHERE m.name = 'Zed' WITH m.city AS city RETURN count(*) AS c", [{"c": 0}]),
    # collect identity is the EMPTY LIST, not null
    ("MATCH (m) WITH m.name AS n WHERE n = 'Zed' RETURN collect(n) AS col", [{"col": []}]),
    # count(DISTINCT ...) has the same identity as count()
    ("MATCH (m) WITH m.name AS n WHERE n = 'Zed' RETURN count(DISTINCT n) AS c", [{"c": 0}]),
    # edge-rooted match
    ("MATCH (a)-[r]->(b) WITH a.name AS n WHERE n = 'Zed' RETURN count(*) AS c", [{"c": 0}]),
    # aggregate stage feeding an aggregate stage
    ("MATCH (m) WITH count(*) AS c WHERE c > 99 RETURN count(*) AS n", [{"n": 0}]),
], ids=["post_with_where", "having", "limit_0", "skip_past_end", "match_where_then_carry",
        "collect", "count_distinct", "edge_rooted", "agg_then_agg"])
def test_multi_stage_ungrouped_aggregate_keeps_identity_row(query, expected, engine):
    """#1909 item 1: the multi-stage lowering used to drop the identity row for
    every one of these emptying mechanisms and return 0 rows."""
    assert _records(_run(query, engine)) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_multi_stage_multi_aggregate_identity_row(engine):
    """All identities in one RETURN: count -> 0, collect -> [], min/max/avg -> null."""
    query = ("MATCH (m) WITH m.city AS city, count(*) AS c WHERE c > 99 "
             "RETURN count(*) AS n, collect(city) AS col, min(c) AS mn, max(c) AS mx, avg(c) AS av")
    assert _records(_run(query, engine)) == [
        {"n": 0, "col": [], "mn": None, "mx": None, "av": None}
    ]


@pytest.mark.parametrize("engine", ENGINES)
def test_multi_stage_post_aggregate_expression_over_identity_row(engine):
    """The identity row flows through the post-aggregate projection: count -> 0,
    so `c + 1` is 1 -- NOT an empty frame."""
    query = ("MATCH (m) WITH m.name AS n WHERE n = 'Zed' "
             "WITH count(*) AS c RETURN c + 1 AS c1")
    assert _records(_run(query, engine)) == [{"c1": 1}]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    # 3 city groups (NY, SF, null), each of size 2 -> all survive c>1 -> n=3
    ("MATCH (m) WITH m.city AS city, count(*) AS c WHERE c > 1 RETURN count(*) AS n",
     [{"n": 3}]),
    # GROUPED aggregate over an emptied stream stays 0 rows: the identity row is
    # for UNGROUPED aggregates only (openCypher yields one row per group, and
    # there are no groups).
    ("MATCH (m) WITH m.name AS n WHERE n = 'Zed' RETURN n, count(*) AS c", []),
    # non-aggregate RETURN over an emptied stream stays 0 rows
    ("MATCH (m) WITH m.name AS n WHERE n = 'Zed' RETURN n", []),
], ids=["nonempty_having", "grouped_stays_empty", "non_aggregate_stays_empty"])
def test_multi_stage_identity_row_negative_controls(query, expected, engine):
    assert _records(_run(query, engine)) == expected


# ===========================================================================
# 2. Terminal SKIP/LIMIT pages the identity row (#1909 item 2)
# ===========================================================================
# openCypher: the identity row is a real row, so SKIP/LIMIT apply to it.
# LIMIT 0 -> 0 rows; LIMIT n>=1 -> the row; SKIP 0 -> the row; SKIP >=1 -> 0 rows.


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("UNWIND [] AS x RETURN count(*) AS c LIMIT 1", [{"c": 0}]),
    ("UNWIND [] AS x RETURN count(*) AS c LIMIT 0", []),
    ("UNWIND [] AS x RETURN count(*) AS c SKIP 0", [{"c": 0}]),
    ("UNWIND [] AS x RETURN count(*) AS c SKIP 1", []),
    ("UNWIND [] AS x RETURN count(*) AS c SKIP 0 LIMIT 5", [{"c": 0}]),
    ("UNWIND [] AS x RETURN count(*) AS c, sum(x) AS s LIMIT 2", [{"c": 0, "s": 0}]),
], ids=["limit_1", "limit_0", "skip_0", "skip_1", "skip_0_limit_5", "multi_agg_limit_2"])
def test_row_only_identity_row_paging(query, expected, engine):
    """#1909 item 2: the row-only path bailed out of synthesizing the identity row
    entirely whenever the final stage had SKIP or LIMIT, so `LIMIT 1` returned 0
    rows. The rule is synthesize-then-page, not decline."""
    assert _records(_run(query, engine)) == expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("MATCH (m) WHERE m.name = 'Zed' RETURN count(*) AS c LIMIT 1", [{"c": 0}]),
    ("MATCH (m) WHERE m.name = 'Zed' RETURN count(*) AS c LIMIT 0", []),
    ("MATCH (m) WHERE m.name = 'Zed' RETURN count(*) AS c SKIP 0", [{"c": 0}]),
    ("MATCH (m) WHERE m.name = 'Zed' RETURN count(*) AS c SKIP 1", []),
    ("MATCH (m) WITH m.name AS n WHERE n = 'Zed' RETURN count(*) AS c LIMIT 1", [{"c": 0}]),
    ("MATCH (m) WITH m.name AS n WHERE n = 'Zed' RETURN count(*) AS c LIMIT 0", []),
    ("MATCH (m) WITH m.name AS n WHERE n = 'Zed' RETURN count(*) AS c SKIP 1", []),
], ids=["single_limit_1", "single_limit_0", "single_skip_0", "single_skip_1",
        "multi_limit_1", "multi_limit_0", "multi_skip_1"])
def test_match_rooted_identity_row_paging(query, expected, engine):
    """MATCH-rooted equivalents of the paging rule. `LIMIT 0` / `SKIP 1` used to
    still emit the identity row on the single-stage path."""
    assert _records(_run(query, engine)) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_sum_identity_over_empty_match_is_zero(engine):
    """openCypher 9 s3.2: sum() over no rows is 0, not null (min/max/avg stay null)."""
    query = ("MATCH (m) WHERE m.name = 'Zed' "
             "RETURN count(*) AS c, sum(m.age) AS s, collect(m.name) AS col, avg(m.age) AS a")
    assert _records(_run(query, engine)) == [{"c": 0, "s": 0, "col": [], "a": None}]


@pytest.mark.parametrize("engine", ENGINES)
def test_nonempty_aggregate_unaffected_by_identity_synthesis(engine):
    """Control: the identity row only ever applies to an EMPTY result. 6 nodes,
    ages 30,30,40,null,40,30 -> count(*) 6, sum 170, avg 170/5 = 34 (nulls skipped)."""
    query = "MATCH (m) RETURN count(*) AS c, sum(m.age) AS s, avg(m.age) AS a"
    assert _records(_run(query, engine)) == [{"c": 6, "s": 170, "a": 34}]


# ===========================================================================
# 3. Cross-engine diagnostics parity (#1909 item 4)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", [
    "MATCH (m) RETURN m.city AS city ORDER BY count(*) DESC",
    "MATCH (m) RETURN m.city AS city, count(*) AS c ORDER BY sum(m.age) DESC",
    "MATCH (m) WITH m.city AS city ORDER BY count(*) DESC RETURN city",
], ids=["no_agg_in_return", "different_agg", "with_stage"])
def test_aggregate_in_order_by_is_a_validation_error_on_both_engines(query, engine):
    """openCypher/Neo4j reject an aggregate introduced by ORDER BY (the sort runs
    AFTER aggregation). #1909 item 4: this used to surface as pandas GFQLTypeError
    [invalid-node-reference] at EXECUTION time and polars NotImplementedError --
    two different exception types and the wrong code for the same rejection."""
    with pytest.raises(GFQLValidationError) as excinfo:
        _run(query, engine)
    assert excinfo.value.code == ErrorCode.E108
    assert "ORDER BY cannot introduce an aggregate" in str(excinfo.value)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", [
    "MATCH (m) RETURN stDev(m.age) AS s",
    "MATCH (m) RETURN stDevP(m.age) AS s",
    "MATCH (m) RETURN percentileCont(m.age, 0.5) AS s",
    "MATCH (m) RETURN percentileDisc(m.age, 0.5) AS s",
    "MATCH (m) WITH m.city AS city, stDev(m.age) AS s RETURN city, s",
], ids=["stdev", "stdevp", "percentile_cont", "percentile_disc", "with_stage"])
def test_unsupported_aggregate_functions_are_validation_errors_on_both_engines(query, engine):
    """Same shape: an openCypher aggregate the local compiler does not lower is a
    compile-time GFQLValidationError [unsupported-cypher-query], not an execution
    -time engine-specific exception (#1909 item 4)."""
    with pytest.raises(GFQLValidationError) as excinfo:
        _run(query, engine)
    assert excinfo.value.code == ErrorCode.E108
    assert "aggregate function is not supported" in str(excinfo.value)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    # ORDER BY may NAME an aggregate the RETURN already projects -- resolved to the
    # projected output, not rejected. ages 30x3, 40x2, null x1.
    ("MATCH (m) RETURN m.age AS age, count(*) AS cnt ORDER BY count(*) DESC",
     [{"age": 30, "cnt": 3}, {"age": 40, "cnt": 2}, {"age": None, "cnt": 1}]),
    # ... including inside a larger expression
    ("MATCH (m) RETURN m.age AS age, count(*) AS cnt ORDER BY age + count(*)",
     [{"age": 30, "cnt": 3}, {"age": 40, "cnt": 2}, {"age": None, "cnt": 1}]),
], ids=["named_aggregate", "aggregate_inside_expression"])
def test_order_by_naming_a_projected_aggregate_still_serves(query, expected, engine):
    """Positive control for the rejection above: only a NEWLY INTRODUCED aggregate
    is an error. (null sorts last under openCypher ASC null placement.)"""
    assert _records(_run(query, engine)) == expected


# ===========================================================================
# 4. DELIBERATE documented extension: sum()/avg() over booleans (#1909 item 3)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    # ages 30,30,40,null,40,30 -> (age > 35) is F,F,T,null,T,F
    ("MATCH (m) RETURN sum(m.age > 35) AS out", 2),
    ("MATCH (m) RETURN avg(m.age > 35) AS out", 0.4),
], ids=["sum", "avg"])
def test_sum_and_avg_over_booleans_coerce_on_purpose(query, expected, engine):
    """DELIBERATE EXTENSION, not a bug: openCypher/Neo4j REJECT sum()/avg() over
    booleans ("Cannot handle booleans"). pygraphistry keeps the numeric coercion
    on purpose -- `sum(x > k)` is a core dataframe idiom and the answer is
    coherent (False=0, True=1, nulls skipped, so avg is 2/5). Owner call recorded
    in #1909 item 3: conforming would break a useful idiom for no correctness
    gain. Change this only with an explicit decision, never as a "conformance" fix.
    """
    assert _scalar(_run(query, engine)["out"][0]) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_min_max_over_booleans_is_plain_opencypher(engine):
    """Contrast: min/max over booleans IS openCypher (booleans are orderable),
    and count() counts the 5 non-null comparison results."""
    query = "MATCH (m) RETURN min(m.age > 35) AS mn, max(m.age > 35) AS mx, count(m.age > 35) AS c"
    assert _records(_run(query, engine)) == [{"mn": False, "mx": True, "c": 5}]


# ===========================================================================
# 5. Residual: a post-aggregate WHERE is undecidable at compile time (#1909)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_post_aggregate_where_over_empty_stream_is_a_known_residual(engine):
    """Control for the residual pinned below: with 6 nodes the count is 6, the
    `c = 0` filter removes it, and 0 rows is CORRECT."""
    assert _records(_run("MATCH (m) WITH count(*) AS c WHERE c = 0 RETURN c", engine)) == []


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.xfail(strict=True, reason="#1909 residual: when a WHERE follows the ungrouped "
                                       "aggregate, whether the identity row survives depends on "
                                       "the real aggregate value, which the compiler cannot see; "
                                       "the synthesis declines rather than guess")
def test_post_aggregate_where_keeping_the_identity_row_is_unsupported(engine):
    """openCypher: no node is named 'Zed', so count(*) is 0 and `WHERE c = 0` KEEPS
    the identity row -> 1 row {c: 0}. We return 0 rows; deciding this needs the
    emptiness of the pre-aggregate stream at runtime, not compile time."""
    query = "MATCH (m) WHERE m.name = 'Zed' WITH count(*) AS c WHERE c = 0 RETURN c"
    assert _records(_run(query, engine)) == [{"c": 0}]


# ===========================================================================
# 6. polars declines pandas serves (#1909 item 5) -- pinned, not invisible
# ===========================================================================
# All three are the same root cause: the polars row pipeline has no native
# cypher expression engine, so `with_` / `order_by` over a projected expression
# raise NotImplementedError (parity-or-error by design). Serving them is the
# general native-polars row-expression work, not a local fix -- pinned strict so
# they flip loudly when that lands.


@pytest.mark.parametrize("query,expected", [
    # whole-entity grouping: each of the 6 nodes is its own group
    ("MATCH (m) RETURN m, count(*) AS c", 6),
    # aggregate over a MISSING property: openCypher counts non-null values -> 0
    ("MATCH (m) RETURN count(m.nosuch) AS c", 1),
    # ORDER BY after a collect-of-collect
    ("MATCH (m) WITH m.city AS city, collect(m.name) AS names "
     "RETURN collect(names) AS nested ORDER BY nested", 1),
], ids=["whole_entity_grouping", "missing_property_aggregate", "order_by_collect_of_collect"])
def test_pandas_serves_the_polars_decline_families(query, expected):
    assert len(_run(query, "pandas")) == expected


@polars_only
@pytest.mark.parametrize("query", [
    "MATCH (m) RETURN m, count(*) AS c",
    "MATCH (m) RETURN count(m.nosuch) AS c",
    "MATCH (m) WITH m.city AS city, collect(m.name) AS names "
    "RETURN collect(names) AS nested ORDER BY nested",
], ids=["whole_entity_grouping", "missing_property_aggregate", "order_by_collect_of_collect"])
@pytest.mark.xfail(strict=True, raises=NotImplementedError,
                   reason="#1909 item 5: polars row pipeline has no native cypher expression "
                          "engine for with_/order_by; pandas serves these correctly")
def test_polars_declines_these_aggregate_shapes(query):
    _run(query, "polars")


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_property_aggregate_value_is_zero_on_pandas(engine):
    """Hand oracle for the decline family above: count() over a property no node
    has counts zero non-null values -- it is not an error."""
    if engine == "polars":
        with pytest.raises(NotImplementedError):
            _run("MATCH (m) RETURN count(m.nosuch) AS c", "polars")
        return
    assert _records(_run("MATCH (m) RETURN count(m.nosuch) AS c", engine)) == [{"c": 0}]
