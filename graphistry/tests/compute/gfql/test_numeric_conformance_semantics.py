"""Round-004 numeric-operator + expression-typing pins (#1900).

openCypher numeric tower: `%` is TRUNCATED remainder (sign of the DIVIDEND,
Java semantics: -7 % 3 = -1, not Python's floored 2); integer `/` truncates
toward zero; an INTEGER zero divisor is an error (float / 0.0 keeps IEEE
inf, Neo4j parity); ordering a boolean against a number is an incomparable
cross-type comparison (null -> row dropped); simple CASE uses `=`, so a null
subject or `WHEN null` arm never matches and falls to ELSE.

Every expected value is HAND-COMPUTED (issue #1900); engine agreement is not
evidence. polars serves natively or declines with an honest NIE -- never a
different number (parity-or-NIE).
"""
import math

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]


def _run(query: str, engine: str) -> pd.DataFrame:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "rank": [3, 1, 2, 5],
        "neg": [-7, -8, 7, 8],
        "flag": [True, False, True, False],
        "score": [1.5, -2.0, 0.0, 2.5],
        "name": ["Alice", "bob", None, "ALICE"],
    })
    edges = pd.DataFrame({"src": ["a"], "dst": ["b"]})
    if engine == "polars":
        g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "src", "dst")
    else:
        g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    out = g.gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _scalar(v):
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if math.isinf(v):
            return v
        if v.is_integer():
            return int(v)
        return v
    if hasattr(v, "item"):
        try:
            return _scalar(v.item())
        except (ValueError, AttributeError):
            pass
    return v


def _one(query: str, engine: str, colname: str):
    df = _run(query, engine)
    assert len(df) == 1, f"expected one row, got {df.to_dict('records')}"
    return _scalar(df[colname][0])


def _served_or_nie(query: str, engine: str, colname: str, expected):
    """polars parity-or-NIE: pandas must match the oracle; polars matches or NIEs."""
    if engine == "polars":
        try:
            got = _one(query, "polars", colname)
        except NotImplementedError:
            return
        assert got == expected
    else:
        assert _one(query, engine, colname) == expected


# ===========================================================================
# 1. Modulo: TRUNCATED (Java/openCypher), sign of the dividend
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("RETURN -7 % 3 AS x", -1),
    ("RETURN 7 % -3 AS x", 1),
    ("RETURN 7 % 3 AS x", 1),
    ("MATCH (n {id:'a'}) RETURN n.neg % 3 AS x", -1),
    ("MATCH (n {id:'c'}) RETURN n.neg % 3 AS x", 1),
    ("MATCH (n {id:'b'}) RETURN n.score % 3 AS x", -2.0),
], ids=["lit_neg_dividend", "lit_neg_divisor", "lit_pos", "col_neg", "col_pos", "col_float_neg"])
def test_modulo_is_truncated_not_floored(query, expected, engine):
    """Python-floored -7 % 3 = 2 was the bug; Java/openCypher say -1."""
    assert _one(query, engine, "x") == expected


# ===========================================================================
# 2. Integer division: truncation toward zero for int/int; float stays true
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("RETURN 5 / 2 AS x", 2),
    ("MATCH (n {id:'d'}) RETURN n.rank / 2 AS x", 2),
    ("MATCH (n {id:'a'}) RETURN n.neg / 2 AS x", -3),
    ("MATCH (n {id:'a'}) RETURN n.score / 2 AS x", 0.75),
], ids=["literal", "col_int", "col_neg_toward_zero", "col_float_true_division"])
def test_integer_division_truncates_toward_zero(query, expected, engine):
    """-7 / 2 = -3 (toward zero, NOT floored -4); float operands keep true division."""
    assert _one(query, engine, "x") == expected


# ===========================================================================
# 3. Integer zero divisor: error (float / 0.0 keeps IEEE infinity)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", [
    "MATCH (n {id:'a'}) RETURN n.rank / 0 AS x",
    "MATCH (n {id:'a'}) RETURN n.rank % 0 AS x",
    "RETURN 1 / 0 AS x",
], ids=["col_div_zero", "col_mod_zero", "literal_div_zero"])
def test_integer_zero_divisor_is_an_error_not_inf(query, engine):
    """openCypher mandates an error; the old inf/NaN was silent-wrong. pandas
    raises the typed error; polars declines honestly (its `// 0` would yield
    null, so non-provably-nonzero divisors never run natively)."""
    with pytest.raises((GFQLValidationError, NotImplementedError)) as exc:
        _run(query, engine)
    if isinstance(exc.value, GFQLValidationError):
        assert "by zero" in str(exc.value)


@pytest.mark.parametrize("engine", ENGINES)
def test_float_division_by_zero_keeps_ieee_infinity(engine):
    """Neo4j parity: float / 0.0 -> Infinity, not an error."""
    assert _one("MATCH (n {id:'d'}) RETURN n.score / 0.0 AS x", engine, "x") == float("inf")


# ===========================================================================
# 4. bool-vs-int ordering: incomparable -> null (rows drop)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_boolean_vs_number_ordering_is_null(engine):
    """WHERE n.flag > 0 must match NOTHING (cross-type comparison -> null),
    never coerce true to 1; equality on booleans stays served."""
    assert len(_run("MATCH (n) WHERE n.flag > 0 RETURN n.id AS i", engine)) == 0
    assert len(_run("MATCH (n) WHERE n.flag >= 0 RETURN n.id AS i", engine)) == 0
    eq = _run("MATCH (n) WHERE n.flag = true RETURN n.id AS i", engine)
    assert sorted(eq["i"]) == ["a", "c"]


@pytest.mark.parametrize("engine", ENGINES)
def test_numeric_ordering_control_unchanged(engine):
    """Negative control: plain numeric ordering still serves."""
    out = _run("MATCH (n) WHERE n.rank > 2 RETURN n.id AS i", engine)
    assert sorted(out["i"]) == ["a", "d"]


# ===========================================================================
# 5. Simple CASE: '=' semantics -- null never matches
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_simple_case_when_null_never_matches(engine):
    """openCypher simple CASE compares with '='; null = null is null, so a
    null subject falls through to ELSE -- the old deliberate null==null match
    contradicted Neo4j. n.missing is null on every row."""
    q = ("MATCH (n {id:'a'}) "
         "RETURN CASE n.missing WHEN null THEN 'was-null' ELSE 'other' END AS c")
    _served_or_nie(q, engine, "c", "other")


@pytest.mark.parametrize("engine", ENGINES)
def test_simple_case_value_match_control(engine):
    q = ("MATCH (n {id:'a'}) "
         "RETURN CASE n.rank WHEN 3 THEN 'three' ELSE 'other' END AS c")
    _served_or_nie(q, engine, "c", "three")


@pytest.mark.parametrize("engine", ENGINES)
def test_simple_case_null_subject_vs_value_falls_to_else(engine):
    """A null subject never matches a non-null WHEN either."""
    q = ("MATCH (n {id:'c'}) "
         "RETURN CASE n.name WHEN 'Alice' THEN 'hit' ELSE 'miss' END AS c")
    _served_or_nie(q, engine, "c", "miss")


# ===========================================================================
# 6. Expression-lane typing polish
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_one_plus_null_is_null(engine):
    _served_or_nie("RETURN 1 + null AS x", engine, "x", None)


@pytest.mark.parametrize("engine", ENGINES)
def test_tointeger_unparseable_string_is_null(engine):
    _served_or_nie("RETURN toInteger('x1') AS x", engine, "x", None)


@pytest.mark.parametrize("engine", ENGINES)
def test_cross_property_starts_with_declines_typed(engine):
    """col-vs-col STARTS WITH: typed decline, never a raw ValueError."""
    with pytest.raises((GFQLValidationError, NotImplementedError)):
        _run("MATCH (n) WHERE n.name STARTS WITH n.name RETURN n.id AS i", engine)
