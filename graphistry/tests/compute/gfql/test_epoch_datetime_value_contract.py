"""Contract tests for explicit epoch datetime values (DateTimeValue.from_epoch).

Contract (see plans/gfql-ic4-perf/epoch-design-pressure-test/handoff.md):
- strict bare numeric-vs-datetime rejection (E302), no predicate-level epoch flags;
- DateTimeValue.from_epoch(value, *, unit) with required unit "s"|"ms", UTC only;
- wire datetime numeric value + required unit, JSON-safe range on wire numerics;
- canonical serialization to the existing typed ISO UTC form (semantic round-trip);
- native Polars DateTimeValue comparison parity, incl. aware columns and exact ms.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

import graphistry
from graphistry.compute.chain import Chain
from graphistry.compute.ast_temporal import DateTimeValue, temporal_value_from_json
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError, GFQLTypeError
from graphistry.compute.predicates.comparison import between, eq, ge, gt, le, lt, ne
from graphistry.compute.predicates.from_json import from_json as predicate_from_json
from graphistry.models.gfql.coercions.temporal import to_ast


EPOCH_JAN_9_MS = 1578528000000   # 2020-01-09T00:00:00Z
EPOCH_JAN_10_MS = 1578614400000  # 2020-01-10T00:00:00Z
EPOCH_JAN_1_S = 1577836800       # 2020-01-01T00:00:00Z


def _edges(**extra) -> pd.DataFrame:
    df = pd.DataFrame({
        "s": ["a", "b", "c", "d", "e"],
        "d": ["b", "c", "d", "a", "a"],
        "event_time": pd.to_datetime([
            "2020-01-01", "2020-01-08", "2020-01-10", "2020-01-15", None
        ]),
    })
    df.update(pd.DataFrame(extra))
    return df


def _json_chain(predicate: dict, col: str = "event_time") -> dict:
    return {
        "type": "Chain",
        "chain": [{
            "type": "Edge", "direction": "forward", "hops": 1,
            "edge_match": {col: predicate},
        }],
    }


def _ids(result) -> set:
    return set(result._edges["s"])


# ---------- Python factory ----------

def test_from_epoch_units_equivalent_and_distinct():
    ms = DateTimeValue.from_epoch(1577836800000, unit="ms")
    s = DateTimeValue.from_epoch(1577836800, unit="s")
    iso = DateTimeValue("2020-01-01T00:00:00+00:00")
    assert ms.as_pandas_value() == s.as_pandas_value() == iso.as_pandas_value()
    assert DateTimeValue.from_epoch(1577836800, unit="ms").value != s.value  # 1577836800 ms = 1970-01-19


def test_from_epoch_canonical_utc_iso_serialization():
    v = DateTimeValue.from_epoch(0, unit="ms")
    assert v.value == "1970-01-01T00:00:00+00:00" and v.timezone == "UTC"
    vneg = DateTimeValue.from_epoch(-1, unit="ms")
    assert vneg.value == "1969-12-31T23:59:59.999000+00:00"


@pytest.mark.parametrize("bad", [True, False, 1000.5, float("nan"), float("inf"), -float("inf"), None, "1000"])
def test_from_epoch_rejects_invalid_values(bad):
    with pytest.raises(GFQLTypeError) as e:
        DateTimeValue.from_epoch(bad, unit="ms")
    assert e.value.code == ErrorCode.E201


def test_from_epoch_requires_unit_no_default():
    # unit is keyword-only and required: the Python API cannot omit it
    with pytest.raises(TypeError):
        DateTimeValue.from_epoch(1578528000000)
    with pytest.raises(GFQLTypeError) as e:
        DateTimeValue.from_epoch(1578528000000, unit="us")
    assert e.value.code == ErrorCode.E201


def test_from_epoch_native_int_exact_but_representability_checked():
    # Native ints are exact; overflow is rejected against the scalar's own resolution.
    with pytest.raises(GFQLTypeError):
        DateTimeValue.from_epoch(10**19, unit="ms")  # beyond even ms-resolution range
    # Boundary: scalar year 5138 works against scalar/date columns; ns-column fallback is per-cell instants.
    assert DateTimeValue.from_epoch(10**14, unit="ms").value.startswith("5138-11-16")
    assert DateTimeValue.from_epoch(253402300799, unit="s").value.startswith("9999-12-31")


# ---------- wire decode: both entrances, positive + negative ----------

def test_wire_epoch_variant_both_entrances():
    d = {"type": "datetime", "value": EPOCH_JAN_9_MS, "unit": "ms"}
    a = temporal_value_from_json(d)
    b = to_ast(d)
    assert a.value == b.value == "2020-01-09T00:00:00+00:00"


def test_wire_semantic_roundtrip_normalizes_to_iso_utc():
    d = {"type": "datetime", "value": EPOCH_JAN_1_S, "unit": "s"}
    v = temporal_value_from_json(d)
    j = v.to_json()
    assert j == {"type": "datetime", "value": "2020-01-01T00:00:00+00:00", "timezone": "UTC"}
    assert temporal_value_from_json(j).as_pandas_value() == v.as_pandas_value()


@pytest.mark.parametrize("bad", [
    {"type": "datetime", "value": 1578528000000},                        # missing unit
    {"type": "datetime", "value": 1578528000000, "unit": "us"},          # unknown unit
    {"type": "datetime", "value": 1578528000000, "unit": "ms", "timezone": "UTC"},  # tz+numeric
    {"type": "datetime", "value": "2020-01-09", "unit": "ms"},           # unit on ISO
    {"type": "datetime", "value": 2**53, "unit": "ms"},                  # JSON-safe range
    {"type": "datetime", "value": True, "unit": "ms"},                   # bool
    {"type": "datetime", "value": 1000.5, "unit": "ms"},                 # fractional
])
def test_wire_epoch_rejections(bad):
    for decode in (temporal_value_from_json, to_ast):
        with pytest.raises(GFQLTypeError) as e:
            decode(bad)
        assert e.value.code == ErrorCode.E201


def test_wire_integral_float_equals_int():
    a = temporal_value_from_json({"type": "datetime", "value": 1e3, "unit": "ms"})
    b = temporal_value_from_json({"type": "datetime", "value": 1000, "unit": "ms"})
    assert a.as_pandas_value() == b.as_pandas_value()


def test_iso_variant_unchanged():
    d = {"type": "datetime", "value": "2020-01-09T00:00:00", "timezone": "UTC"}
    assert temporal_value_from_json(d).value == "2020-01-09T00:00:00"
    assert to_ast(d).value == "2020-01-09T00:00:00"


# ---------- predicates over pandas + polars engines (JSON chain) ----------

def _wire_ge(val) -> dict:
    return {"type": "GE", "val": val}


@pytest.mark.parametrize("val,expected", [
    ({"type": "datetime", "value": EPOCH_JAN_9_MS, "unit": "ms"}, {"c", "d"}),
    ({"type": "datetime", "value": EPOCH_JAN_1_S, "unit": "s"}, {"a", "b", "c", "d"}),
])
def test_json_epoch_value_all_engines(val, expected):
    chain = Chain.from_json(_json_chain(_wire_ge(val)))
    pandas_g = graphistry.edges(_edges(), "s", "d")
    assert _ids(pandas_g.gfql(chain, engine="pandas")) == expected
    pl = pytest.importorskip("polars")
    polars_g = graphistry.edges(pl.from_pandas(_edges()), "s", "d")
    assert set(polars_g.gfql(chain, engine="polars")._edges["s"].to_list()) == expected


def _typed_edges() -> pd.DataFrame:
    return pd.DataFrame({
        "s": ["a", "b", "c", "d", "e"],
        "d": ["b", "c", "d", "a", "a"],
        "event_time": pd.to_datetime([
            "2020-01-01", "2020-01-08", "2020-01-10", "2020-01-15", None
        ]),
    })


@pytest.mark.parametrize("pred,expected", [
    (ge(DateTimeValue.from_epoch(EPOCH_JAN_9_MS, unit="ms")), {"c", "d"}),
    (gt(DateTimeValue.from_epoch(EPOCH_JAN_9_MS, unit="ms")), {"c", "d"}),
    (le(DateTimeValue.from_epoch(EPOCH_JAN_9_MS, unit="ms")), {"a", "b"}),
    (lt(DateTimeValue.from_epoch(EPOCH_JAN_9_MS, unit="ms")), {"a", "b"}),
    (eq(DateTimeValue.from_epoch(EPOCH_JAN_10_MS, unit="ms")), {"c"}),
    (ne(DateTimeValue.from_epoch(EPOCH_JAN_9_MS, unit="ms")), {"a", "b", "c", "d"}),
    (between(DateTimeValue.from_epoch(EPOCH_JAN_1_S, unit="s"),
             DateTimeValue.from_epoch(EPOCH_JAN_10_MS, unit="ms")), {"a", "b", "c"}),
    (between(DateTimeValue.from_epoch(EPOCH_JAN_1_S, unit="s"),
             DateTimeValue.from_epoch(EPOCH_JAN_10_MS, unit="ms"), inclusive=False), {"b"}),
])
def test_all_operators_pandas(pred, expected):
    g = graphistry.edges(_typed_edges(), "s", "d")
    chain = Chain.from_json(_json_chain(pred.to_json()))
    assert _ids(g.gfql(chain, engine="pandas")) == expected


# ---------- strict numeric-vs-datetime rejection ----------

def test_numeric_vs_datetime_rejected_no_flag_pandas():
    g = graphistry.edges(_typed_edges(), "s", "d")
    chain = Chain.from_json(_json_chain(ge(EPOCH_JAN_9_MS).to_json()))
    with pytest.raises(GFQLSchemaError) as e:
        g.gfql(chain, engine="pandas")
    assert e.value.code == ErrorCode.E302


def test_numeric_vs_datetime_rejected_polars():
    pl = pytest.importorskip("polars")
    g = graphistry.edges(pl.from_pandas(_typed_edges()), "s", "d")
    chain = Chain.from_json(_json_chain(ge(EPOCH_JAN_9_MS).to_json()))
    with pytest.raises(GFQLSchemaError) as e:
        g.gfql(chain, engine="polars")
    assert e.value.code == ErrorCode.E302


def test_numeric_column_control_unchanged():
    df = _typed_edges()
    df["weight"] = [1, 5, 10, 20, None]
    g = graphistry.edges(df, "s", "d")
    chain = Chain.from_json(_json_chain(ge(5).to_json(), col="weight"))
    assert _ids(g.gfql(chain, engine="pandas")) == {"b", "c", "d"}


def test_predicate_from_json_rejects_removed_epoch_ms_flag():
    with pytest.raises(GFQLTypeError) as e:
        predicate_from_json({"type": "GE", "val": 1578528000000, "epoch_ms": True})
    assert e.value.code == ErrorCode.E201


# ---------- boundaries: exact ms vs finer resolution, tz/DST, empty ----------

def _fine_edges() -> pd.DataFrame:
    return pd.DataFrame({
        "s": ["a", "b", "c"],
        "d": ["b", "c", "a"],
        "event_time": pd.to_datetime([
            "2020-01-09 00:00:00.000000000",   # exactly the ms bound
            "2020-01-09 00:00:00.001000000",   # ms+1us... exactly 1 ms after midnight? no: 1 ms
            "2020-01-09 00:00:00.001000001",   # 1 ms + 1 ns
        ]),
    })


def test_exact_millisecond_boundary_vs_ns_column():
    pred = eq(DateTimeValue.from_epoch(EPOCH_JAN_9_MS + 1, unit="ms"))
    g = graphistry.edges(_fine_edges(), "s", "d")
    chain = Chain.from_json(_json_chain(pred.to_json()))
    assert _ids(g.gfql(chain, engine="pandas")) == {"b"}
    pl = pytest.importorskip("polars")
    gp = graphistry.edges(pl.from_pandas(_fine_edges()), "s", "d")
    assert set(gp.gfql(chain, engine="polars")._edges["s"].to_list()) == {"b"}


def test_ge_includes_exact_millisecond_against_finer_column():
    pred = ge(DateTimeValue.from_epoch(EPOCH_JAN_9_MS + 1, unit="ms"))
    g = graphistry.edges(_fine_edges(), "s", "d")
    chain = Chain.from_json(_json_chain(pred.to_json()))
    assert _ids(g.gfql(chain, engine="pandas")) == {"b", "c"}


def test_tz_aware_dst_instants():
    df = pd.DataFrame({
        "s": ["a", "b"],
        "d": ["b", "a"],
        "event_time": pd.to_datetime([
            "2021-03-14 06:30:00+00:00",  # before spring-forward (01:30 EST)
            "2021-03-14 07:30:00+00:00",  # after (03:30 EDT)
        ]).tz_convert("America/New_York"),
    })
    g = graphistry.edges(df, "s", "d")
    # GT 06:30 UTC keeps only the second instant
    chain = Chain.from_json(_json_chain(gt(DateTimeValue.from_epoch(1615703400, unit="s")).to_json()))
    assert _ids(g.gfql(chain, engine="pandas")) == {"b"}
    pl = pytest.importorskip("polars")
    gp = graphistry.edges(pl.from_pandas(df), "s", "d")
    assert set(gp.gfql(chain, engine="polars")._edges["s"].to_list()) == {"b"}


def test_naive_column_read_as_utc_policy():
    df = pd.DataFrame({
        "s": ["a", "b"],
        "d": ["b", "a"],
        "event_time": pd.to_datetime(["2021-03-14 06:30:00", "2021-03-14 06:29:59"]),
    })
    g = graphistry.edges(df, "s", "d")
    chain = Chain.from_json(_json_chain(ge(DateTimeValue.from_epoch(1615703400, unit="s")).to_json()))
    assert _ids(g.gfql(chain, engine="pandas")) == {"a"}
    pl = pytest.importorskip("polars")
    gp = graphistry.edges(pl.from_pandas(df), "s", "d")
    assert set(gp.gfql(chain, engine="polars")._edges["s"].to_list()) == {"a"}


def test_empty_and_all_null_frames_with_typed_bounds():
    df = _typed_edges()
    df.loc[:, "event_time"] = pd.to_datetime([None, None, None, None, None])
    g = graphistry.edges(df, "s", "d")
    chain = Chain.from_json(_json_chain(ge(DateTimeValue.from_epoch(EPOCH_JAN_9_MS, unit="ms")).to_json()))
    assert _ids(g.gfql(chain, engine="pandas")) == set()
    # schema-based rejection still applies on all-null datetime columns
    chain_num = Chain.from_json(_json_chain(ge(1578528000000).to_json()))
    with pytest.raises(GFQLSchemaError):
        g.gfql(chain_num, engine="pandas")


def test_chain_from_json_serialization_roundtrip():
    pred = ge({"type": "datetime", "value": EPOCH_JAN_9_MS, "unit": "ms"})
    chain = Chain.from_json(_json_chain(pred.to_json()))
    j = chain.to_json()
    # normalized serialization: typed ISO UTC
    val = j["chain"][0]["edge_match"]["event_time"]["val"]
    assert val == {"type": "datetime", "value": "2020-01-09T00:00:00+00:00", "timezone": "UTC"}
    chain2 = Chain.from_json(j)
    g = graphistry.edges(_typed_edges(), "s", "d")
    assert _ids(g.gfql(chain2, engine="pandas")) == {"c", "d"}


@pytest.mark.parametrize("bad", [
    {"type": "datetime"},                                    # missing value
    {"type": "datetime", "value": "2020-01-09", "timezone": 123},  # non-string timezone
    {"type": "datetime", "value": [1000], "unit": "ms"},     # non-numeric epoch value
    {"type": "date", "value": 123},                          # non-string date
    {"type": "time", "value": 123},                          # non-string time
])
def test_wire_shape_rejections(bad):
    with pytest.raises(GFQLTypeError) as e:
        temporal_value_from_json(bad)
    assert e.value.code == ErrorCode.E201
