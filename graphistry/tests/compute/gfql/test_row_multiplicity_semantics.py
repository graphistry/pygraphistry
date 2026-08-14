"""Round-004 row-multiplicity + aggregate-identity pins (#1899).

openCypher RETURN/WITH operate on the BAG of pattern-match rows: a projection
of one endpoint keeps one row per match (never a deduplicated node set), an
ungrouped aggregate over an empty row stream still emits its identity row
(count -> 0, sum -> 0, collect -> []), and list subscripts index from the end
for negative keys.

Every expected value is HAND-COMPUTED on the fixture below (issue #1899);
engine agreement is not evidence. Fixture: nodes 1-5 (Ann/Bob/Cat/Dan/Eve,
null age/city cases), edges 1->2, 1->3, 2->3, 3->4.
"""
import math

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError  # noqa: F401  (negative controls)

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]


def _run(query: str, engine: str) -> pd.DataFrame:
    nodes = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Ann", "Bob", "Cat", "Dan", "Eve"],
        "age": [30.0, 40.0, 25.0, None, 35.0],
        "city": ["NYC", "NYC", "SF", "SF", None],
    })
    edges = pd.DataFrame({"s": [1, 1, 2, 3], "d": [2, 3, 3, 4]})
    if engine == "polars":
        g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    else:
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    out = g.gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _scalar(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if hasattr(v, "item"):
        try:
            return _scalar(v.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(v, (list, tuple)) or type(v).__name__ == "ndarray":
        return [_scalar(x) for x in v]
    return v


def _bag(df: pd.DataFrame, col: str):
    return sorted((_scalar(v) for v in df[col]), key=lambda x: (x is None, str(x)))


# ===========================================================================
# 1. Single-endpoint projection multiplicity (bag semantics)
# ===========================================================================
# Edge bag: (1,2),(1,3),(2,3),(3,4). a-side bag [1,1,2,3]; b-side [2,3,3,4].


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("MATCH (a)-->(b) RETURN a.id AS id", [1, 1, 2, 3]),
    ("MATCH (a)-->(b) RETURN b.id AS id", [2, 3, 3, 4]),
    ("MATCH (a)-->() RETURN a.id AS id", [1, 1, 2, 3]),
    ("MATCH (a)-->(b) WITH a.id AS id RETURN id", [1, 1, 2, 3]),
    ("MATCH (a)-->(b) WITH a RETURN a.id AS id", [1, 1, 2, 3]),
    ("MATCH (a)-->(b) WHERE b.id >= 3 RETURN a.id AS id", [1, 2, 3]),
], ids=["a_side", "b_side", "anon_endpoint", "with_property_stage", "with_pure_carry", "post_where"])
def test_single_endpoint_projection_keeps_multiplicity(query, expected, engine):
    """#1899 item 1: one row per pattern match -- node 1 has two out-edges, so
    its id appears twice; the old node-set projection silently deduplicated."""
    assert _bag(_run(query, engine), "id") == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_single_endpoint_distinct_still_dedupes(engine):
    """Negative control: RETURN DISTINCT is the user-requested dedupe."""
    assert _bag(_run("MATCH (a)-->(b) RETURN DISTINCT a.id AS id", engine), "id") == [1, 2, 3]


@pytest.mark.parametrize("engine", ENGINES)
def test_single_endpoint_order_by_keeps_multiplicity(engine):
    """ORDER BY over the multiplicity bag: ordered compare."""
    df = _run("MATCH (a)-->(b) RETURN a.id AS id ORDER BY id", engine)
    assert [_scalar(v) for v in df["id"]] == [1, 1, 2, 3]


@pytest.mark.parametrize("engine", ENGINES)
def test_pair_projection_control_unchanged(engine):
    """Control: the pair projection was already row-correct and must stay so."""
    df = _run("MATCH (a)-->(b) RETURN a.id AS x, b.id AS y", engine)
    got = sorted((int(r["x"]), int(r["y"])) for r in df.to_dict("records"))
    assert got == [(1, 2), (1, 3), (2, 3), (3, 4)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.xfail(strict=True, reason="#1899 residual: whole-row endpoint projection still "
                   "collapses multiplicity (b bound twice to node 3 must yield two rows)")
def test_whole_row_endpoint_projection_multiplicity_residual(engine):
    """Residual pin: `RETURN b` (whole entity) over the same match should be a
    4-row bag (node 3 twice). Flip when the whole-row lane joins binding rows."""
    df = _run("MATCH (a)-->(b) RETURN b", engine)
    assert len(df) == 4


# ===========================================================================
# 2. Grouped-aggregate fast path: output column named like the node id
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_grouped_count_output_named_id_serves(engine):
    """#1899 item 2: `a.id AS id` collides with the node-id column inside the
    single-hop fast path's lookup rename -- was a raw pandas KeyError."""
    df = _run("MATCH (a)-->(b) RETURN a.id AS id, count(*) AS c", engine)
    got = sorted((int(r["id"]), int(r["c"])) for r in df.to_dict("records"))
    assert got == [(1, 2), (2, 1), (3, 1)]


@pytest.mark.parametrize("engine", ENGINES)
def test_grouped_count_output_id_sourced_from_other_prop(engine):
    """The reverse collision: an output NAMED `id` sourced from another
    property must not corrupt the join key."""
    df = _run("MATCH (a)-->(b) RETURN a.city AS id, count(*) AS c", engine)
    got = sorted(((_scalar(r["id"]), int(r["c"])) for r in df.to_dict("records")),
                 key=lambda t: str(t[0]))
    assert got == [("NYC", 3), ("SF", 1)]


# ===========================================================================
# 3. Empty-UNWIND ungrouped aggregate identities
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("UNWIND [] AS x RETURN count(*) AS out", 0),
    ("UNWIND [] AS x RETURN count(x) AS out", 0),
    ("UNWIND [] AS x RETURN sum(x) AS out", 0),
    ("UNWIND [] AS x RETURN min(x) AS out", None),
], ids=["count_star", "count_x", "sum", "min"])
def test_empty_unwind_ungrouped_aggregate_identity_row(query, expected, engine):
    """#1899 item 3: an ungrouped aggregate over zero rows emits ONE row with
    the aggregate identity (count -> 0, sum -> 0, min -> null)."""
    df = _run(query, engine)
    assert len(df) == 1
    assert _scalar(df["out"][0]) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_unwind_collect_identity_row(engine):
    df = _run("UNWIND [] AS x RETURN collect(x) AS out", engine)
    assert len(df) == 1
    assert _scalar(df["out"][0]) == []


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_unwind_multi_aggregate_identity_row(engine):
    df = _run("UNWIND [] AS x RETURN count(*) AS c, sum(x) AS s", engine)
    assert df.to_dict("records") == [{"c": 0, "s": 0}]


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_unwind_non_aggregate_still_zero_rows(engine):
    """Negative control: a non-aggregate RETURN over the empty stream stays
    zero rows -- the identity row is aggregate-only."""
    assert len(_run("UNWIND [] AS x RETURN x", engine)) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_nonempty_unwind_aggregate_unaffected(engine):
    df = _run("UNWIND [1, 2] AS x RETURN count(*) AS c, sum(x) AS s", engine)
    assert df.to_dict("records") == [{"c": 2, "s": 3}]


# ===========================================================================
# 4. Negative list subscripts (pandas serves; polars parity-or-NIE)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("UNWIND [[1,2,3]] AS l RETURN l[-1] AS v", [3]),
    ("UNWIND [[1,2,3]] AS l RETURN l[-3] AS v", [1]),
    ("UNWIND [[1,2,3]] AS l RETURN l[-4] AS v", [None]),
    ("UNWIND [[1,2,3]] AS l RETURN l[2] AS v", [3]),
    ("UNWIND [[1,2,3]] AS l RETURN l[3] AS v", [None]),
    ("UNWIND [[1,2],[5]] AS l RETURN l[-1] AS v", [2, 5]),
], ids=["last", "neg_first", "neg_out_of_range", "pos_last", "pos_out_of_range", "per_row"])
def test_negative_list_subscript_indexes_from_end(query, expected, engine):
    """#1899 item 4: l[-1] is the last element (openCypher); out-of-range in
    either direction is null. polars declines UNWIND honestly (parity-or-NIE)."""
    if engine == "polars":
        with pytest.raises(NotImplementedError):
            _run(query, "polars")
        return
    df = _run(query, engine)
    assert [_scalar(v) for v in df["v"]] == expected
