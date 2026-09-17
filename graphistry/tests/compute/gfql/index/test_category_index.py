"""Category indexes answer scalar equality exactly as the canonical filter does.

The index turns a label or edge-type predicate into a code compare. Every case here
checks it against the filter it replaces, including the places where a code compare
could quietly disagree: nulls, a value the column never holds, boolean-versus-integer
conflation, and an index left over from a frame that has since changed.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.ast import e_forward, e_reverse, n, rows, select
from graphistry.compute.gfql.index.api import get_registry
from graphistry.compute.gfql.index.build import build_category_index
from graphistry.tests.compute.gfql.routes.switch import routes_off

pl = pytest.importorskip("polars")

ROUTE = "polars-bindings-select"

#: The end-to-end cases assert the array route and the category lookup SERVE, so they
#: are engagement pins: with that route off both legs run the canonical filter and the
#: comparison proves nothing. The build/registry unit tests below do not need the route
#: and keep running, which is why the mark is applied per test rather than per module.
engagement_pin = pytest.mark.route_engaged("polars-bindings-select", "indexed-kernel")


def _codes(frame, column):
    return build_category_index(frame, column, "nodes", Engine.POLARS)


@pytest.mark.parametrize("values,dtype", [
    ([True, False, True], pl.Boolean),
    ([True, None, False], pl.Boolean),
    (["a", "b", "a"], pl.String),
    (["a", None, "b"], pl.String),
    ([1, 2, 1], pl.Int64),
    ([1, None, 2], pl.Int64),
])
def test_codes_round_trip_every_indexable_dtype(values, dtype):
    frame = pl.DataFrame({"c": pl.Series(values, dtype=dtype)})
    index = _codes(frame, "c")
    assert index is not None
    assert index.n_rows == len(values)
    for row, value in enumerate(values):
        if value is None:
            # The null code is reserved: it is never handed out for a queried value.
            assert index.codes[row] not in set(index.value_codes.values())
        else:
            assert index.codes[row] == index.value_codes[value]


@pytest.mark.parametrize("column,reason", [
    ("floats", "float equality is not a category"),
    ("wide", "cardinality above the cap"),
    ("absent", "no such column"),
])
def test_declines_what_a_code_compare_should_not_answer(column, reason):
    frame = pl.DataFrame({
        "floats": [1.0, 2.0, 3.0],
        "wide": [str(value) for value in range(3)],
    })
    if column == "wide":
        frame = pl.DataFrame({"wide": [str(value) for value in range(200)]})
    assert _codes(frame, column) is None, reason


def test_declines_on_non_polars_engines():
    frame = pl.DataFrame({"c": [True, False]})
    assert build_category_index(frame, "c", "nodes", Engine.PANDAS) is None
    assert build_category_index(frame, "c", "nodes", Engine.CUDF) is None


def test_index_goes_stale_with_the_frame():
    nodes = pl.DataFrame({"id": [1, 2, 3], "kind": ["a", "b", "a"]})
    edges = pl.DataFrame({"s": [1, 2], "d": [2, 3]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql_index_all(engine="polars")
    registry = get_registry(g)
    assert registry.get_category_valid("nodes", "kind", g._nodes, Engine.POLARS) is not None
    # A different frame object with the same content is still a different binding.
    assert registry.get_category_valid("nodes", "kind", nodes.clone(), Engine.POLARS) is None
    # And a reshaped frame fails the fingerprint.
    assert registry.get_category_valid(
        "nodes", "kind", nodes.head(2), Engine.POLARS,
    ) is None
    assert registry.get_category_valid("nodes", "kind", None, Engine.POLARS) is None


NODES = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "label__Person": [True, None, None, None, True, True],
    "label__Message": [None, True, True, True, None, None],
    "kind": ["person", "msg", "msg", "msg", "person", "person"],
    "flag": [1, 0, 1, 0, 1, 0],
    "name": list("abcdef"),
})
EDGES = pd.DataFrame({
    "s": [2, 3, 4, 4, 2],
    "d": [1, 1, 1, 5, 6],
    "type": ["HAS_CREATOR", "HAS_CREATOR", "HAS_CREATOR", "LIKES", "LIKES"],
    "w": [1, 2, 3, 4, 5],
})


def _graph(nodes=NODES, edges=EDGES):
    g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    return g.gfql_index_all(engine="polars")


def _both(g, ops, expect_served=True, **kwargs):
    """(served, canonical) frames, having PROVEN the served leg actually took the path.

    A differential case whose fast path silently stopped serving compares the canonical
    route against itself and passes while proving nothing, so engagement is asserted here
    rather than assumed. ``expect_served=False`` is for cases that must decline.
    """
    import graphistry.compute.gfql.index.array_bindings as array_bindings
    import graphistry.compute.gfql.lazy.engine.polars.chain as polars_chain

    kwargs.setdefault("index_policy", "force")
    seen = {"specialization": 0, "category": 0}
    specialization = polars_chain.try_bindings_select_polars
    category = array_bindings._positions_via_category_index

    def counting_specialization(*args, **kwargs_):
        result = specialization(*args, **kwargs_)
        seen["specialization"] += result is not None
        return result

    def counting_category(*args, **kwargs_):
        result = category(*args, **kwargs_)
        seen["category"] += result is not None
        return result

    polars_chain.try_bindings_select_polars = counting_specialization
    array_bindings._positions_via_category_index = counting_category
    try:
        # point-rows admits some of these shapes first; this file is about the array route.
        with routes_off(["polars-point-rows", "point-rows"]):
            served = g.gfql(ops, engine="polars", **kwargs)
    finally:
        polars_chain.try_bindings_select_polars = specialization
        array_bindings._positions_via_category_index = category
    if expect_served:
        assert seen["specialization"], "the array route never served; this case proves nothing"
        assert seen["category"], "the category index never answered; this case proves nothing"
    with routes_off(["polars-point-rows", "point-rows", ROUTE, "indexed-kernel", "index-hop"]):
        canonical = g.gfql(ops, engine="polars", **{**kwargs, "index_policy": "off"})
    return served._nodes, canonical._nodes


# Node 1 has no OUTGOING edge, so these walk in reverse. A corpus that traverses the empty
# direction compares nothing to nothing, which is why `expect_rows` is asserted per case.
@pytest.mark.parametrize("seed_filter,edge_match,end_filter,expect_rows", [
    ({"id": 1, "label__Person": True}, {"type": "HAS_CREATOR"}, {"label__Message": True}, 3),
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"kind": "msg"}, 3),
    ({"id": 1}, {}, {}, 3),
    ({"id": 1}, {"type": "NOPE"}, {}, 0),                        # value the column never holds
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"kind": "ghost"}, 0),   # endpoint value never held
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"flag": 0}, 2),         # integer category
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"label__Message": True}, 3),
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"label__Person": True}, 0),  # all-null for these rows
])
@engagement_pin
def test_indexed_predicates_match_the_canonical_filter(seed_filter, edge_match, end_filter, expect_rows):
    ops = [
        n(seed_filter, name="a"),
        e_reverse(edge_match or None, name="e"),
        n(end_filter or None, name="b"),
        rows(),
        select([("aid", "a"), ("bid", "b"), ("bkind", "b.kind")]),
    ]
    served, canonical = _both(_graph(), ops)
    assert served.columns == canonical.columns
    assert served.schema == canonical.schema
    assert served.rows() == canonical.rows()
    assert canonical.height == expect_rows, "the oracle moved; the case no longer means what it says"


@pytest.mark.parametrize("end_filter,expect_rows", [
    ({"flag": True}, 1),            # integer column, boolean scalar: polars coerces, True == 1
    ({"flag": 1.0}, 1),             # integer column, float scalar
    ({"flag": False}, 2),           # integer column, boolean scalar, the zero side
    ({"label__Message": 1}, 3),     # boolean column, integer scalar
    ({"label__Message": 1.0}, 3),   # boolean column, float scalar
    ({"flag": 1}, 1),               # same type: the code compare answers it
    ({"label__Message": True}, 3),  # same type: the code compare answers it
])
@engagement_pin
def test_a_cross_type_scalar_answers_exactly_as_the_canonical_filter(end_filter, expect_rows):
    """A code compare is equality within one type; coercion belongs to the engine.

    ``True == 1`` in Python AND in a polars filter, so a code compare that found no
    same-type key must DECLINE, not report "matches nothing". Reporting nothing is a
    silent wrong answer, and these are the shapes that produce it.
    """
    ops = [
        n({"id": 1}, name="a"), e_reverse({"type": "HAS_CREATOR"}, name="e"),
        n(end_filter, name="b"), rows(), select([("bid", "b")]),
    ]
    served, canonical = _both(_graph(), ops)
    assert served.rows() == canonical.rows()
    assert canonical.height == expect_rows, "the oracle moved; the case no longer means what it says"


def test_a_cross_type_scalar_declines_rather_than_guessing():
    """The decline is the mechanism, so pin it directly and not only through the answer."""
    from graphistry.compute.gfql.index.array_bindings import _DECLINE, _code_for

    integer_column = {0: 0, 1: 1}
    boolean_column = {False: 0, True: 1}
    assert _code_for(integer_column, 1) == 1
    assert _code_for(integer_column, 7) is None           # that type, absent value: matches nothing
    assert _code_for(integer_column, True) is _DECLINE    # bool against an integer column
    assert _code_for(integer_column, 1.0) is _DECLINE     # float against an integer column
    assert _code_for(boolean_column, True) == 1
    assert _code_for(boolean_column, 1) is _DECLINE       # int against a boolean column
    assert _code_for({}, 1) is _DECLINE                   # all-null column: no type to compare


def test_null_rows_are_never_matched_by_a_scalar_predicate():
    frame = pl.DataFrame({"c": pl.Series([True, None, False], dtype=pl.Boolean)})
    index = _codes(frame, "c")
    assert index is not None
    true_code = index.value_codes[True]
    false_code = index.value_codes[False]
    assert index.codes[1] != true_code and index.codes[1] != false_code
    # The canonical filter agrees: a null row survives neither predicate.
    assert frame.filter(pl.col("c") == True).height == 1  # noqa: E712
    assert frame.filter(pl.col("c") == False).height == 1  # noqa: E712


def test_index_all_builds_categories_and_skips_the_rest():
    g = _graph()
    registry = get_registry(g)
    node_cols = set(registry.category_cols("nodes"))
    edge_cols = set(registry.category_cols("edges"))
    assert {"label__Person", "label__Message", "kind", "flag"} <= node_cols
    assert "type" in edge_cols
    # `name` is all-distinct here but still under the cap, so it is indexed; the cap is
    # what bounds the work, and a wide column is skipped.
    wide = NODES.assign(name=[f"n{value}" for value in range(len(NODES))])
    assert "name" in set(get_registry(_graph(nodes=wide)).category_cols("nodes"))


def test_registry_can_drop_categories():
    g = _graph()
    registry = get_registry(g)
    assert registry.category_cols("nodes")
    assert registry.without_categories().category_cols("nodes") == ()


def _endpoint_fact(g):
    return get_registry(g).get_endpoint_rows_valid(
        str(g._source), str(g._destination), str(g._node), g._edges, g._nodes, Engine.POLARS,
    )


def test_endpoint_rows_resolve_every_edge_endpoint():
    g = _graph()
    fact = _endpoint_fact(g)
    assert fact is not None
    ids = g._nodes.get_column("id").to_numpy()
    assert list(ids[fact.src_rows]) == list(g._edges.get_column("s").to_numpy())
    assert list(ids[fact.dst_rows]) == list(g._edges.get_column("d").to_numpy())


def test_endpoint_rows_decline_when_an_endpoint_has_no_node():
    nodes = pl.DataFrame({"id": [1, 2], "kind": ["a", "b"]})
    edges = pl.DataFrame({"s": [1, 9], "d": [2, 2]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql_index_all(engine="polars")
    assert _endpoint_fact(g) is None


def test_endpoint_rows_go_stale_with_either_frame():
    g = _graph()
    registry = get_registry(g)
    src, dst, node = str(g._source), str(g._destination), str(g._node)
    assert registry.get_endpoint_rows_valid(src, dst, node, g._edges, g._nodes, Engine.POLARS) is not None
    # A clone of either frame is a different binding, and a reshape fails the fingerprint.
    assert registry.get_endpoint_rows_valid(src, dst, node, g._edges.clone(), g._nodes, Engine.POLARS) is None
    assert registry.get_endpoint_rows_valid(src, dst, node, g._edges, g._nodes.clone(), Engine.POLARS) is None
    assert registry.get_endpoint_rows_valid(src, dst, node, g._edges, None, Engine.POLARS) is None
    assert registry.without_endpoint_rows().endpoint_rows == {}


@pytest.mark.parametrize("seed_filter,edge_match,end_filter", [
    ({"id": 1, "label__Person": True}, {"type": "HAS_CREATOR"}, {"label__Message": True}),
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"kind": "msg"}),
    ({"id": 1}, {}, {}),
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"kind": "ghost"}),
])
def test_results_match_with_and_without_the_endpoint_rows_fact(
    seed_filter, edge_match, end_filter, monkeypatch,
):
    """The fact is an accelerator: dropping it must not change a single row."""
    import graphistry.compute.gfql.index.registry as registry_mod
    ops = [
        n(seed_filter, name="a"), e_forward(edge_match or None, name="e"),
        n(end_filter or None, name="b"), rows(),
        select([("aid", "a"), ("bid", "b"), ("bkind", "b.kind")]),
    ]
    g = _graph()
    with_fact = g.gfql(ops, engine="polars", index_policy="force")._nodes
    monkeypatch.setattr(
        registry_mod.GfqlIndexRegistry, "get_endpoint_rows_valid",
        lambda *a, **k: None,
    )
    without_fact = g.gfql(ops, engine="polars", index_policy="force")._nodes
    assert with_fact.columns == without_fact.columns
    assert with_fact.schema == without_fact.schema
    assert with_fact.rows() == without_fact.rows()


def test_temporal_text_verdicts_match_a_direct_scan():
    frame = pl.DataFrame({
        "plain": ["hello", "a (paren) here", "no"],
        "constructor": ["date({year: 1984})", "x", "y"],
        "number": [1, 2, 3],
    })
    g = graphistry.nodes(frame.with_columns(pl.Series("id", [1, 2, 3])), "id").edges(
        pl.DataFrame({"s": [1], "d": [2]}), "s", "d",
    ).gfql_index_all(engine="polars")
    fact = get_registry(g).get_temporal_text_valid("nodes", g._nodes, Engine.POLARS)
    assert fact is not None
    assert fact.verdicts["plain"] is False
    assert fact.verdicts["constructor"] is True
    assert "number" not in fact.verdicts  # not a String column


def test_temporal_text_fact_goes_stale_with_the_frame():
    g = _graph()
    registry = get_registry(g)
    assert registry.get_temporal_text_valid("nodes", g._nodes, Engine.POLARS) is not None
    assert registry.get_temporal_text_valid("nodes", g._nodes.clone(), Engine.POLARS) is None
    assert registry.get_temporal_text_valid("nodes", None, Engine.POLARS) is None
    assert registry.without_temporal_text().temporal_text == {}


def test_projection_of_constructor_text_declines_to_the_canonical_route():
    """A column holding constructor text must not reach the caller raw."""
    nodes = pd.DataFrame({
        "id": [1, 2, 3],
        "label__Person": [True, None, None],
        "label__Message": [None, True, True],
        "kind": ["person", "msg", "msg"],
        "flag": [1, 0, 1],
        "name": ["a", "b", "c"],
        "when": ["date({year: 1984})", "date({year: 1985})", "plain"],
    })
    edges = pd.DataFrame({"s": [2, 3], "d": [1, 1], "type": ["HAS_CREATOR"] * 2, "w": [1, 2]})
    g = _graph(nodes=nodes, edges=edges)
    ops = [
        n({"id": 1}, name="a"), e_forward({"type": "HAS_CREATOR"}, name="e"),
        n({}, name="b"), rows(), select([("w", "b.when")]),
    ]
    import graphistry.compute.gfql.lazy.engine.polars.chain as polars_chain
    assert polars_chain.try_bindings_select_polars(g, ops[:3], ops[3:], None) is None
    # The same query still answers, through the canonical route.
    served = g.gfql(ops, engine="polars", index_policy="force")
    with routes_off([ROUTE]):
        canonical = g.gfql(ops, engine="polars", index_policy="force")
    assert served._nodes.rows() == canonical._nodes.rows()


def test_string_literal_that_is_constructor_text_also_declines():
    g = _graph()
    ops = [
        n({"id": 1}, name="a"), e_forward({"type": "HAS_CREATOR"}, name="e"),
        n({}, name="b"), rows(), select([("lit", "date({year: 1984})"), ("bid", "b")]),
    ]
    import graphistry.compute.gfql.lazy.engine.polars.chain as polars_chain
    # A bare string item is an expression, not a literal, so this declines on the plan;
    # the point is that neither route can emit raw constructor text.
    assert polars_chain.try_bindings_select_polars(g, ops[:3], ops[3:], None) is None


def test_a_wide_column_declines_without_paying_the_exact_distinct_pass():
    """A column the cap will clearly reject is ruled out without an exact distinct pass.

    Eligible columns are the minority of a wide frame, so deciding the clear rejections
    by sketch is what keeps the build from scaling with columns nobody queries. The
    exact pass still decides every column near the cap.
    """
    import polars as pl

    calls = {"exact": 0}
    original = pl.Series.unique

    def counting_unique(self, *args, **kwargs):
        calls["exact"] += 1
        return original(self, *args, **kwargs)

    wide = pl.DataFrame({"c": [f"v{value}" for value in range(5000)]})
    pl.Series.unique = counting_unique
    try:
        assert _codes(wide, "c") is None
        assert calls["exact"] == 0, "a clearly-over-cap column paid the exact distinct pass"
        calls["exact"] = 0
        narrow = pl.DataFrame({"c": [f"v{value % 64}" for value in range(5000)]})
        assert _codes(narrow, "c") is not None
        assert calls["exact"] == 1, "an eligible column must still be decided exactly"
    finally:
        pl.Series.unique = original


@pytest.mark.parametrize("dtype", ["String", "Categorical", "Int64", "UInt8"])
@pytest.mark.parametrize("distinct", [63, 64, 65, 200])
def test_the_cardinality_cap_is_decided_exactly_at_its_own_boundary(dtype, distinct):
    """The sketch may not move the cap, on ANY admitted dtype: 64 indexes, 65 does not.

    The sketch that rules out clearly-over-cap columns is approximate, so the cap is
    checked per dtype rather than on strings alone: a sketch that read differently on
    integers would silently cost those indexes.
    """
    builders = {
        "String": lambda: pl.Series([f"v{value % distinct}" for value in range(4000)], dtype=pl.String),
        "Categorical": lambda: pl.Series([f"v{value % distinct}" for value in range(4000)], dtype=pl.Categorical),
        "Int64": lambda: pl.Series([value % distinct for value in range(4000)], dtype=pl.Int64),
        "UInt8": lambda: pl.Series([value % distinct for value in range(4000)], dtype=pl.UInt8),
    }
    if dtype == "UInt8" and distinct > 255:
        pytest.skip("UInt8 cannot hold that many distinct values")
    index = _codes(pl.DataFrame({"c": builders[dtype]()}), "c")
    assert (index is not None) == (distinct <= 64)
    if index is not None:
        assert len(index.value_codes) == distinct


@engagement_pin
def test_an_all_null_column_declines_rather_than_coding_a_value_it_has_not_got():
    """An all-null column has no value to take a type from, so it cannot answer at all.

    It still BUILDS -- every row carries the reserved null code -- and the temptation is to
    read an empty value map as "matches nothing", which is the right answer here by accident
    and the wrong one as a rule. The lookup declines and the canonical filter decides.
    """
    frame = pl.DataFrame({"c": pl.Series([None, None, None], dtype=pl.String)})
    index = _codes(frame, "c")
    assert index is not None and dict(index.value_codes) == {}
    assert set(index.codes) == {0}, "every row should carry the reserved null code"

    nodes = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "allnull": pl.Series([None] * 4, dtype=pl.String),
        "name": list("abcd"),
    })
    edges = pl.DataFrame({"s": [2, 3, 4], "d": [1, 1, 1]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql_index_all(engine="polars")
    ops = [n({"id": 1}, name="a"), e_reverse({}, name="e"), n({"allnull": "x"}, name="b"),
           rows(), select([("bid", "b")])]
    served, canonical = _both(g, ops)
    assert served.rows() == canonical.rows() == []


def test_the_new_facts_report_stale_and_engine_mismatch_like_every_other_index():
    """A fact that reports itself usable after its frame was rebound would be worse than absent."""
    from graphistry.compute.gfql.index.api import show_indexes

    g = _graph()
    fresh = show_indexes(g, engine="polars")
    new_kinds = fresh[fresh["kind"].isin(["category", "endpoint_rows", "temporal_text"])]
    assert not new_kinds.empty and new_kinds["valid"].all() and new_kinds["usable"].all()

    # Rebinding the NODES frame must stale the node-role facts and the endpoint fact, which
    # spans both frames; the edge-role facts are untouched and must stay valid.
    rebound = show_indexes(g.nodes(g._nodes.clone(), "id"), engine="polars")
    node_role = rebound[rebound["name"].str.startswith(("category:nodes", "temporal_text:nodes"))]
    assert not node_role.empty and not node_role["valid"].any()
    assert not rebound[rebound["kind"] == "endpoint_rows"]["valid"].any()
    edge_role = rebound[rebound["name"].str.startswith(("category:edges", "temporal_text:edges"))]
    assert not edge_role.empty and edge_role["valid"].all()

    # An engine the index was not built for is a decline, with the shared wording.
    mismatched = show_indexes(g, engine="pandas")
    assert not mismatched["usable"].any()
    for _, row in mismatched[mismatched["kind"].isin(
        ["category", "endpoint_rows", "temporal_text"]
    )].iterrows():
        assert row["reason"] == (
            f"resident {row['kind']} index engine=polars, requested engine=pandas -> scan"
        )


def test_endpoint_rows_are_polars_only():
    """The only consumer is the polars array path, so other engines must not pay for it."""
    from graphistry.compute.gfql.index.build import build_endpoint_rows_fact
    from graphistry.compute.gfql.index.registry import NODE_ID

    g = _graph()
    node_index = get_registry(g).get(NODE_ID)
    assert node_index is not None
    for engine in (Engine.PANDAS, Engine.CUDF):
        assert build_endpoint_rows_fact(
            g._edges, g._nodes, str(g._source), str(g._destination), node_index, engine,
        ) is None


def test_show_indexes_accounts_for_every_array_carrying_structure():
    """An index invisible to the memory signal is memory nobody can see they are paying."""
    from graphistry.compute.gfql.index.api import show_indexes

    g = _graph()
    report = show_indexes(g, engine="polars")
    kinds = set(report["kind"])
    assert {"category", "endpoint_rows", "temporal_text"} <= kinds
    per_row_arrays = report[report["kind"].isin(["category", "endpoint_rows"])]
    assert (per_row_arrays["nbytes"] > 0).all(), "a per-row array reported as costing nothing"
    assert report["valid"].all() and report["usable"].all()


@pytest.mark.parametrize("dtype,indexable", [
    (pl.String, True),
    (pl.Categorical, True),
    (pl.Enum(["x", "y", "z"]), False),
])
def test_string_like_dtypes_code_or_decline_as_declared(dtype, indexable):
    """Categorical is admitted alongside String; Enum is not, and must decline cleanly.

    The build joins on a mapping frame it constructs itself, which for Categorical is the
    string-cache-sensitive case, so the codes are checked against the rows they came from
    rather than assumed.
    """
    values = ["x", "y", "x", "z"]
    frame = pl.DataFrame({"c": pl.Series(values, dtype=dtype)})
    index = _codes(frame, "c")
    assert (index is not None) == indexable
    if index is None:
        return
    assert set(index.value_codes) == set(values)
    for row, value in enumerate(values):
        assert index.codes[row] == index.value_codes[value]


@engagement_pin
@pytest.mark.parametrize("end_filter,expect_rows", [
    ({"kind": "msg"}, 2),
    ({"kind": "other"}, 1),
    ({"kind": "ghost"}, 0),  # a value the column never holds
])
def test_a_categorical_predicate_matches_the_canonical_filter(end_filter, expect_rows):
    """End to end on a Categorical column, which no other case here exercises."""
    nodes = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "kind": pl.Series(["seed", "msg", "msg", "other"], dtype=pl.Categorical),
        "name": list("abcd"),
    })
    edges = pl.DataFrame({"s": [2, 3, 4], "d": [1, 1, 1]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql_index_all(engine="polars")
    ops = [n({"id": 1}, name="a"), e_reverse({}, name="e"), n(end_filter, name="b"),
           rows(), select([("bid", "b"), ("bname", "b.name")])]
    served, canonical = _both(g, ops)
    assert served.columns == canonical.columns
    assert served.schema == canonical.schema
    assert served.rows() == canonical.rows()
    assert canonical.height == expect_rows, "the oracle moved; the case no longer means what it says"
