"""The array-side bindings+select path answers exactly what the canonical path answers.

Every case runs twice — once with the specialization and once with it disabled — and
the two frames must be identical in columns, dtypes, values and order. The canonical
path is the oracle; a decline is always allowed, a different answer never is.
"""
import os

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, e_reverse, n, order_by, rows, select
from graphistry.tests.compute.gfql.routes.switch import routes_off

pl = pytest.importorskip("polars")

ROUTE = "polars-bindings-select"

#: Every case here asserts the array route SERVES, so the whole file is an engagement
#: pin: in a lane where that route is off both legs run the canonical path and the
#: comparison proves nothing. It still runs in the point-rows-off lane, which is where
#: it gains cases rather than losing them.
pytestmark = pytest.mark.route_engaged("polars-bindings-select", "indexed-kernel")


NODES = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6, 7],
    "label__Person": [True, False, False, False, True, True, True],
    "label__Message": [False, True, True, True, False, False, False],
    "name": ["ann", "m1", "m2", "m3", "bob", "cid", "dee"],
    "age": [30, 0, 0, 0, 41, 52, 63],
    "score": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
})
EDGES = pd.DataFrame({
    "s": [2, 3, 4, 4, 2, 3, 5, 6],
    "d": [1, 1, 1, 5, 6, 6, 7, 7],
    "type": ["HAS_CREATOR"] * 4 + ["LIKES", "HAS_CREATOR", "KNOWS", "KNOWS"],
    "weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "tag": list("abcdefgh"),
})


def _graph(nodes=NODES, edges=EDGES, index=True):
    g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    return g.gfql_index_all(engine="polars") if index else g


def _both(g, ops, monkeypatch=None, expect_served=True, **kwargs):
    """(served, canonical) frames for the same query.

    Disabling goes through the repo's own route switch, which patches the symbol the chain
    actually calls; patching the defining module would leave the chain's import bound and
    silently compare the fast path against itself.

    ``index_policy="force"`` by default: these corpus graphs sit far below the cost
    gate, so under the default policy BOTH routes decline and the comparison would
    prove nothing. ``test_matches_canonical_under_every_index_policy`` covers the
    gated policies explicitly.
    """
    kwargs.setdefault("index_policy", "force")
    import graphistry.compute.gfql.lazy.engine.polars.chain as polars_chain
    seen = {"n": 0}
    specialization = polars_chain.try_bindings_select_polars

    def counting(*args, **kwargs_):
        result = specialization(*args, **kwargs_)
        seen["n"] += result is not None
        return result

    polars_chain.try_bindings_select_polars = counting
    try:
        # `point-rows` admits several of these shapes first; with it on, both legs would run
        # THAT route and the comparison would say nothing about the code under test.
        with routes_off(["polars-point-rows", "point-rows"]):
            served = g.gfql(ops, engine="polars", **kwargs)
    finally:
        polars_chain.try_bindings_select_polars = specialization
    if expect_served:
        assert seen["n"], "the array route never served; this case proves nothing"
    with routes_off(["polars-point-rows", "point-rows", ROUTE]):
        canonical = g.gfql(ops, engine="polars", **kwargs)
    return served._nodes, canonical._nodes


def _assert_identical(served, canonical):
    assert served.columns == canonical.columns
    assert served.schema == canonical.schema
    assert served.rows() == canonical.rows()


MIDDLES = {
    "one_hop_reverse": [
        n({"id": 1, "label__Person": True}, name="p"),
        e_reverse({"type": "HAS_CREATOR"}, name="r"),
        n({"label__Message": True}, name="m"),
    ],
    "one_hop_forward": [
        n({"id": 2, "label__Message": True}, name="m"),
        e_forward({"type": "HAS_CREATOR"}, name="r"),
        n({"label__Person": True}, name="p"),
    ],
    "two_hops": [
        n({"id": 1, "label__Person": True}, name="p"),
        e_reverse({"type": "HAS_CREATOR"}, name="r"),
        n({"label__Message": True}, name="m"),
        e_forward({}, name="r2"),
        n({}, name="t"),
    ],
    "unfiltered_edges": [
        n({"id": 4}, name="a"),
        e_forward({}, name="e"),
        n({}, name="b"),
    ],
    "dropping_endpoint_filter": [
        n({"id": 1, "label__Person": True}, name="p"),
        e_reverse({}, name="r"),
        n({"label__Message": True, "name": "m1"}, name="m"),
    ],
    "empty_result": [
        n({"id": 1}, name="p"),
        e_reverse({"type": "NOPE"}, name="r"),
        n({}, name="m"),
    ],
    "unaliased_edge": [
        n({"id": 1, "label__Person": True}, name="p"),
        e_reverse({"type": "HAS_CREATOR"}),
        n({"label__Message": True}, name="m"),
    ],
}

PROJECTIONS = {
    "node_columns": lambda: select([("who", "p.name"), ("mid", "m.id")]),
    "bare_alias": lambda: select([("pid", "p"), ("mid", "m")]),
    "literal": lambda: select([("k", 1), ("who", "p.name")]),
    "mixed_types": lambda: select([("who", "p.name"), ("sc", "p.score"), ("ag", "p.age")]),
}


@pytest.mark.parametrize("middle_name", sorted(MIDDLES))
@pytest.mark.parametrize("projection_name", sorted(PROJECTIONS))
def test_matches_canonical_across_shapes(middle_name, projection_name, monkeypatch):
    middle = MIDDLES[middle_name]
    aliases = {op._name for op in middle}
    projection = PROJECTIONS[projection_name]()
    referenced = set()
    for item in projection.params["items"]:
        expression = item[1]
        if isinstance(expression, str):
            referenced.add(expression.partition(".")[0])
    if not referenced <= aliases:
        pytest.skip("projection references an alias this middle does not bind")
    ops = [*middle, rows(), projection]
    served, canonical = _both(_graph(), ops, monkeypatch)
    _assert_identical(served, canonical)


@pytest.mark.parametrize("middle_name", ["one_hop_reverse", "two_hops"])
def test_matches_canonical_with_edge_payload_and_ordering(middle_name, monkeypatch):
    middle = MIDDLES[middle_name]
    edge_alias = next(op._name for op in middle[1::2] if isinstance(op._name, str))
    ops = [
        *middle, rows(),
        select([("w", f"{edge_alias}.weight"), ("t", f"{edge_alias}.tag"), ("mid", "m.id")]),
        order_by([("w", "desc"), ("mid", "asc")]),
    ]
    served, canonical = _both(_graph(), ops, monkeypatch)
    _assert_identical(served, canonical)


@pytest.mark.parametrize("policy,serves", [("use", False), ("force", True), ("off", False)])
def test_matches_canonical_under_every_index_policy(policy, serves, monkeypatch):
    """Each policy answers identically, and each one's SERVING side is stated, not assumed.

    ``off`` must decline by definition. ``use`` declines here too, because this graph is
    small enough that the cost gate refuses it -- which is the honest reading, and pinning it
    is what would catch the gate silently opening or closing.
    """
    ops = [*MIDDLES["one_hop_reverse"], rows(), select([("who", "p.name"), ("mid", "m.id")])]
    served, canonical = _both(
        _graph(), ops, monkeypatch, expect_served=serves, index_policy=policy,
    )
    _assert_identical(served, canonical)


def test_declines_and_matches_on_duplicate_node_ids(monkeypatch):
    nodes = pd.concat([NODES, NODES.iloc[[1]]], ignore_index=True)
    ops = [*MIDDLES["one_hop_reverse"], rows(), select([("who", "p.name"), ("mid", "m.id")])]
    served, canonical = _both(_graph(nodes=nodes), ops, monkeypatch, expect_served=False)
    _assert_identical(served, canonical)


def test_declines_and_matches_on_string_ids(monkeypatch):
    nodes = NODES.assign(id=[f"n{v}" for v in NODES.id])
    edges = EDGES.assign(s=[f"n{v}" for v in EDGES.s], d=[f"n{v}" for v in EDGES.d])
    middle = [
        n({"id": "n1", "label__Person": True}, name="p"),
        e_reverse({"type": "HAS_CREATOR"}, name="r"),
        n({"label__Message": True}, name="m"),
    ]
    ops = [*middle, rows(), select([("who", "p.name"), ("mid", "m.id")])]
    served, canonical = _both(_graph(nodes=nodes, edges=edges), ops, monkeypatch, expect_served=False)
    _assert_identical(served, canonical)


def test_declines_and_matches_with_null_edge_endpoints(monkeypatch):
    edges = EDGES.copy()
    edges.loc[0, "d"] = None
    ops = [*MIDDLES["one_hop_reverse"], rows(), select([("who", "p.name"), ("mid", "m.id")])]
    served, canonical = _both(_graph(edges=edges), ops, monkeypatch, expect_served=False)
    _assert_identical(served, canonical)


def test_declines_and_matches_without_indexes(monkeypatch):
    ops = [*MIDDLES["one_hop_reverse"], rows(), select([("who", "p.name"), ("mid", "m.id")])]
    served, canonical = _both(_graph(index=False), ops, monkeypatch, expect_served=False)
    _assert_identical(served, canonical)


@pytest.mark.route_engaged("polars-bindings-select", "indexed-kernel")
def test_served_path_records_the_same_index_trace(monkeypatch):
    from graphistry.compute.gfql.index.api import index_trace
    ops = [*MIDDLES["one_hop_reverse"], rows(), select([("who", "p.name"), ("mid", "m.id")])]
    g = _graph()
    with index_trace() as served_trace:
        g.gfql(ops, engine="polars", index_policy="force")
    with routes_off([ROUTE]), index_trace() as canonical_trace:
        g.gfql(ops, engine="polars", index_policy="force")

    def decisions(trace):
        return [
            (step.get("seam"), step.get("served"), step.get("reason"), step.get("hop_count"),
             step.get("public_seed_scan"), step.get("hop_details"))
            for step in trace if step.get("op") == "indexed_traversal"
        ]

    assert decisions(served_trace) == decisions(canonical_trace)
    assert any(served for _, served, *_ in decisions(served_trace))


@pytest.mark.parametrize("seed", range(24))
def test_fuzz_matches_canonical(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    node_count = int(rng.integers(3, 14))
    ids = rng.permutation(np.arange(1, node_count + 1))
    nodes = pd.DataFrame({
        "id": ids,
        "flag": rng.integers(0, 2, node_count).astype(bool),
        "kind": rng.choice(["a", "b", "c"], node_count),
        "val": rng.integers(0, 5, node_count),
    })
    edge_count = int(rng.integers(0, 24))
    edges = pd.DataFrame({
        "s": rng.choice(ids, edge_count),
        "d": rng.choice(ids, edge_count),
        "type": rng.choice(["R", "S"], edge_count),
        "w": rng.integers(0, 9, edge_count),
    })
    seed_id = int(rng.choice(ids))
    hops = int(rng.integers(1, 4))
    middle = [n({"id": seed_id}, name="a0")]
    for hop in range(hops):
        direction = e_forward if rng.integers(0, 2) else e_reverse
        match = {"type": str(rng.choice(["R", "S"]))} if rng.integers(0, 2) else {}
        node_filter = {"flag": bool(rng.integers(0, 2))} if rng.integers(0, 2) else {}
        middle.append(direction(match, name=f"e{hop}"))
        middle.append(n(node_filter, name=f"a{hop + 1}"))
    last = f"a{hops}"
    ops = [
        *middle, rows(),
        select([("tail", last), ("kind", f"{last}.kind"), ("w", f"e{hops - 1}.w"), ("seedval", "a0.val")]),
        order_by([("tail", "asc"), ("w", "asc")]),
    ]
    served, canonical = _both(_graph(nodes=nodes, edges=edges), ops, monkeypatch)
    _assert_identical(served, canonical)


@pytest.mark.route_engaged("polars-bindings-select", "indexed-kernel")
def test_the_corpus_actually_takes_the_fast_path(monkeypatch):
    """A differential corpus that silently stopped serving would pass while proving nothing."""
    import graphistry.compute.gfql.lazy.engine.polars.chain as polars_chain
    calls = {"served": 0, "declined": 0}
    original = polars_chain.try_bindings_select_polars

    def counting(*args, **kwargs):
        out = original(*args, **kwargs)
        calls["served" if out is not None else "declined"] += 1
        return out

    # The chain's own symbol, which is what the route switch patches — patching the
    # defining module would leave this import bound and count nothing.
    monkeypatch.setattr(polars_chain, "try_bindings_select_polars", counting)
    g = _graph()
    for middle in MIDDLES.values():
        aliases = {op._name for op in middle}
        if "p" in aliases and "m" in aliases:
            g.gfql([*middle, rows(), select([("who", "p.name"), ("mid", "m.id")])],
                   engine="polars", index_policy="force")
    assert calls["served"] > 0


def test_rows_parameters_the_route_cannot_honor_decline_to_the_canonical_route():
    """An unrecognized ``rows()`` parameter must decline, not be silently ignored.

    ``attach_prop_columns`` and ``attach_prop_aliases`` restrict what the row table carries.
    The canonical polars route refuses them outright (parity-or-error, no silent fallback),
    so a fast path that answers anyway has changed the contract, not just the speed. The
    admission is an allow-list for exactly this reason: a parameter added later declines
    until someone teaches this route what it means.
    """
    middle = [n({"id": 1}, name="a"), e_forward({}, name="e1"), n({}, name="b")]
    projection = select([("x", "a.name"), ("y", "b.name")])
    g = _graph()
    for row_call in (rows(attach_prop_columns={"a": ["name"]}), rows(attach_prop_aliases=["a"])):
        ops = [*middle, row_call, projection]
        with pytest.raises(NotImplementedError):
            g.gfql(ops, engine="polars", index_policy="force")
        with routes_off([ROUTE]):
            with pytest.raises(NotImplementedError):
                g.gfql(ops, engine="polars", index_policy="force")
    # the same shape without the parameter still serves, so the decline is not blanket
    served, canonical = _both(g, [*middle, rows(), projection])
    _assert_identical(served, canonical)


def test_duplicate_projection_names_raise_the_canonical_error():
    """A duplicate output name is a GFQL error, not a raw engine error leaking through."""
    from graphistry.compute.exceptions import GFQLTypeError

    ops = [n({"id": 1}, name="a"), e_forward({}, name="e1"), n({}, name="b"),
           rows(), select([("x", "a.name"), ("x", "b.name")])]
    g = _graph()
    with pytest.raises(GFQLTypeError) as served_error:
        g.gfql(ops, engine="polars", index_policy="force")
    with routes_off([ROUTE]):
        with pytest.raises(GFQLTypeError) as canonical_error:
            g.gfql(ops, engine="polars", index_policy="force")
    assert served_error.value.code == canonical_error.value.code


def test_large_unsigned_ids_are_served_exactly():
    """Ids above 2^53 survive the id->row probe, which a float promotion would alias.

    numpy promotes (int64, uint64) to float64, and float64 cannot tell 2^53+1 from
    2^53+2. The probe only promotes when the two sides differ, so uniform UInt64 ids
    must be compared as integers and land on the right rows.
    """
    base = 2 ** 53
    ids = [base + 1, base + 2, base + 3, base + 4]
    nodes = pl.DataFrame({"id": pl.Series(ids, dtype=pl.UInt64), "name": list("abcd")})
    edges = pl.DataFrame({
        "s": pl.Series(ids[1:], dtype=pl.UInt64),
        "d": pl.Series([ids[0]] * 3, dtype=pl.UInt64),
    })
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql_index_all(engine="polars")
    ops = [n({"id": ids[0]}, name="p"), e_reverse({}, name="r"), n({}, name="m"),
           rows(), select([("mid", "m.id"), ("nm", "m.name")])]
    served, canonical = _both(g, ops)
    _assert_identical(served, canonical)
    assert sorted(canonical.rows()) == [(ids[1], "b"), (ids[2], "c"), (ids[3], "d")]


def test_mismatched_id_dtypes_decline_rather_than_promote():
    """UInt64 node ids against Int64 endpoints would promote to float64; decline instead.

    This is the negative side of the case above, and it is the guard that makes the
    promotion seam unreachable from the public surface rather than merely unlikely.
    """
    base = 2 ** 53
    ids = [base + 1, base + 2, base + 3, base + 4]
    nodes = pl.DataFrame({"id": pl.Series(ids, dtype=pl.UInt64), "name": list("abcd")})
    edges = pl.DataFrame({
        "s": pl.Series(ids[1:], dtype=pl.Int64),
        "d": pl.Series([ids[0]] * 3, dtype=pl.Int64),
    })
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql_index_all(engine="polars")
    ops = [n({"id": ids[0]}, name="p"), e_reverse({}, name="r"), n({}, name="m"),
           rows(), select([("mid", "m.id"), ("nm", "m.name")])]
    served, canonical = _both(g, ops, expect_served=False)
    _assert_identical(served, canonical)
