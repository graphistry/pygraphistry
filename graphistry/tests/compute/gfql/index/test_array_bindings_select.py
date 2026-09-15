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

pl = pytest.importorskip("polars")

SPECIALIZATION = "graphistry.compute.gfql.lazy.engine.polars.chain_specializations.bindings_select"


@pytest.fixture
def no_specialization(monkeypatch):
    """Disable the array path so the same query runs through the canonical route."""
    import graphistry.compute.gfql.lazy.engine.polars.chain as polars_chain
    import importlib
    module = importlib.import_module(SPECIALIZATION)
    monkeypatch.setattr(module, "try_bindings_select_polars", lambda *a, **k: None)
    monkeypatch.setattr(
        polars_chain, "_try_indexed_middle_polars", polars_chain._try_indexed_middle_polars,
    )
    return module


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


def _both(g, ops, monkeypatch, **kwargs):
    """(served, canonical) frames for the same query.

    ``index_policy="force"`` by default: these corpus graphs sit far below the cost
    gate, so under the default policy BOTH routes decline and the comparison would
    prove nothing. ``test_matches_canonical_under_every_index_policy`` covers the
    gated policies explicitly.
    """
    import importlib
    module = importlib.import_module(SPECIALIZATION)
    kwargs.setdefault("index_policy", "force")
    served = g.gfql(ops, engine="polars", **kwargs)
    original = module.try_bindings_select_polars
    monkeypatch.setattr(module, "try_bindings_select_polars", lambda *a, **k: None)
    canonical = g.gfql(ops, engine="polars", **kwargs)
    monkeypatch.setattr(module, "try_bindings_select_polars", original)
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


@pytest.mark.parametrize("kwargs", [{"index_policy": "use"}, {"index_policy": "force"}, {"index_policy": "off"}])
def test_matches_canonical_under_every_index_policy(kwargs, monkeypatch):
    ops = [*MIDDLES["one_hop_reverse"], rows(), select([("who", "p.name"), ("mid", "m.id")])]
    served, canonical = _both(_graph(), ops, monkeypatch, **kwargs)
    _assert_identical(served, canonical)


def test_declines_and_matches_on_duplicate_node_ids(monkeypatch):
    nodes = pd.concat([NODES, NODES.iloc[[1]]], ignore_index=True)
    ops = [*MIDDLES["one_hop_reverse"], rows(), select([("who", "p.name"), ("mid", "m.id")])]
    served, canonical = _both(_graph(nodes=nodes), ops, monkeypatch)
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
    served, canonical = _both(_graph(nodes=nodes, edges=edges), ops, monkeypatch)
    _assert_identical(served, canonical)


def test_declines_and_matches_with_null_edge_endpoints(monkeypatch):
    edges = EDGES.copy()
    edges.loc[0, "d"] = None
    ops = [*MIDDLES["one_hop_reverse"], rows(), select([("who", "p.name"), ("mid", "m.id")])]
    served, canonical = _both(_graph(edges=edges), ops, monkeypatch)
    _assert_identical(served, canonical)


def test_declines_and_matches_without_indexes(monkeypatch):
    ops = [*MIDDLES["one_hop_reverse"], rows(), select([("who", "p.name"), ("mid", "m.id")])]
    served, canonical = _both(_graph(index=False), ops, monkeypatch)
    _assert_identical(served, canonical)


def test_served_path_records_the_same_index_trace(monkeypatch):
    from graphistry.compute.gfql.index.api import index_trace
    ops = [*MIDDLES["one_hop_reverse"], rows(), select([("who", "p.name"), ("mid", "m.id")])]
    g = _graph()
    import importlib
    module = importlib.import_module(SPECIALIZATION)
    with index_trace() as served_trace:
        g.gfql(ops, engine="polars")
    original = module.try_bindings_select_polars
    monkeypatch.setattr(module, "try_bindings_select_polars", lambda *a, **k: None)
    with index_trace() as canonical_trace:
        g.gfql(ops, engine="polars")
    monkeypatch.setattr(module, "try_bindings_select_polars", original)

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


def test_the_corpus_actually_takes_the_fast_path(monkeypatch):
    """A differential corpus that silently stopped serving would pass while proving nothing."""
    import importlib
    module = importlib.import_module(SPECIALIZATION)
    calls = {"served": 0, "declined": 0}
    original = module.try_bindings_select_polars

    def counting(*args, **kwargs):
        out = original(*args, **kwargs)
        calls["served" if out is not None else "declined"] += 1
        return out

    monkeypatch.setattr(module, "try_bindings_select_polars", counting)
    g = _graph()
    for middle in MIDDLES.values():
        aliases = {op._name for op in middle}
        if "p" in aliases and "m" in aliases:
            g.gfql([*middle, rows(), select([("who", "p.name"), ("mid", "m.id")])],
                   engine="polars", index_policy="force")
    assert calls["served"] > 0
