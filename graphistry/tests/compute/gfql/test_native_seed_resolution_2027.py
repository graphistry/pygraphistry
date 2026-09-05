"""Native chain seeded lookups resolve their seed through the resident indexes.

The native surface (op lists) used to scan the node table for a seed predicate on a
property that is not the node binding, and declined named single-node ops and named
seeded hops whenever the traversal indexes were resident. Pins: a lane-shaped graph
(nodes bound on a synthetic key, predicate on ``id``, ``label__X`` columns) is served on
the fast path with parity to the full path, the alias columns sit where the full path
puts them, and the property index is the seam that resolves the seed.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
import graphistry.compute.chain as chain_mod
from graphistry.compute.ast import e_forward, n, rows, select
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.predicates.numeric import GT

ENGINES = ["pandas", "cudf"]


def _lane_graph(engine, n_persons=2000, n_messages=6000):
    rng = np.random.default_rng(1)
    persons = pd.DataFrame({"key": np.arange(n_persons), "id": np.arange(n_persons) + 10_000,
                            "label__Person": True, "label__Message": False,
                            "firstName": [f"f{i}" for i in range(n_persons)]})
    messages = pd.DataFrame({"key": np.arange(n_persons, n_persons + n_messages),
                             "id": np.arange(n_messages) + 50_000,
                             "label__Person": False, "label__Message": True, "firstName": None})
    nodes = pd.concat([persons, messages], ignore_index=True)
    edges = pd.DataFrame({"s": np.arange(n_persons, n_persons + n_messages),
                          "d": rng.integers(0, n_persons, n_messages), "type": "HAS_CREATOR"})
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "key").edges(edges, "s", "d")
    return g.gfql_index_all(engine=engine).gfql_index_node_props(["id"], engine=engine)


def _canon(frame):
    """Value parity: the native fast path may differ from the full path in row order and in
    the rows pivot's int-to-float artifact, never in values or columns."""
    df = frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("float64")
    cols = [str(c) for c in df.columns]
    df.columns = cols
    return df.sort_values(cols).reset_index(drop=True) if len(df) else df


def _run(g, ops, engine, fast):
    real = chain_mod._try_chain_fast_path
    hits = {"n": 0}

    def spy(*a, **k):
        r = real(*a, **k)
        hits["n"] += r is not None
        return r
    chain_mod._try_chain_fast_path = (lambda *a, **k: None) if not fast else spy
    try:
        return g.gfql(ops, engine=engine, index_policy="use"), hits["n"]
    finally:
        chain_mod._try_chain_fast_path = real


SHAPES = {
    "named single node + rows": lambda: [n({"id": 10_007, "label__Person": True}, name="p"), rows(source="p")],
    "named single node": lambda: [n({"id": 10_007, "label__Person": True}, name="p")],
    "seeded typed hop + rows + select": lambda: [
        n({"id": 50_003, "label__Message": True}, name="m"), e_forward({"type": "HAS_CREATOR"}),
        n({"label__Person": True}, name="p"), rows(source="p"),
        select([("personId", "p.id"), ("firstName", "p.firstName")])],
    "seeded typed hop, all named": lambda: [
        n({"id": 50_003, "label__Message": True}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="e"),
        n({"label__Person": True}, name="p")],
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_lane_shapes_are_served_with_exact_parity(engine, shape):
    g = _lane_graph(engine)
    ops = SHAPES[shape]()
    fast, hits = _run(g, ops, engine, fast=True)
    full, _ = _run(g, ops, engine, fast=False)
    assert hits >= 1, f"{shape}: the native fast path must serve"
    assert list(fast._nodes.columns) == list(full._nodes.columns)
    # alias flags are bool on the fast path and object after the full path's merges
    pd.testing.assert_frame_equal(_canon(fast._nodes), _canon(full._nodes), check_dtype=False)
    pd.testing.assert_frame_equal(_canon(fast._edges), _canon(full._edges), check_dtype=False)


@pytest.mark.parametrize("engine", ENGINES)
def test_property_index_resolves_the_seed(engine, monkeypatch):
    import graphistry.compute.gfql.index.bindings as bindings
    g = _lane_graph(engine)
    calls = {"n": 0}
    real = bindings._seed_rows_via_property_index

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(bindings, "_seed_rows_via_property_index", spy)
    out = g.gfql([n({"id": 10_007, "label__Person": True}, name="p")], engine=engine, index_policy="use")
    assert calls["n"] >= 1 and len(out._nodes) == 1
    calls["n"] = 0
    out = g.gfql([n({"id": 50_003, "label__Message": True}, name="m"), e_forward({"type": "HAS_CREATOR"}),
                  n({"label__Person": True}, name="p")], engine=engine, index_policy="use")
    assert calls["n"] >= 1 and len(out._edges) == 1


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("binding_first", [True, False])
def test_named_single_node_alias_layout_matches_the_full_path(engine, binding_first):
    g = _lane_graph(engine)
    if not binding_first:
        columns = list(g._nodes.columns)
        g = g.nodes(g._nodes[[*columns[1:], columns[0]]])
    ops = [n({"id": 10_007}, name="p")]
    fast, hits = _run(g, ops, engine, fast=True)
    full, _ = _run(g, ops, engine, fast=False)
    assert hits == 1
    assert list(fast._nodes.columns) == list(full._nodes.columns)
    assert list(fast._nodes.columns)[:2] == ["key", "p"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("binding_first", [True, False])
def test_named_single_node_alias_overwrites_colliding_property_like_full_path(engine, binding_first):
    g = _lane_graph(engine)
    columns = list(g._nodes.columns)
    nodes = g._nodes.assign(p="shadow")[[columns[0], "p", *columns[1:]]]
    if not binding_first:
        nodes = nodes[["p", *columns[1:], columns[0]]]
    g = g.nodes(nodes).gfql_index_all(engine=engine).gfql_index_node_props(["id"], engine=engine)
    ops = [n({"id": 10_007}, name="p")]
    fast, hits = _run(g, ops, engine, fast=True)
    full, _ = _run(g, ops, engine, fast=False)
    assert hits == 1
    assert list(fast._nodes.columns) == list(full._nodes.columns)
    pd.testing.assert_frame_equal(_canon(fast._nodes), _canon(full._nodes), check_dtype=False)


@pytest.mark.parametrize("engine", ENGINES)
def test_named_hop_aliases_overwrite_nonfinal_properties_like_full_path(engine):
    g = _lane_graph(engine)
    node_columns = list(g._nodes.columns)
    edge_columns = list(g._edges.columns)
    nodes = g._nodes.assign(m="node shadow")[[node_columns[0], "m", *node_columns[1:]]]
    edges = g._edges.assign(e="edge shadow")[[edge_columns[0], "e", *edge_columns[1:]]]
    g = g.nodes(nodes).edges(edges).gfql_index_all(engine=engine).gfql_index_node_props(["id"], engine=engine)
    ops = [
        n({"id": 50_003, "label__Message": True}, name="m"),
        e_forward({"type": "HAS_CREATOR"}, name="e"),
        n({"label__Person": True}, name="p"),
    ]
    fast, hits = _run(g, ops, engine, fast=True)
    full, _ = _run(g, ops, engine, fast=False)
    assert hits == 1
    assert list(fast._nodes.columns) == list(full._nodes.columns)
    assert list(fast._edges.columns) == list(full._edges.columns)
    pd.testing.assert_frame_equal(_canon(fast._nodes), _canon(full._nodes), check_dtype=False)
    pd.testing.assert_frame_equal(_canon(fast._edges), _canon(full._edges), check_dtype=False)


# ---- the other side of each boundary: policy off, stale indexes, non-scalar seeds ----

def _same_values(fast, full):
    """Value parity only: alias-marker placement differs between the lanes on older pandas."""
    a, b = _canon(fast), _canon(full)
    cols = sorted(a.columns)
    assert sorted(b.columns) == cols
    pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False)


def _run_policy(g, ops, engine, fast, index_policy):
    real = chain_mod._try_chain_fast_path
    chain_mod._try_chain_fast_path = real if fast else (lambda *a, **k: None)
    try:
        return g.gfql(ops, engine=engine, index_policy=index_policy)
    finally:
        chain_mod._try_chain_fast_path = real


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_policy_off_keeps_parity_and_uses_no_index(engine, shape):
    g = _lane_graph(engine)
    ops = SHAPES[shape]()
    fast = _run_policy(g, ops, engine, True, "off")
    full = _run_policy(g, ops, engine, False, "off")
    _same_values(fast._nodes, full._nodes)
    _same_values(fast._edges, full._edges)
    report = g.gfql_explain(ops, engine=engine, index_policy="off")
    assert report["used_index"] is False and report["decision_code"] == "policy_off", report


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_stale_indexes_keep_parity_and_are_not_used(engine, shape):
    g = _lane_graph(engine)
    stale = g.nodes(g._nodes.head(len(g._nodes) - 1))  # rebound frame: every resident index is stale
    ops = SHAPES[shape]()
    fast, _ = _run(stale, ops, engine, True)
    full, _ = _run(stale, ops, engine, False)
    _same_values(fast._nodes, full._nodes)
    _same_values(fast._edges, full._edges)
    assert stale.gfql_explain(ops, engine=engine, index_policy="use")["used_index"] is False


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("seed", [
    lambda: {"id": IsIn([10_007, 10_008]), "label__Person": True},
    lambda: {"id": GT(10_007), "label__Person": True},
], ids=["is_in", "gt"])
def test_non_scalar_seed_predicates_keep_parity_without_the_index(engine, seed):
    from graphistry.compute.gfql.index import index_trace
    g = _lane_graph(engine)
    ops = [n(seed(), name="p"), e_forward({"type": "HAS_CREATOR"}, name="e"), n({"label__Person": True}, name="q")]
    fast, _ = _run(g, ops, engine, True)
    full, _ = _run(g, ops, engine, False)
    _same_values(fast._nodes, full._nodes)
    _same_values(fast._edges, full._edges)
    with index_trace() as steps:
        g.gfql(ops, engine=engine, index_policy="use")
    assert not any(s.get("seam") in ("native_seed_lookup", "native_seeded_hop") and s.get("served") for s in steps), steps
