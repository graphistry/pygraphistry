"""The pandas alias-tagging branch is frame-identical to the generic tagging it shortcuts.

``_tag_fast_path_aliases`` attaches the alias flag columns a named seeded hop carries; on
pandas it now builds them with one copy and in-place inserts. Pins: for every alias shape
the pandas branch returns exactly (columns, order, dtypes, index of every touched frame) what the generic
path returns, and the shapes it cannot reproduce in place (colliding alias names, float or
object ids) decline to the generic path.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
import graphistry.compute.chain_fast_paths as cfp
from graphistry.compute.ast import e_forward, n


def _res(nodes, edges):
    return graphistry.nodes(nodes, "k").edges(edges, "s", "d")


def _both(res, aliases, direction="forward"):
    a0, a1, a2 = aliases
    real = cfp._tag_fast_path_aliases_eager
    branch = {"n": 0}

    def spy(*a, **k):
        r = real(*a, **k)
        branch["n"] += r is not None
        return r
    cfp._tag_fast_path_aliases_eager = spy
    try:
        fast = cfp._tag_fast_path_aliases(res, a0, a1, a2, "s", "d", "k", direction)
    finally:
        cfp._tag_fast_path_aliases_eager = real
    cfp._tag_fast_path_aliases_eager = lambda *a, **k: None
    try:
        generic = cfp._tag_fast_path_aliases(res, a0, a1, a2, "s", "d", "k", direction)
    finally:
        cfp._tag_fast_path_aliases_eager = real
    return fast, generic, branch["n"]


NODES = pd.DataFrame({"k": [3, 1, 2, 9], "x": ["a", "b", "c", "d"], "w": [1.5, 2.5, 3.5, 4.5]}, index=[10, 11, 12, 13])
EDGES = pd.DataFrame({"s": [3, 1], "d": [1, 2], "t": ["A", "B"]}, index=[7, 8])

SHAPES = {
    "all named": ("m", "e", "p"),
    "seed only": ("m", None, None),
    "edge only": (None, "e", None),
    "destination only": (None, None, "p"),
    "nodes only": ("m", None, "p"),
}


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("binding_first", [True, False])
def test_pandas_branch_is_frame_identical_to_generic_tagging(shape, direction, binding_first):
    nodes = NODES if binding_first else NODES[["x", "k", "w"]]
    fast, generic, branch = _both(_res(nodes, EDGES), SHAPES[shape], direction)
    assert branch == 1
    pd.testing.assert_frame_equal(fast._nodes, generic._nodes)
    pd.testing.assert_frame_equal(fast._edges, generic._edges)
    a0, a1, a2 = SHAPES[shape]
    if a0 is not None or a2 is not None:
        assert isinstance(fast._nodes.index, pd.RangeIndex)
    if a1 is not None:
        assert isinstance(fast._edges.index, pd.RangeIndex)


def test_dead_end_seed_is_tagged_false_on_both_paths():
    fast, generic, branch = _both(_res(NODES, EDGES.iloc[:0]), ("m", "e", "p"))
    assert branch == 1
    pd.testing.assert_frame_equal(fast._nodes, generic._nodes)
    assert not fast._nodes["m"].any() and not fast._nodes["p"].any()


DECLINE_SHAPES = {
    "colliding node alias": (NODES.assign(m=0), EDGES, ("m", "e", "p")),
    "colliding edge alias": (NODES, EDGES.assign(e=0), ("m", "e", "p")),
    "float ids": (NODES.assign(k=NODES["k"].astype(float)), EDGES, ("m", None, "p")),
    "object ids": (NODES.assign(k=NODES["k"].astype(str)), EDGES.assign(s=EDGES["s"].astype(str), d=EDGES["d"].astype(str)), ("m", None, "p")),
}


@pytest.mark.parametrize("shape", list(DECLINE_SHAPES))
def test_shapes_the_inplace_branch_cannot_reproduce_take_the_generic_path(shape):
    nodes, edges, aliases = DECLINE_SHAPES[shape]
    fast, generic, branch = _both(_res(nodes, edges), aliases)
    assert branch == 0
    pd.testing.assert_frame_equal(fast._nodes, generic._nodes)
    pd.testing.assert_frame_equal(fast._edges, generic._edges)


def test_end_to_end_named_seeded_hop_matches_the_full_path():
    rng = np.random.default_rng(5)
    nodes = pd.DataFrame({"k": np.arange(300), "id": np.arange(300) + 100, "label__P": np.arange(300) % 2 == 0})
    edges = pd.DataFrame({"s": rng.integers(0, 300, 900), "d": rng.integers(0, 300, 900), "type": "T"})
    g = graphistry.nodes(nodes, "k").edges(edges, "s", "d").gfql_index_all(engine="pandas").gfql_index_node_props(["id"], engine="pandas")
    ops = [n({"id": 142}, name="m"), e_forward({"type": "T"}, name="e"), n(name="p")]
    served = g.gfql(ops, engine="pandas", index_policy="use")
    full = g.gfql(ops, engine="pandas", index_policy="off")

    def key(df):
        return df.sort_values(list(df.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(key(served._nodes), key(full._nodes), check_dtype=False)
    pd.testing.assert_frame_equal(key(served._edges), key(full._edges), check_dtype=False)
    assert list(served._nodes.columns) == list(full._nodes.columns)


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("binding_first", [False, True])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_eager_alias_tags_keep_backend_and_inputs(engine, direction, empty, binding_first, shape):
    original_nodes = NODES if binding_first else NODES[["x", "k", "w"]]
    nodes, edges = original_nodes.copy(), EDGES.iloc[:0].copy() if empty else EDGES.copy()
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    fast, generic, served = _both(_res(nodes, edges), SHAPES[shape], direction)
    assert served == 1
    for actual, expected in [(fast._nodes, generic._nodes), (fast._edges, generic._edges)]:
        assert type(actual) is type(expected) is type(nodes)
        if engine == "cudf":
            actual, expected = actual.to_pandas(), expected.to_pandas()
        pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(nodes.to_pandas() if engine == "cudf" else nodes, original_nodes)
    pd.testing.assert_frame_equal(edges.to_pandas() if engine == "cudf" else edges,
                                  EDGES.iloc[:0] if empty else EDGES)


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_repeated_node_alias_uses_generic_overwrite_semantics(engine):
    nodes, edges = NODES.copy(), EDGES.copy()
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    fast, generic, served = _both(_res(nodes, edges), ("m", "e", "m"))
    assert served == 0
    actual, expected = fast._nodes, generic._nodes
    if engine == "cudf":
        actual, expected = actual.to_pandas(), expected.to_pandas()
    pd.testing.assert_frame_equal(actual, expected)
    assert actual["m"].tolist() == [False, True, True, False]


def test_mixed_integer_width_aliases_decline_before_lossy_array_membership():
    nodes = pd.DataFrame({"k": pd.Series([2**63, 2**63 + 1], dtype="uint64")})
    edges = pd.DataFrame({"s": pd.Series([2**63 - 1], dtype="int64"),
                          "d": pd.Series([2**63 - 1], dtype="int64")})
    result = cfp._tag_fast_path_aliases_eager(nodes, edges, "m", None, "p", "s", "d", "k")
    assert result is None
