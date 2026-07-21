"""Parity + gate tests for the cardinality-aware "lean combine" (#1755, slice 5).

The lean path replaces full-node-frame ``safe_merge`` reconciliations in the
two-pass chain executor with ``isin`` membership filters when the wavefront is
much smaller than the full frame. It is byte-identical by construction; these
tests pin that (lean-on vs lean-off, via ``GFQL_LEAN_COMBINE``) and exercise the
narrow applicability gate directly.
"""
import os

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n, e_forward
from graphistry.compute.chain import (
    _is_unique_ids,
    _lean_combine_enabled,
    _lean_engine_ok,
    _lean_intersect_full,
    _lean_prefilter_right,
)
from graphistry.Engine import Engine


@pytest.fixture(autouse=True)
def _restore_lean_env():
    """Keep GFQL_LEAN_COMBINE hermetic across tests."""
    prev = os.environ.get('GFQL_LEAN_COMBINE')
    yield
    if prev is None:
        os.environ.pop('GFQL_LEAN_COMBINE', None)
    else:
        os.environ['GFQL_LEAN_COMBINE'] = prev


def _seeded_graph(n_persons: int = 1000, n_messages: int = 4000, seed: int = 0):
    """Message -> Person HAS_CREATOR graph (same shape as the #1755 probe)."""
    rng = np.random.default_rng(seed)
    persons = pd.DataFrame({"id": np.arange(n_persons), "type": "Person"})
    messages = pd.DataFrame(
        {"id": np.arange(n_persons, n_persons + n_messages), "type": "Message"}
    )
    ndf = pd.concat([persons, messages], ignore_index=True)
    edf = pd.DataFrame({
        "src": np.arange(n_persons, n_persons + n_messages),
        "dst": rng.integers(0, n_persons, n_messages),
        "type": "HAS_CREATOR",
    })
    return graphistry.nodes(ndf, "id").edges(edf, "src", "dst"), n_persons


# ---------------------------------------------------------------------------
# gate unit tests
# ---------------------------------------------------------------------------

def test_lean_combine_enabled_env():
    os.environ.pop('GFQL_LEAN_COMBINE', None)
    assert _lean_combine_enabled() is True          # default on
    os.environ['GFQL_LEAN_COMBINE'] = '0'
    assert _lean_combine_enabled() is False
    os.environ['GFQL_LEAN_COMBINE'] = '1'
    assert _lean_combine_enabled() is True


def test_lean_engine_ok_pandas_only():
    assert _lean_engine_ok(Engine.PANDAS) is True
    assert _lean_engine_ok(Engine.CUDF) is False
    assert _lean_engine_ok(Engine.DASK) is False


def test_is_unique_ids():
    assert _is_unique_ids(pd.Series([], dtype='int64')) is True
    assert _is_unique_ids(pd.Series([7])) is True
    assert _is_unique_ids(pd.Series([1, 2, 3])) is True
    assert _is_unique_ids(pd.Series([1, 2, 2])) is False


def test_lean_intersect_full_matches_inner_merge():
    full = pd.DataFrame({"id": np.arange(1000), "val": np.arange(1000) * 2})
    key_frame = pd.DataFrame({"id": [3, 900, 12]})
    lean = _lean_intersect_full(full, key_frame, "id", Engine.PANDAS)
    merged = full.merge(key_frame[["id"]], on="id", how="inner").reset_index(drop=True)
    assert lean is not None
    pd.testing.assert_frame_equal(lean, merged)


def test_lean_intersect_full_declines_when_not_applicable():
    full = pd.DataFrame({"id": np.arange(100), "val": np.arange(100)})
    # small side not >=4x smaller -> decline
    big_key = pd.DataFrame({"id": np.arange(50)})
    assert _lean_intersect_full(full, big_key, "id", Engine.PANDAS) is None
    # key_frame carries an extra column -> would fan-out/add cols -> decline
    extra = pd.DataFrame({"id": [1, 2], "extra": [9, 9]})
    assert _lean_intersect_full(full, extra, "id", Engine.PANDAS) is None
    # non-unique key -> decline
    dup = pd.DataFrame({"id": [1, 1]})
    assert _lean_intersect_full(full, dup, "id", Engine.PANDAS) is None
    # non-pandas engine -> decline
    assert _lean_intersect_full(full, pd.DataFrame({"id": [1]}), "id", Engine.CUDF) is None
    # disabled -> decline
    os.environ['GFQL_LEAN_COMBINE'] = '0'
    assert _lean_intersect_full(full, pd.DataFrame({"id": [1]}), "id", Engine.PANDAS) is None


def test_lean_prefilter_right_matches_left_merge():
    left = pd.DataFrame({"id": [5, 10]})
    right = pd.DataFrame({"id": np.arange(1000), "val": np.arange(1000)})
    shrunk = _lean_prefilter_right(left, right, "id", Engine.PANDAS)
    # left merge result identical whether right is pre-shrunk or not
    full_merge = left.merge(right, on="id", how="left")
    lean_merge = left.merge(shrunk, on="id", how="left")
    pd.testing.assert_frame_equal(full_merge, lean_merge)
    assert len(shrunk) <= len(right)


def test_lean_prefilter_right_noop_when_left_not_smaller():
    left = pd.DataFrame({"id": np.arange(50)})
    right = pd.DataFrame({"id": np.arange(100), "val": np.arange(100)})
    out = _lean_prefilter_right(left, right, "id", Engine.PANDAS)
    assert out is right  # untouched


# ---------------------------------------------------------------------------
# end-to-end parity: lean-on vs lean-off must be byte-identical
# ---------------------------------------------------------------------------

def _run(chain_ops, lean: str):
    os.environ['GFQL_LEAN_COMBINE'] = lean
    g, _ = _seeded_graph()
    out = g.gfql(chain_ops, engine='pandas')
    return (
        out._nodes.sort_values('id').reset_index(drop=True),
        out._edges.sort_values(['src', 'dst']).reset_index(drop=True),
    )


@pytest.mark.parametrize("ops_name", ["is5_creator", "is5_typed", "expand_both"])
def test_seeded_chain_parity_lean_on_off(ops_name):
    seed_msg = 1000 + 456  # a Message id (n_persons=1000)
    ops = {
        "is5_creator": [n({"id": seed_msg}), e_forward(), n()],
        "is5_typed": [
            n({"id": seed_msg}),
            e_forward(edge_match={"type": "HAS_CREATOR"}),
            n({"type": "Person"}),
        ],
        "expand_both": [n({"id": seed_msg}), e_forward(), n({"type": "Person"})],
    }[ops_name]
    on_nodes, on_edges = _run(ops, lean='1')
    off_nodes, off_edges = _run(ops, lean='0')
    pd.testing.assert_frame_equal(on_nodes, off_nodes)
    pd.testing.assert_frame_equal(on_edges, off_edges)


def test_lean_path_actually_engages(monkeypatch):
    """Guard against a vacuous parity test: on a seeded chain the lean intersect
    must fire at least once (return a non-None frame)."""
    import graphistry.compute.chain as chain_mod

    real = chain_mod._lean_intersect_full
    hits = {"non_none": 0}

    def _spy(full, key_frame, key, engine):
        out = real(full, key_frame, key, engine)
        if out is not None:
            hits["non_none"] += 1
        return out

    monkeypatch.setattr(chain_mod, "_lean_intersect_full", _spy)
    os.environ['GFQL_LEAN_COMBINE'] = '1'
    g, n_persons = _seeded_graph()
    seed_msg = n_persons + 456
    g.gfql([n({"id": seed_msg}), e_forward(), n({"type": "Person"})], engine='pandas')
    assert hits["non_none"] >= 1
