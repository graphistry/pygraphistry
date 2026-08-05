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
from graphistry.compute.ast import n, e_forward, e_reverse
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
    # null in the (small) key frame -> decline (isin matches nulls, merge is
    # version-dependent; declining keeps the equivalence version-independent)
    nullkey = pd.DataFrame({"id": [1.0, np.nan]})
    assert _lean_intersect_full(full.astype({"id": float}), nullkey, "id", Engine.PANDAS) is None
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

    real_intersect = chain_mod._lean_intersect_full
    real_prefilter = chain_mod._lean_prefilter_right
    hits = {"intersect": 0, "prefilter": 0}

    def _spy_intersect(full, key_frame, key, engine):
        out = real_intersect(full, key_frame, key, engine)
        if out is not None:
            hits["intersect"] += 1
        return out

    def _spy_prefilter(left, right, key, engine):
        out = real_prefilter(left, right, key, engine)
        # engaged iff it actually shrank the right frame
        try:
            if len(out) < len(right):
                hits["prefilter"] += 1
        except Exception:
            pass
        return out

    monkeypatch.setattr(chain_mod, "_lean_intersect_full", _spy_intersect)
    monkeypatch.setattr(chain_mod, "_lean_prefilter_right", _spy_prefilter)
    os.environ['GFQL_LEAN_COMBINE'] = '1'
    g, n_persons = _seeded_graph()
    seed_msg = n_persons + 456
    # A MULTI-hop seeded chain with a backward reconciliation: the single-hop
    # degenerate fast path (#1755) intercepts a seeded 1-hop before combine_steps,
    # so lean-combine (which optimizes combine_steps + the backward pass) is
    # exercised by a chain that actually reaches those passes. fwd->typed->rev
    # engages BOTH lean helpers (combine intersect + backward-pass prefilter).
    g.gfql(
        [n({"id": seed_msg}), e_forward(), n({"type": "Person"}), e_reverse(), n()],
        engine='pandas',
    )
    # both lean paths must fire on a seeded chain (else parity tests are vacuous)
    assert hits["intersect"] >= 1
    assert hits["prefilter"] >= 1


# --- empty-left shrink (#1783): both directions --------------------------------------
# The shrink returns a ZERO-ROW slice of `right` when `left` is empty. That is only sound
# because the sole call site merges how='left', which discards unmatched right rows anyway.
# These pin both directions: the empty case must shrink AND stay schema-identical, and the
# non-empty cases must be untouched by the new branch.

def _wide_right(n_rows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "id": np.arange(n_rows, dtype=np.int64),
        "f": rng.random(n_rows),
        "s": np.array([f"v{i % 7}" for i in range(n_rows)]),
        "b": rng.integers(0, 2, n_rows).astype(bool),
    })


def test_empty_left_shrinks_to_zero_rows_preserving_schema():
    """The optimization itself: nothing survives a left merge from an empty left."""
    right = _wide_right()
    left = right.iloc[:0][["id"]]
    out = _lean_prefilter_right(left, right, "id", Engine.PANDAS)

    assert len(out) == 0, "empty left must not carry the full right frame into the merge"
    # Schema-identical, or the downstream merge would produce different columns/dtypes.
    assert list(out.columns) == list(right.columns)
    assert list(out.dtypes) == list(right.dtypes)


def test_empty_left_merge_result_matches_the_unshrunk_merge():
    """End-to-end equivalence at the call site's actual merge type."""
    right = _wide_right()
    left = right.iloc[:0][["id"]]
    shrunk = _lean_prefilter_right(left, right, "id", Engine.PANDAS)

    got = left.merge(shrunk, on="id", how="left")
    expected = left.merge(right, on="id", how="left")
    pd.testing.assert_frame_equal(got, expected)


@pytest.mark.parametrize("n_left", [1, 5, 999, 1000])
def test_non_empty_left_is_unaffected_by_the_empty_branch(n_left):
    """The other direction: adding the empty-left case must not perturb any non-empty
    one — neither the shrink-eligible small lefts nor the ones the ratio gate declines."""
    right = _wide_right()
    left = right[["id"]].head(n_left)
    out = _lean_prefilter_right(left, right, "id", Engine.PANDAS)

    got = left.merge(out, on="id", how="left").sort_values("id").reset_index(drop=True)
    expected = left.merge(right, on="id", how="left").sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected)
    assert len(got) == n_left


def test_empty_left_with_no_matching_keys_is_still_empty():
    """A NON-empty left whose keys miss entirely must still yield null-filled rows, one
    per left row — not a zero-row frame.

    Note on what this does and does not pin: it constrains the RESULT, not the branch
    taken. Shrinking on 'no key overlap' would also be sound under how='left' (both give
    null-filled rows), so this test deliberately cannot distinguish those two
    implementations — verified by mutation. What it does catch is the result-level error:
    dropping the unmatched left rows entirely.
    """
    right = _wide_right()
    left = pd.DataFrame({"id": np.array([10_000, 10_001], dtype=np.int64)})
    out = _lean_prefilter_right(left, right, "id", Engine.PANDAS)

    got = left.merge(out, on="id", how="left")
    expected = left.merge(right, on="id", how="left")
    pd.testing.assert_frame_equal(got, expected)
    assert len(got) == 2 and got["f"].isna().all()


def test_empty_left_shrink_holds_through_the_chain():
    """The shape the fix exists for: a seed that matches nothing, so the intermediate is
    empty and the combine would otherwise join it against the whole frame. Lean-on must
    equal lean-off."""
    g, _ = _seeded_graph()
    q = [n({"id": -12345}), e_forward(), n()]

    os.environ['GFQL_LEAN_COMBINE'] = '0'
    off = g.chain(q, engine='pandas')
    os.environ['GFQL_LEAN_COMBINE'] = '1'
    on = g.chain(q, engine='pandas')

    pd.testing.assert_frame_equal(
        off._nodes.sort_values(list(off._nodes.columns)).reset_index(drop=True),
        on._nodes.sort_values(list(on._nodes.columns)).reset_index(drop=True),
    )
    assert len(on._nodes) == 0


def test_empty_left_shrink_works_on_a_float_index():
    """`right[0:0]` is LABEL-based on a float index — pandas routes those through
    slice_indexer — so the bare-slice spelling returns ONE row there and the shrink
    silently does nothing. The result stays correct either way (a 0-row left still yields
    a 0-row how='left' merge), which is exactly why this needs its own test: the bug is a
    silent perf no-op, invisible to any assertion about the merge result."""
    right = _wide_right()
    right.index = np.arange(len(right), dtype=float)
    left = right.iloc[:0][["id"]]

    out = _lean_prefilter_right(left, right, "id", Engine.PANDAS)
    assert len(out) == 0, (
        f"shrink returned {len(out)} rows on a float index — bare [0:0] label-sliced "
        "instead of position-sliced, so the optimization did not fire"
    )
    pd.testing.assert_frame_equal(
        left.merge(out, on="id", how="left"), left.merge(right, on="id", how="left"))
