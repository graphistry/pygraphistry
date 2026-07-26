"""Semi-join key sides carry duplicates on purpose — pin the invariant that makes that safe.

The polars chain used to `.unique()` every frame it fed into a `how="semi"` join. That is a
no-op for the RESULT (a semi-join emits a left row iff >=1 match exists; duplicates neither
change which rows come back nor multiply them) but a full hash pass over the key column — and
on an unfiltered hop the key side IS the node table, i.e. O(N) work inside an O(degree) query.

These tests pin the boundary rather than the speed:
  * duplicate keys reaching a SEMI key side must not change results or multiply rows, and
  * the one frame that still needs `.unique()` — the alias frame feeding a how="left" join —
    must keep it, because THERE duplicates do multiply.
Pandas is the oracle throughout, so a divergence fails as a parity break, not a guess.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected

pl = pytest.importorskip("polars")

from graphistry.tests.compute.gfql.polars_test_utils import graph_sig  # noqa: E402


def _pair(nodes_pd, edges_pd):
    """Same graph as pandas frames and as polars frames."""
    g_pd = graphistry.edges(edges_pd, "s", "d").nodes(nodes_pd, "key")
    g_pl = graphistry.edges(pl.from_pandas(edges_pd), "s", "d").nodes(
        pl.from_pandas(nodes_pd), "key")
    return g_pd, g_pl


def _clean_frames():
    nodes = pd.DataFrame({"key": [0, 1, 2, 3, 4],
                          "id": ["a", "b", "c", "d", "e"],
                          "grp": [1, 2, 2, 1, 2]})
    edges = pd.DataFrame({"s": [0, 0, 1, 2, 3],
                          "d": [1, 2, 3, 3, 4],
                          "type": ["K", "K", "L", "K", "K"]})
    return nodes, edges


def _dup_key_frames():
    """A node table with the SAME key twice — the row that reaches a semi key side twice."""
    nodes, edges = _clean_frames()
    nodes = pd.concat([nodes, nodes[nodes["key"].isin([1, 3])]], ignore_index=True)
    return nodes, edges


def _dangling_frames():
    """An edge whose destination is absent from the node table: the endpoint gate must still
    exclude it, so this is the case where the semi-join is NOT vacuous."""
    nodes, edges = _clean_frames()
    edges = pd.concat([edges, pd.DataFrame({"s": [0], "d": [999], "type": ["K"]})],
                      ignore_index=True)
    return nodes, edges


SHAPES = {
    "fwd_typed": [n({"id": "a"}, name="m"), e_forward({"type": "K"}, name="r"), n(name="p")],
    "rev_typed": [n({"id": "d"}, name="m"), e_reverse({"type": "K"}, name="r"), n(name="p")],
    "undirected": [n({"id": "a"}, name="m"), e_undirected({"type": "K"}, name="r"), n(name="p")],
    "two_hop": [n({"id": "a"}, name="m"), e_forward({"type": "K"}, hops=2, name="r"), n(name="p")],
    "dest_filter": [n({"id": "a"}, name="m"), e_forward({"type": "K"}, name="r"),
                    n({"grp": 1}, name="p")],
    "src_and_dest_filter": [n({"grp": 1}, name="m"), e_forward({"type": "K"}, name="r"),
                            n({"grp": 2}, name="p")],
    "untyped": [n({"id": "a"}, name="m"), e_forward(name="r"), n(name="p")],
    "fixed_point": [n({"id": "a"}, name="m"),
                    e_forward({"type": "K"}, to_fixed_point=True, name="r"), n(name="p")],
}

BUILDERS = {"clean": _clean_frames, "dup_keys": _dup_key_frames, "dangling": _dangling_frames}


@pytest.mark.parametrize("frames", sorted(BUILDERS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_polars_matches_pandas_when_semi_key_sides_carry_duplicates(frames, shape):
    """Duplicate / dangling keys flow into the semi key sides; polars must still match pandas."""
    g_pd, g_pl = _pair(*BUILDERS[frames]())
    chain = list(SHAPES[shape])
    assert graph_sig(g_pd.chain(chain, engine="pandas")) == \
        graph_sig(g_pl.chain(chain, engine="polars")), \
        f"polars diverged from the pandas oracle [{frames}/{shape}]"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_duplicate_node_keys_do_not_multiply_result_rows(shape):
    """The regression a missing `.unique()` on a how="left" key side would cause.

    Duplicating node rows must not duplicate OUTPUT rows beyond what the duplicated input
    itself accounts for: the alias-flag join is a left join, so an undeduplicated alias frame
    would multiply every matching node. Bound the output by the input duplication instead of
    hardcoding a count, so the test survives changes to the fixture.
    """
    chain = list(SHAPES[shape])
    _, g_clean = _pair(*_clean_frames())
    _, g_dup = _pair(*_dup_key_frames())
    out_clean = g_clean.chain(chain, engine="polars")
    out_dup = g_dup.chain(chain, engine="polars")

    # each key appears at most twice in the duplicated node table
    counts = out_dup._nodes["key"].value_counts()
    worst = int(counts[counts.columns[-1]].max()) if counts.height else 0
    assert worst <= 2, f"[{shape}] a node key came back {worst}x — alias join multiplied rows"
    assert set(out_clean._nodes["key"].to_list()) == set(out_dup._nodes["key"].to_list()), \
        f"[{shape}] duplicating node rows changed WHICH nodes matched"


def test_dangling_endpoint_is_still_excluded():
    """The semi-join is load-bearing, not vacuous: an endpoint missing from the node table
    must not appear in the output. A 'the gate accepts everything, drop it' optimization
    would break exactly here."""
    g_pd, g_pl = _pair(*_dangling_frames())
    chain = [n({"id": "a"}, name="m"), e_forward({"type": "K"}, name="r"), n(name="p")]
    out = g_pl.chain(chain, engine="polars")
    assert 999 not in out._nodes["key"].to_list(), "dangling endpoint leaked into the nodes"
    assert graph_sig(g_pd.chain(chain, engine="pandas")) == graph_sig(out)


@pytest.mark.parametrize("shape", ["fwd_typed", "undirected", "two_hop"])
def test_duplicate_start_nodes_do_not_change_the_result(shape):
    """`start_nodes` reaches a semi key side (pattern_apply); repeating rows in it must be
    inert. This is the one key side a CALLER controls, so it can carry duplicates on any
    schema, not just a malformed graph."""
    _, g_pl = _pair(*_clean_frames())
    chain = list(SHAPES[shape])
    uniq = pl.DataFrame({"key": [0, 1, 2]})
    dup = pl.DataFrame({"key": [0, 1, 1, 2, 2, 2]})
    try:
        a = g_pl.chain(chain, engine="polars", start_nodes=uniq)
        b = g_pl.chain(chain, engine="polars", start_nodes=dup)
    except TypeError:
        pytest.skip("this chain surface does not accept start_nodes")
    assert graph_sig(a) == graph_sig(b), \
        f"[{shape}] duplicate start_nodes changed the result"
