"""Named Polars hops preserve tables across singleton and fallback boundaries."""
import importlib

import pytest

import graphistry
from graphistry.compute import ast

pl = pytest.importorskip("polars")
from polars.testing import assert_frame_equal

chain_polars = importlib.import_module("graphistry.compute.gfql.lazy.engine.polars.chain")


@pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt64, pl.Float64])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shape", ["one", "loop", "parallel", "dangling", "null", "filtered", "multi", "empty"])
@pytest.mark.parametrize("aliases", [("m", "e", "p"), (None, None, None), ("m", None, None), (None, "e", "p")])
def test_named_hop_exact_tables(dtype, reverse, shape, aliases, monkeypatch):
    offset = 2**63 + 1024 if dtype == pl.UInt64 else 0
    nodes = pl.DataFrame({
        "key": pl.Series([offset + i for i in [4, 1, 6, 2, 5, 3]], dtype=dtype),
        "id": [104, 101, 106, 102, 105, 103],
        "kind": ["Message", "Person", "Message", "Person", "Message", "Person"],
        "value": ["a", None, "c", "d", "e", "f"],
    })
    ends = {
        "one": ([4], [1]), "loop": ([4], [4]), "parallel": ([4, 4], [1, 1]),
        "dangling": ([4], [99]), "null": ([4], [None]), "filtered": ([4], [6]),
        "multi": ([4, 4, 4, 6], [1, 2, None, 1]), "empty": ([], []),
    }[shape]
    src, dst = [[None if v is None else offset + v for v in values] for values in ends]
    if reverse:
        src, dst = dst, src
    edges = pl.DataFrame({
        "s": pl.Series(src, dtype=dtype), "d": pl.Series(dst, dtype=dtype),
        "type": pl.Series(["T"] * len(src), dtype=pl.String),
        "v": pl.Series(range(len(src)), dtype=pl.Int64),
    })
    g = graphistry.nodes(nodes, "key").edges(edges, "s", "d").gfql_index_all(engine="polars")
    n0, e, n2 = aliases
    ops = [
        ast.n({"key": offset + 4}, name=n0),
        (ast.e_reverse if reverse else ast.e_forward)({"type": "T"}, name=e),
        ast.n({} if shape == "loop" else {"kind": "Person"}, name=n2),
    ]
    actual = g.gfql(ops, engine="polars", index_policy="use")
    monkeypatch.setattr(chain_polars, "_try_seeded_chain_polars", lambda *args: None)
    expected = g.gfql(ops, engine="polars", index_policy="use")
    assert_frame_equal(actual._nodes, expected._nodes)
    assert_frame_equal(actual._edges, expected._edges)
    assert_frame_equal(g._nodes, nodes)
    assert_frame_equal(g._edges, edges)
