"""Exact nullable integer identifiers stay resolvable on the seeded one-hop fallback."""

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute import e_forward, n, rows, select
from graphistry.tests.compute.gfql.routes.switch import ROUTES, routes_off


@pytest.mark.parametrize(
    "dtype,base,step",
    [
        ("Int64", 2**53, 1),
        ("UInt64", 2**63, 1),
        ("Int64", -(2**53), -1),
    ],
)
@pytest.mark.parametrize("seed_offset", [0, 1, 2])
@pytest.mark.parametrize("disable_routes", [False, True])
@pytest.mark.parametrize("node_null", [False, True])
def test_seeded_hop_preserves_exact_nullable_integer_ids(
    dtype, base, step, seed_offset, disable_routes, node_null
):
    def ids(values):
        return pd.array(np.asarray(values, dtype=object), dtype=dtype)

    neighbor = base + step
    node_ids = [None, neighbor, base] if node_null else [neighbor, base]
    node_values = [None, "neighbor", "base"] if node_null else ["neighbor", "base"]
    nodes = pd.DataFrame({"id": ids(node_ids), "value": node_values})
    edges = pd.DataFrame(
        {
            "s": ids([None, base, neighbor, base]),
            "d": ids([base, neighbor, base, None]),
        }
    )
    graph = (
        graphistry.nodes(nodes, "id")
        .edges(edges, "s", "d")
        .create_index("node_id", engine="pandas")
        .gfql_index_edges("both", engine="pandas")
    )
    seed = base + (seed_offset * step)
    query = [
        n({"id": seed}, name="a"),
        e_forward(),
        n(name="b"),
        rows(source="b"),
        select([("identifier", "b.id"), ("value", "b.value")]),
    ]

    with routes_off(ROUTES if disable_routes else ()):
        result = graph.gfql(query, engine="pandas")._nodes

    expected_value = (
        "neighbor" if seed_offset == 0 else "base" if seed_offset == 1 else None
    )
    expected_id = neighbor if seed_offset == 0 else base
    expected = (
        [] if seed_offset == 2 else [{"identifier": expected_id, "value": expected_value}]
    )
    assert result.to_dict("records") == expected
    if expected:
        assert str(result["identifier"].dtype) == dtype


@pytest.mark.parametrize("node_null", [False, True])
def test_shared_endpoint_pool_preserves_exact_ids_for_unseeded_and_multi_hop(node_null):
    base = 2**53
    neighbor = base + 1
    far = base + 2
    node_ids = [None, base, neighbor, far] if node_null else [base, neighbor, far]
    nodes = pd.DataFrame({"id": pd.array(node_ids, dtype="Int64")})
    edges = pd.DataFrame(
        {
            "s": pd.array([base, neighbor], dtype="Int64"),
            "d": pd.array([neighbor, far], dtype="Int64"),
        }
    )
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

    def edge_pairs(out):
        return set(zip(out._edges["s"].tolist(), out._edges["d"].tolist()))

    unseeded = graph.hop(hops=1, engine="pandas")
    seeded_two_hop = graph.hop(
        nodes=pd.DataFrame({"id": pd.array([base], dtype="Int64")}),
        hops=2,
        engine="pandas",
    )
    expected = {(base, neighbor), (neighbor, far)}
    assert edge_pairs(unseeded) == expected
    assert edge_pairs(seeded_two_hop) == expected
    for out in (unseeded, seeded_two_hop):
        assert str(out._edges["s"].dtype) == "Int64"
        assert str(out._edges["d"].dtype) == "Int64"
