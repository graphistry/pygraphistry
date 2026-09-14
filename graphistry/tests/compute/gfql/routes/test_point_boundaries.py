"""Broadcast a graph-theoretic bag oracle across point-row size boundaries."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, e_reverse, n, rows, select
from graphistry.tests.compute.gfql.routes.registry import to_engine
from graphistry.tests.compute.gfql.routes.switch import ROUTES, routes_off


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("count", [0, 1, 2, 8, 9, 32, 33])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("aliases", [("a", "b"), ("seed_node", "target_node")])
def test_point_joined_bag_matches_input_edge_oracle(engine, count, reverse, aliases):
    if engine == "cudf" and os.environ.get("TEST_CUDF") != "1":
        pytest.skip("cuDF lane runs with TEST_CUDF=1")
    # Repeated destinations are distinct paths, and a different edge type is excluded.
    targets = [1 + i % 3 for i in range(count)]
    nodes = pd.DataFrame({"key": [0, 1, 2, 3], "value": [11, 22, 33, 44]})
    starts, ends = [0] * (count + 1), targets + [3]
    if reverse:
        starts, ends = ends, starts
    edges = pd.DataFrame({"s": starts, "d": ends, "eid": range(count + 1),
                          "type": ["X"] * count + ["Y"]})
    g = graphistry.nodes(to_engine(nodes, engine), "key").edges(
        to_engine(edges, engine), "s", "d", "eid").gfql_index_all(engine=engine)
    a, b = aliases
    ops = [n({"key": 0}, name=a), (e_reverse if reverse else e_forward)({"type": "X"}, name="link"),
           n(name=b), rows(), select([("seed", f"{a}.value"), ("target", f"{b}.value"),
                                      ("edge", "link.eid")])]
    expected = sorted((11, (target + 1) * 11, i) for i, target in enumerate(targets))

    def values(result):
        frame = result._nodes
        if hasattr(frame, "to_dicts"):
            return sorted((r["seed"], r["target"], r["edge"]) for r in frame.to_dicts())
        if hasattr(frame, "to_pandas"):
            frame = frame.to_pandas()
        return sorted(frame[["seed", "target", "edge"]].itertuples(index=False, name=None))

    assert values(g.gfql(ops, engine=engine, index_policy="force")) == expected
    with routes_off(ROUTES):
        assert values(g.gfql(ops, engine=engine, index_policy="force")) == expected


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("reverse_nodes", [False, True])
@pytest.mark.parametrize("reverse_edges", [False, True])
@pytest.mark.parametrize("hops", [1, 2])
@pytest.mark.parametrize("property_index", [False, True])
def test_joined_path_order_matches_input_positions(engine, reverse, reverse_nodes, reverse_edges, hops, property_index):
    if engine == "cudf" and os.environ.get("TEST_CUDF") != "1":
        pytest.skip("cuDF lane runs with TEST_CUDF=1")
    node_ids = [20, 10, 30, 40][::-1 if reverse_nodes else 1]
    edge_rows = [(10, 30, 0), (20, 30, 1), (10, 40, 2),
                 (30, 40, 3), (20, 40, 4), (10, 30, 5)][::-1 if reverse_edges else 1]
    seeds = {30, 40} if reverse else {10, 20}
    nodes = pd.DataFrame({"id": node_ids, "seed": [int(node in seeds) for node in node_ids]})
    edges = pd.DataFrame(edge_rows, columns=["s", "d", "eid"])
    graph = graphistry.nodes(to_engine(nodes, engine), "id").edges(
        to_engine(edges, engine), "s", "d", "eid").gfql_index_all(engine=engine)
    if property_index:
        graph = graph.gfql_index_node_props(["seed"], engine=engine)
    paths = [(node, node, ()) for node in node_ids if node in seeds]
    ops = [n({"seed": 1}, name="a0")]
    for step in range(hops):
        expanded = []
        for seed, current, used in paths:
            for src, dst, eid in edge_rows:
                before, after = (dst, src) if reverse else (src, dst)
                if before == current and eid not in used:
                    expanded.append((seed, after, (*used, eid)))
        paths = expanded
        ops += [(e_reverse if reverse else e_forward)(name=f"e{step}"), n(name=f"a{step + 1}")]
    projection = [("seed", "a0.id"), ("target", f"a{hops}.id")]
    projection += [(f"edge_{step}", f"e{step}.eid") for step in range(hops)]
    ops += [rows(), select(projection)]
    expected = [{"seed": seed, "target": target, **{f"edge_{step}": eid for step, eid in enumerate(used)}}
                for seed, target, used in paths]
    for disabled in ((), ROUTES):
        with routes_off(disabled):
            result = graph.gfql(ops, engine=engine, index_policy="force")._nodes
        assert list(result.columns) == [name for name, _ in projection]
        if engine == "polars":
            actual = result.to_dicts()
        else:
            actual = (result.to_pandas() if engine == "cudf" else result).to_dict("records")
        assert actual == expected, (engine, disabled, property_index)
