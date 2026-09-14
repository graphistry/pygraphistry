"""Node selection preserves source-row identity without traversal set semantics."""
import pandas as pd
import pytest

import graphistry
from graphistry.compute import n
from graphistry.compute.chain import chain
from graphistry.tests.compute.gfql.routes.registry import to_engine
from graphistry.tests.compute.gfql.routes.switch import ROUTES, routes_off


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("selection", ["all", "matching", "missing"])
@pytest.mark.parametrize("seeded", [False, True])
def test_node_selection_retains_source_rows(engine, named, selection, seeded):
    ids = [1, 1, None, 2]
    values = [10, 11, 12, 13]
    nodes = pd.DataFrame({"id": pd.Series(ids, dtype="Int64"), "value": values,
                          "keep": [True, True, True, False]})
    edges = pd.DataFrame({"s": pd.Series([], dtype="Int64"), "d": pd.Series([], dtype="Int64")})
    graph = graphistry.nodes(to_engine(nodes, engine), "id").edges(to_engine(edges, engine), "s", "d")
    start = to_engine(nodes.iloc[[0, 3]], engine) if seeded else None
    filters = {} if selection == "all" else ({"keep": True} if selection == "matching" else {"value": 99})
    expected = [{"id": node_id, "value": value} for position, (node_id, value) in enumerate(zip(ids, values))
                if (not seeded or position in (0, 3))
                and (selection == "all" or (selection == "matching" and value != 13))]
    for disabled in ((), ROUTES):
        with routes_off(disabled):
            ops = [n(filters, name="picked" if named else None)]
            result = (chain(graph, ops, engine=engine, start_nodes=start) if seeded
                      else graph.gfql(ops, engine=engine))
        frame = result._nodes
        expected_columns = (["id", "picked", "value", "keep"] if named and engine != "polars"
                            else ["id", "value", "keep"] + (["picked"] if named else []))
        assert list(frame.columns) == expected_columns
        if engine == "polars":
            actual = frame.select(["id", "value"]).to_dicts()
            flags = frame["picked"].to_list() if named else []
        else:
            frame = frame.to_pandas() if engine == "cudf" else frame
            actual = [{"id": None if pd.isna(node_id) else int(node_id), "value": int(value)}
                      for node_id, value in zip(frame["id"], frame["value"])]
            flags = frame["picked"].tolist() if named else []
        assert actual == expected, (engine, disabled, seeded)
        assert flags == ([True] * len(expected) if named else [])
        assert len(result._edges) == 0
