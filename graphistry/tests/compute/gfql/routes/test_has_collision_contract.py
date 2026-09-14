"""HAS destination narrowing depends on reached collisions before node predicates."""
import pandas as pd
import pytest
import graphistry
from graphistry.tests.compute.gfql.routes.registry import to_engine
from graphistry.tests.compute.gfql.routes.switch import ROUTES, routes_off

@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("reverse_rows", [False, True])
@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("explicit_label", [False, True])
@pytest.mark.parametrize("property_filter", [False, True])
def test_reached_has_collision_before_destination_filter(engine, reverse_rows, collision, explicit_label, property_filter):
    records = [(601, True, False, None), (300, False, False, "f300"),
               (400, False, True, "t4"), (400 if collision else 401, False, False, "f4"),
               (500, False, True, "t5"), (500, False, False, "f5")]
    nodes = pd.DataFrame(records[::-1] if reverse_rows else records,
                         columns=["id", "label__Post", "label__Tag", "name"])
    edges = pd.DataFrame({"s": [601, 601], "d": [400, 300], "type": ["HAS_TAG", "HAS_TAG"]})
    graph = graphistry.nodes(to_engine(nodes, engine), "id").edges(to_engine(edges, engine), "s", "d")
    label = ":Tag" if explicit_label else ""
    predicate = " {name: 'f300'}" if property_filter else ""
    query = "MATCH (p:Post {id: 601})-[:HAS_TAG]->(t" + label + predicate + ") RETURN t.name AS name ORDER BY name"
    expected = (["t4"] if collision or explicit_label else ["f300", "t4"])
    if property_filter:
        expected = [value for value in expected if value == "f300"]
    for disabled in ((), ROUTES):
        with routes_off(disabled):
            frame = graph.gfql(query, engine=engine)._nodes
        values = frame["name"].to_list() if engine == "polars" else (frame.to_pandas() if engine == "cudf" else frame)["name"].tolist()
        assert values == expected, (engine, disabled, collision, explicit_label, property_filter)
