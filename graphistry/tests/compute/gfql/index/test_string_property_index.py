"""Business-key strings use native indexed gathers with canonical scan semantics."""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.gfql.index import get_registry, index_trace
from graphistry.tests.compute.gfql.index.test_edge_property_index import frame_records


@pytest.fixture(params=["pandas", "polars", "cudf", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def graph(engine, dtype="object"):
    emails = pd.Series([f"user{i}@example.test" for i in range(400)], dtype=dtype)
    emails.iloc[7] = emails.iloc[9] = "alice@example.test"
    emails.iloc[11] = ""
    emails.iloc[13] = "é用户🙂"
    emails.iloc[15] = None
    nodes = pd.DataFrame({"id": np.arange(400), "email": emails, "keep": np.arange(400) % 3})
    nodes.index = np.arange(400)[::-1]
    edges = pd.DataFrame({"s": np.arange(399), "d": np.arange(399) + 1, "external_id": emails.iloc[:399].array})
    concrete = Engine(engine)
    return graphistry.nodes(df_to_engine(nodes, concrete), "id").edges(
        df_to_engine(edges, concrete), "s", "d",
    )


@pytest.mark.parametrize("dtype", ["object", "string"])
@pytest.mark.parametrize("query,expected", [
    ("MATCH (a {email: 'alice@example.test'}) RETURN a.id AS id", [7, 9]),
    ("MATCH (a {email: 'alice@example.test', keep: 1}) RETURN a.id AS id", [7]),
    ("MATCH (a {email: ''}) RETURN a.id AS id", [11]),
    ("MATCH (a {email: 'é用户🙂'}) RETURN a.id AS id", [13]),
    ("MATCH (a {email: 'missing'}) RETURN a.id AS id", []),
    ("MATCH (a {email: 'alice@example.test'})-[e]->(b) RETURN b.id AS id", [8, 10]),
    ("MATCH (a) WHERE a.email IN ['alice@example.test', 'alice@example.test', 'é用户🙂'] RETURN a.id AS id", [7, 9, 13]),
])
def test_string_business_key_query_parity_and_engagement(engine, dtype, query, expected):
    base = graph(engine, dtype)
    indexed = base.gfql("CREATE GFQL INDEX FOR node_prop ON (email)", engine=engine)
    assert get_registry(base).is_empty()
    actual = frame_records(indexed.gfql(query, engine=engine)._nodes)
    scan = frame_records(indexed.gfql(query, engine=engine, index_policy="off")._nodes)
    assert actual == scan
    assert [r["id"] for r in actual] == expected
    report = indexed.gfql_explain(query, engine=engine)
    assert report["error"] is None
    assert report["used_index"]


@pytest.mark.parametrize("kind,role,column", [("node_prop", "nodes", "email"), ("edge_prop", "edges", "external_id")])
def test_direct_string_membership_gathers_in_input_order(engine, kind, role, column):
    indexed = graph(engine).create_index(kind, column=column, engine=engine)
    method = indexed.filter_nodes_by_dict if role == "nodes" else indexed.filter_edges_by_dict
    with index_trace() as steps:
        out = method({column: ["é用户🙂", "alice@example.test", "alice@example.test"]}, engine=engine)
    frame = out._nodes if role == "nodes" else out._edges
    id_col = "id" if role == "nodes" else "s"
    assert [r[id_col] for r in frame_records(frame)] == [7, 9, 13]
    assert any(s.get("op") == "property_lookup" and s.get("path") == "index" for s in steps)
    shown = indexed.show_indexes(engine=engine)
    assert shown["valid"].all() and shown["usable"].all()
    assert shown["n_rows"].tolist() == [400 if role == "nodes" else 399]
    assert shown["nbytes"].iloc[0] > 0


def test_string_index_stale_rebinding_declines(engine):
    base = graph(engine)
    indexed = base.create_index("node_prop", column="email", engine=engine)
    rebound = indexed.nodes(graph(engine)._nodes)
    with index_trace() as steps:
        out = rebound.filter_nodes_by_dict({"email": "alice@example.test"}, engine=engine)
    assert [r["id"] for r in frame_records(out._nodes)] == [7, 9]
    assert not any(s.get("path") == "index" for s in steps)


@pytest.mark.parametrize("values", [[], [None, None]])
def test_empty_and_all_null_text_indexes(engine, values):
    nodes = pd.DataFrame({"id": np.arange(len(values)), "email": pd.Series(values, dtype="string")})
    base = graphistry.nodes(df_to_engine(nodes, Engine(engine)), "id")
    indexed = base.create_index("node_prop", column="email", engine=engine)
    assert get_registry(indexed).node_props["email"].n_keys == 0
    assert frame_records(indexed.filter_nodes_by_dict({"email": "missing"}, engine=engine)._nodes) == []


def test_mixed_object_column_is_not_coerced_to_text():
    base = graphistry.nodes(pd.DataFrame({"id": [0, 1], "mixed": ["1", 1]}), "id")
    with pytest.raises(NotImplementedError):
        base.create_index("node_prop", column="mixed")
