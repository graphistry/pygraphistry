from __future__ import annotations

import importlib.util
import os
from typing import Any

import pandas as pd
import numpy as np
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.api import cypher_to_gfql
from graphistry.compute.gfql.cypher.lowering import CompiledCypherMutationQuery, compile_cypher_query
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.cypher.result_postprocess import render_entity_text
from graphistry.tests.compute.gfql.cypher._whole_entity_compat import entity_text_records
from graphistry.compute.gfql.row.entity_props import MUTATION_NODE_ID, ID_PROPERTY_PRESENT


ENGINES = [
    pytest.param("pandas"),
    pytest.param("cudf", marks=pytest.mark.skipif(
        os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
    )),
    pytest.param("polars", marks=pytest.mark.skipif(
        importlib.util.find_spec("polars") is None,
        reason="polars is not installed",
    )),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("label", ["None", "nan", "<NA>"])
def test_node_create_declines_filtered_label_names_without_effects(engine, label):
    source = _graph(engine)
    before_nodes, before_edges = _records(source._nodes), _records(source._edges)
    with pytest.raises(GFQLValidationError) as caught:
        source.gfql(f"CREATE (:`{label}`)", engine=engine)
    assert caught.value.code == ErrorCode.E108
    assert caught.value.context["field"] == "create"
    assert caught.value.context["value"] == [label]
    assert _records(source._nodes) == before_nodes and _records(source._edges) == before_edges


@pytest.mark.parametrize("engine", ENGINES)
def test_node_create_label_name_case_is_preserved(engine):
    result = _graph(engine).gfql("CREATE (:none)", engine=engine)
    assert sum(bool(row.get("label__none")) for row in _records(result._nodes)) == 1


def _records(frame: Any) -> list[dict[str, Any]]:
    if isinstance(frame, pd.DataFrame):
        rows = frame.to_dict(orient="records")
    else:
        rows = frame.to_arrow().to_pylist()
    return [
        {key: None if value is None or not isinstance(value, (list, dict)) and pd.isna(value) else value for key, value in row.items()}
        for row in rows
    ]


def _entity_text_records(result: Any, entities: dict[str, str]) -> list[dict[str, Any]]:
    nodes = result._nodes
    if nodes is not None and not isinstance(nodes, pd.DataFrame):
        result = graphistry.nodes(nodes.to_pandas(), result._node)
    return entity_text_records(result, entities)


def _series_records(series: Any) -> list[Any]:
    if isinstance(series, pd.Series):
        return series.tolist()
    return series.to_pandas().tolist()


def _assert_served(engine: str, frame: Any) -> None:
    expected = "pandas" if engine == "pandas" else engine
    assert type(frame).__module__.split(".")[0] == expected.split("-")[0]


def _graph(engine: str) -> Any:
    nodes = pd.DataFrame({"id": [1, 2], "label__Person": [True, False]})
    edges = pd.DataFrame({"src": [1], "dst": [2], "type": ["KNOWS"]})
    if engine == "cudf":
        import cudf
        nodes = cudf.from_pandas(nodes)
        edges = cudf.from_pandas(edges)
    elif engine == "polars":
        polars = pytest.importorskip("polars")
        nodes = polars.from_pandas(nodes)
        edges = polars.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


@pytest.mark.parametrize("source_engine,target_engine", [
    pytest.param("pandas", "polars", marks=pytest.mark.skipif(
        importlib.util.find_spec("polars") is None,
        reason="polars is not installed",
    )),
    ("polars", "pandas"),
    pytest.param("pandas", "cudf", marks=pytest.mark.skipif(
        os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1",
    )),
])
def test_set_normalizes_target_frames_to_explicit_engine_and_retains_caller(source_engine, target_engine):
    source = _graph(source_engine)
    nodes, edges = source._nodes, source._edges
    before_nodes, before_edges = _records(nodes), _records(edges)

    result = source.gfql(
        "MATCH (n:Person) SET n.score = 20 RETURN n.score AS score", engine=target_engine,
    )

    _assert_served(target_engine, result._nodes)
    _assert_served(target_engine, result.gfql_graph._nodes)
    _assert_served(target_engine, result.gfql_graph._edges)
    assert _records(result._nodes) == [{"score": 20}]
    assert sorted((row["id"], row["score"]) for row in _records(result.gfql_graph._nodes)) == [(1, 20), (2, None)]
    assert _records(result.gfql_graph._edges) == before_edges
    assert source._nodes is nodes and source._edges is edges
    assert _records(nodes) == before_nodes and _records(edges) == before_edges


@pytest.mark.parametrize("engine", ENGINES)
def test_set_list_assignment_declines_before_value_hashing_without_effects(engine):
    source = _graph(engine)
    before_nodes, before_edges = _records(source._nodes), _records(source._edges)

    with pytest.raises(GFQLValidationError) as caught:
        source.gfql("MATCH (n:Person) SET n.values = [1, 2, 3] RETURN n", engine=engine)

    assert caught.value.code == ErrorCode.E108
    assert caught.value.context["field"] == "set"
    assert caught.value.context["value"] == "non_scalar_value"
    assert _records(source._nodes) == before_nodes and _records(source._edges) == before_edges


@pytest.mark.parametrize("engine", ENGINES)
def test_create_nodes_adds_identity_labels_and_properties(engine: str) -> None:
    source = _graph(engine)

    result = source.gfql(
        "CREATE (a:Admin {name: 'Ada', age: 36}), (b:Admin {name: $name})",
        params={"name": "Grace"},
        engine=engine,
        language="cypher",
    )

    assert isinstance(result, type(source))
    assert result._edges is source._edges
    assert _records(result._nodes) == [
        {"id": 1, MUTATION_NODE_ID: 1, ID_PROPERTY_PRESENT: True, "label__Person": True, "label__Admin": False, "name": None, "age": None},
        {"id": 2, MUTATION_NODE_ID: 2, ID_PROPERTY_PRESENT: True, "label__Person": False, "label__Admin": False, "name": None, "age": None},
        {"id": None, MUTATION_NODE_ID: 3, ID_PROPERTY_PRESENT: False, "label__Person": False, "label__Admin": True, "name": "Ada", "age": 36},
        {"id": None, MUTATION_NODE_ID: 4, ID_PROPERTY_PRESENT: False, "label__Person": False, "label__Admin": True, "name": "Grace", "age": None},
    ]
    assert _records(source._nodes) == [
        {"id": 1, "label__Person": True},
        {"id": 2, "label__Person": False},
    ]
    assert result.gfql_graph is result
    assert source.gfql_graph is source
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_nodes_without_existing_nodes_preserves_inferred_graph(engine: str) -> None:
    if engine == "polars":
        pytest.importorskip("polars")

    edges = pd.DataFrame({"src": [7], "dst": [8], "type": ["R"]})
    source = graphistry.edges(edges, "src", "dst")

    result = source.gfql("CREATE (a:L {name: 'x'})", engine=engine, language="cypher")

    assert _records(result._nodes) == [
        {"id": 7, MUTATION_NODE_ID: 7, ID_PROPERTY_PRESENT: True, "label__L": False, "name": None},
        {"id": 8, MUTATION_NODE_ID: 8, ID_PROPERTY_PRESENT: True, "label__L": False, "name": None},
        {"id": None, MUTATION_NODE_ID: 9, ID_PROPERTY_PRESENT: False, "label__L": True, "name": "x"},
    ]
    assert _records(result._edges) == [{"src": 7, "dst": 8, "type": "R"}]
    assert source._nodes is None
    assert result.gfql_graph is result
    _assert_served(engine, result._nodes)
    count = result.gfql("MATCH (a)-[r]->(b) RETURN count(*) AS count", engine=engine)
    assert _records(count._nodes) == [{"count": 1}]


@pytest.mark.parametrize("engine", ENGINES)
def test_create_declines_old_legacy_type_with_new_canonical_label(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": ["old"], "type": ["Old"]}),
        "id",
    )

    with pytest.raises(GFQLValidationError, match="legacy type labels"):
        source.gfql("CREATE (:New)", engine=engine, language="cypher")
    assert _records(source._nodes) == [{"id": "old", "type": "Old"}]


@pytest.mark.parametrize("engine", ENGINES)
def test_plain_read_ignores_legacy_type_when_label_flags_exist(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": ["old"], "type": ["Business"], "label__Foo": [True]}),
        "id",
    )
    rows = source.gfql("MATCH (old) RETURN old", engine=engine, language="cypher")

    assert _entity_text_records(rows, {"old": "nodes"}) == [{"old": "(:Foo)"}]


@pytest.mark.parametrize("engine", ENGINES)
def test_plain_read_retains_legacy_type_without_label_flags(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": ["old"], "type": ["Business"]}),
        "id",
    )
    rows = source.gfql("MATCH (old) RETURN old", engine=engine, language="cypher")

    assert _entity_text_records(rows, {"old": "nodes"}) == [{"old": "(:Business)"}]


@pytest.mark.parametrize("engine", ENGINES)
def test_create_declines_same_legacy_and_canonical_label(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": ["old"], "type": ["Shared"]}),
        "id",
    )

    with pytest.raises(GFQLValidationError, match="legacy type labels"):
        source.gfql("CREATE (:Shared)", engine=engine, language="cypher")


@pytest.mark.parametrize("engine", ENGINES)
def test_create_render_does_not_admit_null_or_nan_legacy_type_labels(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": ["old"], "type": [None]}),
        "id",
    )

    result = source.gfql("CREATE (:New)", engine=engine, language="cypher")
    rows = result.gfql(
        "MATCH (old) WHERE old.id = 'old' RETURN old", engine=engine, language="cypher"
    )

    assert _entity_text_records(rows, {"old": "nodes"}) == [{"old": "()"}]


@pytest.mark.parametrize("engine", ENGINES)
def test_create_declines_any_nonnull_legacy_type_with_new_canonical_label(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": ["old", "new"], "type": ["Old", None]}),
        "id",
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        source.gfql("CREATE (:New)", engine=engine, language="cypher")

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "create"
    assert exc_info.value.context["value"] == ["New"]
    assert _records(source._nodes) == [
        {"id": "old", "type": "Old"},
        {"id": "new", "type": None},
    ]


@pytest.mark.parametrize("engine", ENGINES)
def test_create_relationship_declines_endpoint_legacy_type(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": ["old"], "type": ["Old"]}),
        "id",
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        source.gfql("CREATE (:New)-[:R]->()", engine=engine, language="cypher")

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "create"
    assert exc_info.value.context["value"] == ["New"]


@pytest.mark.parametrize("engine", ENGINES)
def test_create_relationship_all_null_legacy_type_is_admitted(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": ["old"], "type": [None]}),
        "id",
    )

    result = source.gfql("CREATE (:New)-[:R]->()", engine=engine, language="cypher")

    assert "type" in result.gfql_graph._edges.columns


def test_create_preserves_string_existing_identity() -> None:
    source = graphistry.nodes(pd.DataFrame({"id": ["a", "b"]}), "id")
    result = source.gfql("CREATE ()", engine="pandas")
    assert result._nodes[result._node].tolist()[:2] == ["a", "b"]
    assert result._nodes[result._node].nunique() == 3
    assert _records(source._nodes) == [{"id": "a"}, {"id": "b"}]


def test_create_rejects_missing_existing_identity_column() -> None:
    source = graphistry.bind(node="id").nodes(pd.DataFrame({"name": ["Ada"]}))

    with pytest.raises(GFQLValidationError, match="missing identity column 'id'"):
        source.gfql("CREATE (a:L)", engine="pandas", language="cypher")


def test_create_return_has_rows_and_same_execution_graph() -> None:
    source = _graph("pandas")
    result = source.gfql("CREATE (a:L {name:'new'}) RETURN a.name AS name", engine="pandas")
    assert _records(result._nodes) == [{"name": "new"}]


@pytest.mark.parametrize("engine", ENGINES)
def test_create_relationship_adds_nodes_edges_and_bindings(engine: str) -> None:
    source = _graph(engine)

    result = source.gfql(
        "CREATE (a:Person {name: 'Ada'})-[r:KNOWS {since: 2020}]->"
        "(b:Person {name: 'Bob'}) RETURN a, b",
        engine=engine,
        language="cypher",
    )

    assert result.gfql_graph is not result
    assert result.gfql_graph._edge == "__graphistry_edge_identity__"
    rows = _records(result._edges)
    assert not rows
    graph_rows = _records(result.gfql_graph._edges)
    assert graph_rows[-1] == {
        "src": 3,
        "dst": 4,
        "type": "KNOWS",
        "since": 2020,
        "__graphistry_edge_identity__": 0,
    }
    assert graph_rows[:-1] == [
        {
            "src": 1,
            "dst": 2,
            "type": "KNOWS",
            "__graphistry_edge_identity__": -1,
            "since": None,
        },
    ]
    assert _records(source._edges) == [{"src": 1, "dst": 2, "type": "KNOWS"}]
    assert _records(source._nodes) == [{"id": 1, "label__Person": True}, {"id": 2, "label__Person": False}]
    _assert_served(engine, result.gfql_graph._edges)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_reverse_relationship_uses_semantic_endpoint_order(engine: str) -> None:
    source = _graph(engine)

    result = source.gfql(
        "CREATE (a {name:'A'})<-[:T]-(b {name:'B'})",
        engine=engine,
        language="cypher",
    )

    node_ids_by_name = {
        row["name"]: row["__graphistry_node_identity__"]
        for row in _records(result._nodes)
        if row.get("name") is not None
    }
    rows = _records(result._edges)
    assert rows[-1] == {
        "src": node_ids_by_name["B"],
        "dst": node_ids_by_name["A"],
        "type": "T",
        "__graphistry_edge_identity__": 0,
    }
    assert _records(result._nodes)[-1]["name"] in {"A", "B"}


@pytest.mark.parametrize("engine", ENGINES)
def test_create_parallel_and_reverse_relationships_preserve_multiplicity(engine: str) -> None:
    source = _graph(engine)

    result = source.gfql(
        "CREATE (a), (b), (a)-[:T]->(b), (a)<-[:T]-(b), (a)-[:U]->(b)",
        engine=engine,
        language="cypher",
    )

    rows = _records(result._edges)
    assert [(row["src"], row["dst"], row["type"]) for row in rows[1:]] == [
        (3, 4, "T"),
        (4, 3, "T"),
        (3, 4, "U"),
    ]
    assert [row["__graphistry_edge_identity__"] for row in rows[1:]] == [0, 1, 2]


@pytest.mark.parametrize("engine", ENGINES)
def test_create_relationship_reuses_and_allocates_in_source_order(engine: str) -> None:
    source = _graph(engine)

    result = source.gfql(
        "CREATE (a), (b), (a)-[r:T]->(a), ()-[s:T]->(a)",
        engine=engine,
        language="cypher",
    )

    assert len(result._nodes) == 5
    assert [row[MUTATION_NODE_ID] for row in _records(result._nodes)][2:] == [3, 4, 5]
    edges = _records(result._edges)
    assert edges[-2:] == [
        {"src": 3, "dst": 3, "type": "T", "__graphistry_edge_identity__": 0},
        {"src": 5, "dst": 3, "type": "T", "__graphistry_edge_identity__": 1},
    ]
    assert result.gfql_graph is result
    _assert_served(engine, result._edges)
    assert len(result.gfql_graph._nodes) == 5
    graph = result.gfql_graph
    assert [row[MUTATION_NODE_ID] for row in _records(graph._nodes)][2:] == [3, 4, 5]


@pytest.mark.parametrize("engine", ENGINES)
def test_create_backfills_keyless_old_edge_identities(engine: str) -> None:
    source = _graph(engine)

    first = source.gfql("CREATE (a)-[:T]->(b)", engine=engine, language="cypher")
    first_ids = [row["__graphistry_edge_identity__"] for row in _records(first._edges)]
    assert first_ids == [-1, 0]

    second = first.gfql("CREATE (c)-[:T]->(d)", engine=engine, language="cypher")
    second_ids = [row["__graphistry_edge_identity__"] for row in _records(second._edges)]
    assert second_ids == [-1, 0, 1]
    assert len(set(second_ids)) == len(second_ids)


def test_create_relationship_rejects_duplicate_alias_before_effects() -> None:
    source = _graph("pandas")
    before = _records(source._edges)

    with pytest.raises(GFQLValidationError, match="relationship variable is already bound"):
        source.gfql(
            "CREATE (a)-[r:T]->(b), (c)-[r:U]->(d)",
            engine="pandas",
            language="cypher",
        )

    assert _records(source._edges) == before


@pytest.mark.parametrize("decorated", [":Bar", " {name:'again'}", " {}"])
def test_create_rejects_decorated_repeated_alias_before_effects(decorated: str) -> None:
    source = _graph("pandas")
    before_nodes = _records(source._nodes)
    before_edges = _records(source._edges)

    with pytest.raises(GFQLValidationError, match="variable is already bound"):
        source.gfql(
            f"CREATE (n:Foo)\nCREATE (n{decorated})",
            engine="pandas",
            language="cypher",
        )

    assert _records(source._nodes) == before_nodes
    assert _records(source._edges) == before_edges


@pytest.mark.parametrize("decorated", [":Bar", " {name:'again'}", " {}"])
def test_create_rejects_same_clause_decorated_repeated_alias_before_effects(decorated: str) -> None:
    source = _graph("pandas")
    before_nodes = _records(source._nodes)
    before_edges = _records(source._edges)

    with pytest.raises(GFQLValidationError, match="variable is already bound"):
        source.gfql(
            f"CREATE (n:Foo), (n{decorated})",
            engine="pandas",
            language="cypher",
        )

    assert _records(source._nodes) == before_nodes
    assert _records(source._edges) == before_edges


@pytest.mark.parametrize(
    "query",
    [
        "CREATE (a), (a)-[:T]->(a)",
        "CREATE (`a{`), (`a{`)-[:T]->(`a{`)",
        "CREATE (a), (a /* { */)-[:T]->(a)",
    ],
)
def test_compile_create_accepts_structurally_bare_repeated_alias(query: str) -> None:
    compiled = compile_cypher_query(parse_cypher(query))

    assert isinstance(compiled, CompiledCypherMutationQuery)
    assert compiled.creates


@pytest.mark.parametrize(
    "query",
    [
        "CREATE (n:Foo), (n {name:'again'})",
        "CREATE (`a{`), (`a{` {name:'again'})",
        "CREATE (a), (a /* { */ :Bar)",
    ],
)
def test_compile_create_rejects_structurally_decorated_repeated_alias(query: str) -> None:
    with pytest.raises(GFQLValidationError, match="variable is already bound"):
        compile_cypher_query(parse_cypher(query))


def test_create_preserves_bare_repeated_alias_for_self_loop() -> None:
    source = _graph("pandas")

    result = source.gfql("CREATE (a), (a)-[:T]->(a)", engine="pandas", language="cypher")

    assert len(result._nodes) == 3
    assert _records(result._edges)[-1] == {
        "src": 3,
        "dst": 3,
        "type": "T",
        "__graphistry_edge_identity__": 0,
    }


@pytest.mark.parametrize("property_key", ["type", "label__X", "__private"])
def test_create_relationship_rejects_reserved_edge_properties(property_key: str) -> None:
    source = _graph("pandas")
    before = _records(source._edges)

    with pytest.raises(GFQLValidationError):
        source.gfql(
            f"CREATE (a)-[r:T {{{property_key}: 'x'}}]->(b)",
            engine="pandas",
            language="cypher",
        )

    assert _records(source._edges) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_create_relationship_allocates_existing_custom_edge_key(engine: str) -> None:
    edges = pd.DataFrame({"src": [1], "dst": [2], "eid": [100], "type": ["OLD"]})
    nodes = pd.DataFrame({"id": [1, 2]})
    if engine == "cudf":
        import cudf
        edges = cudf.from_pandas(edges)
        nodes = cudf.from_pandas(nodes)
    elif engine == "polars":
        polars = pytest.importorskip("polars")
        edges = polars.from_pandas(edges)
        nodes = polars.from_pandas(nodes)
    source = graphistry.nodes(nodes, "id").edges(edges, "src", "dst", edge="eid")

    result = source.gfql(
        "CREATE (a {name: 'new'})-[r:T {weight: 2}]->(b)",
        engine=engine,
        language="cypher",
    )

    rows = _records(result._edges)
    assert rows == [
        {"src": 1, "dst": 2, "eid": 100, "type": "OLD", "weight": None},
        {"src": 3, "dst": 4, "eid": 101, "type": "T", "weight": 2},
    ]
    assert result._edge == "eid"
    assert result._nodes is not source._nodes
    assert _records(source._edges) == [{"src": 1, "dst": 2, "eid": 100, "type": "OLD"}]
    _assert_served(engine, result._edges)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_relationship_allocates_existing_string_edge_key(engine: str) -> None:
    edges = pd.DataFrame({"src": [1], "dst": [2], "eid": ["e0"], "type": ["OLD"]})
    nodes = pd.DataFrame({"id": [1, 2]})
    if engine == "cudf":
        import cudf
        edges = cudf.from_pandas(edges)
        nodes = cudf.from_pandas(nodes)
    elif engine == "polars":
        polars = pytest.importorskip("polars")
        edges = polars.from_pandas(edges)
        nodes = polars.from_pandas(nodes)
    source = graphistry.nodes(nodes, "id").edges(edges, "src", "dst", edge="eid")

    result = source.gfql(
        "CREATE (a {name:'new'})-[r:T {weight:2}]->(b)",
        engine=engine,
        language="cypher",
    )

    assert _records(result._edges) == [
        {"src": 1, "dst": 2, "eid": "e0", "type": "OLD", "weight": None},
        {"src": 3, "dst": 4, "eid": "__gfql_edge_0", "type": "T", "weight": 2},
    ]
    _assert_served(engine, result._edges)


def test_create_relationship_rejects_shared_node_relationship_alias() -> None:
    source = _graph("pandas")
    before_nodes = _records(source._nodes)
    before_edges = _records(source._edges)

    with pytest.raises(GFQLValidationError, match="variable is already bound"):
        source.gfql("CREATE (a)-[a:T]->(b)", engine="pandas", language="cypher")

    assert _records(source._nodes) == before_nodes
    assert _records(source._edges) == before_edges


def test_compile_create_relationship_rejects_shared_node_relationship_alias() -> None:
    with pytest.raises(GFQLValidationError, match="variable is already bound"):
        compile_cypher_query(parse_cypher("CREATE (a)-[a:T]->(b)"))


def test_create_relationship_rejects_cross_clause_node_reuse_before_effects() -> None:
    source = _graph("pandas")
    before_nodes = _records(source._nodes)
    before_edges = _records(source._edges)

    with pytest.raises(GFQLValidationError, match="another clause"):
        source.gfql(
            "CREATE (a)\nCREATE (b)\nCREATE (a)-[:T]->(b)",
            engine="pandas",
            language="cypher",
        )

    assert _records(source._nodes) == before_nodes
    assert _records(source._edges) == before_edges


def test_create_relationship_rejects_configured_missing_edge_key() -> None:
    edges = pd.DataFrame({"src": [1], "dst": [2]})
    source = graphistry.nodes(pd.DataFrame({"id": [1, 2]}), "id").edges(
        edges, "src", "dst", edge="eid"
    )
    before = _records(source._edges)

    with pytest.raises(GFQLValidationError, match="missing identity column 'eid'"):
        source.gfql("CREATE (a)-[r:T]->(b)", engine="pandas", language="cypher")

    assert _records(source._edges) == before


def test_create_relationship_rejects_edge_key_property_override() -> None:
    edges = pd.DataFrame({"src": [1], "dst": [2], "eid": [100]})
    source = graphistry.nodes(pd.DataFrame({"id": [1, 2]}), "id").edges(
        edges, "src", "dst", edge="eid"
    )
    before = _records(source._edges)

    with pytest.raises(GFQLValidationError, match="Duplicate CREATE edge property 'eid'"):
        source.gfql(
            "CREATE (a)-[r:T {eid: 999}]->(b)",
            engine="pandas",
            language="cypher",
        )

    assert _records(source._edges) == before


@pytest.mark.parametrize("values", [[100, 100], [100, None]], ids=["duplicate", "null"])
def test_create_relationship_rejects_invalid_existing_edge_keys(values: list[object]) -> None:
    edges = pd.DataFrame({"src": [1, 2], "dst": [2, 1], "eid": values})
    source = graphistry.nodes(pd.DataFrame({"id": [1, 2]}), "id").edges(
        edges, "src", "dst", edge="eid"
    )
    before = _records(source._edges)

    with pytest.raises(GFQLValidationError, match="edge identities"):
        source.gfql("CREATE (a)-[r:T]->(b)", engine="pandas", language="cypher")

    assert _records(source._edges) == before


def test_create_relationship_rejects_non_identity_edge_key_family() -> None:
    edges = pd.DataFrame({"src": [1], "dst": [2], "eid": [True]})
    source = graphistry.nodes(pd.DataFrame({"id": [1, 2]}), "id").edges(
        edges, "src", "dst", edge="eid"
    )
    before = _records(source._edges)

    with pytest.raises(GFQLValidationError, match="integer or string edge identities"):
        source.gfql("CREATE (a)-[r:T]->(b)", engine="pandas", language="cypher")

    assert _records(source._edges) == before


def test_create_relationship_return_is_a_declared_deferred_boundary() -> None:
    source = _graph("pandas")
    before_nodes = _records(source._nodes)
    before_edges = _records(source._edges)

    with pytest.raises(GFQLValidationError):
        source.gfql("CREATE (a)-[r:T]->(b) RETURN r", engine="pandas", language="cypher")

    assert _records(source._nodes) == before_nodes
    assert _records(source._edges) == before_edges


def test_cypher_to_gfql_rejects_standalone_create() -> None:
    with pytest.raises(GFQLValidationError, match="Standalone Cypher CREATE"):
        cypher_to_gfql("CREATE (a:L)")


def test_gfql_graph_is_read_only() -> None:
    result = _graph("pandas").gfql(
        "CREATE (a:L)", engine="pandas", language="cypher"
    )

    with pytest.raises(AttributeError):
        result.gfql_graph = result


@pytest.mark.parametrize("engine", ENGINES)
def test_create_preserves_fractional_new_values_and_original_frame(engine: str) -> None:
    source = _graph(engine)
    nodes = source._nodes
    if engine == "polars":
        import polars as pl
        nodes = nodes.with_columns(pl.lit(1).alias("score"))
    else:
        nodes = nodes.assign(score=1)
    source = source.nodes(nodes)
    before = _records(source._nodes)
    result = source.gfql("CREATE (a {score: 1.5})", engine=engine)
    assert [row["score"] for row in _records(result._nodes)] == [1, 1, 1.5]
    assert _records(source._nodes) == before
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_sparse_created_integer_is_not_rounded_before_frame_construction(engine: str) -> None:
    wide = (1 << 60) + 1
    result = _graph(engine).gfql(
        "CREATE (a:Wide {serial: $wide}), (b:Other {name: 'other'})",
        params={"wide": wide}, engine=engine,
    )
    serials = [row["serial"] for row in _records(result._nodes) if row["serial"] is not None]
    assert serials == [wide]
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_declines_boolean_numeric_column_union(engine: str) -> None:
    source = _graph(engine)
    nodes = source._nodes
    if engine == "polars":
        import polars as pl
        nodes = nodes.with_columns(pl.lit(True).alias("flag"))
    else:
        nodes = nodes.assign(flag=True)
    source = source.nodes(nodes)
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (a {flag: 2})", engine=engine)
    assert exc.value.context["field"] == "flag"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("dtype,maximum", [("int8", 127), ("int16", 32767), ("int64", (1 << 63) - 1)])
def test_create_checks_the_entire_allocated_id_range(engine: str, dtype: str, maximum: int) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": pd.Series([maximum - 1], dtype=dtype)}), "id")
    before = _records(source._nodes)
    before_dtypes = list(map(str, source._nodes.dtypes))
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (a), (b)", engine=engine)
    assert exc.value.context["field"] == MUTATION_NODE_ID
    assert _records(source._nodes) == before
    assert list(map(str, source._nodes.dtypes)) == before_dtypes


@pytest.mark.parametrize("suffix", [
    "CALL graphistry.degree.write()",
    "UNWIND [1] AS x",
    "MATCH (n) RETURN n",
    "WITH 1 AS x RETURN x",
])
def test_create_rejects_the_entire_unsupported_mixed_program(suffix: str) -> None:
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    with pytest.raises(GFQLValidationError):
        source.gfql(f"CREATE (a:L) {suffix}")
    pd.testing.assert_frame_equal(source._nodes, before)


def test_strict_polars_gpu_creation_has_explicit_capability_decline() -> None:
    from graphistry.compute.exceptions import ErrorCode, GFQLUnsupportedError
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    with pytest.raises(GFQLUnsupportedError) as exc:
        source.gfql("CREATE (a:L)", engine="polars-gpu")
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "engine"
    pd.testing.assert_frame_equal(source._nodes, before)


@pytest.mark.parametrize("transform", ["nodes", "edges", "bind"])
def test_ordinary_and_transformed_graph_accessors_return_their_own_graph(transform: str) -> None:
    source = _graph("pandas")
    assert source.gfql_graph is source
    created = source.gfql("CREATE (a:L)")
    if transform == "nodes":
        result = created.nodes(created._nodes.iloc[:1].copy())
    elif transform == "edges":
        result = created.edges(created._edges.iloc[:0].copy())
    else:
        result = created.bind()
    assert result is not created and result.gfql_graph is result
    assert created.gfql_graph is created and len(created._nodes) == 3


def test_create_preflight_is_validate_only_and_matches_execution(monkeypatch) -> None:
    import graphistry.compute.gfql_unified as unified
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    with monkeypatch.context() as context:
        def reject_execution(*args, **kwargs):
            pytest.fail("preflight dispatched a mutation executor")
        context.setattr(unified, "execute_create_nodes", reject_execution)
        assert source.gfql_validate("CREATE (a:L)")["compiled_kind"] == "mutation"
    pd.testing.assert_frame_equal(source._nodes, before)
    ordinary = source.gfql("CREATE (a:L)")
    checked = source.gfql("CREATE (a:L)", validate=True)
    pd.testing.assert_frame_equal(ordinary._nodes, checked._nodes)


@pytest.mark.parametrize("validate", [False, True])
def test_remote_mutation_declines_before_any_request(monkeypatch, validate: bool) -> None:
    from importlib import import_module
    remote_module = import_module("graphistry.compute.chain_remote")
    def reject_io(*args, **kwargs):
        pytest.fail("unsupported remote mutation reached HTTP")
    monkeypatch.setattr(remote_module.requests, "post", reject_io)
    with pytest.raises(GFQLValidationError) as exc:
        _graph("pandas").gfql_remote(
            "CREATE (a:L)", dataset_id="fixture", api_token="fixture-token",
            format="json", validate=validate,
        )
    assert exc.value.context["value"] == "mutation"


@pytest.mark.parametrize("engine", ENGINES)
def test_create_converts_pandas_input_to_the_explicit_native_engine(engine: str) -> None:
    if engine == "polars":
        pytest.importorskip("polars")

    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    result = source.gfql("CREATE (a:L)", engine=engine)
    assert len(result._nodes) == 3
    _assert_served(engine, result._nodes)
    _assert_served(engine, result._edges)
    pd.testing.assert_frame_equal(source._nodes, before)


@pytest.mark.parametrize("engine", ENGINES)
def test_wide_existing_integer_with_missing_new_property_remains_exact(engine: str) -> None:
    source = _graph(engine)
    wide = (1 << 60) + 1
    if engine == "polars":
        import polars as pl
        nodes = source._nodes.with_columns(pl.lit(wide).alias("serial"))
    else:
        nodes = source._nodes.assign(serial=wide)
    source = source.nodes(nodes)
    result = source.gfql("CREATE (a:L)", engine=engine)
    assert [row["serial"] for row in _records(result._nodes)] == [wide, wide, None]


@pytest.mark.parametrize("engine", ENGINES)
def test_new_float_does_not_round_existing_wide_integer(engine: str) -> None:
    source = _graph(engine)
    wide = (1 << 60) + 1
    if engine == "polars":
        import polars as pl
        nodes = source._nodes.with_columns(pl.lit(wide).alias("serial"))
    else:
        nodes = source._nodes.assign(serial=wide)
    source = source.nodes(nodes)
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (a {serial: 1.5})", engine=engine)
    assert exc.value.context["field"] == "serial"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_new_integer_does_not_truncate_existing_fractional_values(engine: str) -> None:
    source = _graph(engine)
    if engine == "polars":
        import polars as pl
        nodes = source._nodes.with_columns(pl.Series("score", [1.5, -2.25]))
    else:
        nodes = source._nodes.assign(score=[1.5, -2.25])
    source = source.nodes(nodes)
    before = _records(source._nodes)
    result = source.gfql("CREATE (a {score: 2})", engine=engine)
    assert [row["score"] for row in _records(result._nodes)] == [1.5, -2.25, 2]
    assert _records(source._nodes) == before
    _assert_served(engine, result._nodes)


def _foundation_frame(engine,data):
    value=pd.DataFrame(data)
    if engine=="cudf":
        import cudf
        return cudf.from_pandas(value)
    if engine=="polars":
        pl = pytest.importorskip("polars")
        return pl.from_pandas(value)
    return value

@pytest.mark.parametrize("engine",ENGINES)
@pytest.mark.parametrize("query,count,expected",[
    ("CREATE ()",1,None),
    ("CREATE (), ()",2,None),
    ("CREATE (:A:B), (:B:C)",2,None),
    ("CREATE (a {name:'foo'}) RETURN a.name AS p",1,[{"p":"foo"}]),
    ("CREATE (a {id:12,name:null}) RETURN a.id AS id,a.name AS name",1,[{"id":12,"name":None}]),
    ("CREATE (a {id:4611686018427387905}) RETURN a.id AS id",1,[{"id":4611686018427387905}]),
    ("CREATE (), () RETURN 1 AS x",2,[{"x":1}]),
    ("CREATE (a {name:'a'}),(b {name:'b'}) RETURN a.name AS a,b.name AS b",2,[{"a":"a","b":"b"}]),
])
def test_foundation(engine,query,count,expected):
    source=graphistry.nodes(_foundation_frame(engine,{"id":pd.Series([],dtype="int64")}),"id")
    before=_records(source._nodes)
    result=source.gfql(query,engine=engine)
    assert _records(source._nodes)==before
    graph=result.gfql_graph
    assert len(graph._nodes)==count
    assert graph._node!="id"
    assert type(graph._nodes).__module__.split(".")[0]==engine
    if expected is not None:
        assert _records(result._nodes)==expected
        assert result._gfql_rows_base_graph is None
        assert result._gfql_start_nodes is None
        assert graph.gfql_graph is graph
    else:
        assert result is graph

@pytest.mark.parametrize("engine",ENGINES)
def test_legacy_string_id(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old"]}),"id")
    result=source.gfql("CREATE (a {id:'new'}),(b) RETURN a.id AS explicit,b.id AS missing",engine=engine)
    assert _records(result._nodes)==[{"explicit":"new","missing":None}]
    graph=result.gfql_graph
    assert graph._node!="id" and len(graph._nodes)==3
    assert _records(source._nodes)==[{"id":"old"}]
    again=graph.gfql("CREATE ()",engine=engine)
    assert len(again._nodes)==4 and again._node==graph._node
    assert list(again._nodes.columns)==list(graph._nodes.columns)

@pytest.mark.parametrize("engine", ENGINES[:2])
@pytest.mark.parametrize("old",["old",10])
def test_current_id_membership(engine,old):
    explicit="new" if isinstance(old,str) else 20
    source=graphistry.nodes(_foundation_frame(engine,{"id":[old]}),"id")
    graph=source.gfql("CREATE (a {id:$id}),(b)",params={"id":explicit},engine=engine)
    result=graph.gfql("MATCH (n) RETURN n.id AS id, keys(n) AS keys, properties(n) AS props",engine=engine)
    got=_records(result._nodes)
    assert [row["id"] for row in got]==[old,explicit,None]
    assert [list(row["keys"]) for row in got]==[[] if isinstance(old,str) else ["id"],["id"],[]]
    assert got[0]["props"]==("{}" if isinstance(old,str) else "{id: 10}")
    assert got[1]["props"]==("{id: 'new'}" if isinstance(old,str) else "{id: 20}")
    assert got[2]["props"]=="{}"

@pytest.mark.parametrize("engine",ENGINES)
def test_whole_entity_and_lifecycle(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old"],"label__Old":[True]}),"id")
    result=source.gfql("CREATE (a:L {id:'new'}) RETURN a",engine=engine)
    assert set(result._nodes.columns)=={"a.id","a.label__Old","a.label__L"}
    assert _records(result._nodes)[0]["a.id"]=="new"
    graph=result.gfql_graph
    assert graph.gfql("MATCH (n) RETURN count(*) AS c",engine=engine)._nodes is not None
    count=_records(graph.gfql("MATCH (n) RETURN count(*) AS c",engine=engine)._nodes)
    assert count==[{"c":2}]
    # Result association survives cleanup; graph transformations describe themselves.
    assert result.gfql_graph is graph
    assert graph.nodes(graph._nodes).gfql_graph is not graph
    assert source.gfql_graph is source

@pytest.mark.parametrize("name",["__graphistry_node_identity__","__gfql_id_property_present__"])
def test_reserved_metadata_collision(name):
    source=graphistry.nodes(pd.DataFrame({"id":[1],name:[False]}),"id")
    before=source._nodes.copy(deep=True)
    from graphistry.compute.exceptions import GFQLValidationError
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE ()",engine="pandas")
    assert exc.value.context["field"] == "create"
    assert exc.value.context["value"] == [name]
    pd.testing.assert_frame_equal(source._nodes,before)


@pytest.mark.parametrize("engine",ENGINES)
def test_create_preserves_original_flag_precedence_over_legacy_type(engine):
    source=graphistry.nodes(_foundation_frame(engine,{
        "id":["old"],"label__Foo":[True],"type":["Business"],
    }),"id")
    before=_records(source._nodes)
    result=source.gfql("CREATE (n:B {id:'new'})",engine=engine)
    graph=result.gfql_graph
    old_rows=graph.gfql("MATCH (old) WHERE old.id='old' RETURN old",engine=engine)
    new_rows=graph.gfql("MATCH (new:B) RETURN new",engine=engine)
    assert _entity_text_records(old_rows,{"old":"nodes"})==[{"old":"(:Foo)"}]
    assert all(row["new"].startswith("(:B") for row in _entity_text_records(new_rows,{"new":"nodes"}))
    assert _records(source._nodes)==before


@pytest.mark.parametrize("engine", ENGINES)
def test_create_numeric_existing_identity_matches_fresh_label_oracle(engine):
    source=graphistry.nodes(_foundation_frame(engine,{
        "id":[1],"label__Foo":[True],"type":["Business"],
    }),"id")
    result=source.gfql("CREATE (n:B {id:2})",engine=engine)
    old_rows=result.gfql("MATCH (n) WHERE n.id=1 RETURN n",engine=engine)
    new_rows=result.gfql("MATCH (n:B) RETURN n",engine=engine)
    ids=result.gfql("MATCH (n) RETURN n.id AS id",engine=engine)
    assert _entity_text_records(old_rows,{"n":"nodes"})==[{"n":"(:Foo {id: 1})"}]
    assert _entity_text_records(new_rows,{"n":"nodes"})==[{"n":"(:B {id: 2})"}]
    assert [row["id"] for row in _records(ids._nodes)]==[1,2]


def test_create_rejects_reserved_existing_edge_identity_before_effects():
    source=graphistry.nodes(_foundation_frame("pandas",{"id":[1,2]}),"id")
    source=source.edges(pd.DataFrame({
        "src":[1],"dst":[2],"__graphistry_edge_identity__":[99],
    }),"src","dst")
    before_edges=source._edges.copy(deep=True)
    with pytest.raises(GFQLValidationError,match="reserved edge metadata columns collide"):
        source.gfql("CREATE (a)-[:REL]->(b)",engine="pandas")
    pd.testing.assert_frame_equal(source._edges,before_edges)


@pytest.mark.parametrize("engine",ENGINES[:2])
def test_keyless_existing_edge_backfill_is_native_signed_int64(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id")
    edges=graphistry.edges(_foundation_frame(engine,{
        "src":[1],"dst":[2],"weight":[1.5],
    }),"src","dst")
    source=source.edges(edges._edges,"src","dst")
    result=source.gfql("CREATE (a)-[:REL {weight:2.5}]->(b)",engine=engine)
    assert pd.api.types.is_signed_integer_dtype(
        result._edges["__graphistry_edge_identity__"].dtype
    )
    assert [row["__graphistry_edge_identity__"] for row in _records(result._edges)]==[-1,0]
    again=result.gfql("CREATE (a)-[:REL]->(b)",engine=engine)
    assert [row["__graphistry_edge_identity__"] for row in _records(again._edges)]==[-1,0,1]


@pytest.mark.parametrize("label",["None","nan"])
def test_create_declines_nullish_label_before_effects(label):
    source=graphistry.nodes(_foundation_frame("pandas",{"id":[1,2]}),"id")
    before=_records(source._nodes)
    with pytest.raises(GFQLValidationError,match="null-like label name"):
        source.gfql(f"CREATE (n:{label})",engine="pandas")
    assert _records(source._nodes)==before

@pytest.mark.parametrize("engine",ENGINES)
def test_all_null_old_property_does_not_stringify_new_integer(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1],"p":pd.Series([None],dtype="string")}),"id")
    before=_records(source._nodes)
    result=source.gfql("CREATE (n {p:12}) RETURN n.p AS p",engine=engine)
    assert _records(result._nodes)==[{"p":12}]
    assert _records(source._nodes)==before


@pytest.mark.parametrize("engine",ENGINES)
@pytest.mark.parametrize("expression",["__gfql_created_return__ AS leaked","*"])
def test_anonymous_scope_does_not_invent_a_public_binding(engine,expression):
    source=graphistry.nodes(_foundation_frame(engine,{"id":pd.Series([],dtype="int64")}),"id")
    with pytest.raises(GFQLValidationError):
        source.gfql("CREATE () RETURN "+expression,engine=engine)
    assert len(source._nodes)==0

@pytest.mark.parametrize("engine",ENGINES)
def test_replaced_ordinary_frame_is_readmitted(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id")
    created=source.gfql("CREATE ()",engine=engine)
    replaced=created.nodes(_foundation_frame(engine,{"id":[10,20]}),"id")
    before=_records(replaced._nodes)
    result=replaced.gfql("CREATE ()",engine=engine)
    assert len(result._nodes)==3
    assert _records(replaced._nodes)==before
    assert result._node!="id"
    assert [row[result._node] for row in _records(result._nodes)][:2] == [10, 20]

@pytest.mark.parametrize("engine",ENGINES)
def test_rebinding_metadata_is_typed_not_silently_reused(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id").gfql("CREATE ()",engine=engine)
    rebound=source.bind(node="id")
    before=_records(rebound._nodes)
    with pytest.raises(GFQLValidationError):
        rebound.gfql("CREATE ()",engine=engine)
    assert _records(rebound._nodes)==before


def test_functional_create_documentation_example() -> None:
    from pathlib import Path
    import textwrap
    document = (Path(__file__).resolve().parents[5] / "docs/source/gfql/cypher.rst").read_text()
    section = document.split("Functional Node Creation\n", 1)[1]
    lines = section.split(".. code-block:: python\n", 1)[1].splitlines()
    code = []
    for line in lines:
        if line and not line.startswith("   "):
            break
        code.append(line)
    exec(textwrap.dedent("\n".join(code)), {})


@pytest.mark.parametrize("engine", ENGINES)
def test_multi_alias_whole_entity_preserves_membership_without_helpers(engine):
    source = graphistry.nodes(_foundation_frame(engine, {"id": ["old"]}), "id")
    result = source.gfql("CREATE (a {id:'new'}), (b) RETURN a, b", engine=engine)
    assert set(result._nodes.columns) == {"a.id", "b.id"}
    assert _records(result._nodes) == [{"a.id": "new", "b.id": None}]
    assert len(result.gfql_graph._nodes) == 3
    assert _records(source._nodes) == [{"id": "old"}]


@pytest.mark.parametrize("engine",ENGINES)
@pytest.mark.parametrize("transform",["nodes","edges","bind"])
def test_unrelated_row_result_transforms_do_not_keep_saved_graph(engine,transform):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id")
    rows=source.gfql("CREATE (n {name:'new'}) RETURN n.name AS name",engine=engine)
    saved=rows.gfql_graph
    if transform=="nodes":
        changed=rows.nodes(_foundation_frame(engine,{"id":[10]}),"id")
        count=changed.gfql("MATCH (n) RETURN count(*) AS count",engine=engine)
        assert _records(count._nodes)==[{"count":1}]
        assert count.gfql_graph is count
    elif transform=="edges":
        changed=rows.edges(_foundation_frame(engine,{"src":[],"dst":[]}),"src","dst")
    else:
        changed=rows.bind(node="name")
    assert changed.gfql_graph is changed
    assert rows.gfql_graph is saved and len(saved._nodes)==3
    assert source.gfql_graph is source and len(source._nodes)==2

@pytest.mark.parametrize("engine",ENGINES)
def test_cleanup_and_view_bind_preserve_same_execution_association(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id")
    rows=source.gfql("CREATE (n {name:'new'}) RETURN n.name AS name",engine=engine)
    saved=rows.gfql_graph
    assert rows.bind().gfql_graph is saved
    from graphistry.compute.gfql.exec_context import attach_row_exec_context, clear_row_exec_context
    contextual=attach_row_exec_context(rows,rows_base_graph=source)
    clean=clear_row_exec_context(contextual)
    assert clean.gfql_graph is saved
    assert clean._gfql_rows_base_graph is None and clean._gfql_start_nodes is None


def test_set_compiles_to_node_identity_mutation() -> None:
    parsed = parse_cypher("MATCH (n:L) SET n.x = 1, n.y = $v RETURN n")
    compiled = compile_cypher_query(parsed, params={"v": "ok"})

    assert isinstance(compiled, CompiledCypherMutationQuery)
    assert compiled.creates == ()
    assert compiled.set_alias_kinds == (("n", "node"),)
    assert compiled.output_query is not None
    assert compiled.return_query is not None
    assert compiled.return_query.return_.items[0].alias == "__gfql_set_seed_identity__"
    assert compiled.return_query.return_.items[1].alias == "__gfql_set_identity_0__"
    assert [item.alias for item in compiled.return_query.return_.items[2:]] == [
        "__gfql_set_value_0__",
        "__gfql_set_value_1__",
    ]


def test_set_without_return_compiles_to_the_mutated_graph() -> None:
    parsed = parse_cypher("MATCH (n) SET n.x = 1")
    compiled = compile_cypher_query(parsed)

    assert isinstance(compiled, CompiledCypherMutationQuery)
    assert compiled.output_query is None
    assert compiled.return_query is not None


@pytest.mark.parametrize("query,target", [
    ("MATCH (a)-[r]->(b) SET r.w = 1 RETURN a", "r"),
    ("MATCH (n) SET m.x = 1 RETURN n", "m"),
    ("MATCH (n) SET n.id = 1 RETURN n", "id"),
    ("MATCH (n) SET n.labels = ['L'] RETURN n", "labels"),
    ("MATCH (n) SET n.type = 'L' RETURN n", "type"),
    ("MATCH (n) SET n.label__L = True RETURN n", "label__L"),
    ("MATCH (n) SET n.__secret__ = 1 RETURN n", "__secret__"),
])
def test_set_declines_unfaithful_or_unbound_targets(query: str, target: str) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher(query))
    if target == "r":
        target = "relationship_pattern"
    assert exc.value.context["value"] == target


@pytest.mark.parametrize("query,expected_field", [
    ("MATCH (n) CREATE (m:L) SET n.x = 1", "create"),
    ("MATCH (n) WHERE n.id = 1 MATCH (m) WHERE m.id = 2 SET n.score = 20", "set"),
    ("MATCH (n), (m) SET n.score = 20", "set"),
    ("MATCH (n)-[r]->(m) SET n.score = 20", "set"),
    ("MATCH (n) UNWIND [1] AS x SET n.x = x", "set"),
    ("MATCH (n) WITH n AS m SET m.x = 1 RETURN m", "set"),
])
def test_set_declines_unsupported_mixed_programs(query: str, expected_field: str) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher(query))
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == expected_field


def test_set_declines_optional_match() -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher("MATCH (n) OPTIONAL MATCH (m) SET n.x = 1"))
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "repeated_match"


@pytest.mark.parametrize("engine", ENGINES)
def test_set_updates_literal_expression_and_multiple_properties(engine: str) -> None:
    source = _graph(engine)
    result = source.gfql(
        "MATCH (n) WHERE n.id = 2 SET n.score = n.id * 10, n.name = 'two' RETURN n.id, n.score, n.name",
        engine=engine,
        language="cypher",
    )

    assert _records(result._nodes) == [{"n.id": 2, "n.score": 20, "n.name": "two"}]
    assert result.gfql_graph is not None
    assert len(result.gfql_graph._nodes) == 2
    updated_graph = {row["id"]: row["score"] for row in _records(result.gfql_graph._nodes)}
    assert updated_graph == {1: None, 2: 20}
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_set_without_return_preserves_caller(engine: str) -> None:
    source = _graph(engine)
    before = _records(source._nodes)
    result = source.gfql("MATCH (n) WHERE n.id = 2 SET n.score = 7", engine=engine)

    updated = {row["id"]: row["score"] for row in _records(result._nodes)}
    assert updated == {1: None, 2: 7}
    assert _records(source._nodes) == before
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_set_detects_conflicting_updates_for_the_same_property(engine: str) -> None:
    source = _graph(engine)
    before = _records(source._nodes)
    from graphistry.compute.exceptions import ErrorCode
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("MATCH (n) WHERE n.id = 2 SET n.x = 1 SET n.x = 2", engine=engine)

    assert exc.value.code == ErrorCode.E105
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "repeated_property"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_set_return_preserves_changed_predicate_binding(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": [1, 2], "score": [5, 100]}), "id"
    )
    before = _records(source._nodes)
    result = source.gfql(
        "MATCH (n) WHERE n.score < 10 SET n.score = 100 RETURN n.id, n.score",
        engine=engine,
        language="cypher",
    )
    assert _records(result._nodes) == [{"n.id": 1, "n.score": 100}]
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_set_return_preserves_original_predicate_binding(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": [1, 2], "name": ["Andres", "Andres"]}),
        "id",
    )
    result = source.gfql(
        "MATCH (n) WHERE n.name = 'Andres' SET n.name = 'Michael' RETURN n.id, n.name",
        engine=engine,
        language="cypher",
    )

    assert _records(result._nodes) == [
        {"n.id": 1, "n.name": "Michael"},
        {"n.id": 2, "n.name": "Michael"},
    ]
    assert result.gfql_graph is not None
    updated = {row["id"]: row["name"] for row in _records(result.gfql_graph._nodes)}
    assert updated == {1: "Michael", 2: "Michael"}
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_set_return_preserves_pattern_property_binding(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": [1, 2], "name": ["Andres", "Taylor"]}),
        "id",
    )
    result = source.gfql(
        "MATCH (n {name: 'Andres'}) SET n.name = 'Michael' RETURN n.id, n.name",
        engine=engine,
        language="cypher",
    )

    assert _records(result._nodes) == [{"n.id": 1, "n.name": "Michael"}]
    updated = {row["id"]: row["name"] for row in _records(result.gfql_graph._nodes)}
    assert updated == {1: "Michael", 2: "Taylor"}
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_set_preserves_internal_marker_columns(engine: str) -> None:
    marker_frame = _foundation_frame(
        engine,
        {
            "id": [1, 2],
            "score": [5, 100],
            "__gfql_set_new_score__": ["new", "old"],
            "__gfql_set_matched_score__": [False, True],
        },
    )
    source = graphistry.nodes(marker_frame, "id")
    before = _records(source._nodes)
    result = source.gfql(
        "MATCH (n) WHERE n.score = 5 SET n.score = 10 RETURN n.id, n.score",
        engine=engine,
        language="cypher",
    )

    assert _records(result._nodes) == [{"n.id": 1, "n.score": 10}]
    updated = {
        row["id"]: row for row in _records(result.gfql_graph._nodes)
    }
    assert updated[1]["score"] == 10
    assert updated[1]["__gfql_set_new_score__"] == "new"
    assert updated[1]["__gfql_set_matched_score__"] is False
    assert updated[2]["__gfql_set_new_score__"] == "old"
    assert updated[2]["__gfql_set_matched_score__"] is True
    assert _records(source._nodes) == before


@pytest.mark.parametrize("value",[[1],{"a":1},float("nan")],ids=["list","map","nan"])
@pytest.mark.parametrize("mode",["validate_only","execute","execute_validate"])
def test_known_unsupported_property_parameters_decline_consistently(value,mode):
    source=graphistry.nodes(_foundation_frame("pandas",{"id":[1,2]}),"id")
    before=_records(source._nodes)
    query="CREATE (n {p:$value})"
    with pytest.raises(GFQLValidationError) as exc:
        if mode=="validate_only":
            source.gfql_validate(query,params={"value":value})
        else:
            source.gfql(query,params={"value":value},validate=mode=="execute_validate",engine="pandas")
    from graphistry.compute.exceptions import ErrorCode
    assert exc.value.code == ErrorCode.E108
    assert _records(source._nodes)==before

@pytest.mark.parametrize("engine",ENGINES)
def test_empty_return_retains_updated_graph_same_execution(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id")
    rows=source.gfql("CREATE (a {name:'new'}) RETURN a.name AS name SKIP 1",engine=engine)
    assert _records(rows._nodes)==[]
    graph=rows.gfql_graph
    assert len(graph._nodes)==3 and graph.gfql_graph is graph
    assert len(graph.gfql("CREATE ()",engine=engine)._nodes)==4
    assert _records(source._nodes)==[{"id":1},{"id":2}]


@pytest.mark.parametrize("engine",ENGINES)
def test_create_declines_old_legacy_type_with_new_canonical_label_and_source_retained(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old"],"type":["Old"]}),"id")
    with pytest.raises(GFQLValidationError,match="legacy type labels"):
        source.gfql("CREATE (:New)",engine=engine)
    assert _records(source._nodes)==[{"id":"old","type":"Old"}]


@pytest.mark.parametrize("engine",ENGINES)
def test_create_declines_same_legacy_and_canonical_label_and_source_retained(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old"],"type":["Shared"]}),"id")
    with pytest.raises(GFQLValidationError,match="legacy type labels"):
        source.gfql("CREATE (:Shared)",engine=engine)
    assert _records(source._nodes)==[{"id":"old","type":"Shared"}]


@pytest.mark.parametrize("engine",ENGINES)
def test_create_declines_any_nonnull_legacy_type_with_new_canonical_label_and_source_retained(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old","new"],"type":["Old",None]}),"id")
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (:New)",engine=engine)
    from graphistry.compute.exceptions import ErrorCode
    assert exc.value.code==ErrorCode.E108
    assert exc.value.context["field"]=="create"
    assert exc.value.context["value"]==["New"]
    assert _records(source._nodes)==[{"id":"old","type":"Old"},{"id":"new","type":None}]


def test_parse_declines_standalone_create_combined_with_set():
    with pytest.raises(GFQLValidationError) as exc:
        parse_cypher("CREATE () SET n.x = 1")
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "create_with_set"


@pytest.mark.parametrize("engine",ENGINES)
def test_create_admits_all_null_legacy_type_with_new_canonical_label(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old"],"type":[None]}),"id")
    result=source.gfql("CREATE (:New)",engine=engine)
    assert len(result._nodes)==2
    assert [row["label__New"] for row in _records(result._nodes)]==[False,True]
    assert _records(source._nodes)==[{"id":"old","type":None}]


WIDE_INT = 1152921504606846977  # 2**60 + 1: not representable as float64


def _column_kind(frame: Any, column: str) -> str:
    if isinstance(frame, pd.DataFrame):
        dtype = frame[column].dtype
        if pd.api.types.is_bool_dtype(dtype):
            return "bool"
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "float"
        return "string" if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype) else "other"
    if hasattr(frame, "schema"):
        dtype = frame.schema[column]
        if dtype.is_integer():
            return "integer"
        if dtype.is_float():
            return "float"
        if str(dtype) == "Boolean":
            return "bool"
        return "string" if str(dtype) == "String" else "other"
    return {"i": "integer", "u": "integer", "f": "float", "b": "bool"}.get(frame[column].dtype.kind, "string")


def _set_values(result: Any, column: str = "score") -> dict[Any, Any]:
    return {row["id"]: row[column] for row in _records(result._nodes)}


@pytest.mark.parametrize("engine", ENGINES)
def test_set_partial_new_wide_integer_keeps_exact_integer(engine: str) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2]}), "id")
    before = _records(source._nodes)
    result = source.gfql(f"MATCH (n) WHERE n.id = 1 SET n.score = {WIDE_INT}", engine=engine)

    assert _set_values(result) == {1: WIDE_INT, 2: None}
    assert _column_kind(result._nodes, "score") == "integer"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_set_partial_existing_integer_column_keeps_integer_and_wide_value(engine: str) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "score": [5, 6]}), "id")
    result = source.gfql(f"MATCH (n) WHERE n.id = 1 SET n.score = {WIDE_INT}", engine=engine)

    assert _set_values(result) == {1: WIDE_INT, 2: 6}
    assert _column_kind(result._nodes, "score") == "integer"
    narrow = source.gfql("MATCH (n) WHERE n.id = 1 SET n.score = 7", engine=engine)
    assert _set_values(narrow) == {1: 7, 2: 6}
    assert _column_kind(narrow._nodes, "score") == "integer"
    if engine == "pandas":
        assert str(narrow._nodes["score"].dtype) == "int64"


@pytest.mark.parametrize("engine", ENGINES)
def test_set_fully_matched_wide_integer_control(engine: str) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2]}), "id")
    result = source.gfql(f"MATCH (n) SET n.score = {WIDE_INT}", engine=engine)

    assert _set_values(result) == {1: WIDE_INT, 2: WIDE_INT}
    assert _column_kind(result._nodes, "score") == "integer"
    if engine == "pandas":
        assert str(result._nodes["score"].dtype) == "int64"


@pytest.mark.parametrize("engine", ENGINES)
def test_set_string_column_rejects_integer_before_effects(engine: str) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "score": ["old", "keep"]}), "id")
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("MATCH (n) WHERE n.id = 1 SET n.score = 20", engine=engine)

    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "score"
    assert exc.value.context["value"] == ["integer"]
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_set_all_null_string_column_promotes_to_assigned_integer(engine: str) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": [1, 2], "score": pd.Series([None, None], dtype="string")}), "id"
    )
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.score = 20", engine=engine)

    assert _set_values(result) == {1: 20, 2: None}
    assert _column_kind(result._nodes, "score") == "integer"
    assert _records(source._nodes) == [{"id": 1, "score": None}, {"id": 2, "score": None}]


@pytest.mark.parametrize("engine", ENGINES)
def test_set_float_column_rejects_wide_integer_before_effects(engine: str) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "score": [1.5, 2.25]}), "id")
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql(f"MATCH (n) WHERE n.id = 1 SET n.score = {WIDE_INT}", engine=engine)

    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "score"
    assert exc.value.context["value"] == "incompatible_numeric_types"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_set_float_column_accepts_exactly_representable_integer(engine: str) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "score": [1.5, 2.25]}), "id")
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.score = 20", engine=engine)

    assert _set_values(result) == {1: 20.0, 2: 2.25}
    assert _column_kind(result._nodes, "score") == "float"


def _label_n_graph(engine: str, nums: list[int]) -> Any:
    ids = list(range(1, len(nums) + 1))
    return graphistry.nodes(
        _foundation_frame(engine, {"id": ids, "label__N": [True] * len(nums), "num": nums}), "id"
    )


def _post_graph_nums(result: Any) -> list[Any]:
    return [row["num"] for row in sorted(_records(result.gfql_graph._nodes), key=lambda row: row["id"])]


# openCypher TCK Set6 (59edf2): SKIP/LIMIT/aggregation shape the returned rows only; every matched node is still set.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("tail,nums,expected_rows", [
    ("RETURN n LIMIT 0", [42], 0),
    ("RETURN n SKIP 1", [42], 0),
])
def test_set6_pagination_empties_rows_but_still_sets_every_match(engine, tail, nums, expected_rows) -> None:
    source = _label_n_graph(engine, nums)
    before = _records(source._nodes)
    result = source.gfql(f"MATCH (n:N) SET n.num = 43 {tail}", engine=engine)

    assert len(result._nodes) == expected_rows
    assert _post_graph_nums(result) == [43]
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("tail,expected_rows", [
    ("SKIP 2 LIMIT 2", [{"num": 42}, {"num": 42}]),
    ("SKIP 0 LIMIT 5", [{"num": 42}] * 5),
])
def test_set6_skip_limit_rows_while_all_five_nodes_are_set(engine, tail, expected_rows) -> None:
    source = _label_n_graph(engine, [1, 2, 3, 4, 5])
    result = source.gfql(f"MATCH (n:N) SET n.num = 42 RETURN n.num AS num {tail}", engine=engine)

    assert _records(result._nodes) == expected_rows
    assert _post_graph_nums(result) == [42] * 5


@pytest.mark.parametrize("engine", ENGINES)
def test_set6_aggregate_return_sums_the_updated_values(engine) -> None:
    source = _label_n_graph(engine, [1, 2, 3, 4, 5])
    before = _records(source._nodes)
    result = source.gfql("MATCH (n:N) SET n.num = n.num + 1 RETURN sum(n.num) AS sum", engine=engine)

    assert _records(result._nodes) == [{"sum": 20}]
    assert _post_graph_nums(result) == [2, 3, 4, 5, 6]
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_set_order_by_limit_ranks_the_updated_values(engine) -> None:
    source = _label_n_graph(engine, [1, 2, 3])
    result = source.gfql(
        "MATCH (n:N) SET n.num = 10 - n.num RETURN n.num AS num ORDER BY num DESC LIMIT 2", engine=engine
    )

    assert _records(result._nodes) == [{"num": 9}, {"num": 8}]
    assert _post_graph_nums(result) == [9, 8, 7]


@pytest.mark.parametrize("engine", ENGINES)
def test_set_aggregate_over_no_matches_returns_zero_and_leaves_graph(engine) -> None:
    source = _label_n_graph(engine, [1, 2])
    before = _records(source._nodes)
    result = source.gfql("MATCH (n:Missing) SET n.num = 9 RETURN count(n) AS c", engine=engine)

    assert _records(result._nodes) == [{"c": 0}]
    assert _records(result.gfql_graph._nodes) == before


# openCypher TCK Set2: SET p = null removes the property from the matched node (not before the SET reads it).
@pytest.mark.parametrize("engine", ENGINES)
def test_set2_null_assignment_removes_only_that_property(engine) -> None:
    source = graphistry.nodes(
        _foundation_frame(engine, {"id": [1], "label__A": [True], "property1": [23], "property2": [46]}), "id"
    )
    before = _records(source._nodes)
    result = source.gfql("MATCH (n:A) SET n.property1 = null RETURN n.property2 AS p2", engine=engine)

    assert _records(result._nodes) == [{"p2": 46}]
    assert _records(result.gfql_graph._nodes) == [
        {"id": 1, "label__A": True, "property1": None, "property2": 46}
    ]
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_set2_null_assignment_reads_the_old_value_before_removing_it(engine) -> None:
    source = graphistry.nodes(
        _foundation_frame(
            engine, {"id": [1, 2], "label__A": [True, True], "name": ["Michael", "Other"], "age": [35, 40]}
        ), "id"
    )
    result = source.gfql("MATCH (n) WHERE n.name = 'Michael' SET n.name = null RETURN n.age AS age", engine=engine)

    assert _records(result._nodes) == [{"age": 35}]
    names = {row["id"]: row["name"] for row in _records(result.gfql_graph._nodes)}
    assert names == {1: None, 2: "Other"}


@pytest.mark.parametrize("engine", ENGINES)
def test_set_null_assignment_keeps_integer_dtype_and_ignores_absent_property(engine) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "score": [5, 6]}), "id")
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.score = null", engine=engine)
    assert _set_values(result) == {1: None, 2: 6}
    assert _column_kind(result._nodes, "score") == "integer"

    absent = source.gfql("MATCH (n) SET n.missing = null", engine=engine)
    assert "missing" not in list(absent._nodes.columns)
    assert _records(absent._nodes) == _records(source._nodes)


# Static CREATE admission: every entry point rejects what execution rejects, with the same code and field.
_STATIC_CREATE_REJECTS = [
    ("CREATE (:`nan`)", ErrorCode.E108, "create"),
    ("CREATE (:`None`)", ErrorCode.E108, "create"),
    ("CREATE (a)-[:R]->(:`<NA>`)", ErrorCode.E108, "create"),
    ("CREATE (n {__x__: 1})", ErrorCode.E108, "__x__"),
    ("CREATE (n {labels: 1})", ErrorCode.E108, "labels"),
    ("CREATE (n {type: 1})", ErrorCode.E108, "type"),
    ("CREATE (n {label__X: 1})", ErrorCode.E108, "label__X"),
    ("CREATE (n {p: 1, p: 2})", ErrorCode.E105, "p"),
    ("CREATE (a)-[:R {type: 1}]->(b)", ErrorCode.E108, "type"),
    ("CREATE (a)-[:R {__x__: 1}]->(b)", ErrorCode.E108, "__x__"),
    ("CREATE (a)-[:R {p: 1, p: 2}]->(b)", ErrorCode.E105, "p"),
]


@pytest.mark.parametrize("query,code,field", _STATIC_CREATE_REJECTS)
def test_create_static_preflight_matches_execution_on_every_entry_point(query, code, field) -> None:
    source = _graph("pandas")
    for entry in (
        lambda: compile_cypher_query(parse_cypher(query)),
        lambda: source.gfql_validate(query),
        lambda: source.gfql(query, engine="pandas", validate=True),
        lambda: source.gfql(query, engine="pandas", validate=False),
    ):
        with pytest.raises(GFQLValidationError) as exc:
            entry()
        assert exc.value.code == code
        assert exc.value.context["field"] == field


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,code,field", _STATIC_CREATE_REJECTS[:3] + _STATIC_CREATE_REJECTS[7:8])
def test_create_static_rejects_leave_every_engine_untouched(engine, query, code, field) -> None:
    source = _graph(engine)
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql(query, engine=engine)
    assert exc.value.code == code
    assert _records(source._nodes) == before


@pytest.mark.parametrize("query", [
    "CREATE (n {id: 7})",
    "CREATE (:none)",
    "CREATE (n:Person {name: 'a', other: 1})",
    "CREATE (a {p: 1})-[:R {p: 2}]->(b {p: 3})",
])
def test_create_static_preflight_admits_ordinary_keys_and_labels(query) -> None:
    source = _graph("pandas")
    assert source.gfql_validate(query)["compiled_kind"] == "mutation"
    assert compile_cypher_query(parse_cypher(query)) is not None


def test_set_polars_null_dtype_column_promotes_to_assigned_integer() -> None:
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"id": [1, 2], "score": pl.Series("score", [None, None], dtype=pl.Null)})
    source = graphistry.nodes(frame, "id")
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.score = 20", engine="polars")

    assert _set_values(result) == {1: 20, 2: None}
    assert _column_kind(result._nodes, "score") == "integer"
    assert source._nodes.schema["score"] == pl.Null


@pytest.mark.parametrize("engine", ENGINES)
def test_set_partial_new_boolean_property_stays_boolean_with_nulls(engine: str) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2]}), "id")
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.flag = true", engine=engine)

    assert _set_values(result, "flag") == {1: True, 2: None}
    assert _column_kind(result._nodes, "flag") == "bool"


@pytest.mark.parametrize("engine", ENGINES)
def test_set_partial_existing_boolean_column_keeps_boolean_dtype(engine: str) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "flag": [False, False]}), "id")
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.flag = true", engine=engine)

    assert _set_values(result, "flag") == {1: True, 2: False}
    assert _column_kind(result._nodes, "flag") == "bool"
    if engine == "pandas":
        assert str(result._nodes["flag"].dtype) == "bool"


@pytest.mark.parametrize("engine", ENGINES)
def test_set_boolean_column_rejects_integer_before_effects(engine: str) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "flag": [True, False]}), "id")
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("MATCH (n) WHERE n.id = 1 SET n.flag = 1", engine=engine)

    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["value"] == ["integer"]
    assert _records(source._nodes) == before


# Polars CREATE on a graph with no nodes must fill absent label flags with False (as pandas does), not leave nulls.
@pytest.mark.parametrize("query,expected", [
    ("CREATE (:A), ()", [True, False]),
    ("CREATE (:A)-[:R]->()", [True, False]),
])
def test_polars_create_on_empty_graph_fills_label_flags_like_pandas(query, expected) -> None:
    pl = pytest.importorskip("polars")
    pandas_result = graphistry.bind().gfql(query, engine="pandas")
    polars_result = graphistry.bind().gfql(query, engine="polars")

    assert polars_result._nodes.schema["label__A"] == pl.Boolean
    assert polars_result._nodes["label__A"].to_list() == expected
    assert pandas_result._nodes["label__A"].tolist() == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_create_relationship_with_empty_property_maps_and_labelled_endpoints(engine) -> None:
    source = _graph(engine)
    result = source.gfql("CREATE (a:L {})-[:R {}]->(b)", engine=engine)

    assert len(result._nodes) == len(source._nodes) + 2
    assert len(result._edges) == len(source._edges) + 1
    assert _records(source._nodes) == _records(_graph(engine)._nodes)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", ["CREATE (a)-[:R]->(:`nan`)", "CREATE (:`None`)-[:R]->(b)"])
def test_create_declines_null_like_label_on_relationship_endpoint(engine, query) -> None:
    source = _graph(engine)
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError, match="null-like label name"):
        source.gfql(query, engine=engine)
    assert _records(source._nodes) == before


# Wave-1 W1-F1: every SET right-hand side is evaluated on the pre-SET graph, so a read of a property assigned earlier in
# the same query would silently see the old value. Such programs are declined before any effect.
_SET_READ_AFTER_WRITE = [
    "MATCH (n:N) SET n.num = n.num + 1 SET n.num = n.num + 1 RETURN n.num AS num",
    "MATCH (n:N) SET n.a = 1 SET n.b = n.a RETURN n.b AS b",
    "MATCH (n:N) SET n.a = 1, n.b = n.a RETURN n.b AS b",
    "MATCH (n:N) SET n.x = n.y, n.y = n.x RETURN n.x AS x",
]


@pytest.mark.parametrize("query", _SET_READ_AFTER_WRITE)
def test_set_declines_reads_of_properties_assigned_earlier_in_the_query(query) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher(query))
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "read_after_write"


@pytest.mark.parametrize("engine", ENGINES)
def test_set_read_after_write_leaves_the_caller_untouched_on_every_engine(engine) -> None:
    source = _label_n_graph(engine, [1])
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql(_SET_READ_AFTER_WRITE[0], engine=engine)
    assert exc.value.context["value"] == "read_after_write"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_set_independent_multi_clause_programs_still_run(engine) -> None:
    source = _label_n_graph(engine, [1, 2])
    result = source.gfql(
        "MATCH (n:N) SET n.a = n.num + 1 SET n.b = n.num * 10 RETURN n.a AS a, n.b AS b ORDER BY b", engine=engine
    )
    assert _records(result._nodes) == [{"a": 2, "b": 10}, {"a": 3, "b": 20}]
    graph = {row["id"]: (row["a"], row["b"]) for row in _records(result.gfql_graph._nodes)}
    assert graph == {1: (2, 10), 2: (3, 20)}


# Wave-1 W1-F2: validation must understand SET queries (it only knew CREATE).
def test_gfql_validate_accepts_admitted_set_and_declines_unsupported_set() -> None:
    source = _graph("pandas")
    report = source.gfql_validate("MATCH (n) WHERE n.id = 1 SET n.x = 1")
    assert report["ok"] is True and report["compiled_kind"] == "mutation"
    for query, value in [
        ("MATCH (n)-[r]->(m) SET n.x = 1", "relationship_pattern"),
        ("MATCH (n) SET n.x = 1 SET n.y = n.x", "read_after_write"),
    ]:
        with pytest.raises(GFQLValidationError) as exc:
            source.gfql_validate(query)
        assert exc.value.code == ErrorCode.E108
        assert exc.value.context["field"] == "set" and exc.value.context["value"] == value


@pytest.mark.parametrize("engine", ENGINES)
def test_gfql_validate_true_runs_admitted_set_and_declines_unsupported_set(engine) -> None:
    source = _graph(engine)
    result = source.gfql("MATCH (n) WHERE n.id = 2 SET n.x = 7", engine=engine, validate=True)
    assert {row["id"]: row["x"] for row in _records(result._nodes)} == {1: None, 2: 7}
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("MATCH (n) SET n.x = 1 SET n.y = n.x", engine=engine, validate=True)
    assert exc.value.context["value"] == "read_after_write"
    assert _records(source._nodes) == before


# Wave-1 W1-F3: type resolution must not depend on the match count; an empty match is a no-op on every engine.
@pytest.mark.parametrize("engine", ENGINES)
def test_set_with_no_matches_is_a_no_op_on_every_engine(engine) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "label__N": [True, True], "score": [5, 6]}), "id")
    before = _records(source._nodes)
    for value in ["'text'", "1", "1.5"]:
        result = source.gfql(f"MATCH (n:Missing) SET n.score = {value}", engine=engine)
        assert _records(result._nodes) == before
        assert _column_kind(result._nodes, "score") == "integer"
    absent = source.gfql("MATCH (n:Missing) SET n.fresh = 1", engine=engine)
    assert "fresh" not in list(absent._nodes.columns)


# Wave-1 W1-F11: unsigned and extension dtypes keep their storage or raise; they never silently change.
@pytest.mark.parametrize("engine", ENGINES)
def test_set_unsigned_integer_column_keeps_dtype_or_raises_before_effects(engine) -> None:
    frame = pd.DataFrame({"id": [1, 2], "u": pd.Series([1, 2], dtype="uint8")})
    if engine == "cudf":
        import cudf
        frame = cudf.from_pandas(frame)
    elif engine == "polars":
        frame = pytest.importorskip("polars").from_pandas(frame)
    source = graphistry.nodes(frame, "id")
    before = _records(source._nodes)
    ok = source.gfql("MATCH (n) WHERE n.id = 1 SET n.u = 5", engine=engine)
    assert _set_values(ok, "u") == {1: 5, 2: 2}
    assert "8" in str(ok._nodes["u"].dtype if engine != "polars" else ok._nodes.schema["u"])
    for value in ["300", "-1"]:
        with pytest.raises(GFQLValidationError) as exc:
            source.gfql(f"MATCH (n) WHERE n.id = 1 SET n.u = {value}", engine=engine)
        assert exc.value.code == ErrorCode.E108
        assert exc.value.context["value"] == "integer_out_of_range"
        assert _records(source._nodes) == before


def test_set_pandas_nullable_integer_column_stays_nullable() -> None:
    source = graphistry.nodes(pd.DataFrame({"id": [1, 2], "s": pd.Series([1, None], dtype="Int64")}), "id")
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.s = 7", engine="pandas")
    assert str(result._nodes["s"].dtype) == "Int64"
    assert _set_values(result, "s") == {1: 7, 2: None}


# Wave-1 W1-F5: NaN collapses to null (property removal) identically on every engine; Polars and cuDF keep NaN as a value.
def _nan_frame(engine: str, data: dict[str, list[Any]]) -> Any:
    if engine == "polars":
        return pytest.importorskip("polars").DataFrame(data)
    if engine == "cudf":
        import cudf
        return cudf.DataFrame({key: cudf.Series(values, nan_as_null=False) for key, values in data.items()})
    return pd.DataFrame(data)


@pytest.mark.parametrize("engine", ENGINES)
def test_set_nan_value_removes_the_property_identically_on_every_engine(engine) -> None:
    nan = float("nan")
    source = graphistry.nodes(
        _nan_frame(engine, {"id": [1, 2], "other": [nan, 2.5], "score": [1.0, 1.0]}), "id"
    )
    result = source.gfql("MATCH (n) SET n.score = n.other", engine=engine)
    assert _set_values(result) == {1: None, 2: 2.5}
    assert _column_kind(result._nodes, "score") == "float"
    new = source.gfql("MATCH (n) SET n.fresh = n.other", engine=engine)
    assert _set_values(new, "fresh") == {1: None, 2: 2.5}
    only_nan = source.gfql("MATCH (n) WHERE n.id = 1 SET n.fresh2 = n.other", engine=engine)
    assert "fresh2" not in list(only_nan._nodes.columns)



# Wave-2 W2-F1: the read-after-write check must analyze every right-hand side with the execution parser and fail
# closed; it may never skip an expression it cannot parse.
_W2_READS = [
    ("MATCH (n:N) SET n.a = $v, n.b = n.a + 1 RETURN n.b AS b", {"v": 5}),
    ("MATCH (n:N) SET n.a = 5 SET n.r = 1 < n.a < 10 RETURN n.r AS r", None),
    ("MATCH (n:N) SET n.a = $v SET n.b = $v + n.a RETURN n.b AS b", {"v": 1}),
]


@pytest.mark.parametrize("query,params", _W2_READS)
def test_set_read_after_write_covers_params_and_chained_comparisons(query, params) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher(query), params=params)
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "read_after_write"


@pytest.mark.parametrize("engine", ENGINES)
def test_set_read_after_write_with_params_leaves_the_caller_untouched(engine) -> None:
    source = _label_n_graph(engine, [1])
    before = _records(source._nodes)
    query, params = _W2_READS[0]
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql(query, engine=engine, params=params)
    assert exc.value.context["value"] == "read_after_write"
    assert _records(source._nodes) == before


def test_set_declines_an_expression_it_cannot_analyze_instead_of_skipping_it(monkeypatch) -> None:
    from graphistry.compute.gfql.cypher import lowering

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise GFQLValidationError(ErrorCode.E108, "outside the supported subset", field="set")

    monkeypatch.setattr(lowering, "_parse_row_expr", refuse)
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher("MATCH (n:N) SET n.a = 1 SET n.b = 2"))
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "unanalyzable_value"


# Parameterized SET executes (a raw KeyError at a04811f5c) with exact values on every engine, and bad parameters
# fail before any effect, at validation as well as at execution.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value,kind,expected", [
    (5, "integer", 5),
    (WIDE_INT, "integer", WIDE_INT),
    ("x", "string", "x"),
])
def test_set_parameter_values_execute_with_exact_values(engine, value, kind, expected) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2]}), "id")
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.p = $v", engine=engine, params={"v": value})
    assert _set_values(result, "p") == {1: expected, 2: None}
    assert _column_kind(result._nodes, "p") == kind


@pytest.mark.parametrize("engine", ENGINES)
def test_set_null_parameter_removes_the_property(engine) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "p": [7, 8]}), "id")
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.p = $v", engine=engine, params={"v": None})
    assert _set_values(result, "p") == {1: None, 2: 8}


@pytest.mark.parametrize("engine", ENGINES)
def test_set_parameter_type_mismatch_fails_before_effects(engine) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "p": [7, 8]}), "id")
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("MATCH (n) SET n.p = $v", engine=engine, params={"v": "text"})
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["value"] == ["string"]
    assert _records(source._nodes) == before


@pytest.mark.parametrize("params,code,value", [
    ({"v": [1, 2]}, ErrorCode.E108, "non_scalar_value"),
    ({"v": {"a": 1}}, ErrorCode.E108, "non_scalar_value"),
    ({"v": float("nan")}, ErrorCode.E108, "nan_property"),
    ({}, ErrorCode.E105, "v"),
])
def test_set_bad_parameters_fail_at_compile_validate_and_execute(params, code, value) -> None:
    source = _graph("pandas")
    query = "MATCH (n) SET n.p = $v"
    for entry in (
        lambda: compile_cypher_query(parse_cypher(query), params=params),
        lambda: source.gfql_validate(query, params=params),
        lambda: source.gfql(query, engine="pandas", params=params, validate=True),
        lambda: source.gfql(query, engine="pandas", params=params),
    ):
        with pytest.raises(GFQLValidationError) as exc:
            entry()
        assert exc.value.code == code
        assert exc.value.context["field"] == "set"
        assert exc.value.context["value"] == value


# Wave-2 W2-F2: SET must follow the MATCH that binds its alias.
def test_set_before_match_is_declined() -> None:
    with pytest.raises(GFQLValidationError) as exc:
        parse_cypher("SET n.x = 1 MATCH (n:Admin) RETURN n")
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "set_before_match"


# Wave-2 W2-F3: an empty match still normalizes the frame engine.
def test_set_with_no_matches_still_normalizes_to_the_requested_engine() -> None:
    pl = pytest.importorskip("polars")
    source = graphistry.nodes(pd.DataFrame({"id": [1, 2], "label__N": [True, True]}), "id")
    result = source.gfql("MATCH (n:Missing) SET n.x = 1", engine="polars")
    assert isinstance(result._nodes, pl.DataFrame)


# Wave-2 W2-F4 / W2-F7: static declines with one code on every engine.
@pytest.mark.parametrize("query,value", [
    ("MATCH (n) SET n.b = n", "whole_entity_read"),
    ("MATCH (n) SET n.p = 1, n.p = 'x'", "repeated_property"),
    ("MATCH (n) SET n.p = 1 SET n.p = 1", "repeated_property"),
])
def test_set_whole_entity_value_and_repeated_targets_are_declined_at_compile(query, value) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher(query))
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == value


# Wave-2 W2-F6: no FutureWarning from the matched-row mask on pandas.
def test_set_partial_update_emits_no_future_warning_on_pandas() -> None:
    import warnings
    source = graphistry.nodes(pd.DataFrame({"id": [1, 2], "s": ["a", "b"]}), "id")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.s = 'z'", engine="pandas")
    assert _set_values(result, "s") == {1: "z", 2: "b"}



# Wave-3 W3-F1: the alias may appear in a SET right-hand side only as a direct property access; any other occurrence is
# a whole-entity read (allowlist, not a denylist of read shapes) and is declined before any effect.
_W3_WHOLE_ENTITY = [
    "MATCH (n:N) SET n.a = 5, n.b = head([n]).a",
    "MATCH (n:N) SET n.b = head([n])",
    "MATCH (n:N) SET n.b = coalesce(n)",
    "MATCH (n:N) SET n.b = [n][0].a",
    "MATCH (n:N) SET n.b = [x IN [n] | x.a][0]",
    "MATCH (n:N) SET n.a = 1 SET n.b = n['a']",
]


@pytest.mark.parametrize("query", _W3_WHOLE_ENTITY)
def test_set_declines_every_whole_entity_read_of_the_alias(query) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher(query))
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "whole_entity_read"
    with pytest.raises(GFQLValidationError) as exc:
        _graph("pandas").gfql_validate(query)
    assert exc.value.context["value"] == "whole_entity_read"


@pytest.mark.parametrize("engine", ENGINES)
def test_set_whole_entity_read_leaves_the_caller_untouched_on_every_engine(engine) -> None:
    source = _label_n_graph(engine, [1])
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql(_W3_WHOLE_ENTITY[0], engine=engine)
    assert exc.value.context["value"] == "whole_entity_read"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("query", [
    "MATCH (n:N) SET n.num = n.num + 1",
    "MATCH (n:N) SET n.b = n.num + 1",
    "MATCH (n:N) SET n.b = n.num * 2, n.c = n.num - 1",
])
def test_set_admits_direct_property_reads_of_unassigned_or_self_properties(query) -> None:
    assert compile_cypher_query(parse_cypher(query)) is not None


# Wave-3 W3-F3: SET followed by a WITH re-entry is an unsupported shape (E108), not a syntax error; only a SET before
# the FIRST MATCH is a syntax error.
def test_set_then_with_reentry_is_unsupported_not_a_syntax_error() -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher("MATCH (a) SET a.x = 1 WITH a MATCH (b) RETURN b"))
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "mixed_clauses"


# Wave-3 W3-F4: parameter rendering edge cases.
@pytest.mark.parametrize("engine", ENGINES)
def test_set_negative_parameter_after_a_binary_minus_executes(engine) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "a": [10, 20]}), "id")
    result = source.gfql("MATCH (n) SET n.b = n.a - $v", engine=engine, params={"v": -5})
    assert _set_values(result, "b") == {1: 15, 2: 25}


@pytest.mark.parametrize("engine", ENGINES)
def test_set_string_parameter_that_spells_nan_is_a_string(engine) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2]}), "id")
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.p = $v", engine=engine, params={"v": "NaN"})
    assert _set_values(result, "p") == {1: "NaN", 2: None}


def test_set_nan_scan_ignores_parameter_names_inside_string_literals() -> None:
    compiled = compile_cypher_query(parse_cypher("MATCH (n) SET n.s = 'cost $v'"), params={"v": float("nan")})
    assert compiled is not None


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_set_infinite_parameter_is_declined_precisely(value) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher("MATCH (n) SET n.p = $v"), params={"v": value})
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "non_finite_parameter"


# Wave-3 W3-F6: more behavior coverage.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value,kind,expected", [(True, "bool", True), (2.5, "float", 2.5)])
def test_set_bool_and_float_parameters_execute(engine, value, kind, expected) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2]}), "id")
    result = source.gfql("MATCH (n) WHERE n.id = 1 SET n.p = $v", engine=engine, params={"v": value})
    assert _set_values(result, "p") == {1: expected, 2: None}
    assert _column_kind(result._nodes, "p") == kind


def test_set_declines_an_out_of_range_integer_behaviorally() -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher("MATCH (n:N) SET n.a = 9223372036854775808 SET n.b = n.a"))
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"


@pytest.mark.parametrize("engine", ENGINES)
def test_set_empty_match_still_normalizes_the_engine_and_checks_physical_identity(engine) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"uid": [1, 2], "label__N": [True, True]}), "uid")
    result = source.gfql("MATCH (n:Missing) SET n.x = 1", engine=engine)
    _assert_served(engine, result._nodes)
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("MATCH (n:Missing) SET n.uid = 1", engine=engine)
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["value"] == "physical_identity"
    assert _records(source._nodes) == before


# Wave-3 W3-F8: SET keeps the node row order on every engine.
@pytest.mark.parametrize("engine", ENGINES)
def test_set_keeps_the_original_node_row_order(engine) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [3, 1, 4, 2], "v": [30, 10, 40, 20]}), "id")
    result = source.gfql("MATCH (n) WHERE n.id IN [1, 2, 3] SET n.p = n.v + 1", engine=engine)
    assert [row["id"] for row in _records(result._nodes)] == [3, 1, 4, 2]
    assert [row["p"] for row in _records(result._nodes)] == [31, 11, None, 21]


# Wave-4 W4-F1: aggregation is not a SET value (it would group by node identity and store wrong values).
_W4_AGGREGATES = [
    "MATCH (n:N) SET n.c = count(*)",
    "MATCH (n:N) SET n.t = sum(n.num)",
    "MATCH (n:N) SET n.m = max(n.num) + 1",
]


@pytest.mark.parametrize("query", _W4_AGGREGATES)
def test_set_declines_aggregate_values_at_compile_and_validate(query) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher(query))
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "aggregate_value"
    with pytest.raises(GFQLValidationError) as exc:
        _graph("pandas").gfql_validate(query)
    assert exc.value.context["value"] == "aggregate_value"


@pytest.mark.parametrize("engine", ENGINES)
def test_set_aggregate_value_leaves_the_caller_untouched_on_every_engine(engine) -> None:
    source = _label_n_graph(engine, [1, 2])
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql(_W4_AGGREGATES[0], engine=engine)
    assert exc.value.context["value"] == "aggregate_value"
    assert _records(source._nodes) == before


# Wave-4 W4-F3: SET requires unique non-null node keys before any effect, like CREATE.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("ids", [[1, 1, 2], [1, None, 2]])
def test_set_rejects_duplicate_or_null_node_keys_before_effects(engine, ids) -> None:
    frame = pd.DataFrame({"id": pd.Series(ids, dtype="Int64"), "tag": ["a", "b", "c"]})
    if engine == "cudf":
        import cudf
        frame = cudf.from_pandas(frame)
    elif engine == "polars":
        frame = pytest.importorskip("polars").from_pandas(frame)
    source = graphistry.nodes(frame, "id")
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("MATCH (n) WHERE n.tag = 'a' SET n.x = 1", engine=engine)
    assert exc.value.code == ErrorCode.E105
    assert exc.value.context["value"] == "invalid_identity"
    assert _records(source._nodes) == before


# Wave-4 W4-F5: fail closed on any identifier other than the matched alias used as alias.prop (or a comprehension local).
def test_set_declines_unbound_identifiers_and_admits_comprehension_locals() -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher("MATCH p = (n) SET n.b = head(nodes(p)).a"))
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "unbound_variable"
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher("MATCH (n) SET n.b = stranger + 1"))
    assert exc.value.context["value"] == "unbound_variable"
    assert compile_cypher_query(parse_cypher("MATCH (n) SET n.b = [x IN [1, 2] | x + 1][0]")) is not None


# Wave-4 W4-F2: unspaced / unary / negative-zero parameter forms.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression,value,expected", [
    ("n.a -$v", -5, {1: 15, 2: 25}),
    ("-$v", -5, {1: 5, 2: 5}),
    ("n.a -$v", -0.0, {1: 10.0, 2: 20.0}),
])
def test_set_parameter_forms_execute(engine, expression, value, expected) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": [1, 2], "a": [10, 20]}), "id")
    result = source.gfql(f"MATCH (n) SET n.b = {expression}", engine=engine, params={"v": value})
    assert _set_values(result, "b") == expected



# Wave-5 W5-F4: an edges-only graph (no node table) materializes its nodes like the read path; no raw AssertionError.
def _edges_only(engine: str) -> Any:
    frame = _foundation_frame(engine, {"s": [1, 2], "d": [2, 3]})
    return graphistry.edges(frame, "s", "d")


@pytest.mark.parametrize("engine", ENGINES)
def test_set_on_an_edges_only_graph_materializes_nodes(engine) -> None:
    source = _edges_only(engine)
    result = source.gfql("MATCH (n) WHERE n.id = 2 SET n.x = 5", engine=engine)
    assert _set_values(result, "x") == {1: None, 2: 5, 3: None}
    assert source._nodes is None


@pytest.mark.parametrize("engine", ENGINES)
def test_set_on_an_edges_only_graph_with_no_match_is_a_clean_no_op(engine) -> None:
    source = _edges_only(engine)
    result = source.gfql("MATCH (n) WHERE n.id = 99 SET n.x = 5", engine=engine)
    assert "x" not in list(result._nodes.columns)
    assert source._nodes is None


# Wave-5 W5-F1: a comprehension/quantifier variable that shadows the SET alias is declined (the read rewrite is not
# scope-aware, so it would store the node instead of the value).
@pytest.mark.parametrize("query", [
    "MATCH (n) SET n.b = [n IN ['x'] | n][0]",
    "MATCH (n) SET n.b = size([n IN [1, 2] | n])",
    "MATCH (n) SET n.b = any(n IN [1, 2] WHERE n > 1)",
])
def test_set_declines_a_local_variable_that_shadows_the_matched_alias(query) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        compile_cypher_query(parse_cypher(query))
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "set"
    assert exc.value.context["value"] == "shadowed_alias"


# Wave-5 W5-F2 / W5-F3: parameter edge cases.
@pytest.mark.parametrize("value", [2 ** 1030, -(2 ** 1030)])
def test_set_huge_integer_parameter_fails_structurally(value) -> None:
    with pytest.raises(GFQLValidationError):
        compile_cypher_query(parse_cypher("MATCH (n) SET n.p = $v"), params={"v": value})


@pytest.mark.parametrize("param", [[1, 2], (1, 2), {"a": 1}, [float("inf")], [float("nan")]])
def test_set_declines_list_and_map_parameters_anywhere_in_the_value(param) -> None:
    for text in ("$v", "$v[0]", "head($v)"):
        with pytest.raises(GFQLValidationError) as exc:
            compile_cypher_query(parse_cypher(f"MATCH (n) SET n.p = {text}"), params={"v": param})
        assert exc.value.code == ErrorCode.E108
        assert exc.value.context["value"] == "non_scalar_value"


# Wave-5 W5-F5: the unsupported SET forms and the dead-code guard.
@pytest.mark.parametrize("query", ["MATCH (n) SET n:L", "MATCH (n) SET n = {a: 1}", "MATCH (n) SET n += {a: 1}"])
def test_set_label_and_map_replacement_forms_are_unsupported(query) -> None:
    with pytest.raises(Exception) as exc:
        parse_cypher(query)
    assert getattr(exc.value, "code", None) == ErrorCode.E107


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_set_combine_step_rejects_conflicting_and_repeated_assignments(engine) -> None:
    from graphistry.compute.gfql.mutation import SET_IDENTITY, SET_VALUE, _combine_assignment_frames

    def make(values: list[int]) -> Any:
        frame = pd.DataFrame({SET_IDENTITY: [1] * len(values), SET_VALUE: values})
        return pytest.importorskip("polars").from_pandas(frame) if engine == "polars" else frame

    with pytest.raises(GFQLValidationError) as exc:
        _combine_assignment_frames([make([1, 2])], object())
    assert exc.value.code == ErrorCode.E105 and exc.value.context["value"] == "conflicting_updates"
    with pytest.raises(GFQLValidationError) as exc:
        _combine_assignment_frames([make([1]), make([1])], object())
    assert exc.value.code == ErrorCode.E105 and exc.value.context["value"] == "repeated_property"
