from __future__ import annotations

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.ir.types import ScalarType
from graphistry.schema import EdgeType, GraphSchema, NodeType
from graphistry.schema_inference import InferredProperty, SchemaInferenceReport


def _person_company_graph():
    nodes = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "alice",
                "age": 42,
                "department": None,
                "label__Person": True,
                "label__Company": False,
            },
            {
                "id": 2,
                "name": "acme",
                "age": None,
                "department": "engineering",
                "label__Person": False,
                "label__Company": True,
            },
            {
                "id": 3,
                "name": "bob",
                "age": None,
                "department": None,
                "label__Person": True,
                "label__Company": False,
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {"src": 1, "dst": 2, "since": 2020, "label__WORKS_AT": True},
            {"src": 3, "dst": 2, "since": 2021, "label__WORKS_AT": True},
        ]
    )
    return graphistry.bind(source="src", destination="dst", node="id").edges(edges).nodes(nodes)


def _node_type(schema: GraphSchema, name: str) -> NodeType:
    return next(node_type for node_type in schema.node_types if node_type.name == name)


def _edge_type(schema: GraphSchema, name: str) -> EdgeType:
    return next(edge_type for edge_type in schema.edge_types if edge_type.name == name)


def test_infer_schema_public_function_returns_graph_schema() -> None:
    schema = graphistry.infer_schema(_person_company_graph())

    assert isinstance(schema, GraphSchema)
    assert graphistry.InferredProperty is InferredProperty
    assert graphistry.SchemaInferenceReport is SchemaInferenceReport
    assert schema.metadata["source"] == "inferred"
    assert schema.metadata["inference"]["label_column_prefix"] == "label__"
    assert {node_type.name for node_type in schema.node_types} == {"Company", "Person"}
    assert {edge_type.name for edge_type in schema.edge_types} == {"WORKS_AT"}


def test_plotter_infer_schema_is_pure_and_chainable() -> None:
    g = _person_company_graph()

    schema = g.infer_schema()
    rebound = g.bind(schema=schema)

    assert getattr(g, "_gfql_schema", None) is None
    assert rebound._gfql_schema is schema


def test_module_infer_schema_requires_plotter() -> None:
    with pytest.raises(ValueError, match="requires a plotter"):
        graphistry.infer_schema()


def test_bind_infer_schema_is_opt_in_and_round_trips_into_preflight() -> None:
    g = _person_company_graph()

    inferred = g.bind(infer_schema=True)

    assert isinstance(inferred._gfql_schema, GraphSchema)
    assert inferred.gfql_validate("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name")["ok"] is True
    with pytest.raises(GFQLValidationError) as exc_info:
        inferred.gfql_validate("MATCH (p:Person)-[:WORKS_AT]->(x:Person) RETURN p.name")

    err = exc_info.value
    assert err.code == ErrorCode.E301
    assert err.context["relationship_types"] == ("WORKS_AT",)


def test_infer_node_and_edge_property_types_from_pandas() -> None:
    schema = graphistry.infer_schema(_person_company_graph())
    person = _node_type(schema, "Person")
    works_at = _edge_type(schema, "WORKS_AT")

    assert person.properties["id"] == ScalarType("int64", nullable=False)
    assert person.properties["name"] == ScalarType("string", nullable=False)
    assert person.properties["age"] == ScalarType("float64", nullable=True)
    assert works_at.properties["since"] == ScalarType("int64", nullable=False)
    assert "src" not in works_at.properties
    assert "dst" not in works_at.properties


def test_infer_presence_report_distinguishes_required_optional_and_maybe_absent() -> None:
    _, report = graphistry.infer_schema(_person_company_graph(), return_report=True)

    assert report.node_properties["Person"]["name"].presence == "required"
    assert report.node_properties["Person"]["age"].presence == "optional"
    assert report.node_properties["Person"]["department"].presence == "maybe_absent"
    assert report.edge_properties["WORKS_AT"]["since"].presence == "required"


def test_infer_relationship_topology_from_endpoint_labels() -> None:
    schema = graphistry.infer_schema(_person_company_graph())
    works_at = _edge_type(schema, "WORKS_AT")

    assert works_at.source == frozenset({"Person"})
    assert works_at.destination == frozenset({"Company"})


def test_infer_edge_only_graph_keeps_edge_schema_without_fabricating_topology() -> None:
    g = graphistry.bind(source="src", destination="dst").edges(
        pd.DataFrame({"src": [1], "dst": [2], "weight": [0.5], "label__KNOWS": [True]})
    )

    schema = graphistry.infer_schema(g)
    knows = _edge_type(schema, "KNOWS")

    assert schema.node_types == ()
    assert knows.properties["weight"] == ScalarType("float64", nullable=False)
    assert knows.source == frozenset()
    assert knows.destination == frozenset()


def test_infer_schema_no_data_fallback_is_empty_bound_schema() -> None:
    schema = graphistry.bind(source="src", destination="dst", node="id").infer_schema()

    assert schema == GraphSchema(
        node_id_column="id",
        edge_source_column="src",
        edge_destination_column="dst",
    )
    assert schema.metadata["source"] == "inferred"


def test_declared_schema_is_not_silently_merged_by_bind_infer_schema() -> None:
    declared = GraphSchema(node_types=[NodeType("Declared", {"name": str})])
    g = _person_company_graph().bind(schema=declared)

    with pytest.raises(ValueError, match="schema and infer_schema"):
        g.bind(infer_schema=True)

    assert g.bind()._gfql_schema is declared


def test_infer_schema_declared_override_wins_when_requested() -> None:
    declared = GraphSchema(
        node_types=[NodeType("Person", {"age": "string"})],
        edge_types=[EdgeType("WORKS_AT", source="Person", destination="Company", properties={"since": "float64"})],
    )

    schema = graphistry.infer_schema(_person_company_graph(), schema=declared)

    assert schema.metadata["source"] == "mixed"
    assert schema.metadata["inference"]["property_type_source"] == "dataframe-arrow-schema"
    assert _node_type(schema, "Person").properties["age"] == ScalarType("string")
    assert _edge_type(schema, "WORKS_AT").properties["since"] == ScalarType("float64")
    assert _node_type(schema, "Company").properties["name"] == ScalarType("string", nullable=False)


def test_infer_schema_cudf_matches_pandas_representative_case() -> None:
    cudf = pytest.importorskip("cudf")
    cp = pytest.importorskip("cupy")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"cuDF schema inference requires a CUDA device: {exc}")
    if device_count == 0:
        pytest.skip("cuDF schema inference requires a CUDA device")

    pdf_graph = _person_company_graph()
    gdf_graph = graphistry.bind(source="src", destination="dst", node="id").edges(
        cudf.from_pandas(pdf_graph._edges)
    ).nodes(
        cudf.from_pandas(pdf_graph._nodes)
    )

    assert graphistry.infer_schema(gdf_graph) == graphistry.infer_schema(pdf_graph)
