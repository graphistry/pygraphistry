from __future__ import annotations

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import ScalarType
from graphistry.schema import EdgeType, GraphSchema, NodeType


def _schema(*, strict: bool = True) -> GraphSchema:
    person = NodeType("Person", {"age": int, "id": int, "name": str})
    company = NodeType("Company", {"id": int, "name": str})
    works_at = EdgeType("WORKS_AT", source=person, destination=company, properties={"since": int})
    contracts = EdgeType("CONTRACTS", source=company, destination=person, properties={"fee": int})
    return GraphSchema(
        node_types=[person, company],
        edge_types=[works_at, contracts],
        strict=strict,
        node_id_column="id",
        edge_source_column="src",
        edge_destination_column="dst",
    )


def _graph(schema: GraphSchema):
    nodes = pd.DataFrame(
        [
            {"id": 1, "name": "alice", "label__Person": True, "label__Company": False},
            {"id": 2, "name": "acme", "label__Person": False, "label__Company": True},
        ]
    )
    edges = pd.DataFrame(
        [
            {"src": 1, "dst": 2, "since": 2020, "label__WORKS_AT": True},
        ]
    )
    return graphistry.bind(source="src", destination="dst", node="id", schema=schema).edges(edges).nodes(nodes)


def test_public_schema_imports_are_stable() -> None:
    assert graphistry.NodeType is NodeType
    assert graphistry.EdgeType is EdgeType
    assert graphistry.GraphSchema is GraphSchema


def test_graph_schema_adapts_to_internal_catalog() -> None:
    catalog = _schema(strict=False).to_catalog()

    assert catalog.node_columns == frozenset({"age", "id", "name", "label__Person", "label__Company"})
    assert catalog.edge_columns == frozenset({"fee", "since", "label__CONTRACTS", "label__WORKS_AT"})
    assert catalog.node_id == "id"
    assert catalog.edge_source == "src"
    assert catalog.edge_destination == "dst"
    assert catalog.metadata["strict"] is False
    assert catalog.metadata["edge_types"] == ("WORKS_AT", "CONTRACTS")
    assert catalog.metadata["edge_topologies"] == (
        {
            "relationship_type": "WORKS_AT",
            "source_labels": ("Person",),
            "destination_labels": ("Company",),
        },
        {
            "relationship_type": "CONTRACTS",
            "source_labels": ("Company",),
            "destination_labels": ("Person",),
        },
    )
    assert catalog.metadata["node_columns_by_label"]["Person"] == ("age", "id", "label__Person", "name")
    assert catalog.metadata["node_columns_by_label"]["Company"] == ("id", "label__Company", "name")
    assert catalog.metadata["edge_columns_by_type"]["CONTRACTS"] == ("fee", "label__CONTRACTS")
    assert catalog.metadata["edge_columns_by_type"]["WORKS_AT"] == ("label__WORKS_AT", "since")
    assert catalog.metadata["node_row_schemas"]["Person"] == RowSchema(
        columns={
            "age": ScalarType("int64"),
            "id": ScalarType("int64"),
            "name": ScalarType("string"),
        }
    )
    assert catalog.metadata["edge_row_schemas"]["WORKS_AT"] == RowSchema(
        columns={"since": ScalarType("int64")}
    )


def test_public_schema_accepts_and_exports_arrow_schema() -> None:
    pa = pytest.importorskip("pyarrow")

    person = NodeType(
        "Person",
        pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("name", pa.large_string()),
            ]
        ),
    )
    works_at = EdgeType(
        "WORKS_AT",
        source=person,
        destination="Company",
        properties=pa.schema([pa.field("since", pa.int32(), nullable=False)]),
    )

    assert person.properties["id"] == ScalarType("int64", nullable=False)
    node_arrow = person.to_arrow()
    assert node_arrow.field("id").type == pa.int64()
    assert node_arrow.field("id").nullable is False
    assert node_arrow.field("name").type == pa.large_string()
    assert node_arrow.field("label__Person").type == pa.bool_()

    edge_arrow = works_at.to_arrow()
    assert edge_arrow.field("since").type == pa.int32()
    assert edge_arrow.field("since").nullable is False
    assert edge_arrow.field("label__WORKS_AT").type == pa.bool_()


def test_public_schema_accepts_arrow_mapping_values() -> None:
    pa = pytest.importorskip("pyarrow")

    person = NodeType(
        "Person",
        {
            "id": pa.int64(),
            "display_name": pa.field("name", pa.large_string(), nullable=False),
        },
    )

    assert person.properties["id"] == ScalarType("int64")
    assert person.properties["display_name"] == ScalarType("string", nullable=False)
    exported = person.to_arrow(include_labels=False)
    assert exported.field("id").type == pa.int64()
    assert exported.field("display_name").nullable is False


def test_bind_schema_is_chainable_and_used_by_preflight() -> None:
    schema = _schema()
    g = _graph(schema).bind(point_color="name")

    assert g._gfql_schema is schema
    report = g.gfql_validate("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name AS name")
    assert report["ok"] is True


def test_schema_bound_preflight_rejects_missing_property() -> None:
    g = _graph(_schema())

    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql_validate("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN c.age AS age")

    err = exc_info.value
    assert err.code == ErrorCode.E301
    assert err.context["property"] == "age"
    assert err.context["entity_kind"] == "node"


def test_schema_bound_preflight_rejects_missing_relationship_type() -> None:
    g = _graph(_schema())

    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql_validate("MATCH (p:Person)-[:KNOWS]->(c:Company) RETURN p.name AS name")

    err = exc_info.value
    assert err.code == ErrorCode.E301
    assert err.context["relationship_type"] == "KNOWS"
    assert err.context["available_relationship_types"] == ("CONTRACTS", "WORKS_AT")


def test_schema_bound_preflight_rejects_edge_property_from_different_type() -> None:
    g = _graph(_schema())

    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql_validate("MATCH (p:Person)-[:WORKS_AT {fee: 10}]->(c:Company) RETURN p.name AS name")

    err = exc_info.value
    assert err.code == ErrorCode.E301
    assert err.context["property"] == "fee"
    assert err.context["entity_kind"] == "edge"
    assert err.context["available_columns"] == ("label__WORKS_AT", "since")


def test_schema_bound_preflight_accepts_reverse_topology() -> None:
    g = _graph(_schema())

    report = g.gfql_validate("MATCH (c:Company)<-[:WORKS_AT]-(p:Person) RETURN p.name AS name")

    assert report["ok"] is True


def test_schema_bound_preflight_rejects_topology_mismatch() -> None:
    g = _graph(_schema())

    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql_validate("MATCH (p:Person)-[:WORKS_AT]->(other:Person) RETURN p.name AS name")

    err = exc_info.value
    assert err.code == ErrorCode.E301
    assert err.context["relationship_types"] == ("WORKS_AT",)
    assert err.context["source_labels"] == ("Person",)
    assert err.context["destination_labels"] == ("Person",)


def test_schema_bound_preflight_can_be_called_permissively() -> None:
    g = _graph(_schema(strict=False))

    report = g.gfql_validate("MATCH (p:Unknown) RETURN p")

    assert report["ok"] is True


def test_schema_bound_preflight_explicit_strict_overrides_permissive_schema() -> None:
    g = _graph(_schema(strict=False))

    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql_validate("MATCH (p:Unknown) RETURN p", strict=True)

    err = exc_info.value
    assert err.code == ErrorCode.E301
    assert err.context["label"] == "Unknown"
