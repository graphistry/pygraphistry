from __future__ import annotations

import pandas as pd
import pytest

import graphistry
from graphistry.exceptions import SchemaValidationError
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import ListType, NodeRef, ScalarType
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
            {"id": 1, "age": 42, "name": "alice", "label__Person": True, "label__Company": False},
            {"id": 2, "age": 0, "name": "acme", "label__Person": False, "label__Company": True},
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


def test_node_and_edge_types_round_trip_from_arrow() -> None:
    pa = pytest.importorskip("pyarrow")

    person = NodeType(
        "Person",
        pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("name", pa.large_string())]),
    )
    works_at = EdgeType(
        "WORKS_AT",
        source="Person",
        destination="Company",
        properties=pa.schema([pa.field("since", pa.int32(), nullable=False)]),
    )

    imported_node = NodeType.from_arrow("Person", person.to_arrow())
    imported_edge = EdgeType.from_arrow("WORKS_AT", "Person", "Company", works_at.to_arrow())

    assert imported_node.labels == frozenset({"Person"})
    assert imported_node.properties == person.properties
    assert "label__Person" not in imported_node.properties
    assert imported_edge.source == frozenset({"Person"})
    assert imported_edge.destination == frozenset({"Company"})
    assert imported_edge.properties == works_at.properties
    assert "label__WORKS_AT" not in imported_edge.properties


def test_graph_schema_arrow_declaration_round_trip() -> None:
    pa = pytest.importorskip("pyarrow")

    schema = _schema(strict=False)
    declaration = schema.to_arrow()

    assert declaration["nodes"].field("age").type == pa.int64()
    assert declaration["edges"].field("since").type == pa.int64()
    assert declaration["nodes"].metadata[b"gfql.arrow_bridge.version"] == b"1"
    assert declaration["edges"].metadata[b"gfql.arrow_bridge.version"] == b"1"
    assert declaration["edge_types"]["WORKS_AT"]["source"] == ("Person",)
    assert declaration["edge_types"]["WORKS_AT"]["destination"] == ("Company",)

    imported = GraphSchema.from_arrow(declaration)

    assert imported.strict is False
    assert imported.node_id_column == "id"
    assert imported.edge_source_column == "src"
    assert imported.edge_destination_column == "dst"
    assert imported.node_columns == schema.node_columns
    assert imported.edge_columns == schema.edge_columns
    assert imported.edge_types[0].topology.as_metadata() == schema.edge_types[0].topology.as_metadata()


def test_graph_schema_rejects_conflicting_node_physical_column_types() -> None:
    with pytest.raises(ValueError) as exc_info:
        GraphSchema(
            node_types=[
                NodeType("Cat", {"age": int}),
                NodeType("Dog", {"age": str}),
            ]
        )

    message = str(exc_info.value)
    assert "nodes" in message
    assert "age" in message
    assert "Cat" in message
    assert "Dog" in message
    assert "int64" in message
    assert "string" in message


def test_graph_schema_rejects_conflicting_edge_physical_column_types() -> None:
    with pytest.raises(ValueError) as exc_info:
        GraphSchema(
            edge_types=[
                EdgeType("LIKES", source="Cat", destination="Toy", properties={"weight": int}),
                EdgeType("CHASES", source="Dog", destination="Toy", properties={"weight": str}),
            ]
        )

    message = str(exc_info.value)
    assert "edges" in message
    assert "weight" in message
    assert "LIKES" in message
    assert "CHASES" in message
    assert "int64" in message
    assert "string" in message


def test_graph_schema_from_arrow_rejects_conflicting_node_physical_column_types() -> None:
    pa = pytest.importorskip("pyarrow")

    with pytest.raises(ValueError) as exc_info:
        GraphSchema.from_arrow(
            {
                "node_types": {
                    "Cat": pa.schema([pa.field("age", pa.int64())]),
                    "Dog": pa.schema([pa.field("age", pa.large_string())]),
                },
                "edge_types": {},
            }
        )

    message = str(exc_info.value)
    assert "nodes" in message
    assert "age" in message
    assert "Cat" in message
    assert "Dog" in message


def test_graph_schema_from_arrow_rejects_conflicting_edge_physical_column_types() -> None:
    pa = pytest.importorskip("pyarrow")

    with pytest.raises(ValueError) as exc_info:
        GraphSchema.from_arrow(
            {
                "node_types": {},
                "edge_types": {
                    "LIKES": {
                        "source": ("Cat",),
                        "destination": ("Toy",),
                        "schema": pa.schema([pa.field("weight", pa.int64())]),
                    },
                    "CHASES": {
                        "source": ("Dog",),
                        "destination": ("Toy",),
                        "schema": pa.schema([pa.field("weight", pa.large_string())]),
                    },
                },
            }
        )

    message = str(exc_info.value)
    assert "edges" in message
    assert "weight" in message
    assert "LIKES" in message
    assert "CHASES" in message


def test_graph_schema_rejects_conflicting_list_physical_column_types() -> None:
    with pytest.raises(ValueError) as exc_info:
        GraphSchema(
            node_types=[
                NodeType("Cat", {"tags": ListType(ScalarType("int64"))}),
                NodeType("Dog", {"tags": ListType(ScalarType("string"))}),
            ]
        )

    message = str(exc_info.value)
    assert "nodes" in message
    assert "tags" in message
    assert "Cat" in message
    assert "Dog" in message


def test_graph_schema_rejects_conflicting_node_ref_physical_column_types() -> None:
    with pytest.raises(ValueError) as exc_info:
        GraphSchema(
            node_types=[
                NodeType("Cat", {"friend": NodeRef(frozenset({"Cat"}))}),
                NodeType("Dog", {"friend": NodeRef(frozenset({"Dog"}))}),
            ]
        )

    message = str(exc_info.value)
    assert "nodes" in message
    assert "friend" in message
    assert "Cat" in message
    assert "Dog" in message


def test_graph_schema_allows_type_local_nullability_for_shared_node_column() -> None:
    pa = pytest.importorskip("pyarrow")

    cat = NodeType("Cat", pa.schema([pa.field("lives", pa.int64(), nullable=False)]))
    dog = NodeType("Dog", pa.schema([pa.field("lives", pa.int64(), nullable=True)]))

    schema = GraphSchema(node_types=[cat, dog])

    assert schema.node_types[0].properties["lives"] == ScalarType("int64", nullable=False)
    assert schema.node_types[1].properties["lives"] == ScalarType("int64", nullable=True)
    assert schema.node_arrow().field("lives").nullable is True


def test_graph_schema_absent_property_does_not_change_type_local_nullability() -> None:
    pa = pytest.importorskip("pyarrow")

    cat = NodeType("Cat", pa.schema([pa.field("lives", pa.int64(), nullable=False)]))
    house = NodeType("House", pa.schema([pa.field("address", pa.large_string())]))

    schema = GraphSchema(node_types=[cat, house])

    assert schema.node_types[0].properties["lives"] == ScalarType("int64", nullable=False)
    assert "lives" not in schema.node_types[1].properties


def test_bind_schema_is_chainable_and_used_by_preflight() -> None:
    schema = _schema()
    g = _graph(schema).bind(point_color="name")

    assert g._gfql_schema is schema
    assert g.schema is schema
    report = g.gfql_validate("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name AS name")
    assert report["ok"] is True


def test_schema_accessor_returns_bound_schema() -> None:
    schema = _schema()
    g = _graph(schema)

    assert g.schema is schema


def test_schema_accessor_is_read_only() -> None:
    schema = _schema()
    g = _graph(schema)

    with pytest.raises(AttributeError):
        g.schema = None  # type: ignore[misc]

    assert g.schema is schema


def test_schema_accessor_returns_none_when_unbound() -> None:
    g = graphistry.bind()

    assert g.schema is None


def test_bound_schema_arrow_boundary_strict_passes() -> None:
    pa = pytest.importorskip("pyarrow")
    g = _graph(_schema())

    arr = g.to_arrow(validate="strict", schema_validate="strict")

    assert arr is not None
    assert arr.schema.field("since").type == pa.int64()


def test_bound_schema_arrow_boundary_plot_strict_validates_edges_and_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    g = _graph(_schema())
    monkeypatch.setattr(g._pygraphistry, "refresh", lambda: None)

    uploader = g.plot(skip_upload=True, schema_validate="strict")

    assert uploader.edges is not None
    assert uploader.nodes is not None


def test_bound_schema_arrow_boundary_autofix_casts_declared_column() -> None:
    pa = pytest.importorskip("pyarrow")
    schema = _schema()
    edges = pd.DataFrame(
        [
            {"src": 1, "dst": 2, "since": "2020", "label__WORKS_AT": True},
        ]
    )
    nodes = _graph(schema)._nodes
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id").bind(schema=schema)

    arr = g.to_arrow(validate="autofix", schema_validate="autofix")

    assert arr is not None
    assert arr.schema.field("since").type == pa.int64()
    assert arr.column("since").to_pylist() == [2020]


def test_bound_schema_arrow_boundary_strict_rejects_type_mismatch() -> None:
    schema = _schema()
    edges = pd.DataFrame(
        [
            {"src": 1, "dst": 2, "since": "2020", "label__WORKS_AT": True},
        ]
    )
    nodes = _graph(schema)._nodes
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id").bind(schema=schema)

    with pytest.raises(SchemaValidationError) as exc_info:
        g.to_arrow(validate="strict", schema_validate="strict")

    err = exc_info.value
    assert err.table == "edges"
    assert err.column == "since"
    assert err.reason == "Arrow type mismatch"


def test_bound_schema_arrow_boundary_rejects_missing_declared_column() -> None:
    schema = _schema()
    edges = pd.DataFrame(
        [
            {"src": 1, "dst": 2, "label__WORKS_AT": True},
        ]
    )
    nodes = _graph(schema)._nodes
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id").bind(schema=schema)

    with pytest.raises(SchemaValidationError) as exc_info:
        g.validate_arrow_schema("edges")

    err = exc_info.value
    assert err.table == "edges"
    assert err.column == "since"
    assert err.reason == "missing declared schema column"


def test_bound_schema_arrow_boundary_ignores_inactive_edge_type_columns() -> None:
    schema = _schema()
    edges = pd.DataFrame(
        [
            {
                "src": 1,
                "dst": 2,
                "since": 2020,
                "label__WORKS_AT": True,
                "label__CONTRACTS": False,
            },
        ]
    )
    nodes = _graph(schema)._nodes
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id").bind(schema=schema)

    arr = g.validate_arrow_schema("edges")

    assert arr is not None
    assert arr.column("since").to_pylist() == [2020]


def test_bound_schema_arrow_boundary_non_nullable_property_is_type_local() -> None:
    pa = pytest.importorskip("pyarrow")
    person = NodeType(
        "Person",
        pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("age", pa.int64(), nullable=False),
            ]
        ),
    )
    company = NodeType("Company", pa.schema([pa.field("id", pa.int64(), nullable=False)]))
    schema = GraphSchema(node_types=[person, company], node_id_column="id")
    nodes = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "age": pa.array([42, None], type=pa.int64()),
            "label__Person": pa.array([True, False]),
            "label__Company": pa.array([False, True]),
        }
    )
    g = graphistry.bind(node="id", schema=schema).nodes(nodes)

    arr = g.validate_arrow_schema("nodes")

    assert arr is not None
    assert arr.column("age").to_pylist() == [42, None]


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
