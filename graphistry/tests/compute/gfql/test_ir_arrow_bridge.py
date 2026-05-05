"""Contract tests for GFQL IR Arrow/type bridge helpers (#1312)."""

from __future__ import annotations

import pytest

from graphistry.compute.gfql.ir.arrow_bridge import from_arrow, to_arrow
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import EdgeRef, ListType, NodeRef, PathType, ScalarType

pa = pytest.importorskip("pyarrow")


class TestArrowBridgeRoundTrip:
    def test_round_trip_preserves_logical_types_and_confidence(self) -> None:
        schema = RowSchema(
            columns={
                "n": NodeRef(labels=frozenset({"Person"})),
                "e": EdgeRef(type="KNOWS", src_label="Person", dst_label="Person"),
                "p": PathType(min_hops=1, max_hops=3),
                "age": ScalarType(kind="int64", nullable=False),
                "nick": ScalarType(kind="string", nullable=True),
                "scores": ListType(element_type=ScalarType(kind="float64", nullable=True)),
            }
        )

        arrow_schema = to_arrow(
            schema,
            confidence={
                "n": "declared",
                "e": "declared",
                "p": "propagated",
                "age": "declared",
                "nick": "propagated",
                "scores": "inferred",
            },
            coercion="widen",
        )

        round_trip_schema, round_trip_conf = from_arrow(arrow_schema, coercion="widen")

        assert round_trip_schema == schema
        assert round_trip_conf == {
            "n": "declared",
            "e": "declared",
            "p": "propagated",
            "age": "declared",
            "nick": "propagated",
            "scores": "inferred",
        }

        assert pa.types.is_large_string(arrow_schema.field("n").type)
        assert pa.types.is_large_string(arrow_schema.field("e").type)
        assert pa.types.is_large_string(arrow_schema.field("p").type)
        assert pa.types.is_int64(arrow_schema.field("age").type)
        assert arrow_schema.field("age").nullable is False

    def test_round_trip_without_bridge_metadata_uses_arrow_type_inference(self) -> None:
        arrow_schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("name", pa.large_string(), nullable=True),
                pa.field("weights", pa.list_(pa.float64()), nullable=True),
            ]
        )

        inferred, conf = from_arrow(arrow_schema, coercion="widen")

        assert inferred == RowSchema(
            columns={
                "id": ScalarType(kind="int64", nullable=False),
                "name": ScalarType(kind="string", nullable=True),
                "weights": ListType(element_type=ScalarType(kind="float64", nullable=True)),
            }
        )
        assert conf == {"id": "inferred", "name": "inferred", "weights": "inferred"}


class TestArrowBridgeCoercion:
    def test_strict_export_rejects_structural_types(self) -> None:
        schema = RowSchema(columns={"n": NodeRef(labels=frozenset({"Person"}))})
        with pytest.raises(ValueError, match="Strict Arrow export does not support NodeRef"):
            to_arrow(schema, coercion="strict")

    def test_strict_export_rejects_unknown_scalar_kind(self) -> None:
        schema = RowSchema(columns={"x": ScalarType(kind="uuid", nullable=True)})
        with pytest.raises(ValueError, match="Unsupported ScalarType.kind"):
            to_arrow(schema, coercion="strict")

    def test_widen_export_falls_back_for_unknown_scalar_kind(self) -> None:
        schema = RowSchema(columns={"x": ScalarType(kind="uuid", nullable=True)})
        arrow_schema = to_arrow(schema, coercion="widen")
        assert pa.types.is_large_string(arrow_schema.field("x").type)

    def test_strict_export_rejects_invalid_confidence(self) -> None:
        schema = RowSchema(columns={"x": ScalarType(kind="int64", nullable=True)})
        with pytest.raises(ValueError, match="Unsupported schema confidence"):
            to_arrow(schema, confidence={"x": "opaque"}, coercion="strict")

    def test_widen_export_demotes_invalid_confidence_to_default(self) -> None:
        schema = RowSchema(columns={"x": ScalarType(kind="int64", nullable=True)})
        arrow_schema = to_arrow(schema, confidence={"x": "opaque"}, coercion="widen")
        rt_schema, rt_conf = from_arrow(arrow_schema, coercion="widen")
        assert rt_schema == schema
        assert rt_conf == {"x": "inferred"}

    def test_strict_import_rejects_unsupported_arrow_types_without_metadata(self) -> None:
        arrow_schema = pa.schema([pa.field("x", pa.struct([("a", pa.int64())]))])
        with pytest.raises(ValueError, match="Unsupported Arrow type"):
            from_arrow(arrow_schema, coercion="strict")

    def test_widen_import_unknown_arrow_type_to_unknown_scalar(self) -> None:
        arrow_schema = pa.schema([pa.field("x", pa.struct([("a", pa.int64())]), nullable=False)])
        schema, confidence = from_arrow(arrow_schema, coercion="widen")
        assert schema == RowSchema(columns={"x": ScalarType(kind="unknown", nullable=False)})
        assert confidence == {"x": "inferred"}
