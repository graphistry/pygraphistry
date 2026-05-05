"""Binder-integration tests for T5 #1311 strict-schema env gate.

Pins the wiring between ``graphistry.compute.gfql.rollout`` and
``binder._strict_schema_mode`` so renames or rewrites at either end
break loudly.
"""

from __future__ import annotations

import pytest

from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog, PlanContext
from graphistry.compute.gfql.rollout import STRICT_SCHEMA_ENV
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError


def _ctx_with_catalog(*, strict_metadata: bool = False) -> PlanContext:
    catalog = GraphSchemaCatalog.from_schema_parts(
        node_columns=frozenset({"id", "label__Person"}),
        edge_columns=frozenset({"src", "dst", "label__KNOWS"}),
        node_id_column="id",
        edge_source_column="src",
        edge_destination_column="dst",
        metadata={"strict": True} if strict_metadata else {},
    )
    return PlanContext(catalog=catalog)


def _query_with_unknown_label() -> str:
    return "MATCH (n:UnknownLabel) RETURN n"


class TestBinderEnvGateOff:
    """Default behavior: env unset / false → loose mode (no behavior change)."""

    def test_env_unset_loose_accepts_unknown_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(STRICT_SCHEMA_ENV, raising=False)
        ctx = _ctx_with_catalog()
        ast = parse_cypher(_query_with_unknown_label())
        FrontendBinder().bind(ast, ctx)  # no exception in loose mode

    def test_env_false_loose_accepts_unknown_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "false")
        ctx = _ctx_with_catalog()
        ast = parse_cypher(_query_with_unknown_label())
        FrontendBinder().bind(ast, ctx)


class TestBinderEnvGateOn:
    """Env=true elevates default to strict when caller did not opt in explicitly."""

    def test_env_true_rejects_unknown_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "true")
        ctx = _ctx_with_catalog()
        ast = parse_cypher(_query_with_unknown_label())
        with pytest.raises(GFQLValidationError) as excinfo:
            FrontendBinder().bind(ast, ctx)
        # Check this is a strict-mode diagnostic (label-missing class).
        assert excinfo.value.code in {
            ErrorCode.E301,
            ErrorCode.E302,
            ErrorCode.E103,
        }

    @pytest.mark.parametrize("env_value", ["1", "yes", "on", "TRUE"])
    def test_env_truthy_variants_rejects(
        self, monkeypatch: pytest.MonkeyPatch, env_value: str
    ) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, env_value)
        ctx = _ctx_with_catalog()
        ast = parse_cypher(_query_with_unknown_label())
        with pytest.raises(GFQLValidationError):
            FrontendBinder().bind(ast, ctx)


class TestBinderPrecedence:
    """Explicit param + catalog flag still win regardless of env state."""

    def test_explicit_true_with_env_unset_strict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(STRICT_SCHEMA_ENV, raising=False)
        ctx = _ctx_with_catalog()
        ast = parse_cypher(_query_with_unknown_label())
        with pytest.raises(GFQLValidationError):
            FrontendBinder().bind(ast, ctx, strict_name_resolution=True)

    def test_explicit_true_with_env_false_strict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "false")
        ctx = _ctx_with_catalog()
        ast = parse_cypher(_query_with_unknown_label())
        with pytest.raises(GFQLValidationError):
            FrontendBinder().bind(ast, ctx, strict_name_resolution=True)

    def test_catalog_strict_with_env_unset_strict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(STRICT_SCHEMA_ENV, raising=False)
        ctx = _ctx_with_catalog(strict_metadata=True)
        ast = parse_cypher(_query_with_unknown_label())
        with pytest.raises(GFQLValidationError):
            FrontendBinder().bind(ast, ctx)

    def test_catalog_strict_with_env_false_strict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "false")
        ctx = _ctx_with_catalog(strict_metadata=True)
        ast = parse_cypher(_query_with_unknown_label())
        with pytest.raises(GFQLValidationError):
            FrontendBinder().bind(ast, ctx)

    def test_known_label_passes_under_env_strict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Strict mode should not break valid queries — this is the canary
        # safety net: enabling env=true must not break TCK-passing queries.
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "true")
        ctx = _ctx_with_catalog()
        ast = parse_cypher("MATCH (n:Person) RETURN n")
        FrontendBinder().bind(ast, ctx)
