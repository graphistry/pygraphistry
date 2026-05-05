"""Tests for GFQL rollout gates (T5 #1311)."""

from __future__ import annotations

import pytest

from graphistry.compute.gfql.rollout import (
    ENV_FALSY,
    ENV_TRUTHY,
    STRICT_SCHEMA_ENV,
    env_bool,
    resolve_strict_schema,
    strict_schema_env_default,
)


class TestEnvBool:
    def test_unset_returns_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GRAPHISTRY_TEST_FLAG", raising=False)
        assert env_bool("GRAPHISTRY_TEST_FLAG") is False

    def test_unset_returns_default_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GRAPHISTRY_TEST_FLAG", raising=False)
        assert env_bool("GRAPHISTRY_TEST_FLAG", default=True) is True

    @pytest.mark.parametrize("value", sorted(ENV_TRUTHY))
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("GRAPHISTRY_TEST_FLAG", value)
        assert env_bool("GRAPHISTRY_TEST_FLAG") is True

    @pytest.mark.parametrize("value", sorted(ENV_FALSY - {""}))
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("GRAPHISTRY_TEST_FLAG", value)
        assert env_bool("GRAPHISTRY_TEST_FLAG", default=True) is False

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAPHISTRY_TEST_FLAG", "TRUE")
        assert env_bool("GRAPHISTRY_TEST_FLAG") is True
        monkeypatch.setenv("GRAPHISTRY_TEST_FLAG", "Yes")
        assert env_bool("GRAPHISTRY_TEST_FLAG") is True

    def test_whitespace_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAPHISTRY_TEST_FLAG", "  true  ")
        assert env_bool("GRAPHISTRY_TEST_FLAG") is True

    def test_empty_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAPHISTRY_TEST_FLAG", "")
        assert env_bool("GRAPHISTRY_TEST_FLAG", default=True) is True
        assert env_bool("GRAPHISTRY_TEST_FLAG", default=False) is False

    def test_unrecognized_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAPHISTRY_TEST_FLAG", "maybe")
        assert env_bool("GRAPHISTRY_TEST_FLAG", default=False) is False
        assert env_bool("GRAPHISTRY_TEST_FLAG", default=True) is True


class TestStrictSchemaEnvDefault:
    def test_unset_is_loose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(STRICT_SCHEMA_ENV, raising=False)
        assert strict_schema_env_default() is False

    def test_explicit_false_is_loose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "false")
        assert strict_schema_env_default() is False

    def test_explicit_true_is_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "true")
        assert strict_schema_env_default() is True

    def test_env_name_constant(self) -> None:
        # Pin the public env-var name; renaming would break operators.
        assert STRICT_SCHEMA_ENV == "GRAPHISTRY_GFQL_STRICT_SCHEMA"


class TestResolveStrictSchema:
    def test_default_loose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(STRICT_SCHEMA_ENV, raising=False)
        assert resolve_strict_schema(explicit=False, catalog_strict=None) is False

    def test_explicit_true_wins_over_unset_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(STRICT_SCHEMA_ENV, raising=False)
        assert resolve_strict_schema(explicit=True, catalog_strict=None) is True

    def test_explicit_true_wins_over_env_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "false")
        assert resolve_strict_schema(explicit=True, catalog_strict=False) is True

    def test_catalog_true_wins_over_env_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "false")
        assert resolve_strict_schema(explicit=False, catalog_strict=True) is True

    def test_env_true_with_no_explicit_or_catalog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "true")
        assert resolve_strict_schema(explicit=False, catalog_strict=None) is True

    def test_env_true_with_explicit_false_still_strict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Explicit False does NOT force loose — monotonic widening.
        # Caller passing False is "no preference", not "force loose".
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "true")
        assert resolve_strict_schema(explicit=False, catalog_strict=None) is True

    def test_env_true_with_catalog_false_still_strict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Catalog False is also "no opinion" today (consistent with current
        # binder behavior where catalog.metadata.get('strict') returning a
        # falsy value falls through). Monotonic.
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "true")
        assert resolve_strict_schema(explicit=False, catalog_strict=False) is True

    def test_all_loose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, "false")
        assert resolve_strict_schema(explicit=False, catalog_strict=False) is False
        assert resolve_strict_schema(explicit=False, catalog_strict=None) is False

    @pytest.mark.parametrize("env_value", ["1", "true", "yes", "on", "TRUE", "Yes"])
    def test_truthy_env_values_strict(
        self, monkeypatch: pytest.MonkeyPatch, env_value: str
    ) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, env_value)
        assert resolve_strict_schema(explicit=False, catalog_strict=None) is True

    @pytest.mark.parametrize("env_value", ["0", "false", "no", "off", "FALSE", ""])
    def test_falsy_env_values_loose(
        self, monkeypatch: pytest.MonkeyPatch, env_value: str
    ) -> None:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, env_value)
        assert resolve_strict_schema(explicit=False, catalog_strict=None) is False


class TestPackageSurface:
    """Pin re-exports so consumers wiring rollout knobs have a stable surface."""

    def test_reexports_from_compute_gfql(self) -> None:
        from graphistry.compute.gfql import (
            STRICT_SCHEMA_ENV as PKG_ENV,
            env_bool as pkg_env_bool,
            resolve_strict_schema as pkg_resolve,
            strict_schema_env_default as pkg_default,
        )
        assert PKG_ENV == STRICT_SCHEMA_ENV
        assert pkg_env_bool is env_bool
        assert pkg_resolve is resolve_strict_schema
        assert pkg_default is strict_schema_env_default
