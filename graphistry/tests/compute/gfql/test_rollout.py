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

@pytest.mark.parametrize(
    "value,default,expected",
    [
        (None, False, False),
        (None, True, True),
        ("", False, False),
        ("", True, True),
        ("maybe", False, False),
        ("maybe", True, True),
        ("TRUE", False, True),
        ("Yes", False, True),
        ("  true  ", False, True),
        *[(value, False, True) for value in sorted(ENV_TRUTHY)],
        *[(value, True, False) for value in sorted(ENV_FALSY)],
    ],
)
def test_env_bool(
    value: str | None,
    default: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if value is None:
        monkeypatch.delenv("GRAPHISTRY_TEST_FLAG", raising=False)
    else:
        monkeypatch.setenv("GRAPHISTRY_TEST_FLAG", value)

    assert env_bool("GRAPHISTRY_TEST_FLAG", default=default) is expected

@pytest.mark.parametrize(
    "explicit,catalog_strict,env_value,expected",
    [
        (False, None, None, False),
        (True, None, None, True),
        (True, False, "false", True),
        (False, True, "false", True),
        (False, None, "true", True),
        (False, None, "TRUE", True),
        (False, None, "Yes", True),
        (False, None, "1", True),
        (False, None, "on", True),
        (False, False, "true", True),
        (False, False, "false", False),
        (False, None, "0", False),
        (False, None, "no", False),
        (False, None, "off", False),
        (False, None, "", False),
    ],
)
def test_resolve_strict_schema_precedence(
    explicit: bool,
    catalog_strict: bool | None,
    env_value: str | None,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if env_value is None:
        monkeypatch.delenv(STRICT_SCHEMA_ENV, raising=False)
    else:
        monkeypatch.setenv(STRICT_SCHEMA_ENV, env_value)

    assert resolve_strict_schema(explicit=explicit, catalog_strict=catalog_strict) is expected


def test_reexports_from_compute_gfql(monkeypatch: pytest.MonkeyPatch) -> None:
    from graphistry.compute.gfql import (
        STRICT_SCHEMA_ENV as PKG_ENV,
        env_bool as pkg_env_bool,
        resolve_strict_schema as pkg_resolve,
        strict_schema_env_default as pkg_default,
    )

    assert STRICT_SCHEMA_ENV == "GRAPHISTRY_GFQL_STRICT_SCHEMA"
    assert PKG_ENV == STRICT_SCHEMA_ENV
    assert pkg_env_bool is env_bool
    assert pkg_resolve is resolve_strict_schema
    assert pkg_default is strict_schema_env_default

    monkeypatch.delenv(STRICT_SCHEMA_ENV, raising=False)
    assert pkg_default() is False
    monkeypatch.setenv(STRICT_SCHEMA_ENV, "false")
    assert pkg_default() is False
    monkeypatch.setenv(STRICT_SCHEMA_ENV, "true")
    assert pkg_default() is True
