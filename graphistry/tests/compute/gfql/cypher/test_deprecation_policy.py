"""Compat + policy tests for deprecated public Cypher API exports.

Verifies three invariants:
1. Deprecated symbols still work (backward compatibility).
2. DeprecationWarning fires at the right call site.
3. Warning messages name the removal version (v2.8.0) so the policy is auditable.
"""
import warnings

import pytest


# ---------------------------------------------------------------------------
# compile_cypher()
# ---------------------------------------------------------------------------

def test_compile_cypher_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from graphistry.compute.gfql.cypher.api import compile_cypher
        compile_cypher("MATCH (n) RETURN n.id AS id")
    assert any(issubclass(x.category, DeprecationWarning) for x in w), "compile_cypher() must emit DeprecationWarning"


def test_compile_cypher_warning_names_removal_version():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from graphistry.compute.gfql.cypher.api import compile_cypher
        compile_cypher("MATCH (n) RETURN n.id AS id")
    msgs = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("v2.8.0" in m for m in msgs), f"DeprecationWarning must mention v2.8.0; got: {msgs}"


def test_compile_cypher_still_returns_result():
    from graphistry.compute.gfql.cypher.api import compile_cypher
    from graphistry.compute.gfql.cypher.lowering import CompiledCypherQuery
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = compile_cypher("MATCH (n) RETURN n.id AS id")
    assert isinstance(result, CompiledCypherQuery), "compile_cypher() must remain functional for backward compat"


# ---------------------------------------------------------------------------
# __getattr__ deprecated module exports
# ---------------------------------------------------------------------------

def _access_deprecated_attr(name: str):
    """Access a deprecated attribute on the cypher module, capturing all warnings."""
    import graphistry.compute.gfql.cypher as mod
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        getattr(mod, name)
    return w


@pytest.mark.parametrize("name", [
    "CompiledCypherQuery",
    "CompiledCypherUnionQuery",
    "compile_cypher_query",
    "CompiledCypherProcedureCall",
])
def test_deprecated_module_attr_emits_deprecation_warning(name):
    w = _access_deprecated_attr(name)
    assert any(issubclass(x.category, DeprecationWarning) for x in w), \
        f"Accessing {name} on the cypher module must emit DeprecationWarning"


@pytest.mark.parametrize("name", [
    "CompiledCypherQuery",
    "CompiledCypherUnionQuery",
    "compile_cypher_query",
    "CompiledCypherProcedureCall",
])
def test_deprecated_module_attr_warning_names_removal_version(name):
    w = _access_deprecated_attr(name)
    msgs = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("v2.8.0" in m for m in msgs), \
        f"DeprecationWarning for {name} must mention v2.8.0; got: {msgs}"


@pytest.mark.parametrize("name", [
    "CompiledCypherQuery",
    "CompiledCypherUnionQuery",
    "compile_cypher_query",
    "CompiledCypherProcedureCall",
])
def test_deprecated_module_attr_still_accessible(name):
    import graphistry.compute.gfql.cypher as mod
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        obj = getattr(mod, name)
    assert obj is not None, f"{name} must still be accessible via the module for backward compat"
