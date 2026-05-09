"""Runtime strict-name-resolution baselines for #1357.

``compile_cypher_query`` now binds the post-normalize AST with
``strict_name_resolution=True``. This module pins representative behaviors
that previously depended on loose-mode fallbacks:

- comprehension-local aliases remain admitted after #1371 scope fixes
- CALL/YIELD aliases survive prepass -> normalize -> bind
- unresolved aliases fail at binder-time with structured E204 metadata
"""

from __future__ import annotations

import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.api import compile_cypher


# ---------------------------------------------------------------------------
# Strict runtime admits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # Quantifier predicates — binder doesn't model x as a
        # comprehension-local; in strict mode it raises on unresolved 'x'.
        "MATCH (n) WHERE all(x IN n.labels WHERE x = 'A') RETURN n",
        "MATCH (n) WHERE any(x IN n.labels WHERE x = 'B') RETURN n",
        "MATCH (n) WHERE none(x IN n.labels WHERE x = 'C') RETURN n",
        "MATCH (n) WHERE single(x IN n.labels WHERE x = 'D') RETURN n",
    ],
)
def test_runtime_strict_admits_quantifier_predicates(query: str) -> None:
    compile_cypher(query)


def test_runtime_strict_admits_call_yield_then_return_yield_alias() -> None:
    query = "CALL graphistry.degree() YIELD nodeId RETURN nodeId"
    compile_cypher(query)


# ---------------------------------------------------------------------------
# Strict runtime rejects unresolved aliases at binder time
# ---------------------------------------------------------------------------


def test_compile_rejects_unresolved_return_alias_via_strict_binder() -> None:
    query = "MATCH (a) RETURN ghost"
    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(query)
    assert exc_info.value.code == ErrorCode.E204
    assert exc_info.value.context.get("field") == "identifier"
    assert exc_info.value.context.get("value") == "ghost"


def test_compile_rejects_unresolved_where_alias_via_strict_binder() -> None:
    query = "MATCH (a) WHERE ghost.foo = 1 RETURN a"
    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(query)
    assert exc_info.value.code == ErrorCode.E204
    assert exc_info.value.context.get("field") == "identifier"
    assert exc_info.value.context.get("value") == "ghost"
