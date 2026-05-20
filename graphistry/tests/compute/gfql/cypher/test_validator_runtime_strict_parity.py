"""Validator/runtime strict-mode parity baselines (#1357).

``gfql_validate(..., strict=True)`` runs the FrontendBinder with
``strict_name_resolution=True``; runtime ``compile_cypher_query`` now
uses the same strict post-normalize bind. This module pins parity between
the two surfaces across representative admit/reject cases.

Cross-kind alias rebind and namespaced builtin function handling now have
parity, and are pinned here as positive parity cases.
"""

from __future__ import annotations

import pandas as pd
import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.api import compile_cypher
from graphistry.compute.gfql_validate import gfql_validate
from graphistry.tests.test_compute import CGFull


def _empty_g():
    """Minimal pandas-backed plottable for the validator (it reads
    g._nodes / g._edges schema even when strict=True with empty schema).
    """
    nodes_df = pd.DataFrame({"id": [], "label__S": []})
    edges_df = pd.DataFrame({"s": [], "d": [], "type": []})
    return CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a) MATCH ()-[a]->() RETURN a",
        "MATCH ()-[r]->() MATCH (r) RETURN r",
        "MATCH p = ()-->() MATCH (p) RETURN p",
        "MATCH (a) MATCH a = ()-->() RETURN a",
    ],
)
def test_validator_and_runtime_both_reject_cross_kind_rebind(query: str) -> None:
    """Both surfaces reject. Validator path goes through the strict binder;
    runtime path goes through the same binder via compile_cypher_query.
    Either order works — both must raise."""
    with pytest.raises(GFQLValidationError):
        gfql_validate(_empty_g(), query, strict=True)
    with pytest.raises(GFQLValidationError):
        compile_cypher(query)


@pytest.mark.parametrize(
    "query",
    [
        "RETURN duration.inSeconds(localtime(), localtime()) AS duration",
        "RETURN duration.between(localdatetime('2018-01-01T12:00'), localdatetime('2018-01-02T10:00')) AS duration",
        "RETURN datetime.fromepoch(416779, 999999999) AS dt",
        "RETURN date.truncate('decade', date({year: 1984, month: 10, day: 11}), {day: 2}) AS d",
    ],
)
def test_validator_and_runtime_both_admit_namespaced_builtins(query: str) -> None:
    gfql_validate(_empty_g(), query, strict=True)
    compile_cypher(query)


@pytest.mark.parametrize(
    "query",
    [
        "RETURN all(x IN [1, 2, 3] WHERE x > 1) AS ok",
        "RETURN [x IN [1, 2, 3] WHERE x > 1 | x + 1] AS xs",
        "RETURN {F: -0x162CD4F6} AS literal",
    ],
)
def test_validator_and_runtime_both_admit_comprehension_locals(query: str) -> None:
    """Parity pin for comprehension-local scope handling after #1371 P2."""
    gfql_validate(_empty_g(), query, strict=True)
    compile_cypher(query)


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) WHERE all(x IN n.labels WHERE x = 'A') RETURN n",
        "MATCH (n) WHERE any(x IN n.labels WHERE x = 'B') RETURN n",
        "CALL graphistry.degree() YIELD nodeId RETURN nodeId",
    ],
)
def test_runtime_strict_admits_compile_only_scope_shapes(query: str) -> None:
    compile_cypher(query)


@pytest.mark.parametrize(
    ("query", "identifier"),
    [
        ("MATCH (a) RETURN ghost", "ghost"),
        ("MATCH (a) WHERE ghost.foo = 1 RETURN a", "ghost"),
    ],
)
def test_runtime_strict_rejects_unresolved_aliases_with_e204(query: str, identifier: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(query)
    assert exc_info.value.code == ErrorCode.E204
    assert exc_info.value.context.get("field") == "identifier"
    assert exc_info.value.context.get("value") == identifier
