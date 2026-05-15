from __future__ import annotations

from typing import List, cast

import pandas as pd
import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.policy import CompileSummary, PolicyContext, PolicyException
from graphistry.tests.test_compute import CGFull


class _CypherPolicyGraph(CGFull):
    _dgl_graph = None

    def search_graph(self, query: str, scale: float = 0.5, top_n: int = 100, thresh: float = 5000, broader: bool = False, inplace: bool = False):
        raise NotImplementedError

    def search(self, query: str, cols=None, thresh: float = 5000, fuzzy: bool = True, top_n: int = 10):
        raise NotImplementedError

    def embed(self, relation: str, *args, **kwargs):
        raise NotImplementedError


def _mk_graph() -> _CypherPolicyGraph:
    return cast(
        _CypherPolicyGraph,
        _CypherPolicyGraph()
        .nodes(pd.DataFrame({"id": ["a", "b"]}), "id")
        .edges(pd.DataFrame({"s": ["a"], "d": ["b"]}), "s", "d"),
    )


def test_postcompile_policy_fires_once_with_structured_binder_payload() -> None:
    calls: List[PolicyContext] = []

    def observe(ctx: PolicyContext) -> None:
        calls.append(ctx)

    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_graph().gfql(
            "MATCH (a) MATCH ()-[a]->() RETURN a",
            policy={"postcompile": observe},
        )

    assert exc_info.value.code == ErrorCode.E204
    assert len(calls) == 1
    ctx = calls[0]
    assert ctx["phase"] == "postcompile"
    assert ctx["hook"] == "postcompile"
    assert ctx["compile_language"] == "cypher"
    assert ctx["success"] is False
    assert ctx["error_type"] == "GFQLValidationError"
    assert "Cypher alias rebound" in ctx["error"]
    summary = ctx["compile"]
    assert isinstance(summary, CompileSummary)
    assert summary.language == "cypher"
    assert summary.success is False
    assert summary.error_type == "GFQLValidationError"
    assert summary.compiler_phase == "bind"
    assert summary.code == ErrorCode.E204
    assert summary.message == "Cypher alias rebound as a different entity kind"
    assert summary.context["existing_kind"] == "node"
    assert summary.context["new_kind"] == "edge"
    assert summary.context["new_role"] == "relationship pattern"
    assert summary.context["value"] == "a"
    with pytest.raises(TypeError):
        summary.context["existing_kind"] = "scalar"  # type: ignore[index]
    assert summary.field == "identifier"
    assert summary.value_repr == "'a'"
    assert summary.suggestion is not None
    assert summary.line is None
    assert summary.column is None
    assert summary.param_keys == ()


def test_postcompile_policy_absent_preserves_existing_error_surface() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_graph().gfql("MATCH (a) MATCH ()-[a]->() RETURN a")

    assert exc_info.value.code == ErrorCode.E204
    assert exc_info.value.context["existing_kind"] == "node"
    assert exc_info.value.context["new_kind"] == "edge"
    assert exc_info.value.context["new_role"] == "relationship pattern"


def test_postcompile_policy_fires_for_validate_true_preflight() -> None:
    calls: List[PolicyContext] = []

    def observe(ctx: PolicyContext) -> None:
        calls.append(ctx)

    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_graph().gfql(
            "MATCH (a) MATCH ()-[a]->() RETURN a",
            validate=True,
            policy={"postcompile": observe},
        )

    assert exc_info.value.code == ErrorCode.E204
    assert len(calls) == 1
    assert calls[0]["phase"] == "postcompile"
    assert calls[0]["success"] is False
    summary = calls[0]["compile"]
    assert isinstance(summary, CompileSummary)
    assert summary.compiler_phase == "bind"
    assert summary.context["existing_kind"] == "node"
    assert summary.context["new_kind"] == "edge"


def test_postcompile_policy_fires_once_for_successful_row_query() -> None:
    calls: List[PolicyContext] = []

    def observe(ctx: PolicyContext) -> None:
        calls.append(ctx)

    result = _mk_graph().gfql(
        "UNWIND [1, 2, 3] AS x RETURN x ORDER BY x",
        policy={"postcompile": observe},
    )

    assert len(calls) == 1
    ctx = calls[0]
    assert ctx["phase"] == "postcompile"
    assert ctx["success"] is True
    assert "error" not in ctx
    summary = ctx["compile"]
    assert isinstance(summary, CompileSummary)
    assert summary.language == "cypher"
    assert summary.success is True
    assert summary.error_type is None
    assert summary.message is None
    assert summary.code is None
    assert summary.context == {}
    assert result._nodes is not None
    assert result._nodes[["x"]].to_dict(orient="records") == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_postcompile_policy_fires_before_runtime_hooks() -> None:
    calls: List[str] = []

    def observe_postcompile(ctx: PolicyContext) -> None:
        calls.append(cast(str, ctx["phase"]))

    def observe_runtime(ctx: PolicyContext) -> None:
        calls.append(cast(str, ctx["phase"]))

    with pytest.raises(GFQLValidationError):
        _mk_graph().gfql(
            "MATCH (a) MATCH ()-[a]->() RETURN a",
            policy={
                "postcompile": observe_postcompile,
                "prechain": observe_runtime,
                "precall": observe_runtime,
                "postcall": observe_runtime,
            },
        )

    assert calls == ["postcompile"]


def test_precompile_policy_fires_once_before_compile() -> None:
    calls: List[PolicyContext] = []

    def observe(ctx: PolicyContext) -> None:
        calls.append(ctx)

    result = _mk_graph().gfql(
        "UNWIND [1, 2, 3] AS x RETURN x ORDER BY x",
        policy={"precompile": observe},
    )

    assert len(calls) == 1
    ctx = calls[0]
    assert ctx["phase"] == "precompile"
    assert ctx["hook"] == "precompile"
    assert ctx["compile_language"] == "cypher"
    assert "compile" not in ctx
    assert "success" not in ctx
    assert result._nodes is not None


def test_precompile_policy_can_deny_before_invalid_query_compiles() -> None:
    calls: List[str] = []

    def deny(ctx: PolicyContext) -> None:
        calls.append(cast(str, ctx["phase"]))
        raise PolicyException("precompile", "compiler disabled")

    with pytest.raises(PolicyException) as exc_info:
        _mk_graph().gfql("THIS IS NOT CYPHER", policy={"precompile": deny})

    assert calls == ["precompile"]
    assert exc_info.value.phase == "precompile"
    assert exc_info.value.query_type == "chain"
