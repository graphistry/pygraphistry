from __future__ import annotations

from typing import List, cast

import pandas as pd
import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.policy import CompileErrorSummary, PolicyContext
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


def test_compile_error_policy_fires_once_with_structured_binder_payload() -> None:
    calls: List[PolicyContext] = []

    def observe(ctx: PolicyContext) -> None:
        calls.append(ctx)

    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_graph().gfql(
            "MATCH (a) MATCH ()-[a]->() RETURN a",
            policy={"compile_error": observe},
        )

    assert exc_info.value.code == ErrorCode.E204
    assert len(calls) == 1
    ctx = calls[0]
    assert ctx["phase"] == "compile_error"
    assert ctx["hook"] == "compile_error"
    assert ctx["compile_language"] == "cypher"
    summary = ctx["compile_error"]
    assert isinstance(summary, CompileErrorSummary)
    assert summary.language == "cypher"
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


def test_compile_error_policy_absent_preserves_existing_error_surface() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_graph().gfql("MATCH (a) MATCH ()-[a]->() RETURN a")

    assert exc_info.value.code == ErrorCode.E204
    assert exc_info.value.context["existing_kind"] == "node"
    assert exc_info.value.context["new_kind"] == "edge"
    assert exc_info.value.context["new_role"] == "relationship pattern"


def test_compile_error_policy_fires_for_validate_true_preflight() -> None:
    calls: List[PolicyContext] = []

    def observe(ctx: PolicyContext) -> None:
        calls.append(ctx)

    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_graph().gfql(
            "MATCH (a) MATCH ()-[a]->() RETURN a",
            validate=True,
            policy={"compile_error": observe},
        )

    assert exc_info.value.code == ErrorCode.E204
    assert len(calls) == 1
    summary = calls[0]["compile_error"]
    assert isinstance(summary, CompileErrorSummary)
    assert summary.compiler_phase == "bind"
    assert summary.context["existing_kind"] == "node"
    assert summary.context["new_kind"] == "edge"


def test_compile_error_policy_does_not_fire_for_successful_row_hot_loop_query() -> None:
    calls: List[PolicyContext] = []

    def observe(ctx: PolicyContext) -> None:
        calls.append(ctx)

    result = _mk_graph().gfql(
        "UNWIND [1, 2, 3] AS x RETURN x ORDER BY x",
        policy={"compile_error": observe},
    )

    assert calls == []
    assert result._nodes is not None
    assert result._nodes[["x"]].to_dict(orient="records") == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_compile_error_policy_fires_before_runtime_hooks() -> None:
    calls: List[str] = []

    def observe_compile_error(ctx: PolicyContext) -> None:
        calls.append(cast(str, ctx["phase"]))

    def observe_runtime(ctx: PolicyContext) -> None:
        calls.append(cast(str, ctx["phase"]))

    with pytest.raises(GFQLValidationError):
        _mk_graph().gfql(
            "MATCH (a) MATCH ()-[a]->() RETURN a",
            policy={
                "compile_error": observe_compile_error,
                "prechain": observe_runtime,
                "precall": observe_runtime,
                "postcall": observe_runtime,
            },
        )

    assert calls == ["compile_error"]
