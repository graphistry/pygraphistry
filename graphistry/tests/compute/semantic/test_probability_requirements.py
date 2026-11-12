import json
import math
import os
from types import SimpleNamespace
from typing import Dict, Literal, Optional, Sequence

import pandas as pd
import pytest

import graphistry
from graphistry import call
from graphistry.compute.ast import ASTCall
from graphistry.compute.exceptions import ErrorCode, GFQLCallExecutionError, GFQLTypeError
from graphistry.compute.semantic import bind_providers
from graphistry.compute.semantic.contract import DecisionRequest, ProbabilityKind
from graphistry.compute.semantic.ops import sem_decide
from graphistry.compute.semantic.providers.callable import CallableDecisionProvider
from graphistry.compute.semantic.providers.chat import AnthropicMessagesProvider, OpenAIChatProvider, Strategy, forget_probes
from graphistry.compute.semantic.providers.langchain import LangChainDecisionProvider
from graphistry.compute.semantic.providers.openai_decisions import OpenAIDecisionsProvider
from graphistry.compute.semantic.rules import ProbabilityRequirement
from graphistry.tests.compute.semantic.http_test_utils import FakeResponse, FakeSession
from graphistry.tests.compute.semantic.probability_test_utils import QUESTION, check_probability_policy
from graphistry.tests.compute.semantic.semantic_test_utils import C, P, S, Recording, Scripted, caps, frame, full_answer, result
from graphistry.tests.compute.semantic.test_gfql_bindings import SEL, decide_params, polars_graph, run
from graphistry.tests.compute.semantic.test_langchain_provider import FakeModel, Message, envelope, tool


@pytest.mark.parametrize("requirement", [False, True, "native"])
@pytest.mark.parametrize("kind", ["none", "verbalized", "token_logprob", "native_decision"])
def test_shared_policy_accepts_only_requested_probability_provenance(requirement: ProbabilityRequirement,
                                                                  kind: ProbabilityKind) -> None:
    provider = Recording(full_answer if kind != "none" else lambda e, q: True, probability_kind=kind)
    check_probability_policy(provider, requirement, kind)
    assert provider.answers == 1


@pytest.mark.parametrize("question", [P, C, S])
@pytest.mark.parametrize("requirement", [False, True, "native"])
@pytest.mark.parametrize("kind", ["verbalized", "token_logprob", "native_decision"])
def test_probability_requirement_covers_every_question_kind(question: Dict[str, object],
                                                           requirement: ProbabilityRequirement,
                                                           kind: ProbabilityKind) -> None:
    provider = Recording(full_answer, probability_kind=kind)
    out = sem_decide(frame(1), questions=[question], provider=provider, task_id="t", evidence_cols=["name"],
                     require_probabilities=requirement)
    accepted = requirement is False or (kind in ("native_decision", "token_logprob") and (
        requirement is True or kind == "native_decision"))
    assert out[str(question["id"]) + "__status"].tolist() == (["ok"] if accepted else ["error"])


def test_required_probability_must_include_the_returned_choice_and_clears_confidence() -> None:
    def script(index: int, requests: Sequence[DecisionRequest]):
        return [result(req, "c", "a", {"b": .4}, probability_kind="token_logprob", confidence=.99)
                for req in requests]
    provider = Scripted(script)
    out = sem_decide(frame(1), questions=[C], provider=provider, task_id="t", evidence_cols=["name"],
                     require_probabilities=True, min_p=.2)
    assert out["c__error_code"].tolist() == ["probabilities_unavailable"]
    assert out["c"].isna().all() and out["c__dist"].isna().all() and out["c__p"].isna().all()
    meta = json.loads(out["c__meta"].iloc[0])
    assert meta["confidence"] is None
    assert meta["cost_usd"] == 0 and meta["latency_ms"] == 1


@pytest.mark.parametrize("kind", ["token_logprob", "native_decision"])
def test_zero_probability_is_available(kind: ProbabilityKind) -> None:
    provider = Recording(lambda e, q: (True, {"true": 0.0, "false": 1.0}), probability_kind=kind)
    out = check_probability_policy(provider, True, kind)
    assert out["p__p"].tolist() == [0.0]


@pytest.mark.parametrize("requirement", [True, "native"])
@pytest.mark.parametrize("status,reason,error", [("uncertain", "refusal", None), ("uncertain", "null_answer", None),
                                              ("error", None, "http_401"), ("skipped", "budget", None)])
def test_non_success_verdicts_are_not_overwritten(requirement: ProbabilityRequirement, status: str,
                                               reason: Optional[str], error: Optional[str]) -> None:
    provider = Scripted(lambda index, requests: [result(req, "p", None, status=status, status_reason=reason,
                                                      error_code=error) for req in requests])
    out = sem_decide(frame(1), questions=[P], provider=provider, task_id="t", evidence_cols=["name"],
                     require_probabilities=requirement, min_p=.9)
    assert out["p__status"].tolist() == [status]
    assert out["p__error_code"].iloc[0] == error or (error is None and pd.isna(out["p__error_code"].iloc[0]))
    assert json.loads(out["p__meta"].iloc[0])["status_reason"] == reason


def test_ties_and_invalid_alignment_or_content_keep_their_original_verdict() -> None:
    provider = Scripted(lambda index, requests: [result(req, "p", None, {"true": .5, "false": .5},
                                                      status="uncertain", status_reason="tie") for req in requests])
    out = sem_decide(frame(1), questions=[P], provider=provider, task_id="t", evidence_cols=["name"],
                     require_probabilities="native")
    assert out["p__status"].tolist() == ["uncertain"]
    for scripted in [lambda i, rs: [result(rs[0], "p", "wrong")],
                     lambda i, rs: [result(rs[0], "p", True), result(rs[0], "p", True)]]:
        invalid = sem_decide(frame(1), questions=[P], provider=Scripted(scripted), task_id="t", evidence_cols=["name"],
                             require_probabilities=True)
        assert invalid["p__error_code"].tolist() == ["invalid_provider_result"]


@pytest.mark.parametrize("requirement", [False, True, "native"])
@pytest.mark.parametrize("reason", [None, "min_p"])
def test_uncertain_value_cannot_bypass_probability_requirement(requirement: ProbabilityRequirement,
                                                             reason: Optional[str]) -> None:
    provider = Scripted(lambda index, requests: [result(req, "p", True, status="uncertain", status_reason=reason)
                                               for req in requests])
    out = sem_decide(frame(1), questions=[P], provider=provider, task_id="t", evidence_cols=["name"],
                     require_probabilities=requirement)
    assert out["p__status"].tolist() == (["uncertain"] if requirement is False else ["error"])
    if requirement is not False:
        assert out["p"].isna().all() and out["p__error_code"].tolist() == ["probabilities_unavailable"]


def test_min_p_runs_after_requirement_and_default_is_unchanged() -> None:
    for requirement, expected in [(False, "uncertain"), (True, "error")]:
        out = sem_decide(frame(1), questions=[P], provider=Recording(), task_id="t", evidence_cols=["name"],
                         require_probabilities=requirement, min_p=.5)
        assert out["p__status"].tolist() == [expected]
    out = sem_decide(frame(1), questions=[P], provider=Recording(full_answer, probability_kind="token_logprob"),
                     task_id="t", evidence_cols=["name"], require_probabilities=True, min_p=.95)
    assert out["p__status"].tolist() == ["uncertain"] and out["p__p"].notna().all()
    default = sem_decide(frame(1), questions=[P], provider=Recording(), task_id="t", evidence_cols=["name"])
    explicit = sem_decide(frame(1), questions=[P], provider=Recording(), task_id="t", evidence_cols=["name"],
                          require_probabilities=False)
    pd.testing.assert_frame_equal(default.drop(columns="p__meta"), explicit.drop(columns="p__meta"))


def test_mixed_batch_checks_each_answer_without_poisoning_other_rows_or_questions() -> None:
    def script(index: int, requests: Sequence[DecisionRequest]):
        answers = []
        for req in requests:
            for q in req.questions:
                if req.request_id == "1":
                    answers.append(result(req, q.id, None, status="uncertain", status_reason="refusal"))
                elif q.id == "p":
                    answers.append(result(req, q.id, True, {"true": .8, "false": .2}))
                elif q.id == "c":
                    answers.append(result(req, q.id, "a", {"b": .4}))
                else:
                    answers.append(result(req, q.id, "low"))
        return list(reversed(answers))
    data = frame(2)
    data.index = [7, 7]
    before = data.copy(deep=True)
    out = sem_decide(data, questions=[P, C, S], provider=Scripted(script), task_id="t", evidence_cols=["name"],
                     require_probabilities=True)
    assert out.index.tolist() == [7, 7]
    assert out["p__status"].tolist() == ["ok", "uncertain"]
    assert out["c__status"].tolist() == out["s__status"].tolist() == ["error", "uncertain"]
    assert out["c__error_code"].iloc[0] == out["s__error_code"].iloc[0] == "probabilities_unavailable"
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("requirement", [False, True, "native"])
@pytest.mark.parametrize("kind", ["none", "token_logprob", "native_decision"])
def test_provider_metadata_must_be_valid_before_probability_policy_can_clear_it(
        requirement: ProbabilityRequirement, kind: ProbabilityKind) -> None:
    class InvalidConfidence:
        def __str__(self) -> str:
            raise ValueError("synthetic invalid metadata")
    distribution = None if kind == "none" else {"true": .8, "false": .2}
    provider = Scripted(lambda index, requests: [result(req, "p", True, distribution, probability_kind=kind,
                                                      confidence=InvalidConfidence()) for req in requests])
    out = sem_decide(frame(1), questions=[P], provider=provider, task_id="t", evidence_cols=["name"],
                     require_probabilities=requirement)
    assert out["p__error_code"].tolist() == ["invalid_provider_result"] and out["p"].isna().all()


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", "token_logprob", [], {}, float("nan")])
def test_requirement_parameter_is_strict_and_validated_before_calls(invalid: object) -> None:
    provider = Recording()
    with pytest.raises(ValueError):
        sem_decide(frame(1), questions=[P], provider=provider, task_id="t", evidence_cols=["name"],
                   require_probabilities=invalid)
    with bind_providers({"p": provider}), pytest.raises(GFQLTypeError) as info:
        graphistry.nodes(frame(1)).gfql([call("sem_decide", {
            "questions": [P], "provider": "p", "task_id": "t", "evidence_cols": ["name"],
            "require_probabilities": invalid})])
    assert info.value.code == ErrorCode.E201 and provider.answers == 0


@pytest.mark.parametrize("strategy", ["text", "json_schema", "json_mode", "tool", "function", "vllm_choice",
                                      "vllm_regex", "vllm_grammar", "guided_choice", "guided_regex", "guided_grammar",
                                      "sglang_regex", "sglang_ebnf"])
@pytest.mark.parametrize("requirement", [False, True, "native"])
def test_all_constrained_chat_strategies_never_upgrade_semantics(strategy: Strategy,
                                                               requirement: ProbabilityRequirement) -> None:
    message: Dict[str, object] = {"content": '{"answer":"A"}' if strategy in ("json_schema", "json_mode") else "A"}
    if strategy in ("function", "tool"):
        function = {"name": "answer", "arguments": '{"answer":"A"}'}
        message = {"function_call": function} if strategy == "function" else {"tool_calls": [{"function": function}]}
    session = FakeSession([FakeResponse(200, {"choices": [{"message": message, "finish_reason": "stop"}]})])
    provider = OpenAIChatProvider("dummy-key", "model", session=session, strategy=strategy, max_retries=0)
    check_probability_policy(provider, requirement, "none")
    assert len(session.posts) == (1 if requirement is False else 0)


@pytest.mark.parametrize("requirement", [False, True, "native"])
def test_native_and_actual_token_logprob_paths_match_shared_policy(requirement: ProbabilityRequirement) -> None:
    native = OpenAIDecisionsProvider("dummy-key", "model", session=FakeSession([FakeResponse(200, {
        "answers": [{"type": "predicate", "name": "p", "probability": .7}]})]), max_retries=0)
    check_probability_policy(native, requirement, "native_decision")
    alternatives = [{"token": "A", "logprob": math.log(.7)}, {"token": "B", "logprob": math.log(.3)}]
    session = FakeSession([FakeResponse(200, {"choices": [{"message": {"content": "A"}, "finish_reason": "stop",
        "logprobs": {"content": [{"token": "A", "logprob": math.log(.7), "top_logprobs": alternatives}]}}]})])
    check_probability_policy(OpenAIChatProvider("dummy-key", "model", session=session, max_retries=0),
                             requirement, "token_logprob")
    assert len(session.posts) == (0 if requirement == "native" else 1)


@pytest.mark.parametrize("requirement", [False, True, "native"])
@pytest.mark.parametrize("strategy", ["text", "json_schema", "tool"])
def test_anthropic_never_upgrades_semantics(requirement: ProbabilityRequirement,
                                          strategy: Literal["text", "json_schema", "tool"]) -> None:
    block = {"type": "text", "text": "A" if strategy == "text" else '{"answer":"A"}'}
    if strategy == "tool":
        block = {"type": "tool_use", "name": "answer", "input": {"answer": "A"}}
    session = FakeSession([FakeResponse(200, {"content": [block], "stop_reason": "end_turn"})])
    check_probability_policy(AnthropicMessagesProvider("dummy-key", "model", strategy=strategy,
                                                      session=session, max_retries=0), requirement, "none")
    assert len(session.posts) == (1 if requirement is False else 0)


@pytest.mark.parametrize("requirement", [False, True, "native"])
def test_langchain_and_auto_fallback_never_upgrade_semantics(requirement: ProbabilityRequirement) -> None:
    model = FakeModel(envelope())
    check_probability_policy(LangChainDecisionProvider(model), requirement, "none")
    assert len(model.invocations) == (1 if requirement is False else 0)
    forget_probes()
    try:
        session = FakeSession([FakeResponse(400), FakeResponse(200, {"choices": [
            {"message": {"content": '{"answer":"A"}'}, "finish_reason": "stop"}]})])
        provider = OpenAIChatProvider("dummy-key", "auto-model", session=session, strategy="auto", max_retries=0)
        check_probability_policy(provider, requirement, "none")
        assert len(session.posts) == (0 if requirement == "native" else 3)
    finally:
        forget_probes()


def test_known_gap_preflight_has_no_estimate_spend_or_request_budget_and_keeps_skips(monkeypatch) -> None:
    session = FakeSession([])
    provider = OpenAIChatProvider("dummy-key", "model", strategy="text", session=session)
    def forbidden_estimate(requests: Sequence[DecisionRequest]) -> Optional[float]:
        raise AssertionError("estimate must not receive evidence during capability preflight")
    monkeypatch.setattr(provider, "estimate_cost_usd", forbidden_estimate)
    data = pd.DataFrame({"name": [None, "row1", "row2", "row3"]})
    out = sem_decide(data, questions=[P], provider=provider, task_id="t", evidence_cols=["name"], max_rows=3,
                     require_probabilities=True, budget={"max_cost_usd": 0, "max_requests": 1})
    assert out["p__status"].tolist() == ["skipped", "error", "error", "skipped"]
    metas = [json.loads(s) for s in out["p__meta"]]
    assert [metas[i]["status_reason"] for i in (0, 3)] == ["null_evidence", "max_rows"]
    assert metas[1]["cost_usd"] == 0 and metas[1]["retry_count"] == 0 and session.posts == []


def test_custom_chat_subclass_is_checked_from_actual_results() -> None:
    class CustomChat(OpenAIChatProvider):
        def decide_batch(self, requests, *, deadline_s, budget):
            return [result(req, question.id, None, status="uncertain", status_reason="refusal")
                    for req in requests for question in req.questions]
    session = FakeSession([])
    out = sem_decide(frame(1), questions=[P], provider=CustomChat("dummy-key", "model", strategy="text", session=session),
                     task_id="t", evidence_cols=["name"], require_probabilities=True)
    assert out["p__status"].tolist() == ["uncertain"] and session.posts == []


def test_auto_profile_with_only_value_mechanisms_is_rejected_without_probe() -> None:
    session = FakeSession([])
    provider = OpenAIChatProvider("dummy-key", "model", strategy="auto", probe_strategies=("json_schema", "text"),
                                  session=session)
    check_probability_policy(provider, True, "none")
    assert session.posts == []


@pytest.mark.parametrize("requirement", [False, True, "native"])
def test_requirement_round_trips_through_call_json(requirement: ProbabilityRequirement) -> None:
    node = call("sem_decide", {"questions": [P], "provider": "p", "task_id": "t", "evidence_cols": ["name"],
                               "require_probabilities": requirement})
    wire = json.loads(json.dumps(node.to_json()))
    restored = ASTCall.from_json(wire)
    assert restored.params["require_probabilities"] == requirement
    provider = Recording(full_answer, probability_kind="native_decision")
    with bind_providers({"p": provider}):
        out = graphistry.nodes(frame(1)).gfql([restored])
    assert out._nodes["p__status"].tolist() == ["ok"]


def test_known_gap_preflight_obeys_reached_deadline(monkeypatch) -> None:
    import graphistry.compute.semantic.ops as ops
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(ops, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    session = FakeSession([])
    out = sem_decide(frame(1), questions=[P], provider=OpenAIChatProvider("dummy-key", "model", strategy="text",
                                                                 session=session), task_id="t", evidence_cols=["name"],
                     deadline_s=1, require_probabilities=True)
    assert out["p__status"].tolist() == ["skipped"]
    assert json.loads(out["p__meta"].iloc[0])["status_reason"] == "deadline" and session.posts == []


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf", "polars-gpu"])
def test_requirement_reaches_methods_and_gfql_without_mutating_frames(engine: str) -> None:
    if engine in ("cudf", "polars-gpu") and os.environ.get("TEST_CUDF") != "1":
        pytest.skip("GPU tests require TEST_CUDF=1")
    data = frame(2).assign(id=[2, 1])
    before = data.copy(deep=True)
    nodes = data
    if engine in ("polars", "polars-gpu"):
        nodes = pytest.importorskip("polars").from_pandas(data)
    elif engine == "cudf":
        nodes = pytest.importorskip("cudf").from_pandas(data)
    provider = Recording(full_answer, probability_kind="token_logprob")
    with bind_providers({"p": provider}):
        graph = graphistry.nodes(nodes, "id")
        direct = graph.sem_decide([P], "p", task_id="t", evidence_cols=["name"], require_probabilities="native")
        called = graph.gfql([call("sem_decide", {"questions": [P], "provider": "p", "task_id": "t",
            "evidence_cols": ["name"], "require_probabilities": "native"})], engine=engine)
    for output in [direct._nodes, called._nodes]:
        host = output if isinstance(output, pd.DataFrame) else output.to_pandas()
        assert list(host["id"]) == [2, 1] and host["p__error_code"].tolist() == ["probabilities_unavailable"] * 2
    pd.testing.assert_frame_equal(data, before)


def test_probability_rejection_after_a_provider_call_retains_effect_barrier() -> None:
    provider = Recording()
    with pytest.raises(GFQLCallExecutionError):
        run(provider, [call("sem_decide", decide_params(require_probabilities=True))] + SEL, g=polars_graph())
    assert provider.answers == 3


def test_local_preflight_does_not_claim_external_side_effect() -> None:
    session = FakeSession([])
    provider = OpenAIChatProvider("dummy-key", "model", strategy="text", session=session)
    out = run(provider, [call("sem_decide", decide_params(require_probabilities=True))] + SEL, g=polars_graph())
    assert session.posts == [] and type(out._nodes).__module__.startswith("pandas")
