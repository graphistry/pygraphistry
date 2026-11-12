import json
import math
from typing import Dict, List, Optional

import pandas as pd

import graphistry
from graphistry import call
from graphistry.compute.semantic import bind_providers
import pytest
import requests

from graphistry.compute.semantic.contract import DecisionQuestion, check_decision_result
from graphistry.compute.semantic.ops import sem_decide
from graphistry.compute.semantic.providers.chat import OpenAIChatProvider, Strategy, forget_probes
from graphistry.compute.semantic.providers.chat_constraints import GRAMMAR_FIELDS
from graphistry.tests.compute.semantic.http_test_utils import CHOICE, PRED, SCORE, FakeResponse, FakeSession, request


GRAMMAR_STRATEGIES: List[Strategy] = [
    "vllm_choice", "vllm_regex", "vllm_grammar", "guided_choice", "guided_regex", "guided_grammar",
    "sglang_regex", "sglang_ebnf",
]


@pytest.fixture(autouse=True)
def fresh_probes():
    forget_probes()
    yield
    forget_probes()


def response(text: Optional[str] = "B", *, function: Optional[Dict[str, object]] = None,
             finish: str = "stop", refused: bool = False) -> FakeResponse:
    message: Dict[str, object] = {"role": "assistant", "content": text}
    if function is not None:
        message["function_call"] = function
    if refused:
        message["refusal"] = "refused"
    tokens = [{"token": "B", "logprob": math.log(0.9), "top_logprobs": [
        {"token": "A", "logprob": math.log(0.1)}, {"token": "B", "logprob": math.log(0.9)}]}]
    return FakeResponse(200, {"choices": [{"message": message, "finish_reason": finish,
                                         "logprobs": {"content": tokens}}]})


def provider(session: FakeSession, strategy: Strategy, profile: Optional[List[Strategy]] = None) -> OpenAIChatProvider:
    return OpenAIChatProvider("test-key", "test-model", session=session, strategy=strategy,
                              probe_strategies=profile, max_retries=0)


def posted_body(session: FakeSession, index: int = 0) -> Dict[str, object]:
    body = session.posts[index]["json"]
    assert isinstance(body, dict)
    return body


@pytest.mark.parametrize("strategy", GRAMMAR_STRATEGIES)
@pytest.mark.parametrize("question,expected", [(PRED, False), (CHOICE, "tech"), (SCORE, "mid")])
def test_grammar_wire_and_all_question_kinds_are_value_only(strategy: Strategy, question: DecisionQuestion,
                                                         expected: object) -> None:
    session = FakeSession([response()])
    instance = provider(session, strategy)
    req = request((question,))
    result = instance.decide_batch([req], deadline_s=None, budget=None)[0]
    assert check_decision_result(req, question, result, instance.capabilities()) == []
    assert (result.status, result.value, result.distribution, result.probability_kind) == ("ok", expected, None, "none")
    assert result.extra == {"answer_strategy": strategy, "semantic_level": "grammar_structured"}
    body = posted_body(session)
    field, kind = GRAMMAR_FIELDS[strategy]
    codes = "AB" if question.kind == "predicate" else "ABC"
    expected_constraint = (list(codes) if kind == "choice" else "(" + "|".join(codes) + ")" if kind == "regex"
                           else "root ::= " + " | ".join(json.dumps(code) for code in codes))
    assert body[field] == ({kind: expected_constraint} if field == "structured_outputs" else expected_constraint)
    assert not {"logprobs", "top_logprobs", "response_format", "tools", "functions"} & set(body)
    assert len(session.posts) == 1


@pytest.mark.parametrize("strategy", GRAMMAR_STRATEGIES)
@pytest.mark.parametrize("text", ["A. yes", "b", "AB", '{"answer":"B"}', "Z", "", None])
def test_grammar_replies_must_be_exact_in_domain_codes(strategy: Strategy, text: Optional[str]) -> None:
    session = FakeSession([response(text)])
    result = provider(session, strategy).decide_batch([request((PRED,))], deadline_s=None, budget=None)[0]
    assert (result.status, result.error_code, result.value, result.distribution) == (
        "error", "invalid_provider_result", None, None)


@pytest.mark.parametrize("strategy", ["json_mode", "function"])
@pytest.mark.parametrize("question,expected", [(PRED, False), (CHOICE, "tech"), (SCORE, "mid")])
def test_legacy_json_and_function_use_the_same_code_domain(strategy: Strategy, question: DecisionQuestion,
                                                        expected: object) -> None:
    reply = response('{"answer":"B"}') if strategy == "json_mode" else response(
        None, function={"name": "answer", "arguments": '{"answer":"B"}'})
    session = FakeSession([reply])
    result = provider(session, strategy).decide_batch([request((question,))], deadline_s=None, budget=None)[0]
    assert (result.status, result.value, result.distribution, result.probability_kind) == ("ok", expected, None, "none")
    body = posted_body(session)
    if strategy == "json_mode":
        assert body["response_format"] == {"type": "json_object"}
        assert "JSON" in str(body["messages"])
    else:
        assert body["function_call"] == {"name": "answer"}
        functions = body["functions"]
        assert isinstance(functions, list) and len(functions) == 1
        assert functions[0]["parameters"]["properties"]["answer"]["enum"] == (
            ["A", "B"] if question.kind == "predicate" else ["A", "B", "C"])


@pytest.mark.parametrize("strategy", ["json_schema", "json_mode", "function"])
@pytest.mark.parametrize("arguments", [
    '{"answer":"B","confidence":0.9}', '{"answer":"A","answer":"B"}', '{"other":"B"}',
    '{"answer":true}', '{"answer":"b"}', '{"answer":" B"}', '{"answer":"A. yes"}', "[]", "B",
])
def test_structured_replies_validate_shape_and_exact_enum(strategy: Strategy, arguments: str) -> None:
    reply = response(arguments) if strategy != "function" else response(
        None, function={"name": "answer", "arguments": arguments})
    result = provider(FakeSession([reply]), strategy).decide_batch(
        [request((PRED,))], deadline_s=None, budget=None)[0]
    assert result.error_code == "invalid_provider_result" and result.value is None


def test_wrong_legacy_function_name_is_invalid() -> None:
    session = FakeSession([response(None, function={"name": "other", "arguments": '{"answer":"A"}'})])
    assert provider(session, "function").decide_batch(
        [request((PRED,))], deadline_s=None, budget=None)[0].error_code == "invalid_provider_result"


@pytest.mark.parametrize("strategy", GRAMMAR_STRATEGIES + ["json_mode", "function"])
def test_refusal_and_truncation_never_become_answers(strategy: Strategy) -> None:
    for reply, status, reason in [(response(refused=True), "uncertain", "refusal"),
                                  (response(finish="length"), "error", None)]:
        result = provider(FakeSession([reply]), strategy).decide_batch(
            [request((PRED,))], deadline_s=None, budget=None)[0]
        assert (result.status, result.value, result.distribution, result.status_reason) == (status, None, None, reason)


def test_constraint_extras_cannot_change_the_selected_mechanism() -> None:
    session = FakeSession([response()])
    instance = OpenAIChatProvider("test-key", "test-model", strategy="vllm_choice", session=session,
                                  extra_body={"logprobs": True, "top_logprobs": 20, "regex": "Z",
                                              "response_format": {"type": "json_object"}, "temperature": 0})
    result = instance.decide_batch([request((PRED,))], deadline_s=None, budget=None)[0]
    body = posted_body(session)
    assert not {"logprobs", "top_logprobs", "regex", "response_format"} & set(body)
    assert body["temperature"] == 0 and result.probability_kind == "none"


def test_explicit_profile_probes_and_keeps_a_legacy_strategy() -> None:
    session = FakeSession([FakeResponse(400), response("A"), response("B")])
    instance = provider(session, "auto", ["vllm_choice", "guided_choice", "text"])
    result = instance.decide_batch([request((PRED,))], deadline_s=10, budget=None)[0]
    assert instance.kept_strategy() == "guided_choice" and result.value is False
    assert "structured_outputs" in posted_body(session, 0)
    assert "guided_choice" in posted_body(session, 1) and "guided_choice" in posted_body(session, 2)
    again = FakeSession([response()])
    provider(again, "auto", ["vllm_choice", "guided_choice", "text"]).decide_batch(
        [request((PRED,))], deadline_s=None, budget=None)
    assert len(again.posts) == 1


def test_profile_order_and_membership_isolate_cache_entries() -> None:
    first = provider(FakeSession([response("A")]), "auto", ["vllm_choice", "text"])
    first.decide_batch([request((PRED,))], deadline_s=None, budget=None)
    for profile in [["text", "vllm_choice"], ["guided_choice", "text"]]:
        session = FakeSession([response("A")])
        other = provider(session, "auto", profile)
        assert first.probe_key() != other.probe_key()
        other.decide_batch([request((PRED,))], deadline_s=None, budget=None)
        assert len(session.posts) == 2


def test_default_profile_preserves_the_existing_four_rungs() -> None:
    instance = provider(FakeSession([]), "auto")
    assert instance.probe_strategies == ("logprobs", "json_schema", "tool", "text")
    with pytest.raises(ValueError):
        provider(FakeSession([]), "auto", [])
    with pytest.raises(ValueError):
        provider(FakeSession([]), "auto", ["auto"])


@pytest.mark.parametrize("code", [401, 403, 407, 408, 409, 429, 500, 503])
def test_probe_transport_failures_do_not_send_user_evidence(code: int) -> None:
    session = FakeSession([FakeResponse(code), response("A")])
    results = provider(session, "auto", ["vllm_choice", "text"]).decide_batch(
        [request((PRED, CHOICE))], deadline_s=None, budget=None)
    assert len(session.posts) == 1
    assert [r.error_code for r in results] == [f"http_{code}", f"http_{code}"]
    assert "a box arrived crushed" not in str(posted_body(session))


def test_connection_failure_after_probe_is_not_a_data_retry() -> None:
    session = FakeSession([requests.ConnectionError(), response("A")])
    result = provider(session, "auto", ["vllm_choice", "text"]).decide_batch(
        [request((PRED,))], deadline_s=None, budget=None)[0]
    assert result.error_code == "transport_error" and len(session.posts) == 1


def test_profile_without_text_never_silently_sends_unconstrained_data() -> None:
    session = FakeSession([response("not a code"), response("A")])
    result = provider(session, "auto", ["vllm_choice"]).decide_batch(
        [request((PRED,))], deadline_s=None, budget=None)[0]
    assert result.error_code == "invalid_provider_result" and len(session.posts) == 1


def test_grammar_at_the_kernel_preserves_duplicate_indexes_and_min_p() -> None:
    frame = pd.DataFrame({"evidence": ["one", "two", None]}, index=[4, 4, 4])
    session = FakeSession([response("A"), response("B")])
    output = sem_decide(frame, questions=[PRED], provider=provider(session, "vllm_choice"), task_id="grammar",
                        evidence_cols=["evidence"], min_p=0.8, max_concurrency=1)
    assert output.index.tolist() == [4, 4, 4]
    assert output["p__status"].tolist() == ["uncertain", "uncertain", "skipped"]
    assert output["p__p"].isna().all() and output["p__probability_kind"].tolist() == ["none"] * 3
    assert len(session.posts) == 2 and frame.columns.tolist() == ["evidence"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_grammar_gfql_binding_keeps_rows_and_engine(engine: str) -> None:
    nodes = pd.DataFrame({"id": [1, 2], "evidence": ["one", "two"]})
    frame = pytest.importorskip("polars").from_pandas(nodes) if engine == "polars" else nodes
    session = FakeSession([response("A"), response("B")])
    instance = provider(session, "sglang_ebnf")
    with bind_providers({"grammar": instance}):
        output = graphistry.nodes(frame, "id").gfql([call("sem_decide", {
            "questions": [{"id": "decision", "kind": "predicate", "text": "Is it damaged?"}],
            "provider": "grammar", "task_id": "grammar", "evidence_cols": ["evidence"], "max_concurrency": 1,
        })], engine=engine)
    assert output._nodes is not None
    result = output._nodes.to_pandas() if engine == "polars" else output._nodes
    assert result["id"].tolist() == [1, 2] and result["decision"].tolist() == [True, False]
    assert result["decision__p"].isna().all() and len(session.posts) == 2
    assert type(output._nodes) is type(frame)


def test_request_parse_uses_a_snapshot_of_the_selected_strategy() -> None:
    instance = provider(FakeSession([]), "vllm_choice")
    def change_strategy(body: Dict[str, object]) -> FakeResponse:
        instance.strategy_in_use = "logprobs"
        return response()
    session = FakeSession([change_strategy])
    instance._session = session
    result = instance.decide_batch([request((PRED,))], deadline_s=None, budget=None)[0]
    assert result.value is False and result.probability_kind == "none"
    assert result.extra == {"answer_strategy": "vllm_choice", "semantic_level": "grammar_structured"}


def test_an_expired_profile_sends_no_probe_or_row() -> None:
    session = FakeSession([response()])
    result = provider(session, "auto", ["vllm_choice", "text"]).decide_batch(
        [request((PRED,))], deadline_s=0.0, budget=None)[0]
    assert result.error_code == "deadline_exceeded" and session.posts == []
