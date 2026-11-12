import json
from typing import Dict, List

import pandas as pd
import pytest

import graphistry
from graphistry import call
from graphistry.compute.semantic import bind_providers
from graphistry.compute.semantic.contract import DecisionQuestion, check_decision_result
from graphistry.compute.semantic.providers.chat import AnthropicMessagesProvider, AnthropicStrategy
from graphistry.compute.semantic.providers.http import TokenPrices
from graphistry.tests.compute.semantic.http_test_utils import CHOICE, PRED, SCORE, FakeResponse, FakeSession, request


def text(value: object) -> Dict[str, object]:
    return {"type": "text", "text": value}


def tool(value: object, name: str = "answer") -> Dict[str, object]:
    return {"type": "tool_use", "id": "call-1", "name": name, "input": value}


def response(blocks: List[Dict[str, object]], stop: str = "end_turn") -> FakeResponse:
    return FakeResponse(200, {"model": "served", "content": blocks, "stop_reason": stop,
                              "usage": {"input_tokens": 10, "output_tokens": 5}})


def provider(session: FakeSession, strategy: AnthropicStrategy) -> AnthropicMessagesProvider:
    return AnthropicMessagesProvider("dummy-key", "configured-model", strategy=strategy, session=session,
                                      max_retries=0, prices=TokenPrices(input=1, cached_input=0.5, output=2))


def body(session: FakeSession, index: int = 0) -> Dict[str, object]:
    result = session.posts[index]["json"]
    assert isinstance(result, dict)
    return result


@pytest.mark.parametrize("strategy", ["json_schema", "tool"])
@pytest.mark.parametrize("question,expected", [(PRED, False), (CHOICE, "tech"), (SCORE, "mid")])
def test_structured_and_tool_answers_match_all_question_domains(strategy: AnthropicStrategy,
                                                              question: DecisionQuestion, expected: object) -> None:
    reply = response([text('{"answer":"B"}')]) if strategy == "json_schema" else response(
        [tool({"answer": "B"})], "tool_use")
    session = FakeSession([reply])
    instance = provider(session, strategy)
    req = request((question,))
    result = instance.decide_batch([req], deadline_s=None, budget=None)[0]
    assert check_decision_result(req, question, result, instance.capabilities()) == []
    assert (result.status, result.value, result.distribution, result.probability_kind) == ("ok", expected, None, "none")
    assert result.confidence is None and result.cost_usd == pytest.approx(20/1e6)
    assert result.extra == {"cached_input_tokens": 0, "answer_strategy": strategy,
                            "semantic_level": "provider_structured" if strategy == "json_schema" else "tool_structured"}
    sent = body(session)
    if strategy == "json_schema":
        configured = sent["output_config"]
        assert isinstance(configured, dict)
        schema = configured["format"]["schema"]
        assert "tools" not in sent and "tool_choice" not in sent and "output_format" not in sent
    else:
        tools = sent["tools"]
        assert isinstance(tools, list) and len(tools) == 1
        schema = tools[0]["input_schema"]
        assert sent["tool_choice"] == {"type": "tool", "name": "answer"}
        assert "output_config" not in sent
    assert schema["additionalProperties"] is False and schema["required"] == ["answer"]
    assert schema["properties"]["answer"]["enum"] == (["A", "B"] if question.kind == "predicate" else ["A", "B", "C"])
    assert len(session.posts) == 1 and "dummy-key" not in str(sent)


@pytest.mark.parametrize("value", [{"answer": "b"}, {"answer": "A. yes"}, {"answer": True}, {"answer": "Z"},
                                   {"answer": "A", "confidence": 0.99}, {"other": "A"}, [], "A", None])
@pytest.mark.parametrize("strategy", ["json_schema", "tool"])
def test_generated_confidence_and_invalid_shapes_are_rejected(strategy: AnthropicStrategy, value: object) -> None:
    reply = response([text(json.dumps(value))]) if strategy == "json_schema" else response([tool(value)], "tool_use")
    result = provider(FakeSession([reply]), strategy).decide_batch(
        [request((PRED,))], deadline_s=None, budget=None)[0]
    assert (result.status, result.error_code, result.value, result.distribution, result.confidence) == (
        "error", "invalid_provider_result", None, None, None)


@pytest.mark.parametrize("strategy", ["json_schema", "tool"])
@pytest.mark.parametrize("stop", ["max_tokens", "model_context_window_exceeded", "pause_turn", "refusal"])
def test_refusal_or_unfinished_content_never_becomes_an_answer(strategy: AnthropicStrategy, stop: str) -> None:
    content = [text('{"answer":"A"}')] if strategy == "json_schema" else [tool({"answer": "A"})]
    result = provider(FakeSession([response(content, stop)]), strategy).decide_batch(
        [request((PRED,))], deadline_s=None, budget=None)[0]
    assert result.value is None and result.distribution is None
    assert result.status == ("uncertain" if stop == "refusal" else "error")


@pytest.mark.parametrize("blocks", [
    [tool({"answer": "A"}, "other")], [tool({"answer": "A"}), tool({"answer": "A"})],
    [tool({"answer": "A"}), tool({"answer": "B"})], [text('{"answer":"A"}')],
    [{"type": "server_tool_use", "name": "answer", "input": {"answer": "A"}}],
])
def test_only_one_named_client_tool_is_an_answer(blocks: List[Dict[str, object]]) -> None:
    result = provider(FakeSession([response(blocks, "tool_use")]), "tool").decide_batch(
        [request((PRED,))], deadline_s=None, budget=None)[0]
    assert result.error_code == "invalid_provider_result"


def test_json_joins_text_fragments_but_rejects_tools_and_duplicate_keys() -> None:
    session = FakeSession([
        response([{"type": "thinking", "thinking": "irrelevant"}, text('{"answer":'), text('"B"}')]),
        response([text('{"answer":"A"}'), tool({"answer": "B"})]),
        response([text('{"answer":"A","answer":"B"}')]),
    ])
    instance = provider(session, "json_schema")
    first = instance.decide_batch([request((PRED,))], deadline_s=None, budget=None)[0]
    assert first.value is False
    for _ in range(2):
        result = instance.decide_batch([request((PRED,))], deadline_s=None, budget=None)[0]
        assert result.error_code == "invalid_provider_result"


@pytest.mark.parametrize("status", [400, 401, 403, 429, 503])
def test_host_errors_are_preserved_without_switching_strategy(status: int) -> None:
    session = FakeSession([FakeResponse(status), response([text('{"answer":"A"}')])])
    result = provider(session, "tool").decide_batch([request((PRED,))], deadline_s=None, budget=None)[0]
    assert result.error_code == f"http_{status}" and len(session.posts) == 1


def test_cost_estimate_includes_the_selected_constraint_payload() -> None:
    req = request((CHOICE,))
    plain = provider(FakeSession([]), "text").estimate_cost_usd([req])
    for strategy in ("json_schema", "tool"):
        constrained = provider(FakeSession([]), strategy).estimate_cost_usd([req])
        assert plain is not None and constrained is not None and constrained > plain


def test_strategy_controls_format_fields_and_keeps_effort_settings() -> None:
    session = FakeSession([response([text('{"answer":"A"}')])])
    instance = AnthropicMessagesProvider("dummy", "model", session=session, strategy="json_schema",
                                         extra_body={"output_config": {"effort": "low", "format": {"type": "text"}},
                                                     "tools": [], "output_format": {}, "temperature": 0})
    instance.decide_batch([request((PRED,))], deadline_s=None, budget=None)
    sent = body(session)
    configured = sent["output_config"]
    assert isinstance(configured, dict) and configured["effort"] == "low"
    assert configured["format"]["type"] == "json_schema" and sent["temperature"] == 0
    assert "tools" not in sent and "output_format" not in sent


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_structured_anthropic_gfql_keeps_rows_and_missing_probabilities(engine: str) -> None:
    nodes = pd.DataFrame({"id": [1, 2], "evidence": ["one", "two"]})
    frame = pytest.importorskip("polars").from_pandas(nodes) if engine == "polars" else nodes
    session = FakeSession([response([tool({"answer": "A"})], "tool_use"),
                           response([tool({"answer": "B"})], "tool_use")])
    with bind_providers({"anthropic": provider(session, "tool")}):
        out = graphistry.nodes(frame, "id").gfql([call("sem_decide", {
            "provider": "anthropic", "questions": [{"id": "p", "kind": "predicate", "text": "Is it damaged?"}],
            "task_id": "test", "evidence_cols": ["evidence"], "max_concurrency": 1,
        })], engine=engine)
    assert out._nodes is not None
    result = out._nodes.to_pandas() if engine == "polars" else out._nodes
    assert result["p"].tolist() == [True, False] and result["id"].tolist() == [1, 2]
    assert result["p__p"].isna().all() and type(out._nodes) is type(frame)
