from dataclasses import asdict, dataclass, field
import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
import pytest

import graphistry
from graphistry import call
from graphistry.compute.semantic import bind_providers
from graphistry.compute.semantic.contract import DecisionQuestion, check_decision_result
from graphistry.compute.semantic.providers.langchain import LangChainDecisionProvider, LangChainStrategy
from graphistry.compute.semantic.providers.http import TokenPrices
from graphistry.tests.compute.semantic.http_test_utils import CHOICE, PRED, SCORE, request


@dataclass
class Message:
    content: object = ""
    tool_calls: object = field(default_factory=list)
    invalid_tool_calls: object = field(default_factory=list)
    additional_kwargs: Dict[str, object] = field(default_factory=dict)
    response_metadata: Dict[str, object] = field(default_factory=dict)
    usage_metadata: Optional[Dict[str, object]] = None


def envelope(code: object = "A", raw: object = None, error: object = None) -> Dict[str, object]:
    return {"raw": Message(json.dumps({"answer": code})) if raw is None else raw,
            "parsed": {"answer": code}, "parsing_error": error}


def tool(code: object = "A", name: str = "answer") -> Dict[str, object]:
    return {"name": name, "args": {"answer": code}, "id": "call-1"}


class FakeModel:
    def __init__(self, response: object, *, structured: bool = True, tools: bool = True) -> None:
        self.response, self.structured, self.tools = response, structured, tools
        self.bindings: List[Dict[str, object]] = []
        self.invocations: List[Dict[str, object]] = []

    def with_structured_output(self, schema: Dict[str, object], **kwargs: object) -> object:
        self.bindings.append({"schema": schema, **kwargs})
        if not self.structured:
            raise NotImplementedError()
        return self

    def bind_tools(self, tools: List[Dict[str, object]], **kwargs: object) -> object:
        self.bindings.append({"tools": tools, **kwargs})
        if not self.tools:
            raise NotImplementedError()
        return self

    def invoke(self, input: List[Dict[str, str]], **kwargs: object) -> object:
        self.invocations.append({"messages": input, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def answer(instance: LangChainDecisionProvider, question: DecisionQuestion = PRED):
    return instance.decide_batch([request((question,))], deadline_s=None, budget=None)[0]


@pytest.mark.parametrize("strategy", ["structured", "json_schema", "function_calling", "json_mode", "tool", "text"])
@pytest.mark.parametrize("question,expected", [(PRED, False), (CHOICE, "tech"), (SCORE, "mid")])
def test_all_domains_and_strategies_are_value_only(strategy: LangChainStrategy,
                                                 question: DecisionQuestion, expected: object) -> None:
    raw = (Message(content="B") if strategy == "text" else Message(content='{"answer":"B"}')
           if strategy in ("json_schema", "json_mode") else Message(tool_calls=[tool("B")]))
    response = raw if strategy in ("tool", "text") else envelope("B", raw)
    model = FakeModel(response)
    instance = LangChainDecisionProvider(model, strategy=strategy)
    result = answer(instance, question)
    assert check_decision_result(request((question,)), question, result, instance.capabilities()) == []
    assert (result.status, result.value, result.probability_kind, result.distribution, result.confidence) == (
        "ok", expected, "none", None, None)
    assert result.retry_count == 0 and result.extra["answer_strategy"] == strategy
    assert len(model.invocations) == 1
    assert [m["role"] for m in model.invocations[0]["messages"]] == ["system", "user"]
    if strategy not in ("tool", "text"):
        assert model.bindings[0]["include_raw"] is True
        assert model.bindings[0].get("method") == (None if strategy == "structured" else strategy)
        schema = model.bindings[0]["schema"]
        assert schema["additionalProperties"] is False and schema["required"] == ["answer"]
        assert schema["properties"]["answer"]["enum"] == (["A", "B"] if question.kind == "predicate" else ["A", "B", "C"])
    if strategy == "tool":
        assert model.bindings[0]["tool_choice"] == "answer"


@pytest.mark.parametrize("strategy", ["structured", "json_schema", "function_calling", "json_mode"])
@pytest.mark.parametrize("parsed", [{"answer": "Z"}, {"answer": True}, {"answer": "a"},
                                    {"answer": "A", "confidence": .99}, {}, "A", None, []])
def test_structured_parser_validates_exact_object_and_domain(strategy: LangChainStrategy, parsed: object) -> None:
    response = {"raw": Message(), "parsed": parsed, "parsing_error": None}
    model = FakeModel(response)
    result = answer(LangChainDecisionProvider(model, strategy=strategy))
    assert result.error_code == "invalid_provider_result" and result.value is None
    assert len(model.invocations) == 1


@pytest.mark.parametrize("response", [
    {"answer": "A"}, envelope(error=ValueError()), envelope(raw=Message(content=[None])),
    envelope(raw=Message(tool_calls=[tool("B")])), envelope(raw=Message(tool_calls=[tool(), tool()])),
    envelope(raw=Message(tool_calls=[tool(name="other")])), envelope(raw=Message(invalid_tool_calls=[tool()])),
])
def test_malformed_or_conflicting_structured_response_fails(response: object) -> None:
    assert answer(LangChainDecisionProvider(FakeModel(response))).error_code == "invalid_provider_result"


@pytest.mark.parametrize("raw", [Message('{"answer":"B"}'), Message('{"answer":"B","answer":"A"}'),
                                Message('{"answer":"A","confidence":0.9}'), Message(""), Message("A")])
def test_structured_parsed_object_cannot_hide_invalid_raw_json(raw: object) -> None:
    assert answer(LangChainDecisionProvider(FakeModel(envelope(raw=raw)))).error_code == "invalid_provider_result"


@pytest.mark.parametrize("strategy", ["json_schema", "json_mode"])
def test_json_methods_reject_foreign_tool_mechanism(strategy: LangChainStrategy) -> None:
    result = answer(LangChainDecisionProvider(FakeModel(envelope(raw=Message(tool_calls=[tool()]))),
                                            strategy=strategy))
    assert result.error_code == "invalid_provider_result"


@pytest.mark.parametrize("calls", [[], [tool(name="other")], [tool(), tool()], [tool(True)], [tool("Z")], "bad"])
def test_tool_shape_and_identity(calls: object) -> None:
    result = answer(LangChainDecisionProvider(FakeModel(Message(tool_calls=calls)), strategy="tool"))
    assert result.error_code == "invalid_provider_result"


def test_legacy_raw_tool_calls_and_duplicate_arguments() -> None:
    for arguments, expected in [('{"answer":"A"}', "ok"), ('{"answer":"A","answer":"B"}', "error")]:
        message = Message(additional_kwargs={"tool_calls": [
            {"function": {"name": "answer", "arguments": arguments}}]})
        assert answer(LangChainDecisionProvider(FakeModel(message), strategy="tool")).status == expected


@pytest.mark.parametrize("strategy", ["structured", "tool", "text"])
@pytest.mark.parametrize("stop", ["length", "max_tokens", "model_context_window_exceeded", "pause_turn", "error"])
def test_unfinished_messages_cannot_supply_answers(strategy: LangChainStrategy, stop: str) -> None:
    raw = Message(content="A", tool_calls=[] if strategy == "text" else [tool()],
                  response_metadata={"finish_reason": stop})
    result = answer(LangChainDecisionProvider(FakeModel(envelope(raw=raw) if strategy == "structured" else raw),
                                            strategy=strategy))
    assert result.error_code == "invalid_provider_result"


@pytest.mark.parametrize("raw", [Message(additional_kwargs={"refusal": "no"}),
                                Message(content=[{"type": "refusal", "refusal": "no"}]),
                                Message(response_metadata={"finish_reason": "content_filter"}),
                                Message(response_metadata={"stop_reason": "refusal"})])
def test_refusal_precedes_parsed_answer(raw: object) -> None:
    result = answer(LangChainDecisionProvider(FakeModel(envelope(raw=raw))))
    assert (result.status, result.status_reason, result.value) == ("uncertain", "refusal", None)


@pytest.mark.parametrize("strategy", ["structured", "text", "tool"])
def test_refusal_exception_retains_raw_signal_for_each_strategy(strategy: LangChainStrategy) -> None:
    class RefusalError(Exception):
        raw_message = Message(content=[{"type": "refusal", "refusal": "no"}])
    result = answer(LangChainDecisionProvider(FakeModel(RefusalError()), strategy=strategy))
    assert (result.status, result.status_reason, result.value) == ("uncertain", "refusal", None)


@pytest.mark.parametrize("key", ["tools", "tool_choice", "response_format", "input", "logprobs"])
def test_invocation_options_cannot_replace_managed_constraint(key: str) -> None:
    with pytest.raises(ValueError):
        LangChainDecisionProvider(FakeModel(envelope()), invoke_kwargs={key: {}})
    with pytest.raises(ValueError):
        LangChainDecisionProvider(FakeModel(envelope()), timeout_kwarg=key)


def test_auto_uses_only_local_binding_fallback_and_caches_domain() -> None:
    model = FakeModel(Message(tool_calls=[tool()]), structured=False)
    instance = LangChainDecisionProvider(model)
    assert answer(instance).value is True
    assert answer(instance).value is True
    assert len(model.bindings) == 2 and len(model.invocations) == 2
    assert answer(LangChainDecisionProvider(FakeModel(Message("A"), structured=False, tools=False))).value is True
    explicit = FakeModel(Message("A"), structured=False)
    assert answer(LangChainDecisionProvider(explicit, strategy="structured")).error_code == "unsupported_strategy"
    assert not explicit.invocations and len(explicit.bindings) == 1


@pytest.mark.parametrize("error,code", [(TimeoutError(), "deadline_exceeded"), (RuntimeError(), "provider_exception")])
def test_invocation_errors_do_not_fallback_or_retry(error: Exception, code: str) -> None:
    model = FakeModel(error)
    assert answer(LangChainDecisionProvider(model)).error_code == code
    assert len(model.bindings) == len(model.invocations) == 1


def test_structured_http_error_keeps_status() -> None:
    class HttpError(Exception):
        status_code = 429
    model = FakeModel(HttpError())
    assert answer(LangChainDecisionProvider(model)).error_code == "http_429"
    assert len(model.invocations) == 1


def test_deadline_and_explicit_integration_timeout() -> None:
    model = FakeModel(envelope())
    instance = LangChainDecisionProvider(model, timeout_kwarg="timeout", timeout_s=30,
                                        invoke_kwargs={"config": {"tags": ["decisions"]}})
    assert instance.decide_batch([request((PRED,))], deadline_s=0, budget=None)[0].error_code == "deadline_exceeded"
    assert not model.bindings and not model.invocations
    assert instance.decide_batch([request((PRED,))], deadline_s=2, budget=None)[0].value is True
    assert 0 < model.invocations[0]["timeout"] <= 2
    assert model.invocations[0]["config"] == {"tags": ["decisions"]}
    default = FakeModel(envelope())
    answer(LangChainDecisionProvider(default))
    assert set(default.invocations[0]) == {"messages"}


def test_binding_lock_wait_obeys_deadline() -> None:
    model = FakeModel(envelope())
    instance = LangChainDecisionProvider(model)
    instance._binding_lock.acquire()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(instance.decide_batch, [request((PRED,))], deadline_s=0.02, budget=None)
            assert future.result(timeout=1)[0].error_code == "deadline_exceeded"
    finally:
        instance._binding_lock.release()
    assert not model.invocations


def test_plain_runnable_needs_no_binding_methods() -> None:
    class Runnable:
        def invoke(self, input: List[Dict[str, str]], **kwargs: object) -> object:
            return "B. no"
    assert answer(LangChainDecisionProvider(Runnable())).value is False


@pytest.mark.parametrize("strategy", ["tool", "structured"])
@pytest.mark.parametrize("raw_calls", [
    [{"function": {"name": "answer", "arguments": '{"answer":"B"}'}}],
    [{"function": {"name": "other", "arguments": '{"answer":"A"}'}}],
    [], None, {}, [tool(), tool()],
])
def test_conflicting_or_malformed_raw_tool_view_cannot_hide_behind_normalized_calls(
        strategy: LangChainStrategy, raw_calls: object) -> None:
    raw = Message(tool_calls=[tool()], additional_kwargs={"tool_calls": raw_calls})
    model = FakeModel(raw if strategy == "tool" else envelope(raw=raw))
    result = answer(LangChainDecisionProvider(model, strategy=strategy))
    assert result.error_code == "invalid_provider_result" and result.value is None
    assert len(model.invocations) == 1


@pytest.mark.parametrize("strategy", ["tool", "structured"])
def test_matching_raw_and_normalized_tool_views_remain_valid(strategy: LangChainStrategy) -> None:
    raw = Message(tool_calls=[tool()], additional_kwargs={"tool_calls": [
        {"function": {"name": "answer", "arguments": '{"answer":"A"}'}}]})
    result = answer(LangChainDecisionProvider(FakeModel(raw if strategy == "tool" else envelope(raw=raw)),
                                            strategy=strategy))
    assert result.value is True and result.status == "ok"


@pytest.mark.parametrize("strategy", ["text", "tool", "structured"])
@pytest.mark.parametrize("finish", [None, "stop"])
@pytest.mark.parametrize("stop", ["max_tokens", "model_context_window_exceeded", "pause_turn"])
def test_each_unfinished_metadata_signal_is_checked(strategy: LangChainStrategy, finish: object, stop: str) -> None:
    raw = Message(content="A" if strategy == "text" else '{"answer":"A"}',
                  tool_calls=[tool()] if strategy == "tool" else [],
                  response_metadata={"finish_reason": finish, "stop_reason": stop})
    model = FakeModel(envelope(raw=raw) if strategy == "structured" else raw)
    result = answer(LangChainDecisionProvider(model, strategy=strategy))
    assert result.error_code == "invalid_provider_result" and result.value is None
    assert len(model.invocations) == 1


def test_usage_and_estimation_with_caller_prices() -> None:
    prices = TokenPrices(input=2, cached_input=1, output=3)
    raw = Message(content='{"answer":"A"}', usage_metadata={"input_tokens": 100, "output_tokens": 10,
                                 "input_token_details": {"cache_read": 20}})
    instance = LangChainDecisionProvider(FakeModel(envelope(raw=raw)), prices=prices)
    assert answer(instance).cost_usd == pytest.approx(210/1e6)
    assert instance.estimate_cost_usd([request((PRED,))]) > 0
    assert answer(LangChainDecisionProvider(FakeModel(envelope()), prices=prices)).cost_usd is None


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf", "polars-gpu"])
def test_gfql_workflow_preserves_rows_and_input(engine: str) -> None:
    if engine in ("cudf", "polars-gpu") and os.environ.get("TEST_CUDF") != "1":
        pytest.skip("GPU tests require TEST_CUDF=1")
    if engine in ("polars", "polars-gpu"):
        pytest.importorskip("polars")
    frame = pd.DataFrame({"id": [3, 1], "text": ["sky", "water"]}, index=[8, 8])
    before = frame.copy(deep=True)
    nodes = frame
    if engine in ("polars", "polars-gpu"):
        nodes = pytest.importorskip("polars").from_pandas(frame)
    elif engine == "cudf":
        nodes = pytest.importorskip("cudf").from_pandas(frame)
    provider = LangChainDecisionProvider(FakeModel(envelope("B")))
    with bind_providers({"model": provider}):
        result = graphistry.nodes(nodes, "id").gfql([call("sem_decide", {
            "provider": "model", "questions": [asdict(PRED)], "evidence_cols": ["text"],
            "task_id": "lc", "task_version": "1"})], engine=engine)
    output = result._nodes if isinstance(result._nodes, pd.DataFrame) else result._nodes.to_pandas()
    assert list(output["id"]) == [3, 1]
    assert list(output["p"]) == [False, False]
    assert output["p__p"].isna().all()
    pd.testing.assert_frame_equal(frame, before)
