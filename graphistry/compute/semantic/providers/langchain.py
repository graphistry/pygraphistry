"""Optional decision adapter for a supplied LangChain chat model or Runnable."""
import json
import threading
import time
from dataclasses import replace
from typing import Dict, List, Literal, Mapping, NamedTuple, Optional, Protocol, Sequence, Tuple, get_args, runtime_checkable

from graphistry.compute.semantic.contract import (
    Budget, DecisionQuestion, DecisionRequest, DecisionResult, ExtraValue, ProviderCapabilities, compute_decision_hash,
    finite_float, unanswered_result)
from graphistry.compute.semantic.providers.chat import (
    SYSTEM_PROMPT, _answer_field, _json_answer, answer_codes, answer_labels, exact_code_index, letter_prompt, reply_index)
from graphistry.compute.semantic.providers.chat_constraints import CONSTRAINT_KEYS, JSON_INSTRUCTION, answer_schema
from graphistry.compute.semantic.providers.http import TokenPrices, TokenUsage, token_count


LangChainStrategy = Literal["auto", "structured", "json_schema", "function_calling", "json_mode", "tool", "text"]


@runtime_checkable
class ChatRunnable(Protocol):
    def invoke(self, input: List[Dict[str, str]], **kwargs: object) -> object:
        ...


@runtime_checkable
class StructuredModel(Protocol):
    def with_structured_output(self, schema: Dict[str, object], **kwargs: object) -> object:
        ...


@runtime_checkable
class ToolModel(Protocol):
    def bind_tools(self, tools: List[Dict[str, object]], **kwargs: object) -> object:
        ...


class Binding(NamedTuple):
    runner: ChatRunnable
    strategy: str


class UnsupportedBinding(Exception):
    pass


def _field(value: object, name: str) -> object:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def _calls_code(calls: object) -> object:
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return None
    call = calls[0]
    if "function" in call:
        function = _mapping(call.get("function"))
        return _json_answer(function.get("arguments")) if function.get("name") == "answer" else None
    return _answer_field(call.get("args")) if call.get("name") == "answer" else None


def _tool_code(message: object) -> object:
    if _field(message, "invalid_tool_calls"):
        return None
    normalized = _field(message, "tool_calls")
    additional = _mapping(_field(message, "additional_kwargs"))
    raw = additional.get("tool_calls")
    if normalized and "tool_calls" in additional and _calls_code(normalized) != _calls_code(raw):
        return None
    return _calls_code(normalized or raw)


def _content(message: object) -> Optional[str]:
    content = message if isinstance(message, str) else _field(message, "content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or any(not isinstance(block, dict) or not isinstance(block.get("type"), str)
                                            for block in content):
        return None
    texts = [block.get("text") for block in content if block.get("type") == "text"]
    return "".join(texts) if all(isinstance(text, str) for text in texts) else None


def _refused(message: object) -> bool:
    content = _field(message, "content")
    return bool(_mapping(_field(message, "additional_kwargs")).get("refusal")) or (
        isinstance(content, list) and any(isinstance(block, dict) and block.get("type") == "refusal"
                                         for block in content)) or (
        _mapping(_field(message, "response_metadata")).get("stop_reason") == "refusal") or (
        _mapping(_field(message, "response_metadata")).get("finish_reason") == "content_filter")


def _usage(message: object) -> Optional[TokenUsage]:
    usage = _field(message, "usage_metadata")
    if not isinstance(usage, dict) or any(isinstance(usage.get(key), bool) or not isinstance(usage.get(key), int)
                                        or usage[key] < 0 for key in ("input_tokens", "output_tokens")):
        return None
    details = _mapping(usage.get("input_token_details"))
    read = min(token_count(details.get("cache_read")), token_count(usage.get("input_tokens")))
    write = min(token_count(details.get("cache_creation")), token_count(usage.get("input_tokens")) - read)
    return TokenUsage(token_count(usage.get("input_tokens")) - read - write, read, write,
                      token_count(usage.get("output_tokens")))


def _invoke(runner: ChatRunnable, messages: List[Dict[str, str]], kwargs: Dict[str, object]
            ) -> Tuple[object, Optional[str]]:
    """Contain integration exceptions at the supplied model invocation boundary; never retry or resend."""
    try:
        return runner.invoke(messages, **kwargs), None
    except Exception as error:
        raw = _field(error, "raw_message")
        if raw is not None and _refused(raw):
            return raw, "refusal"
        status = _field(error, "status_code")
        if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
            return None, f"http_{status}"
        return None, "transport_error" if isinstance(error, (TimeoutError, ConnectionError)) else "provider_exception"


class LangChainDecisionProvider:
    """Wrap a supplied chat model without importing LangChain or executing tools.

    ``auto`` tries local structured binding, then tool binding, then text, before any invocation. Only an absent
    interface or ``NotImplementedError`` permits that local fallback. Explicit strategies never downgrade, and
    invocation/parse errors never resend. Structured methods request ``include_raw=True`` and validate both raw
    message signals and the parsed answer. All answers are value-only, even if the model returns logprobs.

    The supplied model owns network timeouts, retries and tracing. Configure those before passing it here.
    ``timeout_kwarg`` explicitly enables an integration's supported invocation timeout, capped by the remaining
    deadline. Generic Runnable config has no portable hard timeout; the GFQL driver abandons overdue results.
    ``TokenPrices`` estimates input/schema cost, or uses ``per_question``; allow for output and client retries.
    Only normalized ``usage_metadata`` supplies actual token cost. No optional package version is inferred.
    """

    def __init__(self, chat_model: object, *, model: str = "langchain", name: str = "langchain",
                 strategy: LangChainStrategy = "auto", prices: Optional[TokenPrices] = None,
                 concurrency_limit: int = 4, invoke_kwargs: Optional[Dict[str, object]] = None,
                 timeout_kwarg: Optional[str] = None, timeout_s: float = 60) -> None:
        if not isinstance(chat_model, ChatRunnable) or not callable(chat_model.invoke):
            raise ValueError("chat_model must have invoke")
        if strategy not in get_args(LangChainStrategy):
            raise ValueError("unsupported LangChain strategy")
        if not isinstance(model, str) or not model or not isinstance(name, str) or not name:
            raise ValueError("model and name must be non-empty strings")
        if isinstance(concurrency_limit, bool) or not isinstance(concurrency_limit, int) or concurrency_limit < 1:
            raise ValueError("concurrency_limit must be a positive int")
        timeout = finite_float(timeout_s)
        if timeout is None or timeout <= 0 or (timeout_kwarg is not None and (
                not isinstance(timeout_kwarg, str) or not timeout_kwarg or timeout_kwarg in (
                    CONSTRAINT_KEYS | {"input", "config"}))):
            raise ValueError("invalid timeout settings")
        if prices is not None and not isinstance(prices, TokenPrices):
            raise ValueError("prices must be TokenPrices")
        self.chat_model, self.model, self.name, self.strategy = chat_model, model, name, strategy
        self.prices, self.concurrency_limit = prices or TokenPrices(), concurrency_limit
        self.invoke_kwargs = dict(invoke_kwargs or {})
        if set(self.invoke_kwargs) & (CONSTRAINT_KEYS | {"input"}) or (
                timeout_kwarg is not None and timeout_kwarg in self.invoke_kwargs):
            raise ValueError("invoke_kwargs cannot override input, constraints or the managed timeout")
        self.timeout_kwarg, self.timeout_s = timeout_kwarg, timeout
        self._bindings: Dict[str, Binding] = {}
        self._binding_lock = threading.Lock()

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(supports_native_probabilities=False, max_questions_per_request=1,
                                    concurrency_limit=self.concurrency_limit, max_rows_per_request=1,
                                    max_choice_options=26, max_levels=26)

    def _binding(self, codes: str, deadline_at: Optional[float]) -> Binding:
        remaining = None if deadline_at is None else max(0.0, deadline_at - time.monotonic())
        acquired = self._binding_lock.acquire() if remaining is None else self._binding_lock.acquire(
            timeout=min(remaining, threading.TIMEOUT_MAX))
        if not acquired:
            raise TimeoutError()
        try:
            if codes in self._bindings:
                return self._bindings[codes]
            profile = ("structured", "tool", "text") if self.strategy == "auto" else (self.strategy,)
            for strategy in profile:
                try:
                    runner: object = None
                    schema = dict(answer_schema(codes), title="answer", description="The code of the answer.")
                    if strategy == "text":
                        runner = self.chat_model
                    elif strategy == "tool" and isinstance(self.chat_model, ToolModel):
                        runner = self.chat_model.bind_tools([
                            {"type": "function", "function": {"name": "answer", "description": "The answer code.",
                                                              "parameters": answer_schema(codes)}}], tool_choice="answer")
                    elif strategy != "tool" and isinstance(self.chat_model, StructuredModel):
                        options: Dict[str, object] = {"include_raw": True}
                        if strategy != "structured":
                            options["method"] = strategy
                        runner = self.chat_model.with_structured_output(schema, **options)
                except NotImplementedError:
                    continue
                if isinstance(runner, ChatRunnable) and callable(runner.invoke):
                    binding = Binding(runner, strategy)
                    self._bindings[codes] = binding
                    return binding
            raise UnsupportedBinding()
        finally:
            self._binding_lock.release()

    def estimate_cost_usd(self, requests: Sequence[DecisionRequest]) -> Optional[float]:
        texts = [text for request in requests for question in request.questions if answer_codes(question) is not None
                 for text in [SYSTEM_PROMPT + JSON_INSTRUCTION, letter_prompt(request.evidence, question),
                              json.dumps(answer_schema((answer_codes(question) or "")[:len(answer_labels(question))]))]
                 ]
        return self.prices.estimate_usd(texts, sum(len(request.questions) for request in requests))

    def decide_batch(self, requests: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        deadline_at = None if deadline_s is None else time.monotonic() + deadline_s
        return [self._one(request, question, deadline_at) for request in requests for question in request.questions]

    def _one(self, request: DecisionRequest, question: DecisionQuestion, deadline_at: Optional[float]) -> DecisionResult:
        started = time.monotonic()
        failure = unanswered_result(request, question, provider=self.name, model=self.model, retry_count=0,
                                    error_code="invalid_provider_result")
        if deadline_at is not None and started >= deadline_at:
            return replace(failure, error_code="deadline_exceeded")
        alphabet = answer_codes(question)
        if alphabet is None:
            return replace(failure, error_code="unsupported_question")
        codes = alphabet[:len(answer_labels(question))]
        try:
            binding = self._binding(codes, deadline_at)
        except UnsupportedBinding:
            return replace(failure, error_code="unsupported_strategy")
        except TimeoutError:
            return replace(failure, error_code="deadline_exceeded")
        remaining = None if deadline_at is None else deadline_at - time.monotonic()
        if remaining is not None and remaining <= 0:
            return replace(failure, error_code="deadline_exceeded")
        kwargs = dict(self.invoke_kwargs)
        if self.timeout_kwarg is not None:
            kwargs[self.timeout_kwarg] = self.timeout_s if remaining is None else min(self.timeout_s, remaining)
        messages = [{"role": "system", "content": SYSTEM_PROMPT + (
            "" if binding.strategy in ("tool", "text") else JSON_INSTRUCTION)},
                    {"role": "user", "content": letter_prompt(request.evidence, question)}]
        response, error = _invoke(binding.runner, messages, kwargs)
        latency = (time.monotonic() - started) * 1000
        if error is not None and error != "refusal":
            if error == "transport_error" and deadline_at is not None and time.monotonic() >= deadline_at:
                error = "deadline_exceeded"
            return replace(failure, error_code=error, latency_ms=latency)
        structured = binding.strategy not in ("text", "tool")
        envelope = _mapping(response) if structured else {}
        raw = envelope.get("raw") if structured and error != "refusal" else response
        cost = self.prices.cost_usd(_usage(raw))
        failure = replace(failure, latency_ms=latency, cost_usd=cost)
        metadata = _mapping(_field(raw, "response_metadata"))
        extra: Dict[str, ExtraValue] = {"answer_strategy": binding.strategy, "semantic_level": (
            "prompt_parse" if binding.strategy == "text" else "tool_structured" if binding.strategy == "tool"
            else "model_structured")}
        reason = "refusal" if _refused(raw) else None
        if reason is None and any(metadata.get(field) in (
                "length", "max_tokens", "model_context_window_exceeded", "pause_turn", "content_filter", "error")
                                  for field in ("finish_reason", "stop_reason")):
            return failure
        if reason is None and _content(raw) is None:
            return failure
        calls = _field(raw, "tool_calls") or _mapping(_field(raw, "additional_kwargs")).get("tool_calls")
        if reason is not None:
            index = None
        elif structured:
            code = _answer_field(envelope.get("parsed"))
            if raw is None or envelope.get("parsing_error") is not None or (calls and _tool_code(raw) != code) \
                    or _field(raw, "invalid_tool_calls") or (
                        not calls and _json_answer(_content(raw)) != code) or (
                        calls and binding.strategy in ("json_schema", "json_mode")) or (
                        not calls and binding.strategy == "function_calling"):
                return failure
            index = exact_code_index(code, codes)
        elif binding.strategy == "tool":
            index = exact_code_index(_tool_code(raw), codes)
        else:
            index = None if calls or _field(raw, "invalid_tool_calls") else reply_index(_content(raw), codes, question)
        if index is None and reason is None:
            return failure
        return DecisionResult(request_id=request.request_id, question_id=question.id,
                              value=None if index is None else answer_labels(question)[index], distribution=None,
                              confidence=None, probability_kind="none", status="uncertain" if reason else "ok",
                              status_reason="refusal" if reason else None, error_code=None, provider=self.name,
                              model=self.model, task_version=request.task_version,
                              input_hash=compute_decision_hash(request.evidence, question, request.task_version),
                              latency_ms=latency, cost_usd=cost, cache_hit=None, retry_count=0, extra=extra)
