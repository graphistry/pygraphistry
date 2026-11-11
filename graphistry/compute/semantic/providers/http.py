"""Shared base for HTTP decision providers: JSON POST with bounded retries inside the deadline, and token prices."""
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, NamedTuple, Optional, Protocol, Sequence, Tuple

import requests

from graphistry.compute.semantic.contract import (
    Budget, DecisionQuestion, DecisionRequest, DecisionResult, Evidence, ProviderCapabilities, finite_float,
    unanswered_result)


RETRYABLE_STATUSES = frozenset({408, 409, 429})
MAX_RETRY_DELAY_S = 60.0
JSONObject = Dict[str, object]


class HttpPoster(Protocol):
    """Anything with a ``requests``-style ``post`` (a ``requests.Session``, or a test double)."""

    def post(self, url: str, *, json: JSONObject, headers: Dict[str, str], timeout: float) -> requests.Response:
        ...


class RequestFailure(Exception):
    def __init__(self, error_code: str, retry_count: int, latency_ms: float) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.retry_count = retry_count
        self.latency_ms = latency_ms


class TokenUsage(NamedTuple):
    uncached_input: int
    cached_input: int
    cache_write: int
    output: int


def token_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


@dataclass(frozen=True)
class TokenPrices:
    """Caller-supplied USD prices per million tokens; there are no defaults.

    ``cost_usd`` needs the input and cached-input rates (else None, and a cost cap cannot be enforced); it is a
    lower bound that ignores surcharges. ``cache_write`` defaults to the input rate; output is free when None.
    The estimate is about four characters per input token at the uncached rate, or ``per_question`` when set.
    """

    input: Optional[float] = None
    cached_input: Optional[float] = None
    cache_write: Optional[float] = None
    output: Optional[float] = None
    per_question: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("input", "cached_input", "cache_write", "output", "per_question"):
            value = getattr(self, name)
            number = finite_float(value)
            if value is not None and (number is None or number < 0):
                raise ValueError(f"{name} price must be None or a finite number >= 0, got {value!r}")

    def cost_usd(self, usage: Optional[TokenUsage]) -> Optional[float]:
        if usage is None or self.input is None or self.cached_input is None:
            return None
        write_rate = self.input if self.cache_write is None else self.cache_write
        return (usage.uncached_input * self.input + usage.cached_input * self.cached_input
                + usage.cache_write * write_rate + usage.output * (self.output or 0.0)) / 1e6

    def estimate_usd(self, texts: Sequence[str], questions: int) -> Optional[float]:
        if self.per_question is not None:
            return self.per_question * questions
        if self.input is None or self.cached_input is None:
            return None
        return sum(math.ceil(len(text) / 4.0) for text in texts) * self.input / 1e6


def evidence_text(evidence: Evidence) -> str:
    """Evidence as sent: strings verbatim, mappings as compact JSON with sorted keys."""
    if isinstance(evidence, str):
        return evidence
    return json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _attempt(session: Optional[HttpPoster], url: str, body: JSONObject, headers: Dict[str, str],
             timeout: float) -> Tuple[Optional[JSONObject], str, bool, Optional[float]]:
    """One POST: ``(payload, failure code, retryable, Retry-After seconds)``; payload is set only on success."""
    post = session.post if session is not None else requests.post
    try:
        response = post(url, json=body, headers=headers, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ContentDecodingError, requests.exceptions.RetryError):
        return None, "transport_error", True, None
    except requests.HTTPError as error:
        if error.response is None:
            return None, "request_error", False, None
        response = error.response
    except requests.RequestException:
        return None, "request_error", False, None
    status = response.status_code
    if status != 200:
        retry_after = response.headers.get("Retry-After") if response.headers is not None else None
        try:
            delay = finite_float(float(retry_after)) if retry_after is not None else None
        except ValueError:
            delay = None
        return None, f"http_{status}", status in RETRYABLE_STATUSES or status >= 500, delay
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return (payload, "", False, None) if isinstance(payload, dict) else (None, "invalid_provider_result", False, None)


def post_json(url: str, body: JSONObject, headers: Dict[str, str], *, deadline_at: Optional[float],
              timeout_s: float, max_retries: int, backoff_s: float,
              session: Optional[HttpPoster] = None) -> Tuple[JSONObject, int, float]:
    """``(payload, retries, latency_ms)``; retries 408, 409, 429, 5xx and connection errors, honouring
    ``Retry-After`` (capped at 60 s) within the deadline, then raises ``RequestFailure``."""
    started = time.monotonic()
    retries = 0
    while True:
        remaining = None if deadline_at is None else deadline_at - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise RequestFailure("deadline_exceeded", retries, (time.monotonic() - started) * 1000)
        payload, code, retryable, retry_after = _attempt(
            session, url, body, headers, timeout_s if remaining is None else min(timeout_s, remaining))
        if payload is not None:
            return payload, retries, (time.monotonic() - started) * 1000
        delay = min(MAX_RETRY_DELAY_S, retry_after if retry_after is not None else backoff_s * 2 ** retries)
        remaining = None if deadline_at is None else deadline_at - time.monotonic()
        if not retryable or retries >= max_retries or (remaining is not None and delay >= remaining):
            raise RequestFailure(code, retries, (time.monotonic() - started) * 1000)
        if delay > 0:
            time.sleep(delay)
        retries += 1


class ResponseMeta(NamedTuple):
    model: str
    latency_ms: float
    retries: int
    cost_usd: Optional[float]
    cached_tokens: Optional[int]


class HttpDecisionProvider:
    """Base for decision providers that answer over HTTP; subclasses build bodies and map responses.

    Transport failures after retries become ``error`` results with ``error_code`` ``http_<status>``,
    ``transport_error``, ``request_error`` (not retried), ``deadline_exceeded`` or ``invalid_provider_result``.
    The API key is read only from the constructor and never appears in the body or ``repr``.
    """

    name = "http"
    default_base_url = ""
    path = ""

    def __init__(self, api_key: str, model: str, *, base_url: Optional[str] = None, prices: Optional[TokenPrices] = None,
                 concurrency_limit: int = 4, timeout_s: float = 60, max_retries: int = 3,
                 retry_backoff_s: float = 0.5, session: Optional[HttpPoster] = None, name: Optional[str] = None) -> None:
        if not api_key or not model:
            raise ValueError("api_key and model must be non-empty strings")
        self._api_key = api_key
        self.model = model
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.prices = prices or TokenPrices()
        self.concurrency_limit = concurrency_limit
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._session = session
        self.name = name or self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r})"

    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    def estimate_cost_usd(self, requests_: Sequence[DecisionRequest]) -> Optional[float]:
        texts = [text for request in requests_ for text in
                 [evidence_text(request.evidence)] + [question.text for question in request.questions]]
        return self.prices.estimate_usd(texts, sum(len(request.questions) for request in requests_))

    def decide_batch(self, requests_: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        deadline_at = None if deadline_s is None else time.monotonic() + deadline_s
        return [result for request in requests_ for questions in self.question_groups(request)
                for result in self._decide(request, questions, deadline_at)]

    def _decide(self, request: DecisionRequest, questions: Sequence[DecisionQuestion],
                deadline_at: Optional[float]) -> List[DecisionResult]:
        try:
            payload, retries, latency_ms = post_json(
                f"{self.base_url}{self.path}", self.body(request, questions), self.headers(),
                deadline_at=deadline_at, timeout_s=self.timeout_s, max_retries=self.max_retries,
                backoff_s=self.retry_backoff_s, session=self._session)
        except RequestFailure as failure:
            return [unanswered_result(request, question, provider=self.name, model=self.model,
                                      error_code=failure.error_code, latency_ms=failure.latency_ms,
                                      retry_count=failure.retry_count) for question in questions]
        usage = self.usage(payload)
        cost = self.prices.cost_usd(usage)
        model = payload.get("model")
        meta = ResponseMeta(model=model if isinstance(model, str) else self.model,
                            latency_ms=latency_ms, retries=retries,
                            cost_usd=None if cost is None else cost / len(questions),
                            cached_tokens=None if usage is None else usage.cached_input)
        return self.results(request, questions, payload, meta)

    def invalid(self, request: DecisionRequest, question: DecisionQuestion, meta: ResponseMeta) -> DecisionResult:
        return unanswered_result(request, question, provider=self.name, model=meta.model,
                                 error_code="invalid_provider_result", latency_ms=meta.latency_ms,
                                 retry_count=meta.retries)

    def question_groups(self, request: DecisionRequest) -> List[Sequence[DecisionQuestion]]:
        """Questions answered by one HTTP request."""
        raise NotImplementedError

    def headers(self) -> Dict[str, str]:
        raise NotImplementedError

    def body(self, request: DecisionRequest, questions: Sequence[DecisionQuestion]) -> JSONObject:
        raise NotImplementedError

    def usage(self, payload: Mapping[str, object]) -> Optional[TokenUsage]:
        raise NotImplementedError

    def results(self, request: DecisionRequest, questions: Sequence[DecisionQuestion], payload: Mapping[str, object],
                meta: ResponseMeta) -> List[DecisionResult]:
        raise NotImplementedError
