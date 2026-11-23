"""Shared base for HTTP decision providers: JSON POST with bounded retries inside the deadline, and token prices."""
import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Mapping, NamedTuple, Optional, Protocol, Sequence, Tuple, Union

import requests

from graphistry.compute.semantic.cache import fingerprint_of
from graphistry.compute.semantic.contract import (
    Budget, DecisionQuestion, DecisionRequest, DecisionResult, Evidence, ExtraValue, ProbabilityKind,
    ProviderCapabilities, StatusReason, compute_decision_hash, finite_float, unanswered_result)


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


def token_count(value: object) -> int:  # hygiene-ok: object-annotation -- untrusted host or user input, narrowed by isinstance
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2 ** 53 else 0


@dataclass(frozen=True)
class TokenPrices:
    """Caller-supplied USD prices per million tokens; there are no defaults.

    ``cost_usd`` comes from returned token usage and needs the input and cached-input rates (else None); it is a
    lower bound that ignores surcharges. ``cache_write`` defaults to the input rate; output is free when None.
    The estimate (what a cost cap reserves) is ``per_question`` per question when set, else (with both the input and
    cached-input rates) about four characters per input token at the uncached rate, else None.
    """

    input: Optional[float] = None
    cached_input: Optional[float] = None
    cache_write: Optional[float] = None
    output: Optional[float] = None
    per_question: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("input", "cached_input", "cache_write", "output", "per_question"):
            value = getattr(self, name)  # hygiene-ok: dynamic-attr -- duck-typed host object with no common base
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


def _error_code(response: requests.Response) -> object:  # hygiene-ok: object-annotation -- untrusted host or user input, narrowed by isinstance
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    return error.get("code") if isinstance(error, dict) else None


def _attempt(session: Optional[HttpPoster], url: str, body: JSONObject, headers: Dict[str, str],
             timeout: float) -> Tuple[Optional[JSONObject], str, bool, Optional[float]]:
    """One POST: ``(payload, failure code, retryable, Retry-After seconds)``; payload is set only on success."""
    try:
        # The default POST never follows a redirect, so a 3xx cannot forward the key header or evidence to another host.
        response = (session.post(url, json=body, headers=headers, timeout=timeout) if session is not None
                    else requests.post(url, json=body, headers=headers, timeout=timeout, allow_redirects=False))
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
        code = "content_filter" if status == 400 and _error_code(response) == "content_filter" else f"http_{status}"
        return None, code, status in RETRYABLE_STATUSES or (status >= 500 and status != 501), delay
    try:
        payload = response.json()
    except (ValueError, RecursionError):
        payload = None
    return (payload, "", False, None) if isinstance(payload, dict) else (None, "invalid_provider_result", False, None)


def post_json(url: str, body: JSONObject, headers: Dict[str, str], *, deadline_at: Optional[float],
              timeout_s: float, max_retries: int, backoff_s: float,
              session: Optional[HttpPoster] = None) -> Tuple[JSONObject, int, float]:
    """``(payload, retries, latency_ms)``; retries 408, 409, 429, 5xx except 501 and connection errors, honouring
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
        delay = min(MAX_RETRY_DELAY_S, retry_after if retry_after is not None else backoff_s * 2 ** min(retries, 16))
        remaining = None if deadline_at is None else deadline_at - time.monotonic()
        if not retryable or retries >= max_retries or (remaining is not None and delay >= remaining):
            raise RequestFailure(code, retries, (time.monotonic() - started) * 1000)
        if delay > 0:
            time.sleep(delay)
        retries += 1


def _checked_base_url(value: object) -> str:  # hygiene-ok: object-annotation -- untrusted host or user input, narrowed by isinstance
    if not (isinstance(value, str) and value.lower().startswith(("http://", "https://")) and not any(c.isspace() for c in value)
            and value.split("://", 1)[1].strip("/")):
        raise ValueError(f"base_url must be an http:// or https:// URL without whitespace, got {value!r}")
    return value.rstrip("/")


class ResponseMeta(NamedTuple):
    model: str
    latency_ms: float
    retries: int
    cost_usd: Optional[float]
    cached_tokens: Optional[int]


RESENT_REPLY_ERRORS = ("invalid_provider_result", "answer_truncated")


class HttpDecisionProvider(ABC):
    """Base for decision providers that answer over HTTP; subclasses build bodies and map responses.

    Transport failures after retries become ``error`` results with ``error_code`` ``http_<status>``,
    ``transport_error``, ``request_error`` (not retried), ``deadline_exceeded`` or ``invalid_provider_result``.
    A request the host's content filter rejects (HTTP 400 with error code ``content_filter``) is an ``uncertain``
    result with reason ``refusal``, like a filtered reply.
    The API key is read only from the constructor and never appears in the body or ``repr``.

    :param api_key: API key, sent only in a request header
    :param model: model id (a deployment name on Azure)
    :param base_url: API base URL; default the provider's public endpoint
    :param prices: :class:`TokenPrices` for ``cost_usd`` and the cost estimate; no defaults
    :param concurrency_limit: concurrent requests one operator run may send
    :param timeout_s: per-request timeout, capped by the remaining deadline
    :param max_retries: retries on 408, 409, 429, 5xx except 501, connection errors and timeouts; each attempt may be billed
    :param reply_retries: times a request is sent again when its reply holds no readable answer or is cut off at the
        answer-token limit (default 2); each is billed and counted in ``retry_count``
    :param retry_backoff_s: base delay between retries when the server sends no ``Retry-After``
    :param session: object with a ``requests``-style ``post``; default ``requests``
    :param name: provider name stamped on results; default the class's ``name``
    """

    name = "http"
    default_base_url = ""
    path = ""

    def __init__(self, api_key: str, model: str, *, base_url: Optional[str] = None, prices: Optional[TokenPrices] = None,
                 concurrency_limit: int = 4, timeout_s: float = 60, max_retries: int = 3,
                 retry_backoff_s: float = 0.5, session: Optional[HttpPoster] = None, name: Optional[str] = None,
                 reply_retries: int = 2) -> None:
        if not (isinstance(api_key, str) and api_key):
            raise ValueError("api_key must be a non-empty string")
        if not (isinstance(model, str) and model):
            raise ValueError(f"model must be a non-empty string, got {model!r}")
        if not (prices is None or isinstance(prices, TokenPrices)):
            raise ValueError(f"prices must be a TokenPrices, got {type(prices).__name__}")
        if not (session is None or callable(getattr(session, "post", None))):  # hygiene-ok: dynamic-attr -- duck-typed host object with no common base
            raise ValueError("session must have a requests-style post")
        timeout, backoff = finite_float(timeout_s), finite_float(retry_backoff_s)
        if timeout is None or timeout <= 0:
            raise ValueError(f"timeout_s must be a finite number > 0, got {timeout_s!r}")
        if backoff is None or backoff < 0:
            raise ValueError(f"retry_backoff_s must be a finite number >= 0, got {retry_backoff_s!r}")
        if not (isinstance(max_retries, int) and not isinstance(max_retries, bool) and max_retries >= 0):
            raise ValueError(f"max_retries must be an int >= 0, got {max_retries!r}")
        if not (isinstance(reply_retries, int) and not isinstance(reply_retries, bool) and reply_retries >= 0):
            raise ValueError(f"reply_retries must be an int >= 0, got {reply_retries!r}")
        self.reply_retries = reply_retries
        self._api_key = api_key
        self.model = model
        self.base_url = _checked_base_url(base_url or self.default_base_url)
        self.prices = prices or TokenPrices()
        self.concurrency_limit = concurrency_limit
        self.timeout_s = timeout
        self.max_retries = max_retries
        self.retry_backoff_s = backoff
        self._session = session
        self.name = name or self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r})"

    def answer_settings(self) -> Dict[str, object]:
        """Settings, besides the model and the URL, that can change an answer; subclasses extend this."""
        return {}

    def cache_fingerprint(self) -> Optional[str]:
        """A digest of what this provider's answers depend on: its class, name, model, base URL and
        ``answer_settings()``. It never holds the API key or the headers. Result caches key on it."""
        return fingerprint_of({"class": type(self).__name__, "name": self.name, "model": self.model,
                               "base_url": self.base_url, **self.answer_settings()})

    def estimate_cost_usd(self, requests_: Sequence[DecisionRequest]) -> Optional[float]:
        """Prices the text of every HTTP request a batch sends: one per question group of each row."""
        texts = [text for request in requests_ for questions in self.question_groups(request)
                 for text in self.prompt_texts(request, questions)]
        return self.prices.estimate_usd(texts, sum(len(request.questions) for request in requests_))

    def prompt_texts(self, request: DecisionRequest, questions: Sequence[DecisionQuestion]) -> List[str]:
        """The input texts one HTTP request carries, for the cost estimate."""
        return [evidence_text(request.evidence)] + [question.text for question in questions]

    def decide_batch(self, requests_: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        deadline_at = None if deadline_s is None else time.monotonic() + deadline_s
        return [result for request in requests_ for questions in self.question_groups(request)
                for result in self._decide(request, questions, deadline_at)]

    def _decide(self, request: DecisionRequest, questions: Sequence[DecisionQuestion],
                deadline_at: Optional[float]) -> List[DecisionResult]:
        return self.send([(request, question) for question in questions], self.body(request, questions),
                         lambda payload, meta: self.results(request, questions, payload, meta), deadline_at)

    def send(self, pairs: Sequence[Tuple[DecisionRequest, DecisionQuestion]], body: JSONObject,
             read: Callable[[JSONObject, ResponseMeta], List[DecisionResult]],
             deadline_at: Optional[float]) -> List[DecisionResult]:
        """POST one body answering ``pairs`` (row, question); ``read`` maps the payload, with the cost split evenly
        over the pairs. A transport failure errors every pair. A reply in which every pair is unread or cut off is
        sent again up to ``reply_retries`` times, with costs and latencies summed. The cost sums the attempts that
        reported usage, so every outcome of a resend keeps what the earlier replies billed."""
        asks, resent, waited = 0, 0, 0.0
        spent: Optional[float] = None

        def share(total: Optional[float]) -> Optional[float]:
            return None if total is None else total / len(pairs)

        while True:
            try:
                payload, retries, latency_ms = post_json(
                    f"{self.base_url}{self.path}", body, self.headers(),
                    deadline_at=deadline_at, timeout_s=self.timeout_s, max_retries=self.max_retries,
                    backoff_s=self.retry_backoff_s, session=self._session)
            except RequestFailure as failure:
                if failure.error_code == "content_filter":
                    blocked = ResponseMeta(model=self.model, latency_ms=failure.latency_ms + waited,
                                           retries=failure.retry_count + resent, cost_usd=share(spent),
                                           cached_tokens=None)
                    return [self.answered(request, question, blocked, value=None, distribution=None,
                                          probability_kind="none", reason="refusal",
                                          extra={"refusal_source": "request_content_filter"})
                            for request, question in pairs]
                return [replace(unanswered_result(request, question, provider=self.name, model=self.model,
                                                  error_code=failure.error_code, latency_ms=failure.latency_ms + waited,
                                                  retry_count=failure.retry_count + resent), cost_usd=share(spent))
                        for request, question in pairs]
            usage = self.usage(payload)
            cost = self.prices.cost_usd(usage)
            total = spent if cost is None else cost + (spent or 0.0)
            model = payload.get("model")
            meta = ResponseMeta(model=model if isinstance(model, str) else self.model,
                                latency_ms=latency_ms + waited, retries=retries + resent, cost_usd=share(total),
                                cached_tokens=None if usage is None else usage.cached_input)
            results = read(payload, meta)
            unread = bool(results) and all(result.error_code in RESENT_REPLY_ERRORS for result in results)
            if not unread or asks >= self.reply_retries or (deadline_at is not None and time.monotonic() >= deadline_at):
                return results
            asks, resent, spent, waited = asks + 1, resent + 1 + retries, total, waited + latency_ms

    def answered(self, request: DecisionRequest, question: DecisionQuestion, meta: ResponseMeta, *,
                 value: Union[bool, str, None], distribution: Optional[Dict[str, float]], probability_kind: ProbabilityKind,
                 confidence: Optional[float] = None, reason: Optional[StatusReason] = None,
                 extra: Optional[Dict[str, ExtraValue]] = None) -> DecisionResult:
        """A result for one answered question, with the request's shared metadata."""
        extra = dict(extra or {})
        if meta.cached_tokens is not None:
            extra["cached_input_tokens"] = meta.cached_tokens
        return DecisionResult(
            request_id=request.request_id, question_id=question.id, value=value, distribution=distribution,
            confidence=confidence, probability_kind=probability_kind if distribution is not None else "none",
            status="uncertain" if reason is not None else "ok", error_code=None, model=meta.model, provider=self.name,
            task_version=request.task_version,
            input_hash=compute_decision_hash(request.evidence, question, request.task_version),
            latency_ms=meta.latency_ms, cost_usd=meta.cost_usd,
            cache_hit=None if meta.cached_tokens is None else meta.cached_tokens > 0, retry_count=meta.retries,
            status_reason=reason, extra=extra)

    def invalid(self, request: DecisionRequest, question: DecisionQuestion, meta: ResponseMeta,
                error_code: str = "invalid_provider_result") -> DecisionResult:
        return replace(unanswered_result(request, question, provider=self.name, model=meta.model,
                                         error_code=error_code, latency_ms=meta.latency_ms,
                                         retry_count=meta.retries), cost_usd=meta.cost_usd)

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        ...

    @abstractmethod
    def question_groups(self, request: DecisionRequest) -> List[Sequence[DecisionQuestion]]:
        """Questions answered by one HTTP request."""

    @abstractmethod
    def headers(self) -> Dict[str, str]:
        ...

    @abstractmethod
    def body(self, request: DecisionRequest, questions: Sequence[DecisionQuestion]) -> JSONObject:
        ...

    @abstractmethod
    def usage(self, payload: Mapping[str, object]) -> Optional[TokenUsage]:
        ...

    @abstractmethod
    def results(self, request: DecisionRequest, questions: Sequence[DecisionQuestion], payload: Mapping[str, object],
                meta: ResponseMeta) -> List[DecisionResult]:
        ...
