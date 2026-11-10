"""Client for the OpenAI Decisions API (``POST /v1/decisions``) as a ``DecisionProvider``.

Public API guide: https://developers.openai.com/api/docs/guides/decisions
API reference: https://developers.openai.com/api/reference/resources/decisions/methods/create
"""
import json
import math
import time
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import requests

from graphistry.compute.semantic.contract import (
    Budget,
    DecisionQuestion,
    DecisionRequest,
    DecisionResult,
    ExtraValue,
    ProviderCapabilities,
    StatusReason,
    compute_decision_hash,
    error_result,
    finite_float,
)


MAX_QUESTIONS_PER_REQUEST = 400
MAX_CHOICE_OPTIONS = 255
MAX_LEVELS = 10
RETRYABLE_STATUSES = frozenset({408, 409, 429})
MAX_RETRY_DELAY_S = 60.0
JSONObject = Dict[str, object]


class HttpPoster(Protocol):
    """Anything with a ``requests``-style ``post`` (a ``requests.Session``, or a test double)."""

    def post(self, url: str, *, json: JSONObject, headers: Dict[str, str], timeout: float) -> requests.Response:
        ...


class _RequestFailure(Exception):
    """A transport-level failure for one HTTP request, after retries."""

    def __init__(self, error_code: str, retry_count: int, latency_ms: Optional[float]) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.retry_count = retry_count
        self.latency_ms = latency_ms


def evidence_text(evidence: Union[str, Mapping[str, object]]) -> str:
    """Evidence as sent: strings verbatim, mappings as compact JSON with sorted keys."""
    if isinstance(evidence, str):
        return evidence
    return json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def question_payload(question: DecisionQuestion) -> JSONObject:
    payload: JSONObject = {"type": question.kind, "name": question.id, "instructions": question.text}
    if question.kind == "choice":
        payload["choices"] = [{"value": option} for option in question.options or ()]
    elif question.kind == "score":
        payload["levels"] = [{"label": level} for level in question.levels or ()]
    return payload


def _estimated_tokens(text: str) -> int:
    return int(math.ceil(len(text) / 4.0))


def _number(value: object) -> Optional[float]:
    return finite_float(value)


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


class OpenAIDecisionsProvider:
    """Decision provider that calls the OpenAI Decisions API over HTTPS.

    One HTTP request per row carries all of that row's questions (at most 400 per request).
    Prices are caller-supplied in USD per million tokens; there are no default prices or
    model. ``cost_usd`` and the token-rate estimate need both the input and the cached-input rate (else None;
    a cost cap then cannot be enforced). ``cost_usd`` is computed from returned token usage and excludes surcharges (it is a
    lower bound on the billed amount); the cost estimate is a heuristic (about four characters
    per token, every input token at the uncached rate), not a bound. ``price_per_question_usd``
    is a flat per-question price; when set it is used for the estimate.

    :param api_key: API key; only read from this argument
    :param model: model id
    :param base_url: API base URL
    :param input_usd_per_mtok: uncached input price per million tokens
    :param cached_input_usd_per_mtok: cached input price per million tokens
    :param cache_write_usd_per_mtok: cache-write price per million tokens; defaults to the input price
    :param output_usd_per_mtok: output price per million tokens; output adds no cost when None
    :param price_per_question_usd: flat per-question price used for estimates
    :param concurrency_limit: concurrent requests the kernel may send
    :param timeout_s: per-request timeout, capped by the remaining deadline
    :param max_retries: retries on 408, 409, 429, 5xx and connection errors; failures become ``error``
        results with ``error_code`` ``http_<status>``, ``transport_error``, ``request_error`` (a request that cannot
        be sent, such as a malformed URL; not retried), ``deadline_exceeded`` or ``invalid_provider_result``
        (malformed or misaligned responses)
    :param retry_backoff_s: base delay between retries when the server sends no ``Retry-After``
    :param session: object with a ``requests``-style ``post``; default ``requests``

    **Example**
        ::

            provider = OpenAIDecisionsProvider(api_key=key, model=model_id,
                                               input_usd_per_mtok=in_rate, cached_input_usd_per_mtok=cached_rate)
            with bind_providers({"openai": provider}):
                out = g.gfql([call("sem_decide", {..., "provider": "openai"})])
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        input_usd_per_mtok: Optional[float] = None,
        cached_input_usd_per_mtok: Optional[float] = None,
        cache_write_usd_per_mtok: Optional[float] = None,
        output_usd_per_mtok: Optional[float] = None,
        price_per_question_usd: Optional[float] = None,
        concurrency_limit: int = 4,
        timeout_s: float = 60,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
        session: Optional[HttpPoster] = None,
        name: str = "openai_decisions",
    ) -> None:
        if not isinstance(api_key, str) or api_key == "":
            raise ValueError("api_key must be a non-empty string")
        if not isinstance(model, str) or model == "":
            raise ValueError("model must be a non-empty string")
        for label, price in (("input_usd_per_mtok", input_usd_per_mtok),
                             ("cached_input_usd_per_mtok", cached_input_usd_per_mtok),
                             ("cache_write_usd_per_mtok", cache_write_usd_per_mtok),
                             ("output_usd_per_mtok", output_usd_per_mtok),
                             ("price_per_question_usd", price_per_question_usd)):
            if price is not None and (_number(price) is None or price < 0):
                raise ValueError(f"{label} must be None or a finite number >= 0, got {price!r}")
        if _number(timeout_s) is None or timeout_s <= 0:
            raise ValueError(f"timeout_s must be a finite number > 0, got {timeout_s!r}")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError(f"max_retries must be an int >= 0, got {max_retries!r}")
        if _number(retry_backoff_s) is None or retry_backoff_s < 0:
            raise ValueError(f"retry_backoff_s must be a finite number >= 0, got {retry_backoff_s!r}")
        self._api_key = api_key
        self.model = model
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.input_usd_per_mtok = input_usd_per_mtok
        self.cached_input_usd_per_mtok = cached_input_usd_per_mtok
        self.cache_write_usd_per_mtok = cache_write_usd_per_mtok
        self.output_usd_per_mtok = output_usd_per_mtok
        self.price_per_question_usd = price_per_question_usd
        self.timeout_s = float(timeout_s)
        self.max_retries = max_retries
        self.retry_backoff_s = float(retry_backoff_s)
        self._session = session
        self._capabilities = ProviderCapabilities(
            supports_native_probabilities=True, max_questions_per_request=MAX_QUESTIONS_PER_REQUEST,
            max_evidence_tokens=None, supports_shared_prefix_cache=True, concurrency_limit=concurrency_limit,
            max_rows_per_request=1, max_choice_options=MAX_CHOICE_OPTIONS, max_levels=MAX_LEVELS,
        )

    def __repr__(self) -> str:
        return f"OpenAIDecisionsProvider(model={self.model!r}, base_url={self.base_url!r})"

    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def estimate_cost_usd(self, requests_: Sequence[DecisionRequest]) -> Optional[float]:
        questions = sum(len(request.questions) for request in requests_)
        if self.price_per_question_usd is not None:
            return self.price_per_question_usd * questions
        if self.input_usd_per_mtok is None or self.cached_input_usd_per_mtok is None:
            return None
        tokens = 0
        for request in requests_:
            tokens += _estimated_tokens(evidence_text(request.evidence))
            tokens += sum(_estimated_tokens(question.text) for question in request.questions)
        return tokens * self.input_usd_per_mtok / 1e6

    def decide_batch(self, requests_: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        deadline_at = None if deadline_s is None else time.monotonic() + deadline_s
        results: List[DecisionResult] = []
        for request in requests_:
            questions = request.questions
            for start in range(0, len(questions), MAX_QUESTIONS_PER_REQUEST):
                results.extend(self._decide_chunk(request, questions[start:start + MAX_QUESTIONS_PER_REQUEST],
                                                  deadline_at))
        return results

    def _decide_chunk(self, request: DecisionRequest, questions: Sequence[DecisionQuestion],
                      deadline_at: Optional[float]) -> List[DecisionResult]:
        body: JSONObject = {
            "model": self.model,
            "input": evidence_text(request.evidence),
            "questions": [question_payload(question) for question in questions],
        }
        try:
            payload, retries, latency_ms = self._post(body, deadline_at)
        except _RequestFailure as failure:
            return [error_result(request, question, error_code=failure.error_code, provider=self.name,
                                 model=self.model, latency_ms=failure.latency_ms, retry_count=failure.retry_count)
                    for question in questions]
        answers = payload.get("answers")
        usage = payload.get("usage")
        model = payload.get("model") if isinstance(payload.get("model"), str) else self.model
        aligned = isinstance(answers, list) and len(answers) == len(questions) and all(
            isinstance(answer, dict) and answer.get("name") == question.id
            for answer, question in zip(answers, questions)
        )
        if not aligned or not isinstance(answers, list):
            return [error_result(request, question, error_code="invalid_provider_result", provider=self.name,
                                 model=str(model), latency_ms=latency_ms, retry_count=retries)
                    for question in questions]
        cached_tokens = self._cached_tokens(usage) if isinstance(usage, dict) else None
        cost = self._cost_usd(usage)
        share = None if cost is None else cost / len(questions)
        return [self._map_answer(request, question, answer, str(model), latency_ms, retries, share, cached_tokens)
                for question, answer in zip(questions, answers)]

    def _post(self, body: JSONObject, deadline_at: Optional[float]) -> Tuple[JSONObject, int, float]:
        url = f"{self.base_url}/decisions"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        retries = 0
        started = time.monotonic()
        while True:
            remaining = None if deadline_at is None else deadline_at - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise _RequestFailure("deadline_exceeded", retries, (time.monotonic() - started) * 1000)
            timeout = self.timeout_s if remaining is None else min(self.timeout_s, remaining)
            retry_after: Optional[float] = None
            try:
                if self._session is not None:
                    response = self._session.post(url, json=body, headers=headers, timeout=timeout)
                else:
                    response = requests.post(url, json=body, headers=headers, timeout=timeout)
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ContentDecodingError, requests.exceptions.RetryError):
                failure_code = "transport_error"
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else None
                if status is None:
                    raise _RequestFailure("request_error", retries, (time.monotonic() - started) * 1000)
                failure_code = f"http_{status}"
                if status not in RETRYABLE_STATUSES and status < 500:
                    raise _RequestFailure(failure_code, retries, (time.monotonic() - started) * 1000)
            except requests.RequestException:
                raise _RequestFailure("request_error", retries, (time.monotonic() - started) * 1000)
            else:
                status = response.status_code
                if status == 200:
                    try:
                        payload = response.json()
                    except ValueError:
                        raise _RequestFailure("invalid_provider_result", retries,
                                              (time.monotonic() - started) * 1000)
                    if not isinstance(payload, dict):
                        raise _RequestFailure("invalid_provider_result", retries,
                                              (time.monotonic() - started) * 1000)
                    return payload, retries, (time.monotonic() - started) * 1000
                failure_code = f"http_{status}"
                if status not in RETRYABLE_STATUSES and status < 500:
                    raise _RequestFailure(failure_code, retries, (time.monotonic() - started) * 1000)
                retry_after = _number(_header(response, "Retry-After"))
            if retries >= self.max_retries:
                raise _RequestFailure(failure_code, retries, (time.monotonic() - started) * 1000)
            delay = min(MAX_RETRY_DELAY_S,
                        retry_after if retry_after is not None else self.retry_backoff_s * (2 ** retries))
            remaining = None if deadline_at is None else deadline_at - time.monotonic()
            if remaining is not None and delay >= remaining:
                raise _RequestFailure(failure_code, retries, (time.monotonic() - started) * 1000)
            if delay > 0:
                time.sleep(delay)
            retries += 1

    def _cached_tokens(self, usage: object) -> int:
        details = usage.get("input_tokens_details") if isinstance(usage, dict) else None
        return _int_or_zero(details.get("cached_tokens")) if isinstance(details, dict) else 0

    def _cost_usd(self, usage: object) -> Optional[float]:
        if self.input_usd_per_mtok is None or self.cached_input_usd_per_mtok is None or not isinstance(usage, dict):
            return None
        details = usage.get("input_tokens_details")
        details = details if isinstance(details, dict) else {}
        input_tokens = _int_or_zero(usage.get("input_tokens"))
        cached = _int_or_zero(details.get("cached_tokens"))
        cache_write = _int_or_zero(details.get("cache_write_tokens"))
        output_tokens = _int_or_zero(usage.get("output_tokens"))
        uncached = max(0, input_tokens - cached - cache_write)
        write_rate = self.cache_write_usd_per_mtok if self.cache_write_usd_per_mtok is not None \
            else self.input_usd_per_mtok
        total = uncached * self.input_usd_per_mtok + cached * self.cached_input_usd_per_mtok \
            + cache_write * write_rate
        if self.output_usd_per_mtok is not None:
            total += output_tokens * self.output_usd_per_mtok
        return total / 1e6

    def _map_answer(self, request: DecisionRequest, question: DecisionQuestion, answer: Mapping[str, object],
                    model: str, latency_ms: float, retries: int, cost: Optional[float],
                    cached_tokens: Optional[int]) -> DecisionResult:
        def invalid() -> DecisionResult:
            return error_result(request, question, error_code="invalid_provider_result", provider=self.name,
                                model=model, latency_ms=latency_ms, retry_count=retries)

        extra: Dict[str, ExtraValue] = {} if cached_tokens is None else {"cached_input_tokens": cached_tokens}
        kind = answer.get("type")
        value: Union[bool, str, None] = None
        distribution: Optional[Dict[str, float]] = None
        confidence: Optional[float] = None
        status_reason: Optional[StatusReason] = None
        if kind == "refusal":
            status_reason = "refusal"
        elif kind != question.kind:
            return invalid()
        elif kind == "predicate":
            probability = _number(answer.get("probability"))
            if probability is None or not 0.0 <= probability <= 1.0:
                return invalid()
            distribution = {"true": probability, "false": 1.0 - probability}
            value = None if probability == 0.5 else probability > 0.5
            status_reason = "tie" if value is None else None
        elif kind == "choice":
            chosen = answer.get("choice")
            if not isinstance(chosen, str) or chosen not in (question.options or ()):
                return invalid()
            value = chosen
            valid, parsed = self._label_probabilities(answer.get("probabilities"), question.options or (),
                                                      by_index=False)
            if not valid:
                return invalid()
            distribution = parsed
            confidence = _number(answer.get("confidence"))
        else:
            levels = question.levels or ()
            valid, parsed = self._label_probabilities(answer.get("probabilities"), levels, by_index=True)
            if not valid or parsed is None or len(parsed) != len(levels):
                return invalid()
            distribution = parsed
            top = max(parsed.values())
            leaders = [level for level in levels if parsed[level] == top]
            value = leaders[0] if len(leaders) == 1 else None
            status_reason = "tie" if value is None else None
            score = _number(answer.get("score"))
            if score is not None:
                extra["score_index"] = score
            confidence = _number(answer.get("confidence"))
        return DecisionResult(
            request_id=request.request_id, question_id=question.id, value=value, distribution=distribution,
            confidence=confidence, probability_kind="native_decision" if distribution is not None else "none",
            status="uncertain" if status_reason is not None else "ok", error_code=None, model=model,
            provider=self.name, task_version=request.task_version,
            input_hash=compute_decision_hash(request.evidence, question, request.task_version),
            latency_ms=latency_ms, cost_usd=cost, cache_hit=None if cached_tokens is None else cached_tokens > 0,
            retry_count=retries,
            status_reason=status_reason, extra=extra,
        )

    @staticmethod
    def _label_probabilities(entries: object, labels: Sequence[str],
                             by_index: bool) -> Tuple[bool, Optional[Dict[str, float]]]:
        """``(valid, labelled probabilities)``; probabilities are None for a missing or empty list."""
        if entries is None:
            return True, None
        if not isinstance(entries, list):
            return False, None
        if not entries:
            return True, None
        out: Dict[str, float] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                return False, None
            probability = _number(entry.get("probability"))
            if probability is None or not 0.0 <= probability <= 1.0:
                return False, None
            if by_index:
                index = entry.get("value")
                label = entry.get("label")
                if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(labels) \
                        or label != labels[index]:
                    return False, None
                key = labels[index]
            else:
                key_value = entry.get("value")
                if not isinstance(key_value, str) or key_value not in labels:
                    return False, None
                key = key_value
            if key in out:
                return False, None
            out[key] = probability
        return True, out


def _header(response: object, name: str) -> object:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get(name)
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
