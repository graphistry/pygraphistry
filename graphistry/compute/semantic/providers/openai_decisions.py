"""OpenAI Decisions API (``POST /v1/decisions``) as a ``DecisionProvider``.

Public API guide: https://developers.openai.com/api/docs/guides/decisions
API reference: https://developers.openai.com/api/reference/resources/decisions/methods/create
"""
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from graphistry.compute.semantic.contract import (
    DecisionQuestion, DecisionRequest, DecisionResult, ProviderCapabilities, StatusReason,
    finite_float)
from graphistry.compute.semantic.providers.http import (
    HttpDecisionProvider, HttpPoster, JSONObject, ResponseMeta, TokenPrices, TokenUsage, evidence_text, token_count)


MAX_QUESTIONS_PER_REQUEST = 400
Answer = Tuple[Union[bool, str, None], Optional[Dict[str, float]], Optional[float], Optional[StatusReason]]


def question_payload(question: DecisionQuestion) -> JSONObject:
    payload: JSONObject = {"type": question.kind, "name": question.id, "instructions": question.text}
    if question.kind == "choice":
        payload["choices"] = [{"value": option} for option in question.options or ()]
    elif question.kind == "score":
        payload["levels"] = [{"label": level} for level in question.levels or ()]
    return payload


def _probability(entry: object) -> Optional[float]:
    number = finite_float(entry.get("probability")) if isinstance(entry, dict) else None
    return number if number is not None and 0.0 <= number <= 1.0 else None


def _keyed_probabilities(entries: object, key_of: Callable[[Dict[str, object]], Optional[str]]
                         ) -> Optional[Dict[str, float]]:
    """Probabilities by label, ``{}`` for a missing or empty list, None when any entry is malformed or repeated."""
    if entries is None or entries == []:
        return {}
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        return None
    out: Dict[str, float] = {}
    for entry in entries:
        key, probability = key_of(entry), _probability(entry)
        if key is None or probability is None or key in out:
            return None
        out[key] = probability
    return out


def _predicate(question: DecisionQuestion, answer: Mapping[str, object]) -> Optional[Answer]:
    probability = _probability(answer)
    if probability is None:
        return None
    value = None if probability == 0.5 else probability > 0.5
    return value, {"true": probability, "false": 1.0 - probability}, None, "tie" if value is None else None


def _choice(question: DecisionQuestion, answer: Mapping[str, object]) -> Optional[Answer]:
    options = question.options or ()
    chosen = answer.get("choice")
    def option_of(entry: Dict[str, object]) -> Optional[str]:
        value = entry.get("value")
        return value if isinstance(value, str) and value in options else None

    distribution = _keyed_probabilities(answer.get("probabilities"), option_of)
    if not isinstance(chosen, str) or chosen not in options or distribution is None:
        return None
    return chosen, distribution or None, finite_float(answer.get("confidence")), None


def _score(question: DecisionQuestion, answer: Mapping[str, object]) -> Optional[Answer]:
    levels = question.levels or ()

    def level_of(entry: Dict[str, object]) -> Optional[str]:
        index = entry.get("value")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(levels):
            return None
        return levels[index] if entry.get("label") == levels[index] else None

    distribution = _keyed_probabilities(answer.get("probabilities"), level_of)
    if not distribution or len(distribution) != len(levels):
        return None
    top = max(distribution.values())
    leaders = [level for level in levels if distribution[level] == top]
    value = leaders[0] if len(leaders) == 1 else None
    return value, distribution, finite_float(answer.get("confidence")), "tie" if value is None else None


MAPPERS = {"predicate": _predicate, "choice": _choice, "score": _score}


class OpenAIDecisionsProvider(HttpDecisionProvider):
    """Calls the OpenAI Decisions API; one HTTP request per row carries that row's questions (up to
    ``max_questions_per_request``, default 400).

    Prices come from ``prices`` (a ``TokenPrices``); there are no default prices or model. Servers with the same
    API and smaller limits (for example a self-hosted engine at ``base_url``) set ``max_questions_per_request``,
    ``max_choice_options`` and ``max_levels``. The other constructor parameters are those of
    :class:`~graphistry.compute.semantic.providers.http.HttpDecisionProvider`.

    **Example**
        ::

            provider = OpenAIDecisionsProvider(api_key=key, model=model_id,
                                               prices=TokenPrices(input=in_rate, cached_input=cached_rate))
            with bind_providers({"openai": provider}):
                out = g.gfql([call("sem_decide", {..., "provider": "openai"})])
    """

    name = "openai_decisions"
    default_base_url = "https://api.openai.com/v1"
    path = "/decisions"

    def __init__(self, api_key: str, model: str, *, base_url: Optional[str] = None, prices: Optional[TokenPrices] = None,
                 max_questions_per_request: int = MAX_QUESTIONS_PER_REQUEST, max_choice_options: int = 255,
                 max_levels: int = 10, concurrency_limit: int = 4, timeout_s: float = 60, max_retries: int = 3,
                 retry_backoff_s: float = 0.5, session: Optional[HttpPoster] = None, name: Optional[str] = None) -> None:
        super().__init__(api_key, model, base_url=base_url, prices=prices, concurrency_limit=concurrency_limit,
                         timeout_s=timeout_s, max_retries=max_retries, retry_backoff_s=retry_backoff_s,
                         session=session, name=name)
        for setting, value, least in (("max_questions_per_request", max_questions_per_request, 1),
                                      ("max_choice_options", max_choice_options, 2), ("max_levels", max_levels, 2)):
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ValueError(f"{setting} must be an int >= {least}, got {value!r}")
        self.max_questions_per_request = max_questions_per_request
        self.max_choice_options = max_choice_options
        self.max_levels = max_levels

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(supports_native_probabilities=True,
                                    max_questions_per_request=self.max_questions_per_request,
                                    concurrency_limit=self.concurrency_limit, max_rows_per_request=1,
                                    max_choice_options=self.max_choice_options, max_levels=self.max_levels)

    def question_groups(self, request: DecisionRequest) -> List[Sequence[DecisionQuestion]]:
        questions, size = request.questions, self.max_questions_per_request
        return [questions[start:start + size] for start in range(0, len(questions), size)]

    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def body(self, request: DecisionRequest, questions: Sequence[DecisionQuestion]) -> JSONObject:
        return {"model": self.model, "input": evidence_text(request.evidence),
                "questions": [question_payload(question) for question in questions]}

    def usage(self, payload: Mapping[str, object]) -> Optional[TokenUsage]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        details = usage.get("input_tokens_details")
        details = details if isinstance(details, dict) else {}
        cached, written = token_count(details.get("cached_tokens")), token_count(details.get("cache_write_tokens"))
        return TokenUsage(uncached_input=max(0, token_count(usage.get("input_tokens")) - cached - written),
                          cached_input=cached, cache_write=written, output=token_count(usage.get("output_tokens")))

    def results(self, request: DecisionRequest, questions: Sequence[DecisionQuestion], payload: Mapping[str, object],
                meta: ResponseMeta) -> List[DecisionResult]:
        answers = payload.get("answers")
        if not (isinstance(answers, list) and len(answers) == len(questions) and all(
                isinstance(answer, dict) and answer.get("name") == question.id
                for answer, question in zip(answers, questions))):
            return [self.invalid(request, question, meta) for question in questions]
        return [self._result(request, question, answer, meta) for question, answer in zip(questions, answers)]

    def _result(self, request: DecisionRequest, question: DecisionQuestion, answer: Mapping[str, object],
                meta: ResponseMeta) -> DecisionResult:
        answer_type = answer.get("type")
        if answer_type == "refusal":
            mapped: Optional[Answer] = (None, None, None, "refusal")
        elif answer_type == question.kind:
            mapped = MAPPERS[question.kind](question, answer)
        else:
            mapped = None
        if mapped is None:
            return self.invalid(request, question, meta)
        value, distribution, confidence, reason = mapped
        score_index = finite_float(answer.get("score")) if answer_type == "score" else None
        return self.answered(request, question, meta, value=value, distribution=distribution,
                             probability_kind="native_decision",
                             confidence=confidence, reason=reason,
                             extra={} if score_index is None else {"score_index": score_index})
