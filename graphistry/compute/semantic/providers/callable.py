"""Providers backed by plain Python functions, for tests, rules and local models."""
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from graphistry.compute.semantic.contract import (
    Budget,
    DecisionQuestion,
    DecisionRequest,
    DecisionResult,
    Evidence,
    GenerationRequest,
    GenerationResult,
    ProbabilityKind,
    ProviderCapabilities,
    canonicalize,
    compute_decision_hash,
    compute_generation_hash,
    error_result,
    generation_error_result,
)


CallableAnswer = Union[bool, str, None, Tuple[Union[bool, str, None], Optional[Mapping[str, float]]]]
DecisionFn = Callable[[Evidence, DecisionQuestion], CallableAnswer]
GenerationFn = Callable[[Evidence, str], Optional[str]]


def _default_capabilities(native: bool) -> ProviderCapabilities:
    return ProviderCapabilities(supports_native_probabilities=native, max_questions_per_request=1000,
                                max_evidence_tokens=None, supports_shared_prefix_cache=False, concurrency_limit=4)


class CallableDecisionProvider:
    """Decision provider that calls ``fn(evidence, question)`` once per (request, question).

    ``fn`` returns a value (bool for predicates, a label for choice/score, None for no answer) or a
    ``(value, distribution)`` pair. An exception from ``fn`` makes only that result an ``error``
    (``callable_exception``). A distribution returned while ``probability_kind="none"`` is an
    ``error`` (``invalid_provider_result``); a value without a distribution reports kind ``none``.
    A value that is not None, a bool or a string, or a non-numeric probability next to an answer, is
    ``invalid_provider_result``; with no answer (``None``) the result is ``uncertain`` and any
    distribution is dropped.

    **Example**
        ::

            provider = CallableDecisionProvider(lambda evidence, q: "acme" in str(evidence).lower())
    """

    def __init__(self, fn: DecisionFn, *, name: str = "callable", probability_kind: ProbabilityKind = "none",
                 capabilities: Optional[ProviderCapabilities] = None, model: Optional[str] = None) -> None:
        if not callable(fn):
            raise ValueError("fn must be callable")
        if probability_kind not in ("native_decision", "token_logprob", "verbalized", "none"):
            raise ValueError(f"unknown probability_kind {probability_kind!r}")
        self.fn = fn
        self.name = name
        self.model = model if model is not None else name
        self.probability_kind: ProbabilityKind = probability_kind
        self._capabilities = capabilities or _default_capabilities(probability_kind == "native_decision")

    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def estimate_cost_usd(self, requests: Sequence[DecisionRequest]) -> Optional[float]:
        return 0.0

    def decide_batch(self, requests: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        return [self._decide_one(request, question) for request in requests for question in request.questions]

    def _decide_one(self, request: DecisionRequest, question: DecisionQuestion) -> DecisionResult:
        try:
            answer = self.fn(request.evidence, question)
        except Exception:
            return error_result(request, question, error_code="callable_exception", provider=self.name,
                                model=self.model)

        def invalid() -> DecisionResult:
            return error_result(request, question, error_code="invalid_provider_result", provider=self.name,
                                model=self.model)

        try:
            raw_value, raw_distribution = answer if isinstance(answer, tuple) else (answer, None)
            value = canonicalize(raw_value)
            probabilities = None if not raw_distribution else {
                str(key): canonicalize(probability) for key, probability in raw_distribution.items()
            }
        except (TypeError, ValueError, AttributeError):
            return invalid()
        if not (value is None or isinstance(value, (bool, str))):
            return invalid()
        if probabilities is not None and self.probability_kind == "none":
            return invalid()
        distribution: Optional[Dict[str, float]] = None
        if probabilities is not None and value is not None:
            distribution = {}
            for key, number in probabilities.items():
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    return invalid()
                distribution[key] = number
        return DecisionResult(
            request_id=request.request_id, question_id=question.id, value=value,
            distribution=distribution,
            confidence=None, probability_kind=self.probability_kind if distribution is not None else "none",
            status="ok" if value is not None else "uncertain", error_code=None, model=self.model,
            provider=self.name, task_version=request.task_version,
            input_hash=compute_decision_hash(request.evidence, question, request.task_version),
            latency_ms=None, cost_usd=0.0, cache_hit=None, retry_count=0,
            status_reason=None if value is not None else "null_answer",
        )


class CallableGenerationProvider:
    """Generation provider that calls ``fn(evidence, instruction)`` once per request.

    ``fn`` returns a string, or None for no answer (``uncertain``). An exception makes only that
    result an ``error`` (``callable_exception``); a non-string value is ``invalid_provider_result``.
    """

    def __init__(self, fn: GenerationFn, *, name: str = "callable",
                 capabilities: Optional[ProviderCapabilities] = None, model: Optional[str] = None) -> None:
        if not callable(fn):
            raise ValueError("fn must be callable")
        self.fn = fn
        self.name = name
        self.model = model if model is not None else name
        self._capabilities = capabilities or _default_capabilities(False)

    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def estimate_cost_usd(self, requests: Sequence[GenerationRequest]) -> Optional[float]:
        return 0.0

    def generate_batch(self, requests: Sequence[GenerationRequest], *, deadline_s: Optional[float],
                       budget: Optional[Budget]) -> List[GenerationResult]:
        return [self._generate_one(request) for request in requests]

    def _generate_one(self, request: GenerationRequest) -> GenerationResult:
        try:
            value = self.fn(request.evidence, request.instruction)
        except Exception:
            return generation_error_result(request, error_code="callable_exception", provider=self.name,
                                           model=self.model)
        if value is not None and not isinstance(value, str):
            return generation_error_result(request, error_code="invalid_provider_result", provider=self.name,
                                           model=self.model)
        return GenerationResult(
            request_id=request.request_id, value=value, status="ok" if value is not None else "uncertain",
            error_code=None, model=self.model, provider=self.name, task_version=request.task_version,
            input_hash=compute_generation_hash(request.evidence, request.instruction, request.task_version),
            latency_ms=None, cost_usd=0.0, cache_hit=None, retry_count=0,
            status_reason=None if value is not None else "null_answer",
        )
