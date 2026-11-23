"""Providers backed by plain Python functions, for tests, rules and local models."""
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union

from graphistry.compute.semantic.contract import (
    PROBABILITY_KINDS, Budget, DecisionQuestion, DecisionRequest, DecisionResult, Evidence, GenerationRequest,
    GenerationResult, ProbabilityKind, ProviderCapabilities, canonicalize, compute_decision_hash, compute_generation_hash,
    unanswered_generation_result, unanswered_result)


CallableAnswer = Union[bool, str, None, Tuple[Union[bool, str, None], Optional[Mapping[str, float]]]]
DecisionFn = Callable[[Evidence, DecisionQuestion], CallableAnswer]
GenerationFn = Callable[[Evidence, str], Optional[str]]


class CallableDecisionProvider:
    """Decision provider that calls ``fn(evidence, question)`` once per (row, question).

    ``fn`` returns the answer (a bool for predicates, a label otherwise, None for no answer) or an
    ``(answer, distribution)`` pair. No answer is ``uncertain`` and drops any distribution. An exception
    from ``fn`` fails only that answer (``callable_exception``); the operator rejects answers that break
    the contract (``invalid_provider_result``), such as a distribution while ``probability_kind="none"``.
    ``fn`` may run on up to ``capabilities.concurrency_limit`` threads at once within one operator run (default 4).
    A predicate distribution needs both ``"true"`` and ``"false"``.

    **Example**
        ::

            provider = CallableDecisionProvider(lambda evidence, q: "acme" in str(evidence).lower())
    """

    def __init__(self, fn: DecisionFn, *, name: str = "callable", probability_kind: ProbabilityKind = "none",
                 capabilities: Optional[ProviderCapabilities] = None, model: Optional[str] = None) -> None:
        if not callable(fn) or probability_kind not in PROBABILITY_KINDS:
            raise ValueError(f"fn must be callable and probability_kind one of {PROBABILITY_KINDS}")
        self.fn = fn
        self.name = name
        self.model = model if model is not None else name
        self.probability_kind: ProbabilityKind = probability_kind
        self._capabilities = capabilities or ProviderCapabilities(
            supports_native_probabilities=probability_kind == "native_decision", max_questions_per_request=1000,
            concurrency_limit=4)

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
            return unanswered_result(request, question, provider=self.name, model=self.model,
                                     error_code="callable_exception", retry_count=0)
        raw_value, raw_distribution = answer if isinstance(answer, tuple) and len(answer) == 2 else (answer, None)
        value = canonicalize(raw_value)
        distribution = canonicalize(raw_distribution) if value is not None else None
        if not isinstance(value, (bool, str, type(None))) or not isinstance(distribution, (dict, type(None))):
            return unanswered_result(request, question, provider=self.name, model=self.model,
                                     error_code="invalid_provider_result", retry_count=0)
        distribution = distribution or None
        return DecisionResult(
            request_id=request.request_id, question_id=question.id, value=value, distribution=distribution,
            confidence=None, probability_kind=self.probability_kind if distribution is not None else "none",
            status="ok" if value is not None else "uncertain", error_code=None, model=self.model,
            provider=self.name, task_version=request.task_version,
            input_hash=compute_decision_hash(request.evidence, question, request.task_version),
            latency_ms=None, cost_usd=0.0, cache_hit=None, retry_count=0,
            status_reason=None if value is not None else "null_answer",
        )


class CallableGenerationProvider:
    """Generation provider that calls ``fn(evidence, instruction)`` once per row.

    ``fn`` returns a string, or None for no answer (``uncertain``). An exception fails only that row
    (``callable_exception``); a value that is not a string is ``invalid_provider_result``.
    """

    def __init__(self, fn: GenerationFn, *, name: str = "callable",
                 capabilities: Optional[ProviderCapabilities] = None, model: Optional[str] = None) -> None:
        if not callable(fn):
            raise ValueError("fn must be callable")
        self.fn = fn
        self.name = name
        self.model = model if model is not None else name
        self._capabilities = capabilities or ProviderCapabilities(
            supports_native_probabilities=False, max_questions_per_request=1, concurrency_limit=4)

    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def generate_batch(self, requests: Sequence[GenerationRequest], *, deadline_s: Optional[float],
                       budget: Optional[Budget]) -> List[GenerationResult]:
        return [self._generate_one(request) for request in requests]

    def _generate_one(self, request: GenerationRequest) -> GenerationResult:
        try:
            value = canonicalize(self.fn(request.evidence, request.instruction))
        except Exception:
            return unanswered_generation_result(request, provider=self.name, model=self.model,
                                                error_code="callable_exception", retry_count=0)
        if not isinstance(value, (str, type(None))):
            return unanswered_generation_result(request, provider=self.name, model=self.model,
                                                error_code="invalid_provider_result", retry_count=0)
        return GenerationResult(
            request_id=request.request_id, value=value, status="ok" if value is not None else "uncertain",
            error_code=None, model=self.model, provider=self.name, task_version=request.task_version,
            input_hash=compute_generation_hash(request.evidence, request.instruction, request.task_version),
            latency_ms=None, cost_usd=0.0, cache_hit=None, retry_count=0,
            status_reason=None if value is not None else "null_answer",
        )
