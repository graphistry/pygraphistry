"""Shared questions, frames and providers for the semantic operator tests."""
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from graphistry.compute.semantic.contract import (
    Budget, DecisionQuestion, DecisionRequest, DecisionResult, GenerationRequest, GenerationResult, ProviderCapabilities)
from graphistry.compute.semantic.providers.callable import CallableDecisionProvider, CallableGenerationProvider


P = {"id": "p", "kind": "predicate", "text": "Is it notable?"}
C = {"id": "c", "kind": "choice", "text": "Which group?", "options": ["a", "b", "c"]}
S = {"id": "s", "kind": "score", "text": "How much?", "levels": ["low", "mid", "high"]}
# pandas' native text dtypes: object on pandas 2 (and for all-null columns), str on pandas 3
TEXT_DTYPES = ("object", "str")


def caps(native: bool = False, **limits: object) -> ProviderCapabilities:
    fields: Dict[str, object] = dict(supports_native_probabilities=native, max_questions_per_request=1000,
                                     concurrency_limit=8)
    fields.update(limits)
    return ProviderCapabilities(**fields)  # type: ignore[arg-type]


def frame(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame({"name": [f"n{i}" for i in range(rows)], "note": [f"note {i}" for i in range(rows)]})


def value_only(evidence: object, question: DecisionQuestion) -> object:
    return True if question.kind == "predicate" else (question.labels() or ("",))[0]


def full_answer(evidence: object, question: DecisionQuestion) -> Tuple[object, Dict[str, float]]:
    if question.kind == "predicate":
        return True, {"true": 0.9, "false": 0.1}
    labels = question.labels() or ()
    return labels[1], {label: 0.6 if i == 1 else 0.4 / (len(labels) - 1) for i, label in enumerate(labels)}


class Recording(CallableDecisionProvider):
    """Callable provider that records every call: request ids, question ids, budget, deadline, peak concurrency."""

    def __init__(self, fn: Callable = value_only, delay: float = 0.0, **kwargs: object) -> None:
        super().__init__(fn, **kwargs)  # type: ignore[arg-type]
        self.calls: List[List[str]] = []
        self.task_ids: List[str] = []
        self.question_sets: List[List[str]] = []
        self.budgets: List[Optional[Budget]] = []
        self.deadlines: List[Optional[float]] = []
        self.answers = 0
        self.delay = delay
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def decide_batch(self, requests: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        with self.lock:
            self.calls.append([request.request_id for request in requests])
            self.task_ids.extend(request.task_id for request in requests)
            self.question_sets.append([question.id for question in requests[0].questions])
            self.budgets.append(budget)
            self.deadlines.append(deadline_s)
            self.answers += sum(len(request.questions) for request in requests)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            return super().decide_batch(requests, deadline_s=deadline_s, budget=budget)
        finally:
            with self.lock:
                self.active -= 1

    @property
    def request_ids(self) -> List[str]:
        return [request_id for call in self.calls for request_id in call]


class RecordingGenerator(CallableGenerationProvider):
    """Callable generation provider that records every request, the limits each call received and peak concurrency."""

    def __init__(self, fn: Callable[[object, str], object] = lambda evidence, instruction: "x", delay: float = 0.0,
                 **kwargs: object) -> None:
        super().__init__(fn, **kwargs)  # type: ignore[arg-type]
        self.calls: List[List[GenerationRequest]] = []
        self.limits: List[Tuple[Optional[float], Optional[Budget]]] = []
        self.delay = delay
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def generate_batch(self, requests: Sequence[GenerationRequest], *, deadline_s: Optional[float],
                       budget: Optional[Budget]) -> List[GenerationResult]:
        with self.lock:
            self.calls.append(list(requests))
            self.limits.append((deadline_s, budget))
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            return super().generate_batch(requests, deadline_s=deadline_s, budget=budget)
        finally:
            with self.lock:
                self.active -= 1


class Scripted:
    """Provider whose results come from ``script(call_index, requests)``, for crafting malformed returns."""

    name = "raw"
    model = "raw-m"

    def __init__(self, script: Callable[[int, Sequence[DecisionRequest]], object],
                 capabilities: Optional[ProviderCapabilities] = None) -> None:
        self.script = script
        self._capabilities = capabilities or caps(native=True)
        self.count = 0

    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def estimate_cost_usd(self, requests: Sequence[DecisionRequest]) -> Optional[float]:
        return 0.0

    def decide_batch(self, requests: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        self.count += 1
        return self.script(self.count - 1, requests)  # type: ignore[return-value]


def result(request: DecisionRequest, question_id: str, value: object, dist: Optional[Dict[str, float]] = None,
           **overrides: object) -> DecisionResult:
    fields: Dict[str, object] = dict(
        request_id=request.request_id, question_id=question_id, value=value, distribution=dist, confidence=None,
        probability_kind="native_decision" if dist is not None else "none", status="ok", error_code=None,
        model="raw-m", provider="raw", task_version="1", input_hash="placeholder", latency_ms=1.0, cost_usd=0.0,
        cache_hit=False, retry_count=0)
    fields.update(overrides)
    return DecisionResult(**fields)  # type: ignore[arg-type]
