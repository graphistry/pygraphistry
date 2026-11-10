import ast
import dataclasses
import datetime
import hashlib
import math
import subprocess
import sys
import typing
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from graphistry.compute.semantic import contract as c
from graphistry.compute.semantic.contract import (
    Budget,
    DecisionProvider,
    DecisionQuestion,
    DecisionRequest,
    DecisionResult,
    GenerationProvider,
    GenerationRequest,
    GenerationResult,
    ProviderCapabilities,
)


CONTRACT_PATH = Path(c.__file__)

PRED = DecisionQuestion(id="p", kind="predicate", text="Is it red?")
CHOICE = DecisionQuestion(id="c", kind="choice", text="Which?", options=("a", "b", "c"))
SCORE = DecisionQuestion(id="s", kind="score", text="How much?", levels=("low", "mid", "high"))


def make_request(questions: Sequence[DecisionQuestion] = (PRED,), request_id: str = "r0") -> DecisionRequest:
    return DecisionRequest(request_id=request_id, evidence={"name": "x"}, questions=tuple(questions),
                           task_id="t", task_version="1")


def make_result(question: DecisionQuestion = PRED, **overrides: object) -> DecisionResult:
    fields = dict(
        request_id="r0", question_id=question.id, value=True, distribution={"true": 0.8, "false": 0.2},
        confidence=None, probability_kind="native_decision", status="ok", error_code=None, model="m",
        provider="prov", task_version="1", input_hash="h", latency_ms=1.0, cost_usd=0.0, cache_hit=False,
        retry_count=0,
    )
    fields.update(overrides)
    return DecisionResult(**fields)  # type: ignore[arg-type]


def make_generation_result(**overrides: object) -> GenerationResult:
    fields = dict(
        request_id="r0", value="text", status="ok", error_code=None, model="m", provider="prov",
        task_version="1", input_hash="h", latency_ms=None, cost_usd=None, cache_hit=None, retry_count=None,
    )
    fields.update(overrides)
    return GenerationResult(**fields)  # type: ignore[arg-type]


def native_caps(native: bool = True) -> ProviderCapabilities:
    return ProviderCapabilities(supports_native_probabilities=native, max_questions_per_request=10,
                                max_evidence_tokens=None, supports_shared_prefix_cache=False, concurrency_limit=1)


class TestFrozen:

    @pytest.mark.parametrize("instance,field_name", [
        (PRED, "text"),
        (make_request(), "request_id"),
        (Budget(max_requests=1), "max_requests"),
        (GenerationRequest(request_id="r", evidence="e", instruction="i", task_id="t", task_version="1"), "instruction"),
        (make_generation_result(), "value"),
        (make_result(), "value"),
        (native_caps(), "concurrency_limit"),
    ])
    def test_instances_are_frozen(self, instance: object, field_name: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, None)


class TestConstruction:

    def test_lists_become_tuples(self) -> None:
        q = DecisionQuestion(id="c", kind="choice", text="t", options=["a", "b"])  # type: ignore[arg-type]
        assert q.options == ("a", "b")
        req = DecisionRequest(request_id="r", evidence="e", questions=[q], task_id="t",  # type: ignore[arg-type]
                              task_version="1")
        assert isinstance(req.questions, tuple)
        caps = ProviderCapabilities(True, 1, None, False, 1, supported_kinds=["choice"])  # type: ignore[arg-type]
        assert caps.supported_kinds == ("choice",)

    @pytest.mark.parametrize("kwargs", [
        dict(id="", kind="predicate", text="t"),
        dict(id="q", kind="predicate", text=""),
        dict(id="q", kind="nope", text="t"),
        dict(id="q", kind="choice", text="t", options=("a",)),
        dict(id="q", kind="choice", text="t", options=("a", "a")),
        dict(id="q", kind="choice", text="t", options=("a", "")),
        dict(id="q", kind="choice", text="t", options=("a", "b"), levels=("x", "y")),
        dict(id="q", kind="score", text="t", levels=("x",)),
        dict(id="q", kind="score", text="t", levels=("x", "y"), options=("a", "b")),
        dict(id="q", kind="predicate", text="t", options=("a", "b")),
        dict(id="q", kind="choice", text="t", options="ab"),
        dict(id="q", kind="predicate", text="t", levels=("x", "y")),
        dict(id="q", kind="choice", text="t", options=None),
        dict(id="q", kind="score", text="t", levels=None),
    ])
    def test_question_checks(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            DecisionQuestion(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        dict(request_id=""),
        dict(task_id=""),
        dict(task_version=""),
        dict(questions=()),
        dict(questions=(PRED, PRED)),
        dict(questions=("p",)),
        dict(deadline_s=0),
        dict(deadline_s=float("inf")),
        dict(max_cost_usd=-1.0),
    ])
    def test_request_checks(self, kwargs: dict) -> None:
        fields = dict(request_id="r", evidence="e", questions=(PRED,), task_id="t", task_version="1")
        fields.update(kwargs)
        with pytest.raises(ValueError):
            DecisionRequest(**fields)  # type: ignore[arg-type]

    @pytest.mark.parametrize("kwargs", [
        dict(request_id=""), dict(task_id=""), dict(evidence=5), dict(deadline_s=0), dict(max_cost_usd=-1.0),
    ])
    def test_generation_request_common_checks(self, kwargs: dict) -> None:
        fields = dict(request_id="r", evidence="e", instruction="i", task_id="t", task_version="1")
        fields.update(kwargs)
        with pytest.raises(ValueError):
            GenerationRequest(**fields)  # type: ignore[arg-type]

    def test_boundary_values_are_accepted(self) -> None:
        DecisionRequest(request_id="r", evidence="e", questions=(PRED,), task_id="t", task_version="1",
                        deadline_s=0.5, max_cost_usd=0.0)
        GenerationRequest(request_id="r", evidence="e", instruction="i", task_id="t", task_version="1",
                          max_output_tokens=1, deadline_s=2)
        ProviderCapabilities(True, 1, 1, False, 1, max_rows_per_request=1, max_choice_options=2, max_levels=2)
        DecisionQuestion(id="c", kind="choice", text="t", options=("a", "b"))

    def test_generation_request_checks(self) -> None:
        with pytest.raises(ValueError):
            GenerationRequest(request_id="r", evidence="e", instruction="", task_id="t", task_version="1")
        with pytest.raises(ValueError):
            GenerationRequest(request_id="r", evidence="e", instruction="i", task_id="t", task_version="1", max_output_tokens=0)

    @pytest.mark.parametrize("kwargs", [
        dict(max_questions_per_request=0),
        dict(concurrency_limit=0),
        dict(concurrency_limit=True),
        dict(max_evidence_tokens=0),
        dict(max_rows_per_request=0),
        dict(max_choice_options=1),
        dict(max_levels=1),
        dict(supported_kinds=()),
        dict(supported_kinds=("nope",)),
    ])
    def test_capability_checks(self, kwargs: dict) -> None:
        fields = dict(supports_native_probabilities=True, max_questions_per_request=1, max_evidence_tokens=None,
                      supports_shared_prefix_cache=False, concurrency_limit=1)
        fields.update(kwargs)
        with pytest.raises(ValueError):
            ProviderCapabilities(**fields)  # type: ignore[arg-type]

    def test_more_construction_rules(self) -> None:
        q = DecisionQuestion(id="s", kind="score", text="t", levels=["lo", "hi"])  # type: ignore[arg-type]
        assert q.levels == ("lo", "hi") and q.distribution_keys() == ("lo", "hi")
        assert PRED.distribution_keys() == ("true", "false")
        with pytest.raises(ValueError):
            DecisionQuestion(id="c", kind="choice", text="t", options=(1, 2))  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            DecisionRequest(request_id="r", evidence="e", questions=(PRED,), task_id="t", task_version="1",
                            deadline_s=float("nan"))
        with pytest.raises(ValueError):
            DecisionRequest(request_id="r", evidence=5, questions=(PRED,), task_id="t",  # type: ignore[arg-type]
                            task_version="1")
        for big in (float("inf"), 10 ** 400):
            with pytest.raises(ValueError):
                Budget(max_cost_usd=big)
        with pytest.raises(ValueError):
            ProviderCapabilities("false", 1, None, False, 1)  # type: ignore[arg-type]

    def test_vocabularies_are_exact(self) -> None:
        assert c.QUESTION_KINDS == ("predicate", "choice", "score")
        assert c.PROBABILITY_KINDS == ("native_decision", "token_logprob", "verbalized", "none")
        assert c.RESULT_STATUSES == ("ok", "uncertain", "error", "skipped")
        assert c.UNCERTAIN_REASONS == ("refusal", "null_answer", "tie", "min_p")
        assert c.SKIPPED_REASONS == ("max_rows", "deadline", "budget", "null_evidence")

    def test_literal_tuples_cover_status_reason(self) -> None:
        members = {value for part in typing.get_args(c.StatusReason) for value in typing.get_args(part)}
        assert set(c.UNCERTAIN_REASONS) | set(c.SKIPPED_REASONS) == members

    def test_budget_checks(self) -> None:
        with pytest.raises(ValueError):
            Budget(max_cost_usd=-0.1)
        with pytest.raises(ValueError):
            Budget(max_requests=0)
        assert Budget().max_requests is None

    def test_duplicate_request_ids_in_one_call(self) -> None:
        with pytest.raises(ValueError, match="duplicate request_id"):
            c.check_requests([make_request(), make_request()])
        c.check_requests([make_request(request_id="a"), make_request(request_id="b")])


class TestFromMapping:

    def test_question_from_mapping(self) -> None:
        spec = {"id": "c", "kind": np.str_("choice"), "text": "Which?", "options": ["a", "b", "c"]}
        question = DecisionQuestion.from_mapping(spec)
        assert question == CHOICE and type(question.kind) is str
        direct = DecisionQuestion(id="c", kind=np.str_("choice"), text="Which?", options=("a", "b", "c"))  # type: ignore[arg-type]
        assert direct == CHOICE and type(direct.kind) is str
        for bad, match in (({"id": "p", "kind": "predicate", "text": "t", "extra": 1}, "unknown question keys"),
                           ({"id": 1, "kind": "nope", "text": ""}, "question id"),
                           ({"id": "p", "kind": "nope", "text": ""}, "text"),
                           ({"id": "p", "kind": "nope", "text": "t"}, "kind must be one of"),
                           ({"id": "p", "kind": "choice", "text": "t", "options": "ab"}, "options must be a list")):
            with pytest.raises(ValueError, match=match):
                DecisionQuestion.from_mapping(bad)

    def test_budget_from_mapping(self) -> None:
        budget = Budget.from_mapping({"max_cost_usd": np.float32(0.5), "max_requests": np.int64(3)})
        assert budget == Budget(max_cost_usd=0.5, max_requests=3)
        assert type(budget.max_cost_usd) is float and type(budget.max_requests) is int
        assert Budget.from_mapping({}) == Budget()
        assert Budget.from_mapping({"max_cost_usd": None, "max_requests": None}) == Budget()
        assert Budget.from_mapping({"max_cost_usd": np.float64(2.5)}).max_cost_usd == 2.5
        for missing in (float("nan"), np.float64("nan"), pd.NA, pd.NaT):
            with pytest.raises(ValueError, match="max_cost_usd"):
                Budget.from_mapping({"max_cost_usd": missing})
            with pytest.raises(ValueError, match="max_requests"):
                Budget.from_mapping({"max_requests": missing})
        for bad, match in (({"max_rows": 1}, "unknown budget keys"), ({"max_cost_usd": -1}, "max_cost_usd"),
                           ({"max_cost_usd": True}, "max_cost_usd"), ({"max_requests": 0}, "max_requests"),
                           ({"max_requests": 2.0}, "max_requests")):
            with pytest.raises(ValueError, match=match):
                Budget.from_mapping(bad)


class TestResultChecks:

    def test_valid_results_pass(self) -> None:
        req = make_request((PRED, CHOICE, SCORE))
        results = [
            make_result(PRED),
            make_result(CHOICE, value="b", distribution={"a": 0.1, "b": 0.8, "c": 0.1}, confidence=0.93),
            make_result(SCORE, value="mid", distribution={"mid": 0.6}, probability_kind="verbalized",
                        confidence=1.0, extra={"probability_mechanism": "verbalized", "score_index": 1.5,
                                               "evidence_truncated": True, "cached_input_tokens": 3}),
        ]
        assert c.check_decision_results([req], results, native_caps()) == []
        c.assert_decision_results_valid([req], results, native_caps())

    @pytest.mark.parametrize("question,overrides,fragment", [
        (PRED, dict(status="maybe"), "status must be"),
        (PRED, dict(probability_kind="guess"), "probability_kind must be"),
        (PRED, dict(status="error", error_code=None, value=None, distribution=None, probability_kind="none"),
         "error_code"),
        (PRED, dict(error_code="x"), "error_code must be None"),
        (PRED, dict(status="error", error_code="x", distribution=None, probability_kind="none"), "value None"),
        (PRED, dict(status="skipped", status_reason="budget", value=None), "distribution None"),
        (PRED, dict(status="ok", value=None, distribution=None, probability_kind="none"), "must carry a value"),
        (PRED, dict(value="yes"), "predicate value"),
        (CHOICE, dict(value="zzz", distribution=None, probability_kind="none"), "not one of"),
        (PRED, dict(distribution={"true": 1.3, "false": 0.0}), "in [0, 1]"),
        (PRED, dict(distribution={"true": 0.0, "false": 0.0}), "sum to 1"),
        (PRED, dict(distribution={"yes": 1.0}), "not one of"),
        (PRED, dict(distribution={}), "empty distribution"),
        (PRED, dict(probability_kind="none"), "exactly when distribution is None"),
        (PRED, dict(distribution=None), "exactly when distribution is None"),
        (PRED, dict(confidence=1.5), "confidence"),
        (PRED, dict(status="error", error_code="x", value=None, distribution=None, probability_kind="none",
                    confidence=0.4), "confidence None"),
        (PRED, dict(status="uncertain", status_reason="max_rows"), "not allowed"),
        (PRED, dict(status="ok", status_reason="min_p"), "not allowed"),
        (PRED, dict(status="uncertain", status_reason="refusal", value=None), "refusal result"),
        (PRED, dict(retry_count=-1), "retry_count"),
        (PRED, dict(latency_ms=-1.0), "latency_ms"),
        (PRED, dict(cost_usd=float("nan")), "cost_usd"),
        (PRED, dict(cache_hit="yes"), "cache_hit"),
        (PRED, dict(extra={"k": [1]}), "extra entry"),
        (PRED, dict(question_id="other"), "question_id"),
        (PRED, dict(request_id="other"), "request_id"),
        (PRED, dict(status="skipped", status_reason="budget", value=True, distribution=None,
                    probability_kind="none"), "value None"),
        (PRED, dict(status="skipped", status_reason="budget", value=None, distribution=None,
                    probability_kind="none", confidence=0.3), "confidence None"),
        (PRED, dict(status="skipped", value=None, distribution=None, probability_kind="none"), "needs a status_reason"),
        (PRED, dict(status="skipped", status_reason="tie", value=None, distribution=None, probability_kind="none"),
         "needs a status_reason"),
        (PRED, dict(status="skipped", status_reason="budget", value=None, distribution=None, probability_kind="none",
                    error_code="x"), "error_code must be None"),
        (PRED, dict(status="error", error_code="x", value=None), "distribution None"),
        (PRED, dict(status="uncertain", status_reason="null_answer", value=None), "distribution None"),
        (PRED, dict(status="uncertain", status_reason="tie"), "value None"),
        (SCORE, dict(value="extreme", distribution=None, probability_kind="none"), "not one of"),
        (PRED, dict(distribution={"true": True, "false": 0.0}), "number in [0, 1]"),
        (PRED, dict(distribution={"true": 0.5, "false": 0.511}), "sum to 1"),
        (CHOICE, dict(value="a", distribution={"a": 0.9, "b": 0.9}), "partial distribution"),
        (PRED, dict(distribution=[0.8, 0.2]), "mapping"),
        (PRED, dict(model=3), "model must be a string"),
        (PRED, dict(extra="x"), "extra must be None or a mapping"),
        (PRED, dict(extra={"score_index": "high"}), "wrong type"),
        (PRED, dict(extra={"evidence_truncated": 1}), "wrong type"),
        (PRED, dict(extra={"cached_input_tokens": 1.5}), "wrong type"),
        (PRED, dict(extra={"cached_input_tokens": True}), "wrong type"),
        (PRED, dict(extra={"k": float("inf")}), "finite"),
        (PRED, dict(latency_ms=10 ** 400), "latency_ms"),
        (PRED, dict(retry_count=True), "retry_count"),
        (PRED, dict(retry_count=1.5), "retry_count"),
        (PRED, dict(confidence="x"), "confidence"),
        (PRED, dict(status=pd.NA), "status must be a string"),
        (PRED, dict(status_reason=pd.NA), "status_reason must be a string"),
        (PRED, dict(probability_kind=pd.NA), "probability_kind must be a string"),
        (PRED, dict(status=np.array(["ok", "x"])), "status must be a string"),
        (PRED, dict(question_id=["p"]), "question_id must be a string"),
        (PRED, dict(distribution={pd.NA: 0.5, "false": 0.5}), "not one of"),
        (CHOICE, dict(value="a", distribution={"a": -0.1, "b": 0.6, "c": 0.5}), "number in [0, 1]"),
        (PRED, dict(extra={"foo": "bar", "score_index": None}), None),
        (PRED, dict(probability_kind="token_logprob"), None),
        (PRED, dict(status="uncertain", status_reason="min_p"), None),
        (PRED, dict(status="skipped", status_reason="max_rows", value=None, distribution=None, probability_kind="none"), None),
        (PRED, dict(status="skipped", status_reason="null_evidence", value=None, distribution=None,
                    probability_kind="none"), None),
        (PRED, dict(status="skipped", status_reason="refusal", value=None, distribution=None, probability_kind="none"),
         "needs a status_reason"),
        (PRED, dict(request_id=pd.NA), "request_id must be a string"),
        (PRED, dict(confidence=-0.1), "confidence"),
        (PRED, dict(extra={1: "x"}), "extra entry"),
        (PRED, dict(status="error", error_code="", value=None, distribution=None, probability_kind="none"),
         "error_code"),
        (PRED, dict(status="error", error_code="x", value=None, distribution=None, probability_kind="none",
                    status_reason="budget"), "not allowed"),
        (PRED, dict(status="uncertain", status_reason="max_rows", value=None, distribution=None,
                    probability_kind="none"), "not allowed"),
    ])
    def test_each_check_fails_once(self, question: DecisionQuestion, overrides: dict, fragment: Optional[str]) -> None:
        problems = c.check_decision_result(make_request((question,)), question, make_result(question, **overrides))
        if fragment is None:
            assert problems == []
        else:
            assert len(problems) == 1 and fragment in problems[0], problems

    def test_native_decision_needs_native_capability(self) -> None:
        problems = c.check_decision_result(make_request(), PRED, make_result(), native_caps(native=False))
        assert any("native_decision" in problem for problem in problems)
        assert c.check_decision_result(make_request(), PRED, make_result(), native_caps(native=True)) == []

    def test_confidence_is_kept_verbatim_without_distribution(self) -> None:
        result = make_result(CHOICE, value="a", distribution=None, probability_kind="none", confidence=0.55)
        assert c.check_decision_result(make_request((CHOICE,)), CHOICE, result) == []

    def test_sum_tolerance_edge_and_slack(self) -> None:
        edge = make_result(PRED, distribution={"true": 0.2, "false": 0.81})
        assert c.check_decision_result(make_request(), PRED, edge) == []
        partial = make_result(CHOICE, value="a", distribution={"a": 0.2, "b": 0.81})
        assert c.check_decision_result(make_request((CHOICE,)), CHOICE, partial) == []
        over = make_result(PRED, distribution={"true": 0.2, "false": 0.8101})
        assert c.check_decision_result(make_request(), PRED, over) != []

    def test_sum_tolerance_boundary(self) -> None:
        for second, ok in ((0.509, True), (0.495, True), (0.48, False)):
            result = make_result(PRED, distribution={"true": 0.5, "false": second})
            assert (c.check_decision_result(make_request(), PRED, result) == []) is ok

    def test_robust_to_malformed_returns(self) -> None:
        req = make_request()
        assert c.check_decision_results([req], None) != []  # type: ignore[arg-type]
        assert any("strings" in p for p in c.check_decision_results([req], [make_result(request_id=["r0"])]))
        assert any("DecisionResult" in p for p in c.check_decision_results([req], [object()]))  # type: ignore[list-item]
        assert any("strings" in p for p in c.check_decision_results([req], [make_result(question_id=["p"])]))
        problems = c.check_decision_results([req], [make_result()], native_caps(native=False))
        assert any("native_decision" in p for p in problems)

    def test_partial_distribution_is_not_summed(self) -> None:
        result = make_result(CHOICE, value="a", distribution={"a": 0.4})
        assert c.check_decision_result(make_request((CHOICE,)), CHOICE, result) == []

    def test_alignment_problems(self) -> None:
        req = make_request((PRED, CHOICE))
        ok_p = make_result(PRED)
        ok_c = make_result(CHOICE, value="a", distribution=None, probability_kind="none")
        assert any("missing result" in p for p in c.check_decision_results([req], [ok_p]))
        assert any("duplicate result" in p for p in c.check_decision_results([req], [ok_p, ok_p, ok_c]))
        stray = make_result(PRED, request_id="nope")
        assert any("unexpected result" in p for p in c.check_decision_results([req], [ok_p, ok_c, stray]))
        with pytest.raises(ValueError, match="invalid decision results"):
            c.assert_decision_results_valid([req], [ok_p])

    def test_skipped_and_error_helpers_are_valid(self) -> None:
        req = make_request((PRED, SCORE))
        results = [
            c.error_result(req, PRED, error_code="provider_exception", provider="p", model="m"),
            c.skipped_result(req, SCORE, provider="p", model="m", status_reason="budget"),
        ]
        assert c.check_decision_results([req], results) == []
        assert results[0].input_hash == c.compute_decision_hash(req.evidence, PRED, "1")

    def test_generation_checks(self) -> None:
        req = GenerationRequest(request_id="r0", evidence="e", instruction="i", task_id="t", task_version="1")
        assert c.check_generation_results([req], [make_generation_result()]) == []
        assert any("string or None" in p for p in c.check_generation_result(req, make_generation_result(value=3)))
        assert any("does not apply" in p for p in c.check_generation_result(
            req, make_generation_result(status="uncertain", value=None, status_reason="tie")))
        assert any("does not apply" in p for p in c.check_generation_result(
            req, make_generation_result(status="uncertain", value=None, status_reason="min_p")))
        assert any("missing result" in p for p in c.check_generation_results([req], []))
        good = make_generation_result()
        assert any("duplicate" in p for p in c.check_generation_results([req], [good, good]))
        assert any("unexpected" in p for p in c.check_generation_results([req], [good, make_generation_result(request_id="x")]))
        assert any("does not match" in p for p in c.check_generation_result(req, make_generation_result(request_id="x")))
        for reason in ("refusal", "null_answer"):
            bad = make_generation_result(status="uncertain", status_reason=reason)
            assert any("value None" in p for p in c.check_generation_result(req, bad))
        assert c.check_generation_results([req], None) != []  # type: ignore[arg-type]
        assert c.check_generation_results([req], [make_generation_result(value=3)]) != []
        c.assert_generation_results_valid([req], [make_generation_result()])
        assert any("must be a string" in p for p in c.check_generation_results([req], [make_generation_result(request_id=["r0"])]))
        assert any("status must be a string" in p for p in c.check_generation_result(req, make_generation_result(status=pd.NA)))
        assert any("status_reason must be a string" in p for p in c.check_generation_result(
            req, make_generation_result(status="uncertain", value=None, status_reason=pd.NA)))
        assert any("request_id must be a string" in p for p in c.check_generation_result(
            req, make_generation_result(request_id=pd.NA)))
        assert any("GenerationResult" in p for p in c.check_generation_results([req], [object()]))  # type: ignore[list-item]
        with pytest.raises(TypeError):
            c.generation_skipped_result(req, provider="p", model="m")  # type: ignore[call-arg]
        helpers = [c.generation_error_result(req, error_code="x", provider="p", model="m")]
        assert c.check_generation_results([req], helpers) == []
        skipped = c.generation_skipped_result(req, provider="p", model="m", status_reason="deadline")
        assert c.check_generation_results([req], [skipped]) == []
        with pytest.raises(ValueError, match="invalid generation results"):
            c.assert_generation_results_valid([req], [])


class TestHash:

    def test_golden_value(self) -> None:
        digest = c.compute_decision_hash({"name": "Acme", "n": 3}, CHOICE, "1")
        expected_json = (
            '{"evidence":{"n":3,"name":"Acme"},'
            '"question":{"kind":"choice","levels":null,"options":["a","b","c"],"text":"Which?"},'
            '"task_version":"1"}'
        )
        assert digest == hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
        assert digest == "d387c1409ffbd87ab4f68116fda20e7682efd5508b7d77d8dd81f524f8d78052"

    def test_excludes_question_id(self) -> None:
        renamed = DecisionQuestion(id="other", kind="choice", text="Which?", options=("a", "b", "c"))
        assert c.compute_decision_hash("e", CHOICE, "1") == c.compute_decision_hash("e", renamed, "1")
        assert c.compute_decision_hash("e", CHOICE, "1") != c.compute_decision_hash("e", CHOICE, "2")
        assert c.compute_decision_hash("e", CHOICE, "1") != c.compute_decision_hash("e2", CHOICE, "1")

    @pytest.mark.parametrize("left,right", [
        ({"x": np.int64(3)}, {"x": 3}),
        ({"x": np.float64(1.5)}, {"x": 1.5}),
        ({"x": float("nan")}, {"x": None}),
        ({"x": pd.NA}, {"x": None}),
        ({"x": pd.NaT}, {"x": None}),
        ({"x": (1, 2)}, {"x": [1, 2]}),
        ({"b": 1, "a": 2}, {"a": 2, "b": 1}),
        ({"x": [float("nan"), np.int64(1)]}, {"x": [None, 1]}),
        ({"x": (np.float64(1.5),)}, {"x": [1.5]}),
        ({"x": np.longdouble(1.5)}, {"x": 1.5}),
        ({"x": np.clongdouble(1 + 2j)}, {"x": np.complex128(1 + 2j)}),
    ])
    def test_canonicalization(self, left: dict, right: dict) -> None:
        assert c.compute_decision_hash(left, PRED, "1") == c.compute_decision_hash(right, PRED, "1")

    def test_non_ascii_is_stable(self) -> None:
        assert c.compute_generation_hash("café ☕", "summarize", "1") == c.compute_generation_hash(
            "café ☕", "summarize", "1")
        assert c.compute_generation_hash("café", "summarize", "1") != c.compute_generation_hash(
            "cafe", "summarize", "1")

    def test_canonicalize_arrays_and_scalars(self) -> None:
        assert c.canonicalize(np.array([1, 2])) == [1, 2]
        assert c.canonicalize(np.bool_(True)) is True
        assert math.isclose(c.canonicalize(np.float32(0.5)), 0.5)  # type: ignore[arg-type]

    def test_large_arrays_are_not_truncated(self) -> None:
        a, b = np.arange(2000), np.arange(2000)
        b[1000] = -1
        assert c.compute_decision_hash({"x": a}, PRED, "1") != c.compute_decision_hash({"x": b}, PRED, "1")

    @pytest.mark.parametrize("value", [np.longdouble(1.5), np.clongdouble(1 + 2j)])
    def test_extended_precision_scalars_terminate(self, value: object) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c.canonicalize(value)
        assert len(c.compute_decision_hash({"x": value}, PRED, "1")) == 64
        req = make_request()
        assert c.error_result(req, PRED, error_code="x", provider="p", model="m").status == "error"

    def test_masked_and_out_of_range_scalars(self) -> None:
        assert c.canonicalize(np.ma.masked) is None
        huge = np.longdouble("1e4000")
        if np.isfinite(huge):
            assert c.canonicalize(huge) is huge
            assert c.compute_decision_hash({"x": huge}, PRED, "1") != c.compute_decision_hash(
                {"x": np.longdouble("1e4001")}, PRED, "1")
        assert type(c.canonicalize(np.longdouble("inf"))) is float
        tiny = np.longdouble("1e-4000")
        if tiny != 0:
            assert c.canonicalize(tiny) == 0.0
        assert type(c.canonicalize(np.longdouble("0.1"))) is float

    def test_helpers_stamp_input_hashes(self) -> None:
        req = make_request((PRED, SCORE))
        assert c.skipped_result(req, SCORE, provider="p", model="m", status_reason="budget").input_hash == \
            c.compute_decision_hash(req.evidence, SCORE, "1")
        greq = GenerationRequest(request_id="r0", evidence="e", instruction="i", task_id="t", task_version="1")
        expected = c.compute_generation_hash("e", "i", "1")
        assert c.generation_error_result(greq, error_code="x", provider="p", model="m").input_hash == expected
        assert c.generation_skipped_result(greq, provider="p", model="m", status_reason="deadline").input_hash == expected
        with pytest.raises(TypeError):
            c.skipped_result(req, SCORE, provider="p", model="m")  # type: ignore[call-arg]

    def test_cost_estimate_validation(self) -> None:
        assert c.valid_cost_estimate(0.0) == 0.0 and c.valid_cost_estimate(1) == 1.0
        for bad in (None, float("nan"), -1.0, float("inf"), True, "1"):
            assert c.valid_cost_estimate(bad) is None
        assert c.valid_cost_estimate(np.int64(0)) == 0.0 and c.valid_cost_estimate(np.float32(0.5)) == 0.5

    def test_complex_scalars_keep_their_imaginary_part(self) -> None:
        h = lambda v: c.compute_decision_hash({"x": v}, PRED, "1")  # noqa: E731
        assert h(np.clongdouble(1 + 2j)) != h(np.clongdouble(1 + 3j))
        assert h(np.clongdouble(1 + 2j)) != h(1.0)
        assert h(np.longdouble(1.5)) == h(1.5)

    def test_set_hash_is_independent_of_hash_seed(self) -> None:
        code = (
            "import importlib.util as u, sys; "
            f"s = u.spec_from_file_location('sc', {str(CONTRACT_PATH)!r}); m = u.module_from_spec(s); "
            "sys.modules['sc'] = m; s.loader.exec_module(m); "
            "print(m.compute_generation_hash({'x': {'alpha', 'beta', 'gamma', 'delta'}}, 'i', '1'))"
        )
        digests = {
            subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True,
                           env={"PYTHONHASHSEED": seed}).stdout.strip()
            for seed in ("1", "2", "3")
        }
        assert len(digests) == 1

    def test_generation_golden_value(self) -> None:
        expected_json = '{"evidence":"café","instruction":"Summarize","task_version":"2"}'
        digest = c.compute_generation_hash("café", "Summarize", "2")
        assert digest == hashlib.sha256(expected_json.encode("utf-8")).hexdigest()

    def test_default_str_and_int_keys(self) -> None:
        expected_json = ('{"evidence":{"1":"2024-01-02"},"question":{"kind":"predicate","levels":null,'
                         '"options":null,"text":"Is it red?"},"task_version":"1"}')
        digest = c.compute_decision_hash({1: datetime.date(2024, 1, 2)}, PRED, "1")
        assert digest == hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
        assert c.canonicalize({None: 1, True: 2}) == {"None": 1, "True": 2}


class _FakeDecider:
    name = "fake"
    model = "m"

    def capabilities(self) -> ProviderCapabilities:
        return native_caps()

    def estimate_cost_usd(self, requests: Sequence[DecisionRequest]) -> Optional[float]:
        return None

    def decide_batch(self, requests: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        return []


class _NotAProvider:
    name = "nope"

    def capabilities(self) -> ProviderCapabilities:
        return native_caps()


class _FakeGenerator:
    name = "gen"
    model = "m"

    def capabilities(self) -> ProviderCapabilities:
        return native_caps()

    def estimate_cost_usd(self, requests: Sequence[GenerationRequest]) -> Optional[float]:
        return None

    def generate_batch(self, requests: Sequence[GenerationRequest], *, deadline_s: Optional[float],
                       budget: Optional[Budget]) -> List[GenerationResult]:
        return []


def _without(cls: type, member: str) -> object:
    attrs = {key: value for key, value in vars(cls).items() if key != member and not key.startswith("__")}
    return type("Partial", (), attrs)()


class TestProtocols:

    def test_runtime_checkable(self) -> None:
        assert isinstance(_FakeDecider(), DecisionProvider)
        assert not isinstance(_FakeDecider(), GenerationProvider)
        assert isinstance(_FakeGenerator(), GenerationProvider)
        assert not isinstance(_NotAProvider(), DecisionProvider)

    @pytest.mark.parametrize("member", ["name", "model", "capabilities", "estimate_cost_usd", "decide_batch"])
    def test_each_member_is_required(self, member: str) -> None:
        assert not isinstance(_without(_FakeDecider, member), DecisionProvider)

    @pytest.mark.parametrize("member", ["name", "model", "capabilities", "estimate_cost_usd", "generate_batch"])
    def test_each_generation_member_is_required(self, member: str) -> None:
        assert not isinstance(_without(_FakeGenerator, member), GenerationProvider)


class TestModuleHygiene:

    def test_parses_as_python38(self) -> None:
        ast.parse(CONTRACT_PATH.read_text(encoding="utf-8"), feature_version=(3, 8))

    def test_imports_only_stdlib(self) -> None:
        allowed = {"__future__", "dataclasses", "hashlib", "json", "math", "typing"}
        tree = ast.parse(CONTRACT_PATH.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert imported <= allowed, imported - allowed

    def test_imports_by_path_in_isolated_interpreter(self) -> None:
        code = (
            "import importlib.util as u, sys; "
            f"s = u.spec_from_file_location('semantic_contract', {str(CONTRACT_PATH)!r}); "
            "m = u.module_from_spec(s); sys.modules['semantic_contract'] = m; s.loader.exec_module(m); "
            "assert 'pandas' not in sys.modules and 'graphistry' not in sys.modules; print('ok')"
        )
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "ok"
