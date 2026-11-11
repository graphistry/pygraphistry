import ast
import dataclasses
import datetime
import hashlib
import subprocess
import sys
import typing
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from graphistry.compute.semantic import contract as c
from graphistry.compute.semantic.contract import (
    Budget, DecisionProvider, DecisionQuestion, DecisionRequest, DecisionResult, ProviderCapabilities)
from graphistry.tests.compute.semantic.semantic_test_utils import caps


CONTRACT_PATH = Path(c.__file__)
PRED = DecisionQuestion(id="p", kind="predicate", text="Is it red?")
CHOICE = DecisionQuestion(id="c", kind="choice", text="Which?", options=("a", "b", "c"))
SCORE = DecisionQuestion(id="s", kind="score", text="How much?", levels=("low", "mid", "high"))


def make_request(questions: Sequence[DecisionQuestion] = (PRED,), request_id: str = "r0") -> DecisionRequest:
    return DecisionRequest(request_id=request_id, evidence={"name": "x"}, questions=tuple(questions), task_id="t",
                           task_version="1")


def make_result(question: DecisionQuestion = PRED, **overrides: object) -> DecisionResult:
    fields = dict(request_id="r0", question_id=question.id, value=True, distribution={"true": 0.8, "false": 0.2},
                  confidence=None, probability_kind="native_decision", status="ok", error_code=None, model="m",
                  provider="prov", task_version="1", input_hash="h", latency_ms=1.0, cost_usd=0.0, cache_hit=False,
                  retry_count=0)
    fields.update(overrides)
    return DecisionResult(**fields)  # type: ignore[arg-type]


UNANSWERED = dict(value=None, distribution=None, probability_kind="none")


class TestConstruction:

    @pytest.mark.parametrize("instance,field_name", [
        (PRED, "text"), (make_request(), "request_id"), (Budget(max_requests=1), "max_requests"),
        (make_result(), "value"), (caps(), "concurrency_limit")])
    def test_instances_are_frozen(self, instance: object, field_name: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, None)

    def test_lists_become_tuples_and_numpy_kinds_become_str(self) -> None:
        question = DecisionQuestion(id="c", kind=np.str_("choice"), text="t", options=["a", "b"])  # type: ignore[arg-type]
        assert question.options == ("a", "b") and type(question.kind) is str
        assert isinstance(make_request([PRED]).questions, tuple)  # type: ignore[arg-type]
        assert caps(supported_kinds=["choice"]).supported_kinds == ("choice",)
        assert SCORE.distribution_keys() == ("low", "mid", "high") and PRED.distribution_keys() == ("true", "false")

    @pytest.mark.parametrize("kwargs", [
        dict(id="", kind="predicate", text="t"), dict(id="q", kind="predicate", text=""),
        dict(id="q", kind="nope", text="t"), dict(id="q", kind="choice", text="t", options=("a",)),
        dict(id="q", kind="choice", text="t", options=("a", "a")), dict(id="q", kind="choice", text="t", options=("a", "")),
        dict(id="q", kind="choice", text="t", options=(1, 2)), dict(id="q", kind="choice", text="t", options="ab"),
        dict(id="q", kind="choice", text="t", options=("a", "b"), levels=("x", "y")),
        dict(id="q", kind="choice", text="t"), dict(id="q", kind="score", text="t", levels=("x",)),
        dict(id="q", kind="score", text="t", levels=("x", "y"), options=("a", "b")),
        dict(id="q", kind="predicate", text="t", options=("a", "b")),
        dict(id="q", kind="predicate", text="t", levels=("x", "y")),
    ])
    def test_question_rules(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            DecisionQuestion(**kwargs)

    @pytest.mark.parametrize("limits", [
        dict(max_questions_per_request=0), dict(concurrency_limit=0), dict(concurrency_limit=True),
        dict(max_rows_per_request=0), dict(max_choice_options=1), dict(max_levels=1), dict(supported_kinds=()),
        dict(supported_kinds=("nope",)), dict(native="false"),
    ])
    def test_capability_rules(self, limits: dict) -> None:
        with pytest.raises(ValueError):
            caps(**limits)

    def test_boundary_values_are_accepted(self) -> None:
        caps(max_questions_per_request=1, concurrency_limit=1, max_rows_per_request=1, max_choice_options=2,
             max_levels=2)
        DecisionQuestion(id="c", kind="choice", text="t", options=("a", "b"))
        assert Budget().max_requests is None and Budget(max_requests=1).max_requests == 1
        for bad in (0, 2.5, True):
            with pytest.raises(ValueError):
                Budget(max_requests=bad)  # type: ignore[arg-type]

    def test_vocabularies_are_exact(self) -> None:
        assert c.QUESTION_KINDS == ("predicate", "choice", "score")
        assert c.PROBABILITY_KINDS == ("native_decision", "token_logprob", "verbalized", "none")
        assert c.RESULT_STATUSES == ("ok", "uncertain", "error", "skipped")
        assert c.UNCERTAIN_REASONS == ("refusal", "null_answer", "tie", "min_p")
        assert c.SKIPPED_REASONS == ("max_rows", "deadline", "budget", "null_evidence")
        members = {value for part in typing.get_args(c.StatusReason) for value in typing.get_args(part)}
        assert set(c.UNCERTAIN_REASONS) | set(c.SKIPPED_REASONS) == members


class TestFromMapping:

    @pytest.mark.parametrize("spec,match", [
        ({"id": "p", "kind": "predicate", "text": "t", "extra": 1}, "unknown question keys"),
        ({"id": 1, "kind": "nope", "text": ""}, "question id"),
        ({"id": "p", "kind": "nope", "text": ""}, "text"),
        ({"id": "p", "kind": "nope", "text": "t"}, "kind must be one of"),
        ({"id": "p", "kind": "choice", "text": "t", "options": "ab"}, "options must be a list"),
    ])
    def test_question_errors(self, spec: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            DecisionQuestion.from_mapping(spec)

    def test_question_and_budget_values(self) -> None:
        spec = {"id": "c", "kind": np.str_("choice"), "text": "Which?", "options": ["a", "b", "c"]}
        assert DecisionQuestion.from_mapping(spec) == CHOICE
        budget = Budget.from_mapping({"max_cost_usd": np.float32(0.5), "max_requests": np.int64(3)})
        assert budget == Budget(max_cost_usd=0.5, max_requests=3) and type(budget.max_requests) is int
        assert type(budget.max_cost_usd) is float
        assert Budget.from_mapping({}) == Budget() == Budget.from_mapping({"max_requests": None})

    @pytest.mark.parametrize("spec", [{"max_requests": missing} for missing in (float("nan"), pd.NA, pd.NaT)]
                             + [{"max_cost_usd": missing} for missing in (float("nan"), pd.NA, pd.NaT)]
                             + [{"max_requests": 0}, {"max_requests": 2.0}, {"max_requests": True}, {"max_rows": 1},
                                {"max_cost_usd": -1}, {"max_cost_usd": True}, {"max_cost_usd": float("inf")},
                                {"max_cost_usd": 10 ** 400}])
    def test_budget_errors_never_mean_no_cap(self, spec: dict) -> None:
        with pytest.raises(ValueError):
            Budget.from_mapping(spec)


class TestResultChecks:

    @pytest.mark.parametrize("question,overrides,fragment", [
        (PRED, dict(status="maybe"), "status must be"),
        (PRED, dict(status=pd.NA), "status must be a string"),
        (PRED, dict(status=np.array(["ok", "x"])), "status must be a string"),
        (PRED, dict(probability_kind="guess"), "probability_kind must be"),
        (PRED, dict(probability_kind=pd.NA), "probability_kind must be a string"),
        (PRED, dict(status="error", error_code=None, **UNANSWERED), "error_code"),
        (PRED, dict(status="error", error_code="", **UNANSWERED), "error_code"),
        (PRED, dict(error_code="x"), "error_code must be None"),
        (PRED, dict(status="error", error_code="x", distribution=None, probability_kind="none"), "value None"),
        (PRED, dict(status="error", error_code="x", value=None), "distribution None"),
        (PRED, dict(status="error", error_code="x", confidence=0.4, **UNANSWERED), "confidence None"),
        (PRED, dict(status="error", error_code="x", status_reason="budget", **UNANSWERED), "not allowed"),
        (PRED, dict(status="skipped", status_reason="budget", value=None), "distribution None"),
        (PRED, dict(status="skipped", status_reason="budget", value=True, distribution=None, probability_kind="none"),
         "value None"),
        (PRED, dict(status="skipped", status_reason="budget", confidence=0.3, **UNANSWERED), "confidence None"),
        (PRED, dict(status="skipped", status_reason="budget", error_code="x", **UNANSWERED), "error_code must be None"),
        (PRED, dict(status="skipped", **UNANSWERED), "needs a status_reason"),
        (PRED, dict(status="skipped", status_reason="tie", **UNANSWERED), "needs a status_reason"),
        (PRED, dict(status="ok", value=None, distribution=None, probability_kind="none"), "must carry a value"),
        (PRED, dict(status="ok", status_reason="min_p"), "not allowed"),
        (PRED, dict(status="ok", status_reason=pd.NA), "not allowed"),
        (PRED, dict(status="uncertain", status_reason="max_rows"), "not allowed"),
        (PRED, dict(status="uncertain", status_reason="refusal", value=None), "refusal result"),
        (PRED, dict(status="uncertain", status_reason="null_answer", value=None), "distribution None"),
        (PRED, dict(status="uncertain", status_reason="tie"), "value None"),
        (PRED, dict(value="yes"), "predicate value"),
        (CHOICE, dict(value="zzz", distribution=None, probability_kind="none"), "not one of"),
        (SCORE, dict(value="extreme", distribution=None, probability_kind="none"), "not one of"),
        (PRED, dict(distribution={"true": 1.3, "false": 0.0}), "number in [0, 1]"),
        (PRED, dict(distribution={"true": True, "false": 0.0}), "number in [0, 1]"),
        (CHOICE, dict(value="a", distribution={"a": -0.1, "b": 0.6, "c": 0.5}), "number in [0, 1]"),
        (PRED, dict(distribution={"true": 0.0, "false": 0.0}), "sum to 1"),
        (PRED, dict(distribution={"true": 0.5, "false": 0.511}), "sum to 1"),
        (CHOICE, dict(value="a", distribution={"a": 0.9, "b": 0.9}), "partial distribution"),
        (PRED, dict(distribution={"yes": 1.0}), "not one of"),
        (PRED, dict(distribution={pd.NA: 0.5, "false": 0.5}), "not one of"),
        (PRED, dict(distribution={}), "empty distribution"),
        (PRED, dict(distribution=[0.8, 0.2]), "mapping"),
        (PRED, dict(probability_kind="none"), "exactly when distribution is None"),
        (PRED, dict(distribution=None), "exactly when distribution is None"),
        (PRED, dict(question_id="other"), "question_id"),
        (PRED, dict(question_id=["p"]), "question_id"),
        (PRED, dict(request_id=pd.NA), "request_id"),
        (PRED, dict(probability_kind="token_logprob"), None),
        (PRED, dict(status="uncertain", status_reason="min_p"), None),
        (PRED, dict(status="skipped", status_reason="max_rows", **UNANSWERED), None),
        (PRED, dict(status="skipped", status_reason="null_evidence", **UNANSWERED), None),
        (CHOICE, dict(value="a", distribution=None, probability_kind="none", confidence=0.55), None),
        (CHOICE, dict(value="a", distribution={"a": 0.4}), None),
        (CHOICE, dict(value="a", distribution={"a": 0.2, "b": 0.81}), None),
        (PRED, dict(distribution={"true": 0.2, "false": 0.81}), None),
        (PRED, dict(distribution={"true": 0.5, "false": 0.495}), None),
        (PRED, dict(distribution={"true": 0.2, "false": 0.8101}), "sum to 1"),
    ])
    def test_each_rule_fails_once(self, question: DecisionQuestion, overrides: dict, fragment: Optional[str]) -> None:
        problems = c.check_decision_result(make_request((question,)), question, make_result(question, **overrides))
        assert problems == [] if fragment is None else (len(problems) == 1 and fragment in problems[0]), problems

    def test_native_decision_needs_native_capability(self) -> None:
        assert any("native_decision" in p for p in c.check_decision_result(make_request(), PRED, make_result(),
                                                                           caps(native=False)))
        assert c.check_decision_result(make_request(), PRED, make_result(), caps(native=True)) == []

    def test_unanswered_results_are_valid_and_hashed(self) -> None:
        request = make_request((PRED, SCORE))
        error = c.unanswered_result(request, PRED, provider="p", model="m", error_code="provider_exception")
        skipped = c.unanswered_result(request, SCORE, provider="p", model="m", status_reason="budget")
        assert (error.status, skipped.status) == ("error", "skipped")
        assert c.check_decision_result(request, PRED, error) == c.check_decision_result(request, SCORE, skipped) == []
        assert error.input_hash == c.compute_decision_hash(request.evidence, PRED, "1")


class TestAlignment:

    REQUEST = make_request((PRED, CHOICE))
    OK_P = make_result(PRED)
    OK_C = make_result(CHOICE, value="a", distribution=None, probability_kind="none")

    def test_complete_return_is_keyed_by_request_and_question(self) -> None:
        assert c.align_results([self.REQUEST], [self.OK_C, self.OK_P]) == {"r0": {"p": self.OK_P, "c": self.OK_C}}

    @pytest.mark.parametrize("returned", [
        None, "x", [object()], [make_result(request_id="nope")], [make_result(request_id=["r0"])],
        [make_result(), make_result()],
        [make_result(question_id="zz"), make_result(question_id="zz")],
    ])
    def test_untrustworthy_call(self, returned: object) -> None:
        assert c.align_results([self.REQUEST], returned) is None

    @pytest.mark.parametrize("returned", [[OK_P], [OK_P, OK_C, make_result(question_id="zz")],
                                          [OK_P, OK_C, make_result(question_id=None)]])
    def test_incomplete_or_unknown_row(self, returned: List[DecisionResult]) -> None:
        other = make_request((PRED,), request_id="r1")
        aligned = c.align_results([self.REQUEST, other], returned + [make_result(request_id="r1")])
        assert aligned == {"r0": None, "r1": {"p": make_result(request_id="r1")}}


class TestHash:

    def test_golden_value(self) -> None:
        expected_json = ('{"evidence":{"n":3,"name":"Acme"},'
                         '"question":{"kind":"choice","levels":null,"options":["a","b","c"],"text":"Which?"},'
                         '"task_version":"1"}')
        digest = c.compute_decision_hash({"name": "Acme", "n": 3}, CHOICE, "1")
        assert digest == hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
        assert digest == "d387c1409ffbd87ab4f68116fda20e7682efd5508b7d77d8dd81f524f8d78052"
        stringified = ('{"evidence":{"1":"2024-01-02"},"question":{"kind":"predicate","levels":null,'
                       '"options":null,"text":"Is it red?"},"task_version":"1"}')
        assert c.compute_decision_hash({1: datetime.date(2024, 1, 2)}, PRED, "1") == \
            hashlib.sha256(stringified.encode("utf-8")).hexdigest()

    def test_depends_on_evidence_text_and_version_only(self) -> None:
        renamed = DecisionQuestion(id="other", kind="choice", text="Which?", options=("a", "b", "c"))
        assert c.compute_decision_hash("e", CHOICE, "1") == c.compute_decision_hash("e", renamed, "1")
        assert c.compute_decision_hash("e", CHOICE, "1") != c.compute_decision_hash("e", CHOICE, "2")
        assert c.compute_decision_hash("café", CHOICE, "1") != c.compute_decision_hash("cafe", CHOICE, "1")

    @pytest.mark.parametrize("left,right", [
        ({"x": np.int64(3)}, {"x": 3}), ({"x": np.float64(1.5)}, {"x": 1.5}), ({"x": float("nan")}, {"x": None}),
        ({"x": pd.NA}, {"x": None}), ({"x": pd.NaT}, {"x": None}), ({"x": (1, 2)}, {"x": [1, 2]}),
        ({"b": 1, "a": 2}, {"a": 2, "b": 1}), ({"x": [float("nan"), np.int64(1)]}, {"x": [None, 1]}),
        ({"x": np.array([1.5, 2.5])}, {"x": [1.5, 2.5]}), ({"x": np.bool_(True)}, {"x": True}),
    ])
    def test_canonicalization(self, left: dict, right: dict) -> None:
        assert c.compute_decision_hash(left, PRED, "1") == c.compute_decision_hash(right, PRED, "1")

    def test_large_arrays_are_not_truncated(self) -> None:
        a, b = np.arange(2000), np.arange(2000)
        b[1000] = -1
        assert c.compute_decision_hash({"x": a}, PRED, "1") != c.compute_decision_hash({"x": b}, PRED, "1")
        assert c.canonicalize({None: 1, True: 2}) == {"None": 1, "True": 2}


class _Decider:
    name = "fake"
    model = "m"

    def capabilities(self) -> ProviderCapabilities:
        return caps()

    def estimate_cost_usd(self, requests: Sequence[DecisionRequest]) -> Optional[float]:
        return None

    def decide_batch(self, requests: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        return []


@pytest.mark.parametrize("member", ["name", "model", "capabilities", "estimate_cost_usd", "decide_batch"])
def test_each_protocol_member_is_required(member: str) -> None:
    assert isinstance(_Decider(), DecisionProvider)
    partial = type("Partial", (), {k: v for k, v in vars(_Decider).items() if k != member and not k.startswith("__")})
    assert not isinstance(partial(), DecisionProvider)


class TestModuleHygiene:

    def test_parses_as_python38_and_imports_only_stdlib(self) -> None:
        tree = ast.parse(CONTRACT_PATH.read_text(encoding="utf-8"), feature_version=(3, 8))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert imported <= {"__future__", "dataclasses", "hashlib", "json", "math", "typing"}, imported

    def test_imports_by_path_in_isolated_interpreter(self) -> None:
        code = ("import importlib.util as u, sys; "
                f"s = u.spec_from_file_location('semantic_contract', {str(CONTRACT_PATH)!r}); "
                "m = u.module_from_spec(s); sys.modules['semantic_contract'] = m; s.loader.exec_module(m); "
                "assert 'pandas' not in sys.modules and 'graphistry' not in sys.modules; print('ok')")
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "ok"
