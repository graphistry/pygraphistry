import json
import threading
from collections.abc import Mapping
import time
from dataclasses import replace
from enum import Enum
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

from graphistry.compute.semantic.contract import (
    Budget, DecisionQuestion, DecisionRequest, DecisionResult, compute_decision_hash)
from graphistry.compute.semantic.ops import sem_decide
from graphistry.compute.semantic.providers.callable import CallableDecisionProvider
from graphistry.tests.compute.semantic.semantic_test_utils import (
    C, P, S, TEXT_DTYPES, Recording, Scripted, caps, frame, full_answer, result, value_only)


META_KEYS = {"model", "provider", "task_version", "input_hash", "latency_ms", "cost_usd", "cache_hit",
             "retry_count", "confidence", "status_reason", "extra"}


def decide(df: pd.DataFrame, provider: object, questions: Sequence[object] = (P,), **kwargs: object) -> pd.DataFrame:
    params: Dict[str, object] = dict(evidence_cols=["name"], task_id="t")
    params.update(kwargs)
    return sem_decide(df, questions=list(questions), provider=provider, **params)  # type: ignore[arg-type]


def reasons(out: pd.DataFrame, question_id: str = "p") -> List[object]:
    return [json.loads(meta)["status_reason"] for meta in out[f"{question_id}__meta"]]


class TestOutputs:

    def test_columns_dtypes_and_meta(self) -> None:
        out = decide(frame(), Recording(full_answer, probability_kind="native_decision"), [P, C, S])
        assert len([col for col in out.columns if col not in ("name", "note")]) == 21
        assert str(out["p"].dtype) == "boolean" and str(out["c"].dtype) in TEXT_DTYPES
        assert out["p__p"].dtype == np.float64 and out["p__p"].tolist() == [0.9] * 6 and out["c__p"].tolist() == [0.6] * 6
        assert out["c__dist"].iloc[0] == '{"a":0.2,"b":0.6,"c":0.2}' and out["p__dist"].iloc[0] == '{"true":0.9,"false":0.1}'
        meta = out["s__meta"].iloc[0]
        assert set(json.loads(meta)) == META_KEYS and meta == json.dumps(json.loads(meta), sort_keys=True,
                                                                         separators=(",", ":"))
        assert out["p__error_code"].isna().all() and set(out["p__status"]) == {"ok"}

    def test_value_only_provider_has_no_probabilities(self) -> None:
        out = decide(frame(), Recording(), [P, C, S])
        for q in ("p", "c", "s"):
            assert out[f"{q}__p"].isna().all() and out[f"{q}__dist"].isna().all()
            assert set(out[f"{q}__probability_kind"]) == {"none"} and set(out[f"{q}__status"]) == {"ok"}

    def test_dist_follows_label_order_and_input_is_not_mutated(self) -> None:
        df = frame(1).assign(n=[5])
        before = df.copy()
        provider = Recording(lambda e, q: ("b", {"c": 0.1, "b": 0.6, "a": 0.3}), probability_kind="native_decision")
        out = decide(df, provider, [C])
        assert list(json.loads(out["c__dist"].iloc[0])) == ["a", "b", "c"]
        out.loc[0, "n"] = 99
        out.iloc[0, 0] = "changed"
        pd.testing.assert_frame_equal(df, before)

    def test_kernel_stamps_its_own_input_hash(self) -> None:
        out = decide(frame(1), Scripted(lambda i, reqs: [result(r, "p", True, input_hash="provider-made") for r in reqs]))
        question = DecisionQuestion(id="p", kind="predicate", text="Is it notable?")
        assert json.loads(out["p__meta"].iloc[0])["input_hash"] == compute_decision_hash({"name": "n0"}, question, "1")


class TestRowIdentity:

    def test_order_index_and_batch_invariance(self) -> None:
        index = [3, 1, 3, 0, 9, 9, 2, 7, 5, 5]
        df = pd.DataFrame({"name": [f"n{i}" for i in range(10)]}, index=index)

        def by_name(evidence: Dict[str, object], question: DecisionQuestion) -> object:
            flag = int(str(evidence["name"])[1:]) % 2 == 0
            return flag, {"true": 0.8 if flag else 0.3, "false": 0.2 if flag else 0.7}

        frames = []
        for size in (1, 3, None):
            provider = Recording(by_name, probability_kind="native_decision")
            out = decide(df, provider, batch_size=size)
            assert out.index.tolist() == index and out["name"].tolist() == df["name"].tolist()
            assert sorted(provider.request_ids, key=int) == [str(i) for i in range(10)]
            frames.append(out.drop(columns=["p__meta"]))
        pd.testing.assert_frame_equal(frames[0], frames[1])
        pd.testing.assert_frame_equal(frames[0], frames[2])

    def test_unique_index_labels_become_request_ids(self) -> None:
        provider = Recording()
        decide(pd.DataFrame({"name": ["a", "b"]}, index=["x", 7]), provider)
        assert sorted(provider.request_ids) == ["7", "x"]

    def test_unique_id_col_is_used(self) -> None:
        provider = Recording()
        decide(frame(4).assign(rid=["a", "b", "c", "d"]), provider, id_col="rid")
        assert sorted(provider.request_ids) == ["a", "b", "c", "d"]

    @pytest.mark.parametrize("ids", [[1, 1, 2, 3], [1, None, 2, 3], [1, "1", 2, 3], ["a", "", "b", "c"]])
    def test_bad_ids_raise_before_any_call(self, ids: list) -> None:
        provider = Recording()
        with pytest.raises(ValueError, match="id_col"):
            decide(frame(4).assign(rid=pd.Series(ids, dtype=object)), provider, id_col="rid")
        assert provider.calls == []

    @pytest.mark.parametrize("index", [pd.Index([1, "1", 2], dtype=object), pd.Index([1, 1.0, 2], dtype=object),
                                       pd.Index(["a", "", "b"])])
    def test_index_labels_that_collide_or_are_empty_after_str_use_positions(self, index: pd.Index) -> None:
        df = pd.DataFrame({"name": ["a", "b", "c"]}, index=index)
        frames = []
        for size in (1, 3):
            provider = Recording()
            frames.append(decide(df, provider, batch_size=size).drop(columns=["p__meta"]))
            assert sorted(provider.request_ids) == ["0", "1", "2"]
        pd.testing.assert_frame_equal(frames[0], frames[1])


class TestBatching:

    @pytest.mark.parametrize("limits,rows,sizes", [(dict(max_rows_per_request=2), 5, [1, 2, 2]), ({}, 40, [8, 32])])
    def test_default_batch_size(self, limits: dict, rows: int, sizes: List[int]) -> None:
        provider = Recording(capabilities=caps(**limits))
        decide(frame(rows), provider)
        assert sorted(len(call) for call in provider.calls) == sizes

    def test_questions_beyond_the_cap_go_in_separate_calls(self) -> None:
        provider = Recording(capabilities=caps(max_questions_per_request=2))
        questions = [{"id": f"q{i}", "kind": "predicate", "text": "t"} for i in range(3)]
        out = decide(frame(2), provider, questions)
        assert sorted(provider.question_sets) == [["q0", "q1"], ["q2"]]
        assert all(set(out[f"q{i}__status"]) == {"ok"} for i in range(3))

    def test_numpy_parameters(self) -> None:
        out = decide(frame(5), Recording(), batch_size=np.int64(2), max_rows=np.int64(4),
                     budget={"max_requests": np.int64(2)}, min_p=np.float32(0.7))
        assert out["p__status"].tolist() == ["uncertain"] * 4 + ["skipped"] and set(reasons(out)[:4]) == {"min_p"}

    @pytest.mark.parametrize("limit,kwargs", [(8, dict(max_concurrency=2)), (2, {})])
    def test_exactly_the_allowed_number_of_calls_run_together(self, limit: int, kwargs: dict) -> None:
        both_running = threading.Barrier(2, timeout=10)

        def meet(evidence: object, question: DecisionQuestion) -> bool:
            both_running.wait()
            return True

        provider = Recording(meet, delay=0.05, capabilities=caps(concurrency_limit=limit))
        assert set(decide(frame(4), provider, batch_size=1, **kwargs)["p__status"]) == {"ok"}
        assert provider.peak == 2


class TestAnswers:

    def test_abstention_reasons_and_symmetric_min_p(self) -> None:
        def script(_index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            by_id = {r.request_id: r for r in requests}
            return [result(by_id["0"], "p", None, status="uncertain", status_reason="refusal"),
                    result(by_id["1"], "p", None, {"true": 0.5, "false": 0.5}, status="uncertain", status_reason="tie"),
                    result(by_id["2"], "p", True, {"true": 0.9, "false": 0.1}),
                    result(by_id["3"], "p", False, {"true": 0.3, "false": 0.7}),
                    result(by_id["4"], "p", True)]

        out = decide(frame(5), Scripted(script), min_p=0.8)
        assert out["p__status"].tolist() == ["uncertain", "uncertain", "ok", "uncertain", "uncertain"]
        assert reasons(out) == ["refusal", "tie", None, "min_p", "min_p"]
        assert bool(out["p"].iloc[3]) is False and out["p__p"].iloc[3] == pytest.approx(0.7)
        assert out["p__p"].iloc[[0, 1, 4]].isna().all() and out["p__probability_kind"].iloc[0] == "none"
        q = out["p"]
        picked = ((q & (out["p__p"] >= 0.9)) | (~q & (out["p__p"] >= 0.6))).fillna(False)
        assert picked.tolist() == [False, False, True, True, False]

    @pytest.mark.parametrize("min_p,status", [(0.75, "ok"), (0.0, "ok"), (1.0, "uncertain"), (np.float16(0.5), "ok")])
    def test_min_p_is_inclusive_at_both_ends(self, min_p: float, status: str) -> None:
        provider = Recording(lambda e, q: (True, {"true": 0.75, "false": 0.25}), probability_kind="native_decision")
        assert decide(frame(1), provider, min_p=min_p)["p__status"].tolist() == [status]
        certain = Recording(lambda e, q: (True, {"true": 1.0, "false": 0.0}), probability_kind="native_decision")
        assert decide(frame(1), certain, min_p=1.0)["p__status"].tolist() == ["ok"]

    @pytest.mark.parametrize("answer", [None, (None, {"true": 0.9, "false": 0.1}), (None, {"true": float("nan")}),
                                        (None, {"true": None, "false": 0.1}), (None, {"true": "x"})])
    def test_no_answer_is_uncertain_and_drops_any_distribution(self, answer: object) -> None:
        out = decide(frame(1), Recording(lambda e, q: answer, probability_kind="verbalized"))
        assert out["p__status"].tolist() == ["uncertain"] and out["p__dist"].isna().all()
        assert out["p__probability_kind"].tolist() == ["none"] and reasons(out) == ["null_answer"]

    @pytest.mark.parametrize("kind", ["native_decision", "none"])
    def test_empty_distribution_means_none(self, kind: str) -> None:
        out = decide(frame(1), Recording(lambda e, q: ("a", {}), probability_kind=kind), [C])
        assert out["c"].tolist() == ["a"] and out["c__status"].tolist() == ["ok"]
        assert out["c__probability_kind"].tolist() == ["none"] and out["c__dist"].isna().all()

    @pytest.mark.parametrize("bad", [dict(extra="abc"), dict(extra=5), dict(extra=pd.NA), dict(distribution={"true": 0.1})])
    def test_unserializable_metadata_and_half_predicate_distributions_are_invalid_not_a_crash(self, bad: dict) -> None:
        looped: Dict[str, object] = {}
        looped["self"] = looped
        for overrides in (bad, dict(extra=looped)):
            out = decide(frame(2), Scripted(lambda i, reqs: [result(r, "p", True, {"true": 0.9, "false": 0.1}, **overrides)
                                                            for r in reqs]))
            assert out["p__error_code"].tolist() == ["invalid_provider_result"] * 2

    def test_unusable_provider_identity_and_metadata_keep_kernel_fields_only(self) -> None:
        looped: Dict[str, object] = {}
        looped["self"] = looped
        provider = Scripted(lambda i, reqs: [result(r, "p", True, {"true": 0.9, "false": 0.1}, model=looped)
                                             for r in reqs])
        provider.model = looped  # type: ignore[misc,assignment]
        out = decide(frame(2), provider)
        metas = [json.loads(meta) for meta in out["p__meta"]]
        assert out["p__error_code"].tolist() == ["invalid_provider_result"] * 2
        assert {(m["model"], m["provider"], m["retry_count"]) for m in metas} == {("Scripted", "raw", None)}

    @pytest.mark.parametrize("value,dist,status", [
        (np.array(True), {"true": 0.9, "false": 0.1}, "ok"), (np.bool_(False), None, "ok"),
        (pd.NA, {"true": 0.3, "false": 0.7}, "uncertain"), (float("nan"), {"true": 0.3, "false": 0.7}, "uncertain"),
        (True, {"true": 0.9, 1: 0.1, "false": 0.0}, "ok")])
    def test_answers_are_checked_and_written_as_the_same_plain_values(self, value: object, dist: object,
                                                                      status: str) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            return [result(r, "p", value, dist, status=status,  # type: ignore[arg-type]
                           status_reason="null_answer" if status == "uncertain" else None) for r in requests]

        out = decide(frame(1), Scripted(script))
        if isinstance(dist, dict) and 1 in dist:
            assert out["p__error_code"].tolist() == ["invalid_provider_result"]
        elif status == "uncertain":
            assert out["p__error_code"].tolist() == ["invalid_provider_result"] and out["p__p"].isna().all()
        else:
            assert out["p__status"].tolist() == ["ok"] and str(out["p"].dtype) == "boolean"

    def test_unreadable_answers_and_metadata_are_invalid_not_a_crash(self) -> None:
        class Unreadable:
            def tolist(self) -> object:
                raise RuntimeError("cannot read")

        class BrokenMapping(Mapping):
            def __getitem__(self, key: str) -> object:
                raise LookupError(key)

            def __len__(self) -> int:
                return 1

            def __iter__(self):  # type: ignore[no-untyped-def]
                raise LookupError("broken")

        for overrides in (dict(value=Unreadable()), dict(extra=BrokenMapping()), dict(value="yes", retry_count=2 ** 40)):
            out = decide(frame(1), Scripted(lambda i, reqs: [replace(result(r, "p", True), **overrides) for r in reqs]))
            assert out["p__error_code"].tolist() == ["invalid_provider_result"]
            assert json.loads(out["p__meta"].iloc[0])["retry_count"] in (None, 0)

    @pytest.mark.parametrize("via", ["callable", "raw"])
    @pytest.mark.parametrize("enum_keys", [False, True])
    @pytest.mark.parametrize("min_p", [None, 0.5])
    def test_str_enum_answers_are_plain_strings(self, via: str, enum_keys: bool, min_p: object) -> None:
        class Label(str, Enum):
            A = "a"
            B = "b"
            C = "c"

        dist = {Label.A: 0.05, Label.B: 0.9, Label.C: 0.05} if enum_keys else {"a": 0.05, "b": 0.9, "c": 0.05}
        provider = Recording(lambda e, q: (Label.B, dist), probability_kind="native_decision") if via == "callable" \
            else Scripted(lambda i, reqs: [result(r, "c", Label.B, dist) for r in reqs])  # type: ignore[arg-type]
        out = decide(frame(1), provider, [C], min_p=min_p)
        assert (out["c__status"].tolist(), out["c__p"].tolist(), type(out["c"].iloc[0])) == (["ok"], [0.9], str)

    def test_distribution_keys_that_collide_as_strings_are_invalid(self) -> None:
        question = {"id": "c", "kind": "choice", "text": "Which?", "options": ["1", "2"]}
        out = decide(frame(1), Scripted(lambda i, reqs: [result(r, "c", "1", {"1": 1.7, 1: 0.4}) for r in reqs]),
                     [question])
        assert out["c__error_code"].tolist() == ["invalid_provider_result"] and out["c__p"].isna().all()

    def test_provider_objects_that_raise_when_read_error_their_cells(self) -> None:
        class FloatRaises(float):
            def __float__(self) -> float:
                raise RuntimeError("no floats")

        class HashRaises(str):
            def __hash__(self) -> int:
                raise RuntimeError("no hashing")

        cell = decide(frame(1), Scripted(lambda i, reqs: [result(r, "p", True, {"true": FloatRaises(0.9), "false": 0.1})
                                                         for r in reqs]))
        call = decide(frame(2), Scripted(lambda i, reqs: [replace(result(r, "p", True), request_id=HashRaises(r.request_id))
                                                         for r in reqs]))
        assert cell["p__error_code"].tolist() == ["invalid_provider_result"]
        assert call["p__error_code"].tolist() == ["invalid_provider_result"] * 2

    def test_status_labels_are_checked_as_plain_strings(self) -> None:
        class EqualsAnything(str):
            def __eq__(self, other: object) -> bool:
                return True

            __hash__ = str.__hash__

        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            return [replace(result(r, "p", True, {"true": 0.9, "false": 0.1}), probability_kind=EqualsAnything(kind))
                    for r, kind in zip(requests, ["x", "native_decision"])]

        out = decide(frame(2), Scripted(script))
        assert out["p__status"].tolist() == ["error", "ok"] and out["p__error_code"].iloc[0] == "invalid_provider_result"
        assert [type(kind) for kind in out["p__probability_kind"]] == [str, str]

    def test_result_subclasses_with_their_own_constructor_are_read_by_field(self) -> None:
        class FromResponse(DecisionResult):
            def __init__(self, base: DecisionResult) -> None:
                super().__init__(**vars(base))

        out = decide(frame(1), Scripted(lambda i, reqs: [
            FromResponse(result(r, "p", True, {"true": 0.9, "false": 0.1})) for r in reqs]))
        assert (out["p__status"].tolist(), out["p__p"].tolist()) == (["ok"], [0.9])

    def test_kernel_stamps_its_own_task_version(self) -> None:
        out = decide(frame(1), Scripted(lambda i, reqs: [replace(result(r, "p", True), task_version="999")
                                                         for r in reqs]), task_version="2")
        assert out["p__status"].tolist() == ["ok"] and json.loads(out["p__meta"].iloc[0])["task_version"] == "2"

    def test_numpy_answers_from_any_provider_are_accepted(self) -> None:
        out = decide(frame(2), Scripted(lambda i, reqs: [
            result(r, "p", np.bool_(True), {"true": np.float32(0.75), "false": np.float64(0.25)}) for r in reqs]))
        assert set(out["p__status"]) == {"ok"} and out["p__p"].tolist() == [0.75, 0.75]

    def test_input_hash_is_taken_before_the_provider_sees_the_evidence(self) -> None:
        def scribble(evidence: Dict[str, object], question: DecisionQuestion) -> bool:
            evidence["name"] = "changed"
            return True

        meta = json.loads(decide(frame(1), Recording(scribble))["p__meta"].iloc[0])
        question = DecisionQuestion(id="p", kind="predicate", text="Is it notable?")
        assert meta["input_hash"] == compute_decision_hash({"name": "n0"}, question, "1")

    def test_numpy_answers_are_normalized(self) -> None:
        provider = Recording(lambda e, q: (np.bool_(True), {"true": np.float32(0.75), "false": np.float64(0.25)}),
                             probability_kind="native_decision")
        out = decide(frame(2), provider)
        assert set(out["p__status"]) == {"ok"} and out["p__p"].tolist() == [0.75, 0.75]

    def test_question_named_self_and_numpy_deadlines(self) -> None:
        provider = Recording()
        out = decide(frame(2), provider, [{"id": "self", "kind": "predicate", "text": "t"}], deadline_s=np.float32(30.0))
        assert set(out["self__status"]) == {"ok"} and all(type(d) is float and 0 < d <= 30 for d in provider.deadlines)
        slow = Recording(delay=0.01)
        assert set(decide(frame(2), slow, deadline_s=np.float16(30.0))["p__status"]) == {"ok"}
        assert all(type(d) is float and 0 < d <= 30 for d in slow.deadlines)

    def test_callable_defaults(self) -> None:
        native = CallableDecisionProvider(lambda e, q: True, probability_kind="native_decision").capabilities()
        plain = CallableDecisionProvider(lambda e, q: True).capabilities()
        assert native.supports_native_probabilities and not plain.supports_native_probabilities
        assert plain.max_questions_per_request == 1000 and plain.concurrency_limit == 4
        for bad in (dict(fn="not callable"), dict(probability_kind="guess")):
            with pytest.raises(ValueError):
                CallableDecisionProvider(**dict(dict(fn=lambda e, q: True), **bad))  # type: ignore[arg-type]


class TestFailures:

    def test_per_answer_and_per_row(self) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            out = []
            for request in requests:
                rid = request.request_id
                p_result = result(request, "p", True, {"true": 1.3, "false": -0.3} if (index, rid) == (0, "1")
                                  else {"true": 0.8, "false": 0.2})
                c_result = result(request, "c", "zzz" if (index, rid) == (0, "2") else "a", {"a": 0.7}
                                  if (index, rid) != (0, "2") else None)
                out += [p_result] if (index, rid) == (0, "3") else [p_result, c_result]
            return out

        out = decide(frame(8), Scripted(script), [P, C], batch_size=4)
        status = list(zip(out["p__status"], out["c__status"]))
        assert status[:4] == [("ok", "ok"), ("error", "ok"), ("ok", "error"), ("error", "error")]
        assert all(s == ("ok", "ok") for s in status[4:])
        errors = out["p__status"] == "error"
        assert out.loc[errors, "p"].isna().all() and out.loc[errors, "p__dist"].isna().all()
        assert set(out.loc[errors, "p__probability_kind"]) == {"none"}
        assert set(out.loc[errors, "p__error_code"]) == {"invalid_provider_result"}

    def test_callable_exception_fails_one_answer(self) -> None:
        def fn(evidence: Dict[str, object], question: DecisionQuestion) -> object:
            if evidence["name"] == "n6" and question.id == "p":
                raise RuntimeError("boom")
            return value_only(evidence, question)

        out = decide(frame(8), Recording(fn), [P, C], batch_size=4)
        assert out["p__error_code"].fillna("").tolist() == [""] * 6 + ["callable_exception", ""]
        assert out["c__status"].iloc[6] == "ok"

    @pytest.mark.parametrize("fault", ["unknown_id", "duplicate", "raise", "not_a_list", "wrong_type", "duplicate_unknown"])
    def test_whole_call_faults(self, fault: str) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> object:
            results = [result(r, "p", True, {"true": 0.8, "false": 0.2}) for r in requests]
            if index > 0:
                return results
            if fault == "raise":
                raise RuntimeError("down")
            stray = DecisionRequest("zz", "e", (DecisionQuestion("p", "predicate", "t"),), "t", "1")
            return {"unknown_id": results + [result(stray, "p", True)], "duplicate": results + results[:1],
                    "not_a_list": None, "wrong_type": [object()],
                    "duplicate_unknown": results + [result(requests[0], "zz", True)] * 2}[fault]

        out = decide(frame(8), Scripted(script), batch_size=4)
        assert out["p__status"].tolist() == ["error"] * 4 + ["ok"] * 4
        assert set(out["p__error_code"].iloc[:4]) == {"provider_exception" if fault == "raise" else "invalid_provider_result"}

    @pytest.mark.parametrize("bad", [{"question_id": "zz"}, {"question_id": None}, {"question_id": ["p"]}])
    def test_unknown_or_malformed_question_id_errors_only_that_row(self, bad: dict) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            return [result(r, "p", True, {"true": 0.8, "false": 0.2}) for r in requests] + \
                [replace(result(requests[0], "p", True), **bad)]

        assert decide(frame(4), Scripted(script), batch_size=4)["p__status"].tolist() == ["error", "ok", "ok", "ok"]

    def test_invalid_answer_keeps_valid_metadata_only(self) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            return [result(r, "p", "yes", retry_count=[-1, 2][index], latency_ms=[float("inf"), 5.0][index])
                    for r in requests]

        out = decide(frame(2), Scripted(script), batch_size=1, max_concurrency=1)
        assert out["p__error_code"].tolist() == ["invalid_provider_result"] * 2
        metas = [json.loads(meta) for meta in out["p__meta"]]
        assert [(m["retry_count"], m["latency_ms"]) for m in metas] == [(None, None), (2, 5.0)]

    @pytest.mark.parametrize("answer,native", [
        ("yes", True), ((True, {"true": 0.9, "false": 0.1}), False), ((True, {"true": 0.0, "false": 0.0}), True),
        (5, True), (1.0, True), (["yes"], True), ((True, {"true": "1.0", "false": 0.0}), True),
        ((True, {"true": True, "false": False}), True), ((True, {"true": 1.0}, "extra"), True),
        ((True, [0.9, 0.1]), True),
    ])
    def test_no_default_positive(self, answer: object, native: bool) -> None:
        provider = Recording(lambda e, q: answer, probability_kind="native_decision", capabilities=caps(native=native))
        out = decide(frame(2), provider)
        assert out["p__error_code"].tolist() == ["invalid_provider_result"] * 2 and out["p"].isna().all()
        assert [json.loads(m)["retry_count"] for m in out["p__meta"]] == [0, 0]

    def test_distribution_from_a_value_only_callable_is_invalid(self) -> None:
        out = decide(frame(2), Recording(lambda e, q: (True, {"true": 1.0, "false": 0.0})))
        assert set(out["p__error_code"]) == {"invalid_provider_result"}


class TestLimits:

    def test_max_rows_wins_over_null_evidence(self) -> None:
        out = decide(pd.DataFrame({"name": ["a", None, None]}), Recording(), max_rows=2)
        assert reasons(out)[1:] == ["null_evidence", "max_rows"]

    def test_max_rows(self) -> None:
        provider = Recording()
        out = decide(frame(10), provider, max_rows=3)
        assert out["p__status"].tolist() == ["ok"] * 3 + ["skipped"] * 7 and set(reasons(out)[3:]) == {"max_rows"}
        assert out["p"].iloc[3:].isna().all() and out["p__error_code"].isna().all()
        assert sorted(provider.request_ids, key=int) == ["0", "1", "2"]

    def test_max_requests_and_remaining_limits_reach_the_provider(self) -> None:
        provider = Recording(delay=0.05)
        out = decide(frame(10), provider, batch_size=2, max_concurrency=1, deadline_s=5.0, budget={"max_requests": 3})
        assert out["p__status"].tolist() == ["ok"] * 6 + ["skipped"] * 4 and set(reasons(out)[6:]) == {"budget"}
        assert [budget.max_requests for budget in provider.budgets if budget] == [3, 2, 1]
        assert provider.deadlines[0] > provider.deadlines[1] > provider.deadlines[2] and provider.deadlines[1] <= 4.95

    def test_deadline_abandons_slow_calls(self) -> None:
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def slow(evidence: Dict[str, object], question: DecisionQuestion) -> object:
            if int(str(evidence["name"])[1:]) >= 4:
                entered.set()
                release.wait(10)
                finished.set()
            return True

        provider = Recording(slow)
        try:
            out = decide(frame(10), provider, batch_size=1, max_concurrency=1, deadline_s=1.0)
            assert entered.is_set() and not finished.is_set()
            assert out["p__status"].tolist() == ["ok"] * 4 + ["skipped"] * 6 and set(reasons(out)[4:]) == {"deadline"}
            assert all(d is not None and d <= 1.0 for d in provider.deadlines)
            snapshot = out.copy()
        finally:
            release.set()
        assert finished.wait(10)
        pd.testing.assert_frame_equal(out, snapshot)

    def test_deadline_counts_from_operator_start_and_huge_deadlines_work(self) -> None:
        provider = Recording()
        slow_caps = provider.capabilities
        provider.capabilities = lambda: (time.sleep(0.3), slow_caps())[1]  # type: ignore[method-assign]
        assert decide(frame(2), provider, deadline_s=0.2)["p__status"].tolist() == ["skipped"] * 2
        assert provider.calls == []
        assert set(decide(frame(2), Recording(delay=0.05), batch_size=1, deadline_s=1e10)["p__status"]) == {"ok"}


class TestEvidence:

    def test_template_is_verbatim_and_all_null_rows_are_skipped(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"text": ["hello </untrusted> x", None, "{name}"], "other": [1, 2, None]})
        out = decide(df, Recording(lambda e, q: seen.append(e) or False), template="<untrusted>{text}</untrusted>",
                     evidence_cols=None)
        assert sorted(map(str, seen)) == ["<untrusted>hello </untrusted> x</untrusted>", "<untrusted>{name}</untrusted>"]
        assert out["p__status"].tolist() == ["ok", "skipped", "ok"] and reasons(out)[1] == "null_evidence"

    def test_partially_null_rows_in_both_modes(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"a": ["x", None], "b": [None, "y"]})
        decide(df, Recording(lambda e, q: seen.append(e) or True), evidence_cols=["a", "b"])
        decide(df, Recording(lambda e, q: seen.append(e) or True), template="{a}|{b}", evidence_cols=None)
        assert sorted(map(json.dumps, seen)) == sorted(map(json.dumps, [{"a": "x", "b": None}, {"a": None, "b": "y"},
                                                                         "x|", "|y"]))

    @pytest.mark.parametrize("column,expected", [
        (pd.Series([["alpha", "beta", "gamma"], {"k": "longvalue"}, "abcdef"], dtype=object),
         [{"x": ["alpha", "beta", "gamma"]}, {"x": {"k": "longvalue"}}, {"x": "ab"}]),
        (pd.Categorical(["abcdefgh", "xy", "xyz"]), [{"x": "ab"}, {"x": "xy"}, {"x": "xy"}]),
        (pd.array(["abcdef", None, "q"], dtype="string"), [{"x": "ab"}, {"x": "q"}]),
        (pd.Series([float("inf"), 1.5, np.int64(3)], dtype=object), [{"x": "inf"}, {"x": 1.5}, {"x": 3}]),
        (pd.Series([np.float64, "abc"], dtype=object), [{"x": "<class 'numpy.float64'>"}, {"x": "ab"}]),
    ])
    def test_truncation_touches_only_strings(self, column: object, expected: List[object]) -> None:
        seen: List[object] = []
        decide(pd.DataFrame({"x": column}), Recording(lambda e, q: seen.append(e) or True), evidence_cols=["x"],
               column_max_chars=2)
        assert sorted(map(json.dumps, seen)) == sorted(map(json.dumps, expected))

    def test_template_truncation_mapping_order_and_canonical_values(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"b": ["bbbb", None], "a": ["aaaa", None], "n": [np.int64(3), np.nan],
                           "ts": pd.to_datetime(["2024-01-02", None])})
        record = Recording(lambda e, q: seen.append(e) or True)
        decide(df, record, template="{b}|{a}", evidence_cols=None, column_max_chars=2)
        out = decide(df, record, evidence_cols=["b", "a", "n", "ts"], column_max_chars=3)
        assert seen == ["bb|aa", {"b": "bbb", "a": "aaa", "n": 3.0, "ts": "2024-01-02 00:00:00"}]
        assert out["p__status"].tolist() == ["ok", "skipped"]


class TestValidationBeforeAnyCall:

    @pytest.mark.parametrize("kwargs", [
        dict(template="{name}"), dict(evidence_cols=None), dict(evidence_cols=["missing"]),
        dict(id_col="missing"), dict(evidence_cols=None, template="{0}"),
        dict(evidence_cols=None, template="{name.upper}"), dict(evidence_cols=None, template="{name!r}"),
        dict(evidence_cols=None, template="{name:>5}"), dict(evidence_cols=None, template="{missing}"),
        dict(evidence_cols=None, template="no fields"), dict(evidence_cols="name"), dict(evidence_cols=["name", "name"]),
        dict(questions=[{"id": "q", "kind": "choice", "text": "t", "options": ["a"]}]),
        dict(questions=[{"id": "q", "kind": "score", "text": "t", "levels": ["x", "y"], "options": ["a", "b"]}]),
        dict(questions=[{"id": "__gfql_x__", "kind": "predicate", "text": "t"}]),
        dict(questions=[{"id": "a", "kind": "predicate", "text": "t"}, {"id": "a__p", "kind": "predicate", "text": "t"}]),
        dict(questions=[{"id": "q", "kind": "predicate", "text": "t", "bogus": 1}]),
        dict(questions=[]), dict(questions=["p"]), dict(min_p=1.5), dict(batch_size=0), dict(max_rows=True),
        dict(deadline_s=0), dict(budget={"max_tokens": 3}), dict(budget=5), dict(task_id=""), dict(overwrite="yes"),
    ])
    def test_raises_with_zero_calls(self, kwargs: dict) -> None:
        provider = Recording()
        params: Dict[str, object] = dict(questions=[P])
        params.update(kwargs)
        with pytest.raises(ValueError):
            decide(frame(3), provider, **params)
        assert provider.calls == []

    @pytest.mark.parametrize("kwargs,match", [
        (dict(evidence_cols=None, template="{0}"), "bare column"), (dict(evidence_cols=None, template="{name.upper}"), "bare column"),
        (dict(evidence_cols=None, template="{name!r}"), "bare column"), (dict(questions=[P, P]), "unique"),
        (dict(questions=[{"id": "a", "kind": "predicate", "text": "t"}, {"id": "a__p", "kind": "predicate", "text": "t"}]),
         "collide"), (dict(budget=Budget(max_requests=1), max_rows=0), "max_rows"),
    ])
    def test_each_rule_names_itself(self, kwargs: dict, match: str) -> None:
        params: Dict[str, object] = dict(questions=[P])
        params.update(kwargs)
        with pytest.raises(ValueError, match=match):
            decide(frame(3), Recording(), **params)

    @pytest.mark.parametrize("limits,question,match", [
        (dict(supported_kinds=("choice",)), P, "does not support predicate"),
        (dict(max_choice_options=255), {"id": "q", "kind": "choice", "text": "t", "options": [f"o{i}" for i in range(256)]},
         "more than 255 options"),
        (dict(max_levels=2), S, "more than 2 levels"),
    ])
    def test_capability_limits(self, limits: dict, question: object, match: str) -> None:
        provider = Recording(capabilities=caps(**limits))
        with pytest.raises(ValueError, match=match):
            decide(frame(3), provider, [question])
        assert provider.calls == []
        exact = Recording(capabilities=caps(max_levels=3, max_choice_options=3))
        assert set(decide(frame(1), exact, [S, C])["s__status"]) == {"ok"}

    def test_existing_output_needs_overwrite(self) -> None:
        df = frame(3).assign(p__status="old")
        with pytest.raises(ValueError, match="overwrite"):
            decide(df, Recording())
        assert set(decide(df, Recording(), overwrite=True)["p__status"]) == {"ok"}

    def test_provider_problems(self) -> None:
        with pytest.raises(ValueError, match="alias 'nope' is not bound"):
            decide(frame(3), "nope")
        with pytest.raises(ValueError, match="not a DecisionProvider"):
            decide(frame(3), object())
        provider = Recording()
        provider.capabilities = lambda: (_ for _ in ()).throw(NotImplementedError("x"))  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="capabilities"):
            decide(frame(2), provider)

    def test_duplicate_table_columns_and_dotted_template_fields(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            decide(pd.DataFrame([["a", "b"]], columns=["name", "name"]), Recording())
        with pytest.raises(ValueError, match="bare column"):
            decide(pd.DataFrame({"a.b": ["x"], "a": ["y"]}), Recording(), template="{a.b}", evidence_cols=None)


def test_raw_provider_results_flow_through_unchanged() -> None:
    def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
        return [replace(result(r, "c", "b", {"a": 0.1, "b": 0.9}, confidence=0.4), extra={"k": "v"}) for r in requests]

    meta = json.loads(decide(frame(1), Scripted(script), [C])["c__meta"].iloc[0])
    assert (meta["confidence"], meta["extra"], meta["model"], meta["latency_ms"]) == (0.4, {"k": "v"}, "raw-m", 1.0)
