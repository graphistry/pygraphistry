import json
import threading
from dataclasses import replace as dataclasses_replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from graphistry.compute.semantic.contract import (
    Budget,
    DecisionQuestion,
    DecisionRequest,
    DecisionResult,
    ProviderCapabilities,
)
from graphistry.compute.semantic.ops import sem_decide
from graphistry.compute.semantic.providers.callable import CallableDecisionProvider


P = {"id": "p", "kind": "predicate", "text": "Is it notable?"}
C = {"id": "c", "kind": "choice", "text": "Which group?", "options": ["a", "b", "c"]}
S = {"id": "s", "kind": "score", "text": "How much?", "levels": ["low", "mid", "high"]}
META_KEYS = {"model", "provider", "task_version", "input_hash", "latency_ms", "cost_usd", "cache_hit",
             "retry_count", "confidence", "status_reason", "extra"}


def caps(native: bool = True, concurrency: int = 4, **kwargs: object) -> ProviderCapabilities:
    return ProviderCapabilities(supports_native_probabilities=native, max_questions_per_request=1000,
                                max_evidence_tokens=None, supports_shared_prefix_cache=False,
                                concurrency_limit=concurrency, **kwargs)  # type: ignore[arg-type]


class Recording(CallableDecisionProvider):
    """Callable provider that records every provider call and request id."""

    def __init__(self, fn: Callable, **kwargs: object) -> None:
        super().__init__(fn, **kwargs)  # type: ignore[arg-type]
        self.calls: List[List[str]] = []
        self.deadlines: List[Optional[float]] = []
        self.lock = threading.Lock()
        self.cost_per_request: Optional[float] = 0.0

    def estimate_cost_usd(self, requests: Sequence[DecisionRequest]) -> Optional[float]:
        return None if self.cost_per_request is None else self.cost_per_request * len(requests)

    def decide_batch(self, requests: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        with self.lock:
            self.calls.append([request.request_id for request in requests])
            self.deadlines.append(deadline_s)
        return super().decide_batch(requests, deadline_s=deadline_s, budget=budget)

    @property
    def request_ids(self) -> List[str]:
        return [rid for call in self.calls for rid in call]


class Raw:
    """Provider returning crafted results per call: ``script(call_index, requests) -> results``."""

    name = "raw"
    model = "raw-m"

    def __init__(self, script: Callable[[int, Sequence[DecisionRequest]], List[DecisionResult]],
                 capabilities: Optional[ProviderCapabilities] = None) -> None:
        self.script = script
        self._caps = capabilities or caps()
        self.calls = 0

    def capabilities(self) -> ProviderCapabilities:
        return self._caps

    def estimate_cost_usd(self, requests: Sequence[DecisionRequest]) -> Optional[float]:
        return 0.0

    def decide_batch(self, requests: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        index = self.calls
        self.calls += 1
        return self.script(index, requests)


def result(request: DecisionRequest, question_id: str, value: object, dist: Optional[Dict[str, float]] = None,
           *, status: str = "ok", kind: Optional[str] = None, reason: Optional[str] = None,
           confidence: Optional[float] = None) -> DecisionResult:
    return DecisionResult(
        request_id=request.request_id, question_id=question_id, value=value,  # type: ignore[arg-type]
        distribution=dist, confidence=confidence,
        probability_kind=kind or ("native_decision" if dist is not None else "none"),  # type: ignore[arg-type]
        status=status, error_code=None, model="raw-m", provider="raw", task_version="1",  # type: ignore[arg-type]
        input_hash="placeholder", latency_ms=1.0, cost_usd=0.0, cache_hit=False, retry_count=0,
        status_reason=reason,  # type: ignore[arg-type]
    )


def full_answer(evidence: object, question: DecisionQuestion) -> Tuple[object, Dict[str, float]]:
    if question.kind == "predicate":
        return True, {"true": 0.9, "false": 0.1}
    labels = question.labels() or ()
    rest = 0.4 / (len(labels) - 1)
    return labels[1], {label: (0.6 if i == 1 else rest) for i, label in enumerate(labels)}


def value_only(evidence: object, question: DecisionQuestion) -> object:
    return True if question.kind == "predicate" else (question.labels() or ("",))[0]


def frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({"name": [f"n{i}" for i in range(n)], "note": [f"note {i}" for i in range(n)]})


class TestTypedOutputs:

    def test_native_columns_dtypes_and_meta(self) -> None:
        provider = Recording(full_answer, probability_kind="native_decision")
        out = sem_decide(frame(), questions=[P, C, S], provider=provider, evidence_cols=["name", "note"],
                         task_id="t")
        added = [col for col in out.columns if col not in ("name", "note")]
        assert len(added) == 21
        assert str(out["p"].dtype) == "boolean"
        assert out["c"].dtype == object and out["s"].dtype == object
        assert out["p__p"].dtype == np.float64
        assert set(out["p__status"]) == {"ok"} and set(out["c__status"]) == {"ok"}
        assert out["p__p"].tolist() == [0.9] * 6
        assert out["c__p"].tolist() == [0.6] * 6
        assert list(json.loads(out["c__dist"].iloc[0]).keys()) == ["a", "b", "c"]
        assert list(json.loads(out["p__dist"].iloc[0]).keys()) == ["true", "false"]
        meta = json.loads(out["s__meta"].iloc[0])
        assert set(meta) == META_KEYS and meta["confidence"] is None
        assert out["p__error_code"].isna().all()

    def test_value_only_provider_has_no_probabilities(self) -> None:
        out = sem_decide(frame(), questions=[P, C, S], provider=Recording(value_only),
                         evidence_cols=["name", "note"], task_id="t")
        for q in ("p", "c", "s"):
            assert out[f"{q}__p"].isna().all()
            assert out[f"{q}__dist"].isna().all()
            assert set(out[f"{q}__probability_kind"]) == {"none"}
            assert set(out[f"{q}__status"]) == {"ok"}

    def test_input_is_not_mutated(self) -> None:
        df = frame()
        before = df.copy()
        sem_decide(df, questions=[P], provider=Recording(value_only), evidence_cols=["name"], task_id="t")
        pd.testing.assert_frame_equal(df, before)


class TestRowIdentity:

    def test_order_index_and_batch_invariance(self) -> None:
        index = [3, 1, 3, 0, 9, 9, 2, 7, 5, 5]
        df = pd.DataFrame({"name": [f"n{i}" for i in range(10)]}, index=index)

        def by_name(evidence: Dict[str, object], question: DecisionQuestion) -> Tuple[object, Dict[str, float]]:
            flag = int(str(evidence["name"])[1:]) % 2 == 0
            return flag, {"true": 0.8 if flag else 0.3, "false": 0.2 if flag else 0.7}

        frames = []
        for size in (1, 3, None):
            provider = Recording(by_name, probability_kind="native_decision")
            out = sem_decide(df, questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                             batch_size=size)
            assert out.index.tolist() == index
            assert out["name"].tolist() == df["name"].tolist()
            assert sorted(provider.request_ids, key=int) == [str(i) for i in range(10)]
            frames.append(out.drop(columns=["p__meta"]))
        pd.testing.assert_frame_equal(frames[0], frames[1])
        pd.testing.assert_frame_equal(frames[0], frames[2])

    def test_unique_id_col_is_used(self) -> None:
        df = frame(4).assign(rid=["a", "b", "c", "d"])
        provider = Recording(value_only)
        sem_decide(df, questions=[P], provider=provider, evidence_cols=["name"], id_col="rid", task_id="t")
        assert sorted(provider.request_ids) == ["a", "b", "c", "d"]

    @pytest.mark.parametrize("ids", [[1, 1, 2, 3], [1, None, 2, 3], [1, "1", 2, 3]])
    def test_bad_ids_raise_before_any_call(self, ids: list) -> None:
        df = frame(4).assign(rid=pd.Series(ids, dtype=object))
        provider = Recording(value_only)
        with pytest.raises(ValueError):
            sem_decide(df, questions=[P], provider=provider, evidence_cols=["name"], id_col="rid", task_id="t")
        assert provider.calls == []


class TestRequestIdEdgeCases:

    def test_index_with_colliding_str_labels_uses_positions(self) -> None:
        df = pd.DataFrame({"name": ["a", "b", "c"]}, index=pd.Index([1, "1", 2], dtype=object))
        frames = []
        for size in (1, 3):
            provider = Recording(value_only)
            out = sem_decide(df, questions=[P], provider=provider, evidence_cols=["name"], task_id="t", batch_size=size)
            assert sorted(provider.request_ids) == ["0", "1", "2"]
            frames.append(out.drop(columns=["p__meta"]))
        pd.testing.assert_frame_equal(frames[0], frames[1])

    def test_empty_index_label_uses_positions(self) -> None:
        df = pd.DataFrame({"name": ["a", "b"]}, index=["a", ""])
        out = sem_decide(df, questions=[P], provider=Recording(value_only), evidence_cols=["name"], task_id="t",
                         max_rows=1)
        assert out["p__status"].tolist() == ["ok", "skipped"]

    def test_empty_id_col_value_raises_before_calls(self) -> None:
        provider = Recording(value_only)
        df = pd.DataFrame({"name": ["a", None], "rid": ["a", ""]})
        with pytest.raises(ValueError, match="empty id"):
            sem_decide(df, questions=[P], provider=provider, evidence_cols=["name"], id_col="rid", task_id="t")
        assert provider.calls == []


class TestKernelHashAndNumbers:

    def test_kernel_stamps_its_own_input_hash(self) -> None:
        from graphistry.compute.semantic.contract import compute_decision_hash
        from graphistry.compute.semantic.ops import sem_map
        from graphistry.compute.semantic.contract import GenerationResult, compute_generation_hash
        out = sem_decide(frame(1), questions=[P], provider=Raw(
            lambda i, reqs: [dataclasses_replace(result(r, "p", True), input_hash="provider-made") for r in reqs]),
            evidence_cols=["name"], task_id="t")
        question = DecisionQuestion(id="p", kind="predicate", text="Is it notable?")
        assert json.loads(out["p__meta"].iloc[0])["input_hash"] == compute_decision_hash({"name": "n0"}, question, "1")

        class Gen:
            name = "g"
            model = "m"

            def capabilities(self) -> ProviderCapabilities:
                return caps()

            def estimate_cost_usd(self, requests):  # type: ignore[no-untyped-def]
                return 0.0

            def generate_batch(self, requests, *, deadline_s, budget):  # type: ignore[no-untyped-def]
                return [GenerationResult(request_id=r.request_id, value="x", status="ok", error_code=None, model="m",
                                         provider="g", task_version="1", input_hash="provider-made", latency_ms=None,
                                         cost_usd=None, cache_hit=None, retry_count=None) for r in requests]

        out = sem_map(frame(1), instruction="i", out_col="o", provider=Gen(), evidence_cols=["name"], task_id="t")
        assert json.loads(out["o__meta"].iloc[0])["input_hash"] == compute_generation_hash({"name": "n0"}, "i", "1")

    def test_numpy_min_p_and_index_rules(self) -> None:
        out = sem_decide(frame(2), questions=[P], provider=Recording(value_only), evidence_cols=["name"], task_id="t",
                         min_p=np.float32(0.7))
        assert set(out["p__status"]) == {"uncertain"}
        provider = Recording(value_only)
        sem_decide(pd.DataFrame({"name": ["a", "b"]}, index=pd.Index([1, 1.0], dtype=object)), questions=[P],
                   provider=provider, evidence_cols=["name"], task_id="t")
        assert sorted(provider.request_ids) == ["0", "1"]

    def test_duplicate_column_names_raise_before_calls(self) -> None:
        provider = Recording(value_only)
        df = pd.DataFrame([["a", "b"]], columns=["name", "name"])
        with pytest.raises(ValueError, match="more than once"):
            sem_decide(df, questions=[P], provider=provider, evidence_cols=["name"], task_id="t", column_max_chars=2)
        assert provider.calls == []

    def test_large_runs_fit_an_exact_cost_cap(self) -> None:
        provider = Recording(value_only)
        provider.cost_per_request = 0.01
        out = sem_decide(frame(2432), questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                         batch_size=1, budget={"max_cost_usd": 24.31})
        assert out["p__status"].tolist().count("ok") == 2431


class TestAbstentionAndMinP:

    def test_reasons_and_symmetric_threshold(self) -> None:
        def script(_index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            by_id = {r.request_id: r for r in requests}
            return [
                result(by_id["0"], "p", None, status="uncertain", reason="refusal"),
                result(by_id["1"], "p", None, {"true": 0.5, "false": 0.5}, status="uncertain", reason="tie"),
                result(by_id["2"], "p", True, {"true": 0.9, "false": 0.1}),
                result(by_id["3"], "p", False, {"true": 0.3, "false": 0.7}),
                result(by_id["4"], "p", True),
            ]

        out = sem_decide(frame(5), questions=[P], provider=Raw(script), evidence_cols=["name"], task_id="t",
                         min_p=0.8)
        assert out["p__status"].tolist() == ["uncertain", "uncertain", "ok", "uncertain", "uncertain"]
        reasons = [json.loads(m)["status_reason"] for m in out["p__meta"]]
        assert reasons == ["refusal", "tie", None, "min_p", "min_p"]
        assert out["p"].iloc[3] is False or bool(out["p"].iloc[3]) is False
        assert out["p__p"].iloc[3] == pytest.approx(0.7)
        assert np.isnan(out["p__p"].iloc[4]) and np.isnan(out["p__p"].iloc[0]) and np.isnan(out["p__p"].iloc[1])
        assert out["p__probability_kind"].iloc[0] == "none"
        q = out["p"]
        picked = ((q & (out["p__p"] >= 0.9)) | (~q & (out["p__p"] >= 0.6))).fillna(False)
        assert picked.tolist() == [False, False, True, True, False]


class TestPartialFailure:

    def test_per_answer_and_per_row(self) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            out = []
            for request in requests:
                rid = request.request_id
                p_res = result(request, "p", True, {"true": 0.8, "false": 0.2})
                c_res = result(request, "c", "a", {"a": 0.7})
                if index == 0 and rid == "1":
                    p_res = result(request, "p", True, {"true": 1.3, "false": -0.3})
                if index == 0 and rid == "2":
                    c_res = result(request, "c", "zzz")
                out.append(p_res)
                if not (index == 0 and rid == "3"):
                    out.append(c_res)
            return out

        out = sem_decide(frame(8), questions=[P, C], provider=Raw(script), evidence_cols=["name"], task_id="t",
                         batch_size=4)
        status = list(zip(out["p__status"], out["c__status"]))
        assert status[0] == ("ok", "ok")
        assert status[1] == ("error", "ok")
        assert status[2] == ("ok", "error")
        assert status[3] == ("error", "error")
        assert all(s == ("ok", "ok") for s in status[4:])
        errors = out["p__status"] == "error"
        assert out.loc[errors, "p"].isna().all()
        assert out.loc[errors, "p__dist"].isna().all()
        assert set(out.loc[errors, "p__probability_kind"]) == {"none"}
        assert set(out.loc[errors, "p__error_code"]) == {"invalid_provider_result"}

    def test_callable_exception_hits_one_answer(self) -> None:
        def fn(evidence: Dict[str, object], question: DecisionQuestion) -> object:
            if evidence["name"] == "n6" and question.id == "p":
                raise RuntimeError("boom")
            return value_only(evidence, question)

        out = sem_decide(frame(8), questions=[P, C], provider=Recording(fn), evidence_cols=["name"], task_id="t",
                         batch_size=4)
        assert out["p__status"].tolist().count("error") == 1
        assert out["p__error_code"].iloc[6] == "callable_exception"
        assert out["c__status"].iloc[6] == "ok"

    @pytest.mark.parametrize("fault", ["unknown_id", "duplicate", "raise", "not_a_list", "wrong_type"])
    def test_whole_call_faults(self, fault: str) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            results = [result(r, "p", True, {"true": 0.8, "false": 0.2}) for r in requests]
            if index == 0:
                if fault == "unknown_id":
                    results.append(result(DecisionRequest("zz", "e", (DecisionQuestion("p", "predicate", "t"),),
                                                          "t", "1"), "p", True))
                elif fault == "duplicate":
                    results.append(results[0])
                elif fault == "raise":
                    raise RuntimeError("down")
                elif fault == "not_a_list":
                    return None  # type: ignore[return-value]
                elif fault == "wrong_type":
                    return [object()]  # type: ignore[list-item]
            return results

        out = sem_decide(frame(8), questions=[P], provider=Raw(script), evidence_cols=["name"], task_id="t",
                         batch_size=4)
        expected_code = "provider_exception" if fault == "raise" else "invalid_provider_result"
        assert out["p__status"].tolist() == ["error"] * 4 + ["ok"] * 4
        assert set(out["p__error_code"].iloc[:4]) == {expected_code}


class TestMalformedProviderReturns:

    def test_unhashable_ids_and_bad_metadata_stay_inside_the_run(self) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            if index == 0:
                return [dataclasses_replace(result(r, "p", True), request_id=[r.request_id]) for r in requests]
            if index == 1:
                return [dataclasses_replace(result(r, "p", "yes"), retry_count=-1, latency_ms=float("inf"))
                        for r in requests]
            return [dataclasses_replace(result(r, "p", True), status=pd.NA) for r in requests]

        out = sem_decide(frame(3), questions=[P], provider=Raw(script), evidence_cols=["name"], task_id="t",
                         batch_size=1, max_concurrency=1)
        assert out["p__error_code"].tolist() == ["invalid_provider_result"] * 3
        meta = json.loads(out["p__meta"].iloc[1])
        assert meta["retry_count"] is None and meta["latency_ms"] is None


class TestRowLevelAlignment:

    def test_unknown_question_id_errors_only_that_row(self) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            out = [result(r, "p", True, {"true": 0.8, "false": 0.2}) for r in requests]
            out.append(result(requests[0], "zz", True))
            return out

        out = sem_decide(frame(4), questions=[P], provider=Raw(script), evidence_cols=["name"], task_id="t",
                         batch_size=4)
        assert out["p__status"].tolist() == ["error", "ok", "ok", "ok"]

    @pytest.mark.parametrize("bad", [None, ["p"]])
    def test_non_string_question_id_errors_only_that_row(self, bad: object) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            out = [result(r, "p", True, {"true": 0.8, "false": 0.2}) for r in requests]
            out.append(dataclasses_replace(result(requests[0], "p", True), question_id=bad))
            return out

        out = sem_decide(frame(4), questions=[P], provider=Raw(script), evidence_cols=["name"], task_id="t",
                         batch_size=4)
        assert out["p__status"].tolist() == ["error", "ok", "ok", "ok"]

    def test_duplicate_unknown_key_errors_the_call(self) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[DecisionResult]:
            out = [result(r, "p", True, {"true": 0.8, "false": 0.2}) for r in requests]
            out += [result(requests[0], "zz", True), result(requests[0], "zz", True)]
            return out

        out = sem_decide(frame(4), questions=[P], provider=Raw(script), evidence_cols=["name"], task_id="t",
                         batch_size=4)
        assert set(out["p__status"]) == {"error"}


class TestCostEstimates:

    def test_invalid_estimate_mid_run_stops_new_calls(self) -> None:
        provider = Recording(value_only)
        provider.estimate_cost_usd = lambda requests: (  # type: ignore[method-assign]
            float("nan") if requests[0].request_id == "1" else 0.01)
        out = sem_decide(frame(4), questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                         batch_size=1, max_concurrency=1, budget={"max_cost_usd": 1.0})
        assert out["p__status"].tolist() == ["ok", "skipped", "skipped", "skipped"]

    def test_raising_estimate_counts_as_unknown(self) -> None:
        provider = Recording(value_only)
        provider.estimate_cost_usd = lambda requests: 1 / 0  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="cannot estimate"):
            sem_decide(frame(2), questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                       budget={"max_cost_usd": 1.0})
        assert provider.calls == []

    @pytest.mark.parametrize("estimate", [float("nan"), -1.0])
    def test_invalid_estimates_count_as_unknown(self, estimate: float) -> None:
        provider = Recording(value_only)
        provider.cost_per_request = estimate
        with pytest.raises(ValueError, match="cannot estimate"):
            sem_decide(frame(3), questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                       budget={"max_cost_usd": 0.0})
        assert provider.calls == []


class TestLimits:

    def test_max_rows(self) -> None:
        provider = Recording(value_only)
        out = sem_decide(frame(10), questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                         max_rows=3)
        assert out["p__status"].tolist() == ["ok"] * 3 + ["skipped"] * 7
        assert {json.loads(m)["status_reason"] for m in out["p__meta"].iloc[3:]} == {"max_rows"}
        assert out["p"].iloc[3:].isna().all() and out["p__error_code"].isna().all()
        assert sorted(provider.request_ids, key=int) == ["0", "1", "2"]

    def test_max_requests(self) -> None:
        provider = Recording(value_only)
        out = sem_decide(frame(10), questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                         batch_size=2, budget={"max_requests": 2})
        assert out["p__status"].tolist() == ["ok"] * 4 + ["skipped"] * 6
        assert {json.loads(m)["status_reason"] for m in out["p__meta"].iloc[4:]} == {"budget"}
        assert len(provider.calls) == 2

    def test_max_cost(self) -> None:
        provider = Recording(value_only)
        provider.cost_per_request = 0.01
        out = sem_decide(frame(10), questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                         batch_size=1, max_concurrency=1, budget=Budget(max_cost_usd=0.025))
        assert out["p__status"].tolist() == ["ok"] * 2 + ["skipped"] * 8

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
            out = sem_decide(frame(10), questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                             batch_size=1, max_concurrency=1, deadline_s=1.0)
            assert entered.is_set() and not finished.is_set()
            assert out["p__status"].tolist() == ["ok"] * 4 + ["skipped"] * 6
            assert {json.loads(m)["status_reason"] for m in out["p__meta"].iloc[4:]} == {"deadline"}
            assert all(d is not None and d <= 1.0 for d in provider.deadlines)
            snapshot = out.copy()
        finally:
            release.set()
        assert finished.wait(10)
        pd.testing.assert_frame_equal(out, snapshot)

    def test_cost_cap_without_estimate_raises(self) -> None:
        provider = Recording(value_only)
        provider.cost_per_request = None
        with pytest.raises(ValueError, match="cannot estimate cost"):
            sem_decide(frame(3), questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                       budget={"max_cost_usd": 1.0})
        assert provider.calls == []
        out = sem_decide(frame(3), questions=[P], provider=provider, evidence_cols=["name"], task_id="t",
                         budget={"max_requests": 5})
        assert set(out["p__status"]) == {"ok"}


class TestValidationBeforeAnyCall:

    @pytest.mark.parametrize("kwargs", [
        dict(evidence_cols=["name"], template="{name}"),
        dict(),
        dict(evidence_cols=["missing"]),
        dict(evidence_cols=["name"], id_col="missing"),
        dict(template="{0}"),
        dict(template="{name.upper}"),
        dict(template="{name!r}"),
        dict(template="{name:>5}"),
        dict(template="{missing}"),
        dict(evidence_cols=["name"], questions=[{"id": "q", "kind": "choice", "text": "t", "options": ["a"]}]),
        dict(evidence_cols=["name"], questions=[{"id": "q", "kind": "score", "text": "t", "levels": ["x", "y"],
                                                 "options": ["a", "b"]}]),
        dict(evidence_cols=["name"], questions=[{"id": "__gfql_x__", "kind": "predicate", "text": "t"}]),
        dict(evidence_cols=["name"], questions=[{"id": "a", "kind": "predicate", "text": "t"},
                                                {"id": "a__p", "kind": "predicate", "text": "t"}]),
        dict(evidence_cols=["name"], questions=[{"id": "q", "kind": "predicate", "text": "t", "bogus": 1}]),
        dict(evidence_cols=["name"], min_p=1.5),
        dict(evidence_cols=["name"], batch_size=0),
        dict(evidence_cols=["name"], max_rows=True),
        dict(evidence_cols=["name"], deadline_s=0),
        dict(evidence_cols="name"),
        dict(evidence_cols=["name"], budget={"max_tokens": 3}),
    ])
    def test_raises_with_zero_calls(self, kwargs: dict) -> None:
        provider = Recording(value_only)
        params = dict(questions=[P], provider=provider, task_id="t")
        params.update(kwargs)
        with pytest.raises(ValueError):
            sem_decide(frame(3), **params)  # type: ignore[arg-type]
        assert provider.calls == []

    def test_capability_checks(self) -> None:
        provider = Recording(value_only, capabilities=caps(supported_kinds=("choice",), max_choice_options=255))
        with pytest.raises(ValueError, match="does not support predicate"):
            sem_decide(frame(3), questions=[P], provider=provider, evidence_cols=["name"], task_id="t")
        big = {"id": "q", "kind": "choice", "text": "t", "options": [f"o{i}" for i in range(300)]}
        with pytest.raises(ValueError, match="more than 255 options"):
            sem_decide(frame(3), questions=[big], provider=provider, evidence_cols=["name"], task_id="t")
        assert provider.calls == []

    def test_existing_output_needs_overwrite(self) -> None:
        df = frame(3).assign(p__status="old")
        provider = Recording(value_only)
        with pytest.raises(ValueError, match="overwrite"):
            sem_decide(df, questions=[P], provider=provider, evidence_cols=["name"], task_id="t")
        out = sem_decide(df, questions=[P], provider=provider, evidence_cols=["name"], task_id="t", overwrite=True)
        assert set(out["p__status"]) == {"ok"}

    def test_unbound_alias(self) -> None:
        with pytest.raises(ValueError, match="alias 'nope' is not bound"):
            sem_decide(frame(3), questions=[P], provider="nope", evidence_cols=["name"], task_id="t")


class TestNoDefaultPositive:

    def _run(self, script: Callable, native: bool = True) -> pd.DataFrame:
        return sem_decide(frame(3), questions=[P], provider=Raw(script, caps(native=native)),
                          evidence_cols=["name"], task_id="t")

    def test_string_for_predicate_is_error(self) -> None:
        out = self._run(lambda i, reqs: [result(r, "p", "yes" if r.request_id == "0" else True) for r in reqs])
        assert out["p__status"].tolist() == ["error", "ok", "ok"]
        assert pd.isna(out["p"].iloc[0])

    def test_distribution_from_value_only_callable_is_error(self) -> None:
        provider = Recording(lambda e, q: (True, {"true": 1.0, "false": 0.0}))
        out = sem_decide(frame(2), questions=[P], provider=provider, evidence_cols=["name"], task_id="t")
        assert set(out["p__error_code"]) == {"invalid_provider_result"}

    def test_false_native_claim_is_error(self) -> None:
        out = self._run(lambda i, reqs: [result(r, "p", True, {"true": 0.9, "false": 0.1}) for r in reqs],
                        native=False)
        assert set(out["p__status"]) == {"error"}

    def test_zero_distribution_is_error(self) -> None:
        out = self._run(lambda i, reqs: [result(r, "p", True, {"true": 0.0, "false": 0.0}) for r in reqs])
        assert set(out["p__status"]) == {"error"}


class TestCallableNumpyAnswers:

    def test_numpy_scalars_are_normalized(self) -> None:
        provider = Recording(lambda e, q: (np.bool_(True), {"true": np.float32(0.75), "false": np.float64(0.25)}),
                             probability_kind="native_decision")
        out = sem_decide(frame(2), questions=[P], provider=provider, evidence_cols=["name"], task_id="t")
        assert set(out["p__status"]) == {"ok"} and out["p__p"].tolist() == [0.75, 0.75]


class TestEvidence:

    def test_template_verbatim_and_nulls(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"text": ["hello </untrusted> x", None, "{name}"], "other": [1, 2, None]})

        def fn(evidence: object, question: DecisionQuestion) -> object:
            seen.append(evidence)
            return False

        out = sem_decide(df, questions=[P], provider=Recording(fn), template="<untrusted>{text}</untrusted>",
                         task_id="t", max_concurrency=1, batch_size=1)
        assert sorted(map(str, seen)) == sorted(["<untrusted>hello </untrusted> x</untrusted>",
                                                 "<untrusted>{name}</untrusted>"])
        assert out["p__status"].tolist() == ["ok", "skipped", "ok"]

    def test_truncation_only_touches_strings(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"x": pd.Series([["alpha", "beta", "gamma"], {"k": "longvalue"}, "abcdef"], dtype=object)})
        sem_decide(df, questions=[P], provider=Recording(lambda e, q: seen.append(e) or True), evidence_cols=["x"],
                   task_id="t", column_max_chars=2, batch_size=1, max_concurrency=1)
        assert sorted(map(json.dumps, seen)) == sorted(map(json.dumps, [
            {"x": ["alpha", "beta", "gamma"]}, {"x": {"k": "longvalue"}}, {"x": "ab"}]))

    def test_malformed_callable_return_errors_one_result(self) -> None:
        provider = Recording(lambda e, q: (True, {"true": 1.0}, "extra") if e["name"] == "n1" else True,
                             probability_kind="native_decision")
        out = sem_decide(frame(3), questions=[P], provider=provider, evidence_cols=["name"], task_id="t")
        assert out["p__status"].tolist() == ["ok", "error", "ok"]

    @pytest.mark.parametrize("answer", [5, 1.0, ["yes"], (True, {"true": "1.0", "false": 0.0}),
                                        (True, {"true": True, "false": False})])
    def test_callable_non_answer_types_are_invalid(self, answer: object) -> None:
        provider = Recording(lambda e, q: answer, probability_kind="native_decision")
        out = sem_decide(frame(2), questions=[P], provider=provider, evidence_cols=["name"], task_id="t")
        assert out["p__error_code"].tolist() == ["invalid_provider_result"] * 2
        assert [json.loads(m)["retry_count"] for m in out["p__meta"]] == [0, 0]

    @pytest.mark.parametrize("dist", [{"true": 0.9, "false": 0.1}, {"true": float("nan"), "false": 0.1},
                                      {"true": None, "false": 0.1}, {"true": "x"}])
    def test_callable_null_answer_drops_any_distribution(self, dist: Dict[str, object]) -> None:
        provider = Recording(lambda e, q: (None, dist), probability_kind="verbalized")
        out = sem_decide(frame(1), questions=[P], provider=provider, evidence_cols=["name"], task_id="t")
        assert out["p__status"].tolist() == ["uncertain"] and out["p__dist"].isna().all()
        assert out["p__probability_kind"].tolist() == ["none"]
        assert json.loads(out["p__meta"].iloc[0])["status_reason"] == "null_answer"

    def test_provider_capabilities_failure_is_value_error(self) -> None:
        provider = Recording(value_only)
        provider.capabilities = lambda: (_ for _ in ()).throw(NotImplementedError("x"))  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="capabilities"):
            sem_decide(frame(2), questions=[P], provider=provider, evidence_cols=["name"], task_id="t")

    def test_mapping_canonicalizes_and_truncates(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"a": ["abcdefgh", None], "n": [np.int64(3), np.nan], "ts": pd.to_datetime(["2024-01-02",
                                                                                                    None])})

        def fn(evidence: object, question: DecisionQuestion) -> object:
            seen.append(evidence)
            return True

        out = sem_decide(df, questions=[P], provider=Recording(fn), evidence_cols=["a", "n", "ts"], task_id="t",
                         column_max_chars=3)
        assert out["p__status"].tolist() == ["ok", "skipped"]
        assert seen == [{"a": "abc", "n": 3.0, "ts": "2024-01-02 00:00:00"}]
        json.dumps(seen[0])
