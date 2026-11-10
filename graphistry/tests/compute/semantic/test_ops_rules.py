import json
import threading
import time
from typing import Dict, List, Optional, Sequence

import pandas as pd
import pytest

import graphistry
from graphistry import call, e_forward, n
from graphistry.compute.semantic import CallableDecisionProvider, CallableGenerationProvider, bind_providers
from graphistry.compute.semantic.contract import (
    Budget,
    DecisionQuestion,
    DecisionRequest,
    DecisionResult,
    ProviderCapabilities,
    compute_decision_hash,
)
from graphistry.compute.semantic.ops import sem_decide, sem_map


P = {"id": "p", "kind": "predicate", "text": "t"}


class Spy(CallableDecisionProvider):
    def __init__(self, fn=lambda e, q: True, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
        super().__init__(fn, **kwargs)  # type: ignore[arg-type]
        self.batches: List[List[str]] = []
        self.question_sets: List[List[str]] = []
        self.budgets: List[Optional[Budget]] = []
        self.deadlines: List[Optional[float]] = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.delay = 0.0

    def decide_batch(self, requests: Sequence[DecisionRequest], *, deadline_s: Optional[float],
                     budget: Optional[Budget]) -> List[DecisionResult]:
        with self.lock:
            self.batches.append([r.request_id for r in requests])
            self.question_sets.append([q.id for q in requests[0].questions])
            self.budgets.append(budget)
            self.deadlines.append(deadline_s)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return super().decide_batch(requests, deadline_s=deadline_s, budget=budget)
        finally:
            with self.lock:
                self.active -= 1


def caps(**kwargs: object) -> ProviderCapabilities:
    base = dict(supports_native_probabilities=False, max_questions_per_request=1000, max_evidence_tokens=None,
                supports_shared_prefix_cache=False, concurrency_limit=8)
    base.update(kwargs)
    return ProviderCapabilities(**base)  # type: ignore[arg-type]


def frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({"name": [f"n{i}" for i in range(n_rows)]})


class TestBatchingAndChunking:

    def test_default_batch_size_follows_provider_rows_per_request(self) -> None:
        spy = Spy(capabilities=caps(max_rows_per_request=2))
        sem_decide(frame(5), questions=[P], provider=spy, evidence_cols=["name"], task_id="t")
        assert sorted(len(b) for b in spy.batches) == [1, 2, 2]

    def test_fallback_batch_sizes_without_provider_limit(self) -> None:
        spy = Spy()
        sem_decide(frame(40), questions=[P], provider=spy, evidence_cols=["name"], task_id="t")
        assert sorted(len(b) for b in spy.batches) == [8, 32]

        sizes: List[int] = []
        gen = CallableGenerationProvider(lambda e, i: "x")
        original = gen.generate_batch

        def counting(requests, *, deadline_s, budget):  # type: ignore[no-untyped-def]
            sizes.append(len(requests))
            return original(requests, deadline_s=deadline_s, budget=budget)

        gen.generate_batch = counting  # type: ignore[method-assign]
        sem_map(frame(40), instruction="i", out_col="o", provider=gen, evidence_cols=["name"], task_id="t")
        assert sorted(sizes) == [8, 16, 16]

    def test_numpy_integer_parameters(self) -> None:
        import numpy as np
        spy = Spy()
        out = sem_decide(frame(5), questions=[P], provider=spy, evidence_cols=["name"], task_id="t",
                         batch_size=np.int64(2), max_rows=np.int64(4), budget={"max_requests": 2})
        assert out["p__status"].tolist() == ["ok"] * 4 + ["skipped"]

    def test_questions_beyond_the_cap_go_in_separate_calls(self) -> None:
        spy = Spy(capabilities=caps(max_questions_per_request=2))
        questions = [{"id": f"q{i}", "kind": "predicate", "text": "t"} for i in range(3)]
        out = sem_decide(frame(2), questions=questions, provider=spy, evidence_cols=["name"], task_id="t")
        assert sorted(spy.question_sets) == [["q0", "q1"], ["q2"]]
        assert all(set(out[f"q{i}__status"]) == {"ok"} for i in range(3))

    def test_threads_never_exceed_max_concurrency(self) -> None:
        spy = Spy()
        spy.delay = 0.05
        sem_decide(frame(12), questions=[P], provider=spy, evidence_cols=["name"], task_id="t", batch_size=1,
                   max_concurrency=2)
        assert spy.peak <= 2


class TestLimitsPassedToProviders:

    def test_remaining_budget_and_deadline_are_passed(self) -> None:
        spy = Spy()
        spy.delay = 0.2
        spy.estimate_cost_usd = lambda requests: 0.1 * len(requests)  # type: ignore[method-assign]
        sem_decide(frame(3), questions=[P], provider=spy, evidence_cols=["name"], task_id="t", batch_size=1,
                   max_concurrency=1, deadline_s=5.0, budget={"max_cost_usd": 1.0, "max_requests": 5})
        assert [b.max_requests for b in spy.budgets if b] == [5, 4, 3]
        assert [round(b.max_cost_usd, 6) for b in spy.budgets if b and b.max_cost_usd is not None] == [1.0, 0.9, 0.8]
        assert spy.deadlines[0] > spy.deadlines[1] > spy.deadlines[2]
        assert spy.deadlines[1] <= 5.0 - 0.2

    def test_cost_cap_allows_exact_float_sums(self) -> None:
        spy = Spy()
        spy.estimate_cost_usd = lambda requests: 0.1  # type: ignore[method-assign]
        out = sem_decide(frame(4), questions=[P], provider=spy, evidence_cols=["name"], task_id="t", batch_size=1,
                         max_concurrency=1, budget={"max_cost_usd": 0.3})
        assert out["p__status"].tolist() == ["ok", "ok", "ok", "skipped"]


class TestEvidenceShapes:

    def test_partially_null_rows_in_both_modes(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"a": ["x", None], "b": [None, "y"]})
        sem_decide(df, questions=[P], provider=Spy(lambda e, q: seen.append(e) or True), evidence_cols=["a", "b"],
                   task_id="t", batch_size=1, max_concurrency=1)
        sem_decide(df, questions=[P], provider=Spy(lambda e, q: seen.append(e) or True), template="{a}|{b}",
                   task_id="t", batch_size=1, max_concurrency=1)
        assert sorted(map(json.dumps, seen)) == sorted(map(json.dumps, [
            {"a": "x", "b": None}, {"a": None, "b": "y"}, "x|", "|y"]))

    def test_categorical_strings_are_truncated(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"c": pd.Categorical(["abcdefgh", "xy"])})
        sem_decide(df, questions=[P], provider=Spy(lambda e, q: seen.append(e) or True), evidence_cols=["c"],
                   task_id="t", column_max_chars=3)
        assert sorted(e["c"] for e in seen) == ["abc", "xy"]  # type: ignore[index]

    def test_non_finite_floats_are_sent_as_text(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"v": [float("inf")]})
        sem_decide(df, questions=[P], provider=Spy(lambda e, q: seen.append(e) or True), evidence_cols=["v"],
                   task_id="t")
        assert seen == [{"v": "inf"}]


class TestOutputDetails:

    def test_dist_label_order_and_kernel_hash(self) -> None:
        choice = DecisionQuestion(id="c", kind="choice", text="t", options=("a", "b", "c"))
        provider = Spy(lambda e, q: ("b", {"c": 0.1, "b": 0.6, "a": 0.3}), probability_kind="native_decision",
                       capabilities=caps(supports_native_probabilities=True))
        df = frame(1)
        out = sem_decide(df, questions=[choice], provider=provider, evidence_cols=["name"], task_id="t")
        assert list(json.loads(out["c__dist"].iloc[0])) == ["a", "b", "c"]
        meta = json.loads(out["c__meta"].iloc[0])
        assert meta["input_hash"] == compute_decision_hash({"name": "n0"}, choice, "1")

    def test_max_levels_capability(self) -> None:
        provider = Spy(capabilities=caps(max_levels=2))
        score = {"id": "s", "kind": "score", "text": "t", "levels": ["lo", "mid", "hi"]}
        with pytest.raises(ValueError, match="more than 2 levels"):
            sem_decide(frame(1), questions=[score], provider=provider, evidence_cols=["name"], task_id="t")
        assert provider.batches == []


class TestBindingDefaultsAndOverlay:

    def test_id_defaults_to_the_node_binding(self) -> None:
        spy = Spy()
        g = graphistry.nodes(pd.DataFrame({"nid": ["x", "y"], "name": ["a", "b"]}, index=[5, 5]), "nid")
        g.sem_decide([P], spy, task_id="t", evidence_cols=["name"])
        assert sorted(rid for batch in spy.batches for rid in batch) == ["x", "y"]

    def test_internal_edge_binding_is_treated_as_unbound(self) -> None:
        spy = Spy()
        edges = pd.DataFrame({"s": [1, 2, 1], "d": [2, 3, 3], "w": ["a", "b", "c"]}, index=[7, 7, 9])
        g = graphistry.nodes(pd.DataFrame({"id": [1, 2, 3]}), "id").edges(edges, "s", "d")
        with bind_providers({"p": spy}):
            out = g.gfql([n(), e_forward(), n(), call("sem_decide", {"questions": [P], "provider": "p", "task_id": "t",
                                                                     "kind": "edges", "evidence_cols": ["w"]})])
        assert sorted(rid for batch in spy.batches for rid in batch) == ["0", "1", "2"]
        assert set(out._edges["p__status"]) == {"ok"}

    def test_overlay_is_alias_by_alias(self) -> None:
        a, b, c = Spy(), Spy(), Spy()
        with bind_providers({"x": a, "y": b}):
            with bind_providers({"x": c}):
                sem_decide(frame(1), questions=[P], provider="x", evidence_cols=["name"], task_id="t")
                sem_decide(frame(1), questions=[P], provider="y", evidence_cols=["name"], task_id="t")
        assert len(c.batches) == 1 and len(b.batches) == 1 and a.batches == []


class TestCallableEmptyDistribution:

    @pytest.mark.parametrize("kind", ["native_decision", "none"])
    def test_empty_distribution_means_none(self, kind: str) -> None:
        choice = {"id": "c", "kind": "choice", "text": "t", "options": ["a", "b"]}
        provider = Spy(lambda e, q: ("a", {}), probability_kind=kind,
                       capabilities=caps(supports_native_probabilities=kind == "native_decision"))
        out = sem_decide(frame(1), questions=[choice], provider=provider, evidence_cols=["name"], task_id="t")
        assert out["c"].tolist() == ["a"] and out["c__status"].tolist() == ["ok"]
        assert out["c__probability_kind"].tolist() == ["none"] and out["c__dist"].isna().all()


class TestSemMapChecks:

    def test_duplicate_and_content_checks(self) -> None:
        from graphistry.compute.semantic.contract import GenerationResult

        class Dup:
            name = "dup"
            model = "m"

            def capabilities(self) -> ProviderCapabilities:
                return caps()

            def estimate_cost_usd(self, requests):  # type: ignore[no-untyped-def]
                return 0.0

            def generate_batch(self, requests, *, deadline_s, budget):  # type: ignore[no-untyped-def]
                good = [GenerationResult(request_id=r.request_id, value="x", status="ok", error_code=None, model="m",
                                         provider="dup", task_version="1", input_hash="h", latency_ms=None,
                                         cost_usd=None, cache_hit=None, retry_count=None) for r in requests]
                return good + good[:1]

        out = sem_map(frame(2), instruction="i", out_col="o", provider=Dup(), evidence_cols=["name"], task_id="t")
        assert set(out["o__error_code"]) == {"invalid_provider_result"}
        bad = CallableGenerationProvider(lambda e, i: None if e["name"] == "n0" else "ok")  # type: ignore[index]
        out = sem_map(frame(2), instruction="i", out_col="o", provider=bad, evidence_cols=["name"], task_id="t")
        assert out["o__status"].tolist() == ["uncertain", "ok"]


class TestRemainingRulePins:

    def test_huge_deadline_does_not_overflow(self) -> None:
        spy = Spy()
        spy.delay = 0.05
        out = sem_decide(frame(2), questions=[P], provider=spy, evidence_cols=["name"], task_id="t", batch_size=1,
                         max_concurrency=1, deadline_s=1e10)
        assert set(out["p__status"]) == {"ok"}

    def test_string_dtype_is_truncated(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"s": pd.array(["abcdef", None], dtype="string")})
        sem_decide(df, questions=[P], provider=Spy(lambda e, q: seen.append(e) or True), evidence_cols=["s"],
                   task_id="t", column_max_chars=2)
        assert seen == [{"s": "ab"}]

    def test_threads_never_exceed_concurrency_limit(self) -> None:
        spy = Spy(capabilities=caps(concurrency_limit=2))
        spy.delay = 0.05
        sem_decide(frame(10), questions=[P], provider=spy, evidence_cols=["name"], task_id="t", batch_size=1)
        assert spy.peak <= 2

    def test_min_p_is_inclusive_and_exact_limits_are_allowed(self) -> None:
        provider = Spy(lambda e, q: (True, {"true": 0.75, "false": 0.25}), probability_kind="native_decision",
                       capabilities=caps(supports_native_probabilities=True, max_levels=3, max_choice_options=2))
        out = sem_decide(frame(1), questions=[P], provider=provider, evidence_cols=["name"], task_id="t", min_p=0.75)
        assert out["p__status"].tolist() == ["ok"]
        exact = [{"id": "s", "kind": "score", "text": "t", "levels": ["a", "b", "c"]},
                 {"id": "c", "kind": "choice", "text": "t", "options": ["x", "y"]}]
        provider2 = Spy(lambda e, q: (q.labels() or ("",))[0], capabilities=caps(max_levels=3, max_choice_options=2))
        out = sem_decide(frame(1), questions=exact, provider=provider2, evidence_cols=["name"], task_id="t")
        assert out["s__status"].tolist() == ["ok"] and out["c__status"].tolist() == ["ok"]

    def test_meta_and_dist_json_formats(self) -> None:
        provider = Spy(lambda e, q: (True, {"true": 0.75, "false": 0.25}), probability_kind="native_decision",
                       capabilities=caps(supports_native_probabilities=True))
        out = sem_decide(frame(1), questions=[P], provider=provider, evidence_cols=["name"], task_id="t")
        meta = out["p__meta"].iloc[0]
        assert meta == json.dumps(json.loads(meta), sort_keys=True, separators=(",", ":"))
        assert out["p__dist"].iloc[0] == '{"true":0.75,"false":0.25}'

    def test_sem_map_batch_default_and_output_tokens(self) -> None:
        seen: List[Optional[int]] = []
        sizes: List[int] = []
        gen = CallableGenerationProvider(lambda e, i: "x", capabilities=caps(max_rows_per_request=3))
        original = gen.generate_batch

        def recording(requests, *, deadline_s, budget):  # type: ignore[no-untyped-def]
            sizes.append(len(requests))
            seen.extend(r.max_output_tokens for r in requests)
            return original(requests, deadline_s=deadline_s, budget=budget)

        gen.generate_batch = recording  # type: ignore[method-assign]
        sem_map(frame(7), instruction="i", out_col="o", provider=gen, evidence_cols=["name"], task_id="t",
                max_output_tokens=12)
        assert sorted(sizes) == [1, 3, 3] and set(seen) == {12}

    def test_cost_cap_is_not_loose(self) -> None:
        spy = Spy()
        spy.estimate_cost_usd = lambda requests: 0.01  # type: ignore[method-assign]
        out = sem_decide(frame(3), questions=[P], provider=spy, evidence_cols=["name"], task_id="t", batch_size=1,
                         max_concurrency=1, budget={"max_cost_usd": 0.0195})
        assert out["p__status"].tolist() == ["ok", "skipped", "skipped"]

    def test_overwrite_must_be_bool(self) -> None:
        with pytest.raises(ValueError, match="overwrite"):
            sem_decide(frame(1), questions=[P], provider=Spy(), evidence_cols=["name"], task_id="t",
                       overwrite="yes")  # type: ignore[arg-type]

    def test_numpy_budget_requests(self) -> None:
        import numpy as np
        out = sem_decide(frame(3), questions=[P], provider=Spy(), evidence_cols=["name"], task_id="t", batch_size=1,
                         max_concurrency=1, budget={"max_requests": np.int64(2)})
        assert out["p__status"].tolist() == ["ok", "ok", "skipped"]


class TestDeadlineFromOperatorStart:

    def test_time_before_the_first_call_counts(self) -> None:
        spy = Spy()
        slow_caps = spy.capabilities

        def slow_capabilities() -> ProviderCapabilities:
            time.sleep(0.3)
            return slow_caps()

        spy.capabilities = slow_capabilities  # type: ignore[method-assign]
        out = sem_decide(frame(2), questions=[P], provider=spy, evidence_cols=["name"], task_id="t", deadline_s=0.2)
        assert out["p__status"].tolist() == ["skipped", "skipped"] and spy.batches == []

        gen = CallableGenerationProvider(lambda e, i: "x")
        gen_caps = gen.capabilities

        def slow_gen_capabilities() -> ProviderCapabilities:
            time.sleep(0.3)
            return gen_caps()

        gen.capabilities = slow_gen_capabilities  # type: ignore[method-assign]
        out = sem_map(frame(2), instruction="i", out_col="o", provider=gen, evidence_cols=["name"], task_id="t",
                      deadline_s=0.2)
        assert out["o__status"].tolist() == ["skipped", "skipped"]

    def test_template_attribute_access_is_rejected_by_shape(self) -> None:
        df = pd.DataFrame({"a.b": ["x"], "a": ["y"]})
        with pytest.raises(ValueError, match="bare column"):
            sem_decide(df, questions=[P], provider=Spy(), template="{a.b}", task_id="t")


class TestFinalRulePins:

    def test_template_truncation_and_mapping_order(self) -> None:
        seen: List[object] = []
        df = pd.DataFrame({"b": ["bbbb"], "a": ["aaaa"]})
        sem_decide(df, questions=[P], provider=Spy(lambda e, q: seen.append(e) or True), template="{b}|{a}",
                   task_id="t", column_max_chars=2)
        sem_decide(df, questions=[P], provider=Spy(lambda e, q: seen.append(e) or True), evidence_cols=["b", "a"],
                   task_id="t")
        assert seen[0] == "bb|aa" and list(seen[1]) == ["b", "a"]  # type: ignore[call-overload]

    def test_callable_non_answers(self) -> None:
        out = sem_decide(frame(1), questions=[P], provider=Spy(lambda e, q: None), evidence_cols=["name"], task_id="t")
        assert out["p__status"].tolist() == ["uncertain"]
        assert json.loads(out["p__meta"].iloc[0])["status_reason"] == "null_answer"
        provider = Spy(lambda e, q: (None, {"true": 0.5, "false": 0.5}), probability_kind="native_decision",
                       capabilities=caps(supports_native_probabilities=True))
        out = sem_decide(frame(1), questions=[P], provider=provider, evidence_cols=["name"], task_id="t")
        assert out["p__status"].tolist() == ["uncertain"] and out["p__probability_kind"].tolist() == ["none"]
        assert out["p__dist"].isna().all()

    def test_sem_map_validation(self) -> None:
        with pytest.raises(ValueError, match="not a GenerationProvider"):
            sem_map(frame(1), instruction="i", out_col="o", provider=Spy(), evidence_cols=["name"], task_id="t")  # type: ignore[arg-type]
        gen = CallableGenerationProvider(lambda e, i: "x")
        for kwargs in (dict(deadline_s=0), dict(overwrite="yes")):
            with pytest.raises(ValueError):
                sem_map(frame(1), instruction="i", out_col="o", provider=gen, evidence_cols=["name"], task_id="t",
                        **kwargs)  # type: ignore[arg-type]
        import numpy as np
        out = sem_map(frame(1), instruction="i", out_col="o", provider=gen, evidence_cols=["name"], task_id="t",
                      max_output_tokens=np.int64(5))
        assert out["o__status"].tolist() == ["ok"]

    def test_callable_defaults(self) -> None:
        native = CallableDecisionProvider(lambda e, q: True, probability_kind="native_decision").capabilities()
        plain = CallableDecisionProvider(lambda e, q: True).capabilities()
        assert native.supports_native_probabilities and not plain.supports_native_probabilities
        assert plain.max_questions_per_request == 1000 and plain.concurrency_limit == 4
        out = sem_decide(frame(2), questions=[P], provider=CallableDecisionProvider(lambda e, q: True),
                         evidence_cols=["name"], task_id="t", budget={"max_cost_usd": 0.0})
        assert set(out["p__status"]) == {"ok"}
