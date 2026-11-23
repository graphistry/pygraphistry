"""explain_sem_decide, explain_sem_map, semantic_trace and the verdict helpers."""
import contextvars
import json
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import pandas as pd
import pytest

import graphistry
from graphistry import call
from graphistry.compute.semantic import (
    CallableGenerationProvider, MemorySemanticCache, OpenAIChatProvider, OpenAIDecisionsProvider, SemanticRunStats, TokenPrices, bind_providers,
    explain_sem_decide, explain_sem_map, sem_decide, sem_map, semantic_summary, semantic_trace, semantic_verdicts)
from graphistry.compute.semantic import cache as cache_module, explain as explain_module, observe as observe_module
from graphistry.compute.semantic.contract import DecisionQuestion, DecisionRequest
from graphistry.tests.compute.semantic.http_test_utils import FakeResponse, FakeSession
from graphistry.tests.compute.semantic.semantic_test_utils import (
    C, P, Recording, RecordingGenerator, Scripted, caps, full_answer, result)


Q = [P]


def serial(native: bool = False) -> Dict[str, object]:
    return dict(capabilities=caps(native=native, concurrency_limit=1, max_rows_per_request=1))


def table(names: Sequence[Optional[str]]) -> pd.DataFrame:
    return pd.DataFrame({"name": list(names), "note": [f"n{i}" for i in range(len(names))]})


def decide_args(df: pd.DataFrame, provider: object, questions: Sequence[object] = Q, **kwargs: object) -> Dict[str, object]:
    args: Dict[str, object] = dict(questions=list(questions), provider=provider, task_id="t", evidence_cols=["name"])
    args.update(kwargs)
    return args


def versioned(**kwargs: object) -> Recording:
    return Recording(full_answer, fingerprint="v1", probability_kind="native_decision", **serial(native=True), **kwargs)  # type: ignore[arg-type]


def run_traced(df: pd.DataFrame, **args: object) -> "tuple[pd.DataFrame, SemanticRunStats]":
    with semantic_trace() as runs:
        out = sem_decide(df, **args)  # type: ignore[arg-type]
    assert len(runs) == 1
    return out, runs[0]


class TestExplainSendsNothing:

    def test_no_provider_call_and_no_probe(self) -> None:
        class Spy(Recording):
            probed = 0

            def prepare_run(self, requests: Sequence[DecisionRequest], admit: object, *, deadline_s: object) -> "Spy":
                Spy.probed += 1
                return self
        spy = Spy(fingerprint="v1", **serial())
        plan = explain_sem_decide(table(["a", "b", "a"]), **decide_args(table(["a"]), spy))
        assert spy.request_ids == [] and Spy.probed == 0 and plan["planned_calls"] == 2

    def test_an_http_provider_sends_no_request(self) -> None:
        session = FakeSession([FakeResponse(200, {"answers": []})])
        provider = OpenAIDecisionsProvider("k-secret", "m", session=session, prices=TokenPrices(input=1.0, cached_input=0.1))
        plan = explain_sem_decide(table(["a", "b"]), **decide_args(table(["a"]), provider, questions=[P, C]))
        assert session.posts == [] and plan["planned_calls"] == 2 and plan["planned_questions"] == 4
        assert plan["estimated_input_cost_usd"] is not None and plan["estimated_input_cost_usd"] > 0

    def test_sem_map_sends_nothing_either(self) -> None:
        provider = RecordingGenerator(**serial())
        plan = explain_sem_map(table(["a", "b", "a"]), instruction="i", out_col="o", provider=provider, task_id="t", evidence_cols=["name"])
        assert provider.calls == [] and plan["op"] == "sem_map" and plan["provider_rows"] == 2
        assert plan["estimated_input_cost_usd"] is None

    def test_a_plan_leaves_the_cache_as_it_was(self) -> None:
        cache = MemorySemanticCache()
        df = table(["a", "b"])
        with bind_providers({}, cache=cache):
            sem_decide(df, **decide_args(df, versioned()))
            before = (cache.stats(), list(cache._entries))
            plan = explain_sem_decide(df, **decide_args(df, versioned()))
            assert (cache.stats(), list(cache._entries)) == before
        assert plan["cache_hits"] == 2 and plan["cache_misses"] == 0 and plan["planned_calls"] == 0

    def test_peek_does_not_touch_counters_recency_or_expired_entries(self) -> None:
        now = [0.0]
        cache = MemorySemanticCache(max_entries=2, ttl_s=10, clock=lambda: now[0])
        cache.put_many({"a": "1", "b": "2"})
        assert cache.peek_many(["a", "z"]) == {"a": "1"}
        assert cache.stats()["hits"] == 0 and cache.stats()["misses"] == 0
        cache.put_many({"c": "3"})                                   # "a" is still the oldest, so it goes
        assert cache.peek_many(["a", "b", "c"]) == {"b": "2", "c": "3"}
        now[0] = 11
        assert cache.peek_many(["b"]) == {} and len(cache) == 2

    def test_a_cache_without_peek_still_plans_correctly(self) -> None:
        class Plain:
            def __init__(self) -> None:
                self.data: Dict[str, str] = {}

            def get_many(self, keys: Sequence[str]) -> Dict[str, str]:
                return {k: self.data[k] for k in keys if k in self.data}

            def put_many(self, entries: Dict[str, str]) -> None:
                self.data.update(entries)

            def clear(self) -> None:
                self.data.clear()
        cache, df = Plain(), table(["a", "b"])
        with bind_providers({}, cache=cache):
            sem_decide(df, **decide_args(df, versioned()))
            assert explain_sem_decide(df, **decide_args(df, versioned()))["cache_hits"] == 2

    def test_bad_parameters_and_an_unestimable_cost_cap_raise_like_a_run(self) -> None:
        df = table(["a"])
        with pytest.raises(ValueError, match="batch_size"):
            explain_sem_decide(df, **decide_args(df, versioned()), batch_size=0)
        provider = Scripted(lambda i, requests: [], caps(native=True))
        provider.estimate_cost_usd = lambda requests: None  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="cannot estimate cost"):
            explain_sem_decide(df, **decide_args(df, provider), budget={"max_cost_usd": 1.0})
        with pytest.raises(ValueError, match="cannot estimate cost"):
            sem_decide(df, **decide_args(df, provider), budget={"max_cost_usd": 1.0})


CONFIGS = [
    dict(),
    dict(dedupe=False),
    dict(batch_size=2),
    dict(max_rows=4),
    dict(budget={"max_requests": 2}, batch_size=1),
    dict(budget={"max_requests": 3}, batch_size=1, max_rows=6),
    dict(max_concurrency=2),
    dict(budget={"max_requests": 1}, batch_size=3),
]


class TestPlanEqualsExecution:

    @pytest.mark.parametrize("config", CONFIGS)
    @pytest.mark.parametrize("questions", [[P], [P, C]])
    @pytest.mark.parametrize("warm", [False, True])
    def test_decide(self, config: Dict[str, object], questions: List[object], warm: bool) -> None:
        df = table(["a", "b", "a", None, "c", "b", "d", "a"])
        cache = MemorySemanticCache()
        with bind_providers({}, cache=cache):
            if warm:
                sem_decide(table(["a", "d"]), **decide_args(df, versioned(), questions=questions))
            plan = explain_sem_decide(df, **decide_args(df, versioned(), questions=questions), **config)  # type: ignore[arg-type]
            provider = versioned()
            _out, stats = run_traced(df, **decide_args(df, provider, questions=questions), **config)
        assert plan["total_rows"] == stats["total_rows"] and plan["null_evidence_rows"] == stats["null_evidence_rows"] == 1
        assert plan["max_rows_skipped"] == stats["max_rows_skipped"] and plan["active_rows"] == stats["active_rows"]
        assert plan["dedupe_saved_rows"] == stats["deduped_rows"] and plan["cache_hits"] == stats["cache_hits"]
        assert plan["provider_rows"] == stats["provider_rows"] and plan["would_budget_skip_rows"] == stats["budget_skipped_rows"]
        assert plan["calls_within_budget"] == stats["provider_calls"] == len(provider.calls)
        assert plan["effective_concurrency"] >= stats["peak_in_flight"] and plan["unique_rows"] == plan["active_rows"] - plan["dedupe_saved_rows"]
        if warm and not config.get("max_rows"):
            assert plan["cache_hits"] >= 1

    @pytest.mark.parametrize("config", [dict(), dict(dedupe=False), dict(batch_size=1, budget={"max_requests": 2}), dict(max_rows=3)])
    def test_map(self, config: Dict[str, object]) -> None:
        df = table(["a", "b", "a", "c", None])
        args = dict(instruction="i", out_col="o", task_id="t", evidence_cols=["name"])
        plan = explain_sem_map(df, provider=RecordingGenerator(**serial()), **args, **config)  # type: ignore[arg-type]
        provider = RecordingGenerator(**serial())
        with semantic_trace() as runs:
            sem_map(df, provider=provider, **args, **config)  # type: ignore[arg-type]
        stats = runs[0]
        assert plan["provider_rows"] == stats["provider_rows"] and plan["dedupe_saved_rows"] == stats["deduped_rows"]
        assert plan["would_budget_skip_rows"] == stats["budget_skipped_rows"]
        assert plan["calls_within_budget"] == stats["provider_calls"] == len(provider.calls)

    def test_the_estimated_cost_is_the_cost_the_run_reserves_and_a_cost_cap_skips_the_same_rows(self) -> None:
        def reply(body: Dict[str, object]) -> FakeResponse:
            answers = [{"type": "predicate", "name": q["name"], "probability": 0.9} for q in body["questions"]]  # type: ignore[union-attr]
            return FakeResponse(200, {"model": "m", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 1}})
        df = table([f"company number {i}" for i in range(6)])
        for cap, expected_skipped in ((None, 0), (0.00003, None)):
            session = FakeSession([reply])
            provider = OpenAIDecisionsProvider("k", "m", session=session, prices=TokenPrices(input=1.0, cached_input=0.1))
            kwargs: Dict[str, object] = {} if cap is None else dict(budget={"max_cost_usd": cap})
            plan = explain_sem_decide(df, **decide_args(df, provider), **kwargs)  # type: ignore[arg-type]
            assert session.posts == []
            _out, stats = run_traced(df, **decide_args(df, provider), **kwargs)  # type: ignore[arg-type]
            if cap is not None:
                assert plan["estimated_input_cost_usd"] == pytest.approx(stats["estimated_cost_usd"])
            else:
                assert stats["estimated_cost_usd"] is None and plan["estimated_input_cost_usd"] > 0    # a trace makes no estimate call
            assert plan["would_budget_skip_rows"] == stats["budget_skipped_rows"] and plan["calls_within_budget"] == stats["provider_calls"]
            if expected_skipped is not None:
                assert stats["budget_skipped_rows"] == expected_skipped
            else:
                assert 0 < stats["budget_skipped_rows"] < 6

    def test_a_provider_that_cannot_report_probabilities_plans_no_call(self) -> None:
        session = FakeSession([FakeResponse(200, {"choices": []})])
        provider = OpenAIChatProvider("k", "m", strategy="text", session=session, prices=TokenPrices(input=1.0, cached_input=0.1))
        df = table(["a", "b"])
        plan = explain_sem_decide(df, **decide_args(df, provider), require_probabilities=True)
        assert plan["known_probability_gap"] is True and plan["planned_calls"] == 0 and plan["strategy"] == "text"
        _out, stats = run_traced(df, **decide_args(df, provider), require_probabilities=True)
        assert stats["provider_calls"] == 0 and session.posts == [] and stats["error_codes"] == {"probabilities_unavailable": 2}


class TestTrace:

    def test_funnel_verdicts_and_calls_of_a_direct_run(self) -> None:
        df = table(["a", "a", "b", None, "c"])
        out, stats = run_traced(df, **decide_args(df, versioned()), max_rows=4, min_p=0.95)
        assert (stats["op"], stats["total_rows"], stats["null_evidence_rows"], stats["max_rows_skipped"]) == ("sem_decide", 5, 1, 1)
        assert (stats["active_rows"], stats["deduped_rows"], stats["provider_rows"], stats["provider_calls"]) == (3, 1, 2, 2)
        assert stats["status_counts"] == {"p": {"uncertain": 3, "skipped": 2}}
        assert stats["uncertain_reasons"] == {"min_p": 3} and stats["error_codes"] == {}
        assert stats["probability_kind_counts"] == {"native_decision": 3, "none": 2}
        assert stats["questions"] == 1 and stats["cache_bound"] is False and stats["wall_ms"] > 0
        assert list(out["p__status"]) == ["uncertain", "uncertain", "uncertain", "skipped", "skipped"]

    def test_errors_are_counted_by_code(self) -> None:
        def boom(evidence: object, question: object) -> object:
            raise RuntimeError("down")
        df = table(["a", "b", "a"])
        _out, stats = run_traced(df, **decide_args(df, Recording(boom, **serial())))
        assert stats["status_counts"] == {"p": {"error": 3}} and stats["error_codes"] == {"callable_exception": 3}
        assert stats["billed_cost_usd"] == 0.0

    def test_billed_cost_counts_each_paid_answer_once_for_dedupe_and_cache(self) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[object]:
            return [result(request, "p", True, cost_usd=0.25, latency_ms=40.0, retry_count=1,
                           extra={"cached_input_tokens": 7}) for request in requests]

        class Fingerprinted(Scripted):
            def cache_fingerprint(self) -> str:
                return "s1"
        df, cache = table(["a", "a", "b", "a"]), MemorySemanticCache()
        with bind_providers({}, cache=cache):
            out, cold = run_traced(df, **decide_args(df, Fingerprinted(script, caps(native=True, concurrency_limit=1, max_rows_per_request=1))))
            _out, warm = run_traced(df, **decide_args(df, Fingerprinted(script, caps(native=True, concurrency_limit=1, max_rows_per_request=1))))
        assert cold["billed_cost_usd"] == 0.5 and cold["retries"] == 2 and cold["provider_latency_ms_sum"] == 80.0
        assert cold["cached_input_tokens"] == 14 and cold["deduped_rows"] == 2
        assert sum(json.loads(m)["cost_usd"] for m in out["p__meta"]) == cold["billed_cost_usd"]
        assert warm["billed_cost_usd"] == 0.0 and warm["provider_calls"] == 0 and warm["cache_hits"] == 2
        assert warm["cached_input_tokens"] == 0 and warm["retries"] == 0 and warm["cache_bound"] is True

    def test_a_real_provider_reports_cached_tokens_and_its_probability_kind(self) -> None:
        usage = {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 40}, "output_tokens": 5}
        session = FakeSession([lambda body: FakeResponse(200, {"model": "m", "usage": usage, "answers": [
            {"type": "predicate", "name": q["name"], "probability": 0.8} for q in body["questions"]]})])
        provider = OpenAIDecisionsProvider("k", "m", session=session, prices=TokenPrices(input=1.0, cached_input=0.1))
        df = table(["a", "b"])
        _out, stats = run_traced(df, **decide_args(df, provider))
        assert stats["cached_input_tokens"] == 80 and stats["provider_calls"] == len(session.posts) == 2
        assert stats["probability_kind_counts"] == {"native_decision": 2} and stats["billed_cost_usd"] > 0
        assert stats["strategy_used"] is None

    def test_strategy_and_probe_requests(self) -> None:
        class Probing(Recording):
            strategy = "logprobs"

            def prepare_run(self, requests: Sequence[DecisionRequest], admit: object, *, deadline_s: object) -> "Probing":
                assert admit(0.0) and admit(0.0)                      # type: ignore[operator]
                return self
        df = table(["a", "b", "c"])
        _out, stats = run_traced(df, **decide_args(df, Probing(fingerprint="v1", **serial())), budget={"max_requests": 4})
        assert stats["probe_calls"] == 2 and stats["strategy_used"] == "logprobs"
        assert stats["provider_calls"] == 2 and stats["budget_skipped_rows"] == 1      # 4 requests: 2 probes, 2 calls

    def test_a_probe_that_settles_on_a_strategy_without_probabilities_sends_no_row_call(self) -> None:
        session = FakeSession([FakeResponse(200, {"choices": []})])
        settled = OpenAIChatProvider("k", "m", strategy="text", session=session)

        class Settling(Recording):
            def prepare_run(self, requests: Sequence[DecisionRequest], admit: object, *, deadline_s: object) -> OpenAIChatProvider:
                assert admit(0.0)                                      # type: ignore[operator]
                return settled
        df = table(["a", "b"])
        provider = Settling(fingerprint="v1", **serial(native=True))
        _out, stats = run_traced(df, **decide_args(df, provider), require_probabilities=True)
        assert stats["provider_calls"] == 0 and stats["probe_calls"] == 1 and session.posts == [] and provider.request_ids == []
        assert stats["error_codes"] == {"probabilities_unavailable": 2}

    def test_peak_in_flight_is_the_number_of_calls_at_once(self) -> None:
        df = table([f"r{i}" for i in range(12)])
        provider = Recording(delay=0.1, capabilities=caps(concurrency_limit=4, max_rows_per_request=1))
        _out, stats = run_traced(df, **decide_args(df, provider))
        assert stats["peak_in_flight"] == 4 and stats["provider_calls"] == 12
        _out, stats = run_traced(df, **decide_args(df, Recording(**serial())))
        assert stats["peak_in_flight"] == 1

    def test_methods_and_gfql_calls_are_traced(self) -> None:
        g = graphistry.nodes(table(["a", "a", "b"]).assign(id=[1, 2, 3]), "id")
        question = {"id": "p", "kind": "predicate", "text": "?"}
        with semantic_trace() as runs:
            g.sem_decide([question], Recording(**serial()), task_id="t", evidence_cols=["name"])
            provider = Recording(**serial())
            with bind_providers({"p": provider, "m": RecordingGenerator(**serial())}):
                g.gfql([call("sem_decide", {"questions": [question], "provider": "p", "task_id": "t", "evidence_cols": ["name"]})])
                g.gfql([call("sem_map", {"instruction": "i", "out_col": "o", "provider": "m", "task_id": "t", "evidence_cols": ["name"]})])
        assert [r["op"] for r in runs] == ["sem_decide", "sem_decide", "sem_map"]
        assert [r["deduped_rows"] for r in runs] == [1, 1, 1] and provider.request_ids == ["1", "3"]

    def test_nothing_is_recorded_outside_a_block_and_blocks_nest(self) -> None:
        df = table(["a"])
        assert not observe_module.trace_active()
        sem_decide(df, **decide_args(df, Recording(**serial())))
        with semantic_trace() as outer:
            with semantic_trace() as inner:
                sem_decide(df, **decide_args(df, Recording(**serial())))
            sem_decide(df, **decide_args(df, Recording(**serial())))
        assert len(inner) == 1 and len(outer) == 2 and not observe_module.trace_active()

    def test_a_run_that_raises_before_sending_adds_nothing(self) -> None:
        df = table(["a"])
        with semantic_trace() as runs:
            with pytest.raises(ValueError):
                sem_decide(df, **decide_args(df, Recording(**serial())), batch_size=0)
            with pytest.raises(ValueError):
                sem_decide(df, **{**decide_args(df, Recording(**serial())), "evidence_cols": ["missing"]})  # type: ignore[arg-type]
        assert runs == []

    def test_each_thread_with_its_own_context_has_its_own_trace(self) -> None:
        results: Dict[int, List[SemanticRunStats]] = {}
        errors: List[BaseException] = []

        def work(worker: int) -> None:
            try:
                df = table([f"w{worker}-{i}" for i in range(worker + 2)])
                with semantic_trace() as runs:
                    sem_decide(df, **decide_args(df, Recording(delay=0.01, capabilities=caps(concurrency_limit=3, max_rows_per_request=1))))
                results[worker] = runs
            except BaseException as error:  # pragma: no cover
                errors.append(error)
        threads = [threading.Thread(target=lambda w=w: contextvars.copy_context().run(work, w)) for w in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors and all(len(results[w]) == 1 and results[w][0]["total_rows"] == w + 2 for w in range(6))
        assert all(results[w][0]["provider_calls"] == w + 2 and results[w][0]["peak_in_flight"] <= 3 for w in range(6))

    def test_a_thread_that_you_start_without_copying_the_context_sees_no_trace(self) -> None:
        df = table(["a"])
        seen: List[bool] = []
        with semantic_trace() as runs:
            thread = threading.Thread(target=lambda: (sem_decide(df, **decide_args(df, Recording(**serial()))), seen.append(observe_module.trace_active())))
            thread.start()
            thread.join()
        assert runs == [] and seen == [False]

    def test_deadline_skips_are_counted(self) -> None:
        df = table([f"r{i}" for i in range(6)])
        provider = Recording(delay=0.3, capabilities=caps(concurrency_limit=1, max_rows_per_request=1))
        _out, stats = run_traced(df, **decide_args(df, provider), deadline_s=0.4)
        assert stats["deadline_skipped_rows"] >= 3 and stats["status_counts"]["p"].get("skipped", 0) == stats["deadline_skipped_rows"]


class TestFindingsOfWaveOne:

    def test_a_trace_makes_no_provider_call_that_an_untraced_run_would_not(self) -> None:
        class Counting(Recording):
            estimates = 0

            def estimate_cost_usd(self, requests: Sequence[DecisionRequest]) -> float:
                Counting.estimates += 1
                return 0.0
        df = table(["a", "b", "c"])
        sem_decide(df, **decide_args(df, Counting(fingerprint="v1", **serial())))
        untraced = Counting.estimates
        run_traced(df, **decide_args(df, Counting(fingerprint="v1", **serial())))
        assert Counting.estimates == untraced == 0
        run_traced(df, **decide_args(df, Counting(fingerprint="v1", **serial())), budget={"max_cost_usd": 1.0})
        assert Counting.estimates > 0

    def test_a_provider_property_that_raises_does_not_lose_a_paid_run(self) -> None:
        class Angry(Recording):
            @property
            def strategy(self) -> str:                               # type: ignore[override]
                raise RuntimeError("no")
        df = table(["a", "b"])
        provider = Angry(fingerprint="v1", **serial())
        plan = explain_sem_decide(df, **decide_args(df, provider))
        out, stats = run_traced(df, **decide_args(df, provider))
        assert plan["strategy"] is None and stats["strategy_used"] is None and list(out["p__status"]) == ["ok", "ok"]

    def test_a_row_that_spans_several_calls_counts_once_if_any_call_was_skipped(self) -> None:
        df = table(["a", "b"])
        provider = Recording(fingerprint="v1", capabilities=caps(max_questions_per_request=1, concurrency_limit=1))
        args = decide_args(df, provider, questions=[P, C])
        plan = explain_sem_decide(df, **args, batch_size=2, budget={"max_requests": 1})
        _out, stats = run_traced(df, **args, batch_size=2, budget={"max_requests": 1})
        assert plan["planned_calls"] == 2 and plan["calls_within_budget"] == 1 == stats["provider_calls"]
        assert plan["would_budget_skip_rows"] == 2 == stats["budget_skipped_rows"]
        assert stats["status_counts"]["c"] == {"skipped": 2} and stats["status_counts"]["p"] == {"ok": 2}

    def test_only_a_deadline_counts_as_a_deadline_skip(self) -> None:
        df = table(["a", "b", "c", "d"])
        _out, stats = run_traced(df, **decide_args(df, Recording(**serial())), budget={"max_requests": 2}, batch_size=1)
        assert stats["budget_skipped_rows"] == 2 and stats["deadline_skipped_rows"] == 0

    def test_a_plan_without_a_cache_has_no_misses_and_a_known_gap_costs_nothing(self) -> None:
        df = table(["a", "b"])
        plan = explain_sem_decide(df, **decide_args(df, versioned()))
        assert plan["cache_bound"] is False and plan["cache_hits"] == plan["cache_misses"] == 0
        session = FakeSession([FakeResponse(200, {"choices": []})])
        chat = OpenAIChatProvider("k", "m", strategy="text", session=session, prices=TokenPrices(input=1.0, cached_input=0.1))
        gap = explain_sem_decide(df, **decide_args(df, chat), require_probabilities=True)
        assert gap["known_probability_gap"] is True and gap["estimated_input_cost_usd"] == 0.0 and gap["planned_calls"] == 0

    def test_a_map_plan_leaves_the_cache_as_it_was(self) -> None:
        cache, df = MemorySemanticCache(), table(["a", "b"])
        args = dict(instruction="i", out_col="o", task_id="t", evidence_cols=["name"])
        with bind_providers({}, cache=cache):
            sem_map(df, provider=RecordingGenerator(fingerprint="g1", **serial()), **args)  # type: ignore[arg-type]
            before = (cache.stats(), list(cache._entries))
            plan = explain_sem_map(df, provider=RecordingGenerator(fingerprint="g1", **serial()), **args)  # type: ignore[arg-type]
            assert (cache.stats(), list(cache._entries)) == before
        assert plan["cache_hits"] == 2 and plan["planned_calls"] == 0


class TestHelpersAfterReview:

    def test_a_summary_never_drops_a_row(self) -> None:
        df = pd.DataFrame({"q": pd.array([True, None, None, None, None], dtype="boolean"), "q__status": ["ok", "uncertain", None, "skipped", "error"]})
        with pytest.raises(ValueError, match="unknown status"):
            semantic_summary(df)
        with pytest.raises(ValueError, match="unknown status"):
            semantic_verdicts(df, "q")

    def test_a_positive_label_of_the_wrong_type_raises_and_a_near_miss_warns(self) -> None:
        df = pd.DataFrame({"u": ["a", "b"], "u__status": ["ok", "ok"]})
        with pytest.raises(ValueError, match="label strings"):
            semantic_verdicts(df, "u", positive=True)
        pred = pd.DataFrame({"q": pd.array([True, False], dtype="boolean"), "q__status": ["ok", "ok"]})
        with pytest.raises(ValueError, match="predicate"):
            semantic_verdicts(pred, "q", positive="true")
        with pytest.warns(UserWarning, match="case and spaces"):
            assert list(semantic_verdicts(df, "u", positive="A")) == ["no", "no"]

    def test_a_predicate_in_an_object_column_is_still_a_predicate(self) -> None:
        df = pd.DataFrame({"q": pd.Series([True, False, None], dtype=object), "q__status": ["ok", "ok", "error"]})
        assert list(semantic_verdicts(df, "q")) == ["yes", "no", "error"]


def answered(statuses: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({"q": pd.array([True, False, None, None, None], dtype="boolean")[:len(statuses)], "q__status": list(statuses)})


class TestVerdicts:

    def test_only_an_ok_row_is_yes_or_no(self) -> None:
        df = answered(["ok", "ok", "uncertain", "error", "skipped"])
        verdict = semantic_verdicts(df, "q")
        assert list(verdict) == ["yes", "no", "uncertain", "error", "skipped"]
        assert str(verdict.dtype) == "category" and list(verdict.cat.categories) == ["yes", "no", "uncertain", "error", "skipped"]
        assert int((~df["q"].fillna(False).astype(bool)).sum()) == 4          # the trap: four rows look like "no"
        assert int((verdict == "no").sum()) == 1

    def test_a_choice_needs_the_positive_labels(self) -> None:
        df = pd.DataFrame({"u": ["patch_now", "later", None, None], "u__status": ["ok", "ok", "error", "skipped"]})
        with pytest.raises(ValueError, match="positive"):
            semantic_verdicts(df, "u")
        assert list(semantic_verdicts(df, "u", positive="patch_now")) == ["yes", "no", "error", "skipped"]
        assert list(semantic_verdicts(df, "u", positive=["later", "patch_now"])) == ["yes", "yes", "error", "skipped"]

    def test_unknown_columns_and_statuses_are_refused(self) -> None:
        with pytest.raises(ValueError, match="result columns"):
            semantic_verdicts(pd.DataFrame({"a": [1]}), "q")
        with pytest.raises(ValueError, match="unknown status"):
            semantic_verdicts(pd.DataFrame({"q": [True], "q__status": ["maybe"]}), "q")

    def test_the_summary_counts_unanswered_rows_for_every_question(self) -> None:
        df = table(["a", "b", "a", None])
        out = sem_decide(df, **decide_args(df, versioned(), questions=[P, C]), max_rows=3)
        summary = semantic_summary(out)
        assert list(summary.index) == ["p", "c"] and list(summary.loc["p", ["ok", "skipped", "unanswered"]]) == [3, 1, 1]
        assert summary.loc["p", "unanswered_pct"] == 25.0
        assert list(semantic_summary(out, ["c"]).index) == ["c"]
        with pytest.raises(ValueError):
            semantic_summary(out, ["nope"])

    def test_a_real_result_gives_verdicts_with_the_right_index(self) -> None:
        df = table(["a", "b", None]).set_index(pd.Index([10, 20, 30]))
        out = sem_decide(df, **decide_args(df, versioned()))
        verdict = semantic_verdicts(out, "p")
        assert list(verdict.index) == [10, 20, 30] and list(verdict) == ["yes", "yes", "skipped"]


class TestNoGlobalState:

    def test_the_new_modules_keep_no_mutable_process_state(self) -> None:
        for module in (observe_module, explain_module):
            state = [name for name, value in vars(module).items()
                     if not name.startswith("__") and isinstance(value, (dict, list, set, OrderedDict))]
            assert state == [], (module.__name__, state)

    def test_a_trace_leaves_nothing_behind_even_after_an_error(self) -> None:
        with pytest.raises(RuntimeError):
            with semantic_trace():
                assert observe_module.trace_active()
                raise RuntimeError("stop")
        assert not observe_module.trace_active() and cache_module is not None
