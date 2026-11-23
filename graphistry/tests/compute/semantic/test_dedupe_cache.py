"""Within-call dedupe and the result cache of sem_decide and sem_map."""
import json
import threading
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd
import pytest

import graphistry
from graphistry import call
from graphistry.compute.semantic import (
    CallableDecisionProvider, MemorySemanticCache, OpenAIChatProvider, OpenAIDecisionsProvider, SemanticCache, bind_providers,
    sem_decide, sem_map)
from graphistry.compute.semantic import binding, cache as cache_module
from graphistry.compute.semantic.cache import cache_key, provider_fingerprint
from graphistry.compute.semantic.contract import DecisionQuestion, DecisionRequest, compute_decision_hash, compute_generation_hash
from graphistry.tests.compute.semantic.semantic_test_utils import (
    C, P, Recording, RecordingGenerator, Scripted, caps, full_answer, result)


def decide(df: pd.DataFrame, provider: object, questions: Sequence[object] = (P,), **kwargs: object) -> pd.DataFrame:
    params: Dict[str, object] = dict(evidence_cols=["name"], task_id="t")
    params.update(kwargs)
    return sem_decide(df, questions=list(questions), provider=provider, **params)  # type: ignore[arg-type]


def meta(out: pd.DataFrame, question_id: str = "p") -> List[Dict[str, object]]:
    return [json.loads(cell) for cell in out[f"{question_id}__meta"]]


def repeated(names: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({"name": list(names), "note": [f"note {i}" for i in range(len(names))]})


def serial(native: bool = False) -> Dict[str, object]:
    """Capabilities for a provider that runs one row at a time, so that the order of its calls is the order of the rows."""
    return dict(capabilities=caps(native=native, concurrency_limit=1, max_rows_per_request=1))


def versioned(fn: object = None, **kwargs: object) -> Recording:
    return Recording(fn or full_answer, fingerprint="v1", probability_kind="native_decision", **serial(native=True), **kwargs)  # type: ignore[arg-type]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestDedupe:

    def test_rows_with_the_same_evidence_are_asked_once(self) -> None:
        provider = Recording(**serial())
        out = decide(repeated(["a", "b", "a", "a", "c", "b"]), provider)
        assert provider.request_ids == ["0", "1", "4"]
        assert out["p"].tolist() == [True] * 6 and set(out["p__status"]) == {"ok"}
        assert [m["extra"] for m in meta(out)] == [None, None, {"deduped_from": "0"}, {"deduped_from": "0"}, None,
                                                  {"deduped_from": "1"}]

    def test_the_evidence_decides_not_other_columns(self) -> None:
        provider = Recording(**serial())
        decide(repeated(["a", "a"]), provider, evidence_cols=["name"])
        assert provider.request_ids == ["0"]
        provider = Recording(**serial())
        decide(repeated(["a", "a"]), provider, evidence_cols=["name", "note"])
        assert provider.request_ids == ["0", "1"]

    def test_dedupe_false_asks_for_every_row_so_votes_can_differ(self) -> None:
        answers = iter([True, False, True])
        provider = Recording(lambda evidence, question: next(answers), **serial())
        out = decide(repeated(["a", "a", "a"]), provider, dedupe=False)
        assert provider.request_ids == ["0", "1", "2"] and out["p"].tolist() == [True, False, True]
        assert all(m["extra"] is None for m in meta(out))

    def test_a_copy_costs_nothing_so_the_column_sums_to_the_billed_spend(self) -> None:
        def script(index: int, requests: Sequence[DecisionRequest]) -> List[object]:
            return [result(request, "p", True, cost_usd=0.25, latency_ms=40.0, retry_count=2) for request in requests]
        out = decide(repeated(["a", "a", "b", "a"]), Scripted(script), batch_size=1)
        rows = meta(out)
        assert [(m["cost_usd"], m["latency_ms"], m["retry_count"]) for m in rows] == [
            (0.25, 40.0, 2), (0.0, 0.0, 0), (0.25, 40.0, 2), (0.0, 0.0, 0)]
        assert sum(m["cost_usd"] for m in rows) == 0.5  # type: ignore[misc]

    def test_skipped_rows_never_merge_and_budget_sees_only_the_asked_rows(self) -> None:
        df = repeated(["x", "x", "y", "y", None, "x"])  # type: ignore[list-item]
        provider = Recording(**serial())
        out = decide(df, provider, batch_size=1, max_rows=5, budget={"max_requests": 2})
        assert provider.request_ids == ["0", "2"]
        assert out["p__status"].tolist() == ["ok"] * 4 + ["skipped", "skipped"]
        assert [m["status_reason"] for m in meta(out)][4:] == ["null_evidence", "max_rows"]
        provider = Recording(**serial())
        out = decide(df, provider, batch_size=1, max_rows=5, budget={"max_requests": 2}, dedupe=False)
        assert provider.request_ids == ["0", "1"] and out["p__status"].tolist()[2:4] == ["skipped"] * 2

    def test_copies_of_an_unanswered_row_are_unanswered(self) -> None:
        def boom(evidence: object, question: object) -> object:
            raise RuntimeError("down")
        out = decide(repeated(["a", "a"]), Recording(boom, **serial()))
        assert out["p__status"].tolist() == ["error", "error"]
        assert out["p__error_code"].tolist() == ["callable_exception"] * 2

    def test_each_question_is_kept_for_a_copy(self) -> None:
        out = decide(repeated(["a", "a"]), versioned(), [P, C])
        assert out["c"].tolist() == ["b", "b"] and out["c__p"].tolist() == [0.6, 0.6]
        assert [m["extra"] for m in meta(out, "c")] == [None, {"deduped_from": "0"}]

    def test_sem_map_asks_once_per_distinct_row_and_dedupe_false_opts_out(self) -> None:
        df = repeated(["a", "b", "a"])
        provider = RecordingGenerator(lambda evidence, instruction: f"<{evidence['name']}>", **serial())  # type: ignore[index]
        out = sem_map(df, instruction="i", out_col="o", provider=provider, task_id="t", evidence_cols=["name"])
        assert [r.request_id for call_ in provider.calls for r in call_] == ["0", "1"]
        assert out["o"].tolist() == ["<a>", "<b>", "<a>"] and meta(out, "o")[2]["extra"] == {"deduped_from": "0"}
        provider = RecordingGenerator(**serial())
        sem_map(df, instruction="i", out_col="o", provider=provider, task_id="t", evidence_cols=["name"], dedupe=False)
        assert len([r for call_ in provider.calls for r in call_]) == 3

    def test_every_row_keeps_the_input_hash_of_its_own_evidence(self) -> None:
        names = ["a", "a", None, "b", "c", "b"]  # a duplicate and a null-evidence skip come before unique rows
        question = DecisionQuestion.from_mapping(P)
        expected = [compute_decision_hash({"name": name}, question, "1") for name in names]
        for cache in (None, MemorySemanticCache()):
            for _ in range(2 if cache is not None else 1):
                with bind_providers({}, cache=cache):
                    out = decide(repeated(names), versioned(), max_rows=5)  # type: ignore[arg-type]
                got = [m["input_hash"] for m in meta(out)]
                assert [h for i, h in enumerate(got) if i not in (2, 5)] == [h for i, h in enumerate(expected) if i not in (2, 5)]
                assert out["p__status"].tolist()[2] == "skipped" and out["p__status"].tolist()[5] == "skipped"
        generated = sem_map(repeated(["a", "a", "b"]), instruction="i", out_col="o", provider=RecordingGenerator(**serial()), task_id="t",
                            evidence_cols=["name"])
        assert [m["input_hash"] for m in meta(generated, "o")] == [compute_generation_hash({"name": n}, "i", "1") for n in "aab"]

    def test_dedupe_must_be_a_bool(self) -> None:
        with pytest.raises(ValueError, match="dedupe"):
            decide(repeated(["a"]), Recording(**serial()), dedupe="no")
        with pytest.raises(ValueError, match="dedupe"):
            sem_map(repeated(["a"]), instruction="i", out_col="o", provider=RecordingGenerator(**serial()), task_id="t",
                    evidence_cols=["name"], dedupe=1)  # type: ignore[arg-type]

    def test_method_and_call_take_dedupe(self) -> None:
        g = graphistry.nodes(repeated(["a", "a"]).assign(id=[1, 2]), "id")
        question = {"id": "p", "kind": "predicate", "text": "?"}
        for dedupe, expected in ((True, ["1"]), (False, ["1", "2"])):
            provider = Recording(**serial())
            g.sem_decide([question], provider, task_id="t", evidence_cols=["name"], dedupe=dedupe)
            assert provider.request_ids == expected
            provider = Recording(**serial())
            with bind_providers({"p": provider}):
                g.gfql([call("sem_decide", {"questions": [question], "provider": "p", "task_id": "t",
                                            "evidence_cols": ["name"], "dedupe": dedupe})])
            assert provider.request_ids == expected
        provider = RecordingGenerator(**serial())
        g.sem_map("i", "o", provider, task_id="t", evidence_cols=["name"], dedupe=False)
        assert len([r for call_ in provider.calls for r in call_]) == 2
        with bind_providers({"p": Recording(**serial())}), pytest.raises(Exception, match="dedupe"):
            g.gfql([call("sem_decide", {"questions": [question], "provider": "p", "task_id": "t",
                                        "evidence_cols": ["name"], "dedupe": "no"})])


class TestMemorySemanticCache:

    def test_get_returns_only_the_keys_it_holds(self) -> None:
        cache = MemorySemanticCache()
        cache.put_many({"a": "1", "b": "2"})
        assert cache.get_many(["a", "z", "b"]) == {"a": "1", "b": "2"}
        assert isinstance(cache, SemanticCache)

    def test_least_recently_used_entries_go_first(self) -> None:
        cache = MemorySemanticCache(max_entries=2)
        cache.put_many({"a": "1", "b": "2"})
        cache.get_many(["a"])
        cache.put_many({"c": "3"})
        assert cache.get_many(["a", "b", "c"]) == {"a": "1", "c": "3"}
        assert cache.stats()["evictions"] == 1 and len(cache) == 2

    def test_a_rewrite_replaces_the_value_and_refreshes_the_entry(self) -> None:
        cache = MemorySemanticCache(max_entries=2)
        cache.put_many({"a": "1", "b": "2"})
        cache.put_many({"a": "one"})
        cache.put_many({"c": "3"})
        assert cache.get_many(["a", "b", "c"]) == {"a": "one", "c": "3"} and cache.stats()["bytes"] == (1 + 3) + (1 + 1)

    def test_the_byte_cap_evicts_old_entries_and_refuses_an_entry_that_never_fits(self) -> None:
        cache = MemorySemanticCache(max_bytes=10)
        cache.put_many({"a": "1234", "b": "1234"})
        cache.put_many({"c": "1234"})
        assert set(cache.get_many(["a", "b", "c"])) == {"b", "c"} and cache.stats()["bytes"] == 10
        cache.put_many({"big": "x" * 50})
        stats = cache.stats()
        assert "big" not in cache.get_many(["big"]) and stats["rejected"] == 1 and stats["bytes"] == 10

    def test_entries_expire_after_the_time_to_live(self) -> None:
        clock = Clock()
        cache = MemorySemanticCache(ttl_s=60, clock=clock)
        cache.put_many({"a": "1"})
        clock.now += 59
        assert cache.get_many(["a"]) == {"a": "1"}
        clock.now += 2
        assert cache.get_many(["a"]) == {}
        stats = cache.stats()
        assert stats["expirations"] == 1 and stats["entries"] == 0 and stats["bytes"] == 0

    def test_an_entry_is_gone_at_exactly_its_time_to_live(self) -> None:
        clock = Clock()
        cache = MemorySemanticCache(ttl_s=60, clock=clock)
        cache.put_many({"a": "1"})
        clock.now += 60
        assert cache.get_many(["a"]) == {}

    def test_stats_drop_expired_entries_and_an_oversize_rewrite_drops_the_old_value(self) -> None:
        clock = Clock()
        cache = MemorySemanticCache(max_bytes=20, ttl_s=60, clock=clock)
        cache.put_many({"a": "1", "b": "2"})
        clock.now += 61
        stats = cache.stats()
        assert stats["entries"] == 0 and stats["bytes"] == 0 and stats["expirations"] == 2
        cache.put_many({"k": "v"})
        cache.put_many({"k": "x" * 40})
        assert cache.get_many(["k"]) == {} and cache.stats()["rejected"] == 1 and len(cache) == 0

    def test_a_write_starts_a_new_life_and_none_never_expires(self) -> None:
        clock = Clock()
        cache = MemorySemanticCache(ttl_s=60, clock=clock)
        cache.put_many({"a": "1"})
        clock.now += 50
        cache.put_many({"a": "2"})
        clock.now += 50
        assert cache.get_many(["a"]) == {"a": "2"}
        forever = MemorySemanticCache(ttl_s=None, clock=clock)
        forever.put_many({"a": "1"})
        clock.now += 10 ** 9
        assert forever.get_many(["a"]) == {"a": "1"}

    def test_expired_entries_go_before_live_ones_when_the_cache_is_full(self) -> None:
        clock = Clock()
        cache = MemorySemanticCache(max_entries=2, ttl_s=60, clock=clock)
        cache.put_many({"old": "1"})
        clock.now += 40
        cache.put_many({"mid": "2"})
        cache.get_many(["old"])  # "old" is now the most recently used
        clock.now += 30
        cache.put_many({"new": "3"})
        assert set(cache.get_many(["old", "mid", "new"])) == {"mid", "new"}
        assert cache.stats()["expirations"] >= 1

    def test_clear_stats_and_validation(self) -> None:
        cache = MemorySemanticCache()
        cache.put_many({"a": "1"})
        cache.get_many(["a", "b"])
        stats = cache.stats()
        assert (stats["hits"], stats["misses"], stats["puts"], stats["hit_rate"]) == (1, 1, 1, 0.5)
        cache.clear()
        assert len(cache) == 0 and cache.stats()["bytes"] == 0 and cache.get_many(["a"]) == {}
        assert MemorySemanticCache().stats()["hit_rate"] == 0.0
        for bad in (dict(max_entries=0), dict(max_entries=True), dict(max_bytes=0), dict(ttl_s=0), dict(ttl_s=-1),
                    dict(ttl_s=float("nan")), dict(ttl_s=True)):
            with pytest.raises(ValueError):
                MemorySemanticCache(**bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="strings"):
            cache.put_many({"a": 1})  # type: ignore[dict-item]

    def test_many_threads_keep_the_counters_and_caps_consistent(self) -> None:
        cache = MemorySemanticCache(max_entries=50, max_bytes=2000)
        errors: List[BaseException] = []
        lookups = [0] * 8

        def work(worker: int) -> None:
            try:
                for step in range(400):
                    key = f"k{(worker * 7 + step) % 120}"
                    cache.put_many({key: "v" * (step % 9)})
                    lookups[worker] += 2
                    cache.get_many([key, f"k{step % 120}"])
            except BaseException as error:  # pragma: no cover
                errors.append(error)

        threads = [threading.Thread(target=work, args=(worker,)) for worker in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        stats = cache.stats()
        assert not errors and stats["entries"] <= 50 and stats["bytes"] <= 2000
        assert stats["hits"] + stats["misses"] == sum(lookups) and stats["entries"] == len(cache)
        assert stats["bytes"] == sum(len(key) + len(entry[0]) for key, entry in cache._entries.items())


class FlakyCache:
    """A cache that raises, or answers with junk."""

    def __init__(self, mode: str) -> None:
        self.mode, self.puts = mode, 0

    def get_many(self, keys: Sequence[str]) -> Mapping[str, str]:
        if self.mode == "raise":
            raise RuntimeError("cache is down")
        return {key: 5 for key in keys} if self.mode == "junk" else ["not", "a", "mapping"]  # type: ignore[misc,return-value]

    def put_many(self, entries: Mapping[str, str]) -> None:
        self.puts += 1
        if self.mode == "raise":
            raise RuntimeError("cache is down")

    def clear(self) -> None:
        pass


class TestCacheInDecide:

    def run(self, cache: Optional[SemanticCache], provider: object, df: Optional[pd.DataFrame] = None,
            **kwargs: object) -> pd.DataFrame:
        with bind_providers({}, cache=cache):
            return decide(df if df is not None else repeated(["a", "b", "c"]), provider, **kwargs)

    def test_nothing_is_cached_unless_a_cache_is_bound(self) -> None:
        provider = versioned()
        self.run(None, provider)
        self.run(None, provider)
        assert len(provider.request_ids) == 6

    def test_a_second_run_is_answered_from_the_cache_at_no_cost(self) -> None:
        cache, provider = MemorySemanticCache(), versioned()
        cold = self.run(cache, provider)
        assert provider.request_ids == ["0", "1", "2"] and cache.stats()["puts"] == 3
        warm = self.run(cache, provider)
        assert provider.request_ids == ["0", "1", "2"]
        for column in ("p", "p__p", "p__status", "p__dist"):
            assert warm[column].tolist() == cold[column].tolist()
        assert [(m["cost_usd"], m["latency_ms"], m["retry_count"], m["cache_hit"], m["extra"]) for m in meta(warm)] == \
            [(0.0, 0.0, 0, True, {"cache_source": "local_cache"})] * 3
        assert cache.stats()["hit_rate"] == 0.5

    def test_hits_come_before_the_budget_and_use_none_of_it(self) -> None:
        cache, provider = MemorySemanticCache(), versioned()
        self.run(cache, provider)
        warm = self.run(cache, provider, budget={"max_requests": 1}, batch_size=1)
        assert set(warm["p__status"]) == {"ok"} and len(provider.request_ids) == 3
        more = self.run(cache, provider, repeated(["a", "b", "c", "d", "e"]), budget={"max_requests": 1}, batch_size=1)
        assert more["p__status"].tolist() == ["ok"] * 4 + ["skipped"] and provider.request_ids[3:] == ["3"]

    def test_only_new_rows_reach_the_provider(self) -> None:
        cache, provider = MemorySemanticCache(), versioned()
        self.run(cache, provider)
        self.run(cache, provider, repeated(["c", "z", "a"]))
        assert provider.request_ids[3:] == ["1"]

    def test_duplicates_and_the_cache_work_together(self) -> None:
        cache, provider = MemorySemanticCache(), versioned()
        self.run(cache, provider, repeated(["a", "a", "b"]))
        assert provider.request_ids == ["0", "2"] and cache.stats()["puts"] == 2
        warm = self.run(cache, provider, repeated(["b", "a", "a"]))
        assert provider.request_ids == ["0", "2"]
        assert [m["cost_usd"] for m in meta(warm)] == [0.0, 0.0, 0.0]
        assert [m["extra"].get("deduped_from") for m in meta(warm)] == [None, None, "1"]  # type: ignore[union-attr]

    def test_the_cache_holds_the_answer_before_min_p(self) -> None:
        cache, provider = MemorySemanticCache(), versioned()
        strict = self.run(cache, provider, min_p=0.95)
        assert set(strict["p__status"]) == {"uncertain"} and [m["status_reason"] for m in meta(strict)] == ["min_p"] * 3
        loose = self.run(cache, provider, min_p=0.5)
        assert set(loose["p__status"]) == {"ok"} and len(provider.request_ids) == 3
        again = self.run(cache, provider, min_p=0.95)
        assert set(again["p__status"]) == {"uncertain"} and len(provider.request_ids) == 3
        assert [m["status_reason"] for m in meta(again)] == ["min_p"] * 3

    def test_errors_skips_and_passing_faults_are_not_cached(self) -> None:
        def boom(evidence: object, question: object) -> object:
            raise RuntimeError("down")
        for fn in (boom, lambda e, q: None):
            cache, provider = MemorySemanticCache(), versioned(fn)
            self.run(cache, provider)
            self.run(cache, provider)
            assert len(provider.request_ids) == 6 and cache.stats()["puts"] == 0
        cache, provider = MemorySemanticCache(), versioned()
        out = self.run(cache, provider, repeated(["a", "b", "c"]), max_rows=1)
        assert out["p__status"].tolist() == ["ok", "skipped", "skipped"] and cache.stats()["puts"] == 1

    def test_a_tie_is_cached_as_uncertain(self) -> None:
        calls: List[str] = []

        def script(index: int, requests: Sequence[DecisionRequest]) -> List[object]:
            calls.extend(request.request_id for request in requests)
            return [result(request, "p", None, {"true": 0.5, "false": 0.5}, status="uncertain", status_reason="tie")
                    for request in requests]

        class Tied(Scripted):
            def cache_fingerprint(self) -> str:
                return "tied"
        cache = MemorySemanticCache()
        for _ in range(2):
            out = self.run(cache, Tied(script), repeated(["a"]))
        assert calls == ["0"] and out["p__status"].tolist() == ["uncertain"]
        assert meta(out)[0]["status_reason"] == "tie" and meta(out)[0]["cache_hit"] is True

    def test_another_fingerprint_or_question_or_version_is_another_entry(self) -> None:
        cache = MemorySemanticCache()
        first, second = versioned(), Recording(full_answer, fingerprint="v2", probability_kind="native_decision", **serial(native=True))
        self.run(cache, first)
        self.run(cache, second)
        assert len(second.request_ids) == 3
        self.run(cache, first, repeated(["a"]), questions=[{"id": "p", "kind": "predicate", "text": "Another?"}])
        assert len(first.request_ids) == 4
        self.run(cache, first, task_version="2")
        assert len(first.request_ids) == 7

    def test_a_row_is_served_only_when_every_question_hits(self) -> None:
        cache, provider = MemorySemanticCache(), versioned()
        self.run(cache, provider, repeated(["a"]), questions=[P])
        out = self.run(cache, provider, repeated(["a"]), questions=[P, C])
        assert provider.request_ids == ["0", "0"] and set(meta(out, "p")[0]["extra"] or {}) == set()  # type: ignore[arg-type]
        out = self.run(cache, provider, repeated(["a"]), questions=[P, C])
        assert provider.request_ids == ["0", "0"] and meta(out, "c")[0]["cache_hit"] is True

    def test_a_provider_without_a_fingerprint_is_refused_before_any_call(self) -> None:
        cache, provider = MemorySemanticCache(), Recording(**serial())
        with pytest.raises(ValueError, match="cache_fingerprint"):
            self.run(cache, provider)
        assert provider.request_ids == [] and provider_fingerprint(object()) is None

        class Raising(Recording):
            def cache_fingerprint(self) -> Optional[str]:
                raise RuntimeError("no")
        with pytest.raises(ValueError, match="cache_fingerprint"):
            self.run(cache, Raising())

    @pytest.mark.parametrize("mode", ["raise", "junk", "mapping"])
    def test_a_faulty_cache_costs_a_miss_and_never_fails_the_run(self, mode: str) -> None:
        provider = versioned()
        out = self.run(FlakyCache(mode), provider)
        assert set(out["p__status"]) == {"ok"} and provider.request_ids == ["0", "1", "2"]

    def test_a_foreign_or_broken_payload_is_a_miss(self) -> None:
        provider = versioned()
        keys = [cache_key(provider.cache_fingerprint() or "", meta_["input_hash"]) for meta_ in meta(decide(
            repeated(["a"]), versioned()))]
        for payload in ("not json", "[]", json.dumps({"value": True}),
                        json.dumps({"request_id": "x", "question_id": "p", "value": "maybe", "distribution": None,
                                    "confidence": None, "probability_kind": "none", "status": "ok", "error_code": None,
                                    "model": "m", "provider": "p", "task_version": "1", "input_hash": "h",
                                    "latency_ms": 1.0, "cost_usd": 1.0, "cache_hit": None, "retry_count": 0,
                                    "status_reason": None, "extra": None})):
            cache = MemorySemanticCache()
            cache.put_many({key: payload for key in keys})
            provider = versioned()
            out = self.run(cache, provider, repeated(["a"]))
            assert provider.request_ids == ["0"] and out["p"].tolist() == [True]

    def test_keys_and_values_do_not_hold_the_evidence_or_a_row_id(self) -> None:
        cache = MemorySemanticCache()
        df = repeated(["secret-evidence"]).assign(rid=["tenantX-user-1"])
        self.run(cache, versioned(), df, id_col="rid")
        assert len(cache) == 1
        for key, entry in cache._entries.items():
            assert len(key) == 64 and set(key) <= set("0123456789abcdef")
            assert "secret-evidence" not in entry[0] and "tenantX" not in entry[0]
        again = self.run(cache, versioned(), df.assign(rid=["other-id"]), id_col="rid")
        assert meta(again)[0]["cache_hit"] is True

    def test_binding_rules(self) -> None:
        outer, inner = MemorySemanticCache(), MemorySemanticCache()
        provider = versioned()
        with bind_providers({}, cache=outer):
            with bind_providers({}):
                assert binding.bound_cache() is outer
            with bind_providers({}, cache=inner):
                assert binding.bound_cache() is inner
            assert binding.bound_cache() is outer
        assert binding.bound_cache() is None
        decide(repeated(["a"]), provider)
        decide(repeated(["a"]), provider)
        assert len(provider.request_ids) == 2
        with pytest.raises(ValueError, match="cache"):
            with bind_providers({}, cache=object()):  # type: ignore[arg-type]
                pass

    def test_gfql_call_uses_the_bound_cache(self) -> None:
        g = graphistry.nodes(repeated(["a", "b"]).assign(id=[1, 2]), "id")
        provider, cache = versioned(), MemorySemanticCache()
        query = [call("sem_decide", {"questions": [{"id": "p", "kind": "predicate", "text": "?"}], "provider": "p",
                                     "task_id": "t", "evidence_cols": ["name"]})]
        for _ in range(2):
            with bind_providers({"p": provider}, cache=cache):
                out = g.gfql(query)
        assert provider.request_ids == ["1", "2"] and json.loads(out._nodes["p__meta"].iloc[0])["cache_hit"] is True


class TestCacheInMap:

    def provider(self) -> RecordingGenerator:
        return RecordingGenerator(lambda evidence, instruction: f"<{evidence['name']}>", fingerprint="g1", **serial())  # type: ignore[index]

    def map_(self, cache: Optional[SemanticCache], provider: RecordingGenerator, df: Optional[pd.DataFrame] = None,
             **kwargs: object) -> pd.DataFrame:
        with bind_providers({}, cache=cache):
            return sem_map(df if df is not None else repeated(["a", "b"]), instruction="i", out_col="o",
                           provider=provider, task_id="t", evidence_cols=["name"], **kwargs)  # type: ignore[arg-type]

    def asked(self, provider: RecordingGenerator) -> List[str]:
        return [request.request_id for batch in provider.calls for request in batch]

    def test_a_second_run_is_answered_from_the_cache(self) -> None:
        cache, provider = MemorySemanticCache(), self.provider()
        self.map_(cache, provider)
        warm = self.map_(cache, provider)
        assert self.asked(provider) == ["0", "1"] and warm["o"].tolist() == ["<a>", "<b>"]
        assert [(m["cost_usd"], m["latency_ms"], m["retry_count"], m["cache_hit"]) for m in meta(warm, "o")] == \
            [(0.0, 0.0, 0, True)] * 2

    def test_max_output_tokens_and_a_new_fingerprint_are_other_entries(self) -> None:
        cache, provider = MemorySemanticCache(), self.provider()
        self.map_(cache, provider, max_output_tokens=20)
        self.map_(cache, provider, max_output_tokens=20)
        self.map_(cache, provider, max_output_tokens=40)
        assert self.asked(provider) == ["0", "1", "0", "1"]

    def test_hits_use_no_budget_and_a_provider_without_a_fingerprint_is_refused(self) -> None:
        cache, provider = MemorySemanticCache(), self.provider()
        self.map_(cache, provider)
        out = self.map_(cache, provider, repeated(["a", "b", "c"]), batch_size=1, budget={"max_requests": 1})
        assert out["o__status"].tolist() == ["ok"] * 3 and self.asked(provider)[-1] == "2"
        with pytest.raises(ValueError, match="cache_fingerprint"):
            self.map_(cache, RecordingGenerator(**serial()))


class TestFingerprints:

    def http(self, cls: type, **kwargs: object) -> str:
        base: Dict[str, object] = dict(api_key="key-1", model="m")
        base.update(kwargs)
        return cls(**base).cache_fingerprint()  # type: ignore[no-any-return]

    @pytest.mark.parametrize("cls", [OpenAIChatProvider, OpenAIDecisionsProvider])
    def test_the_key_and_the_headers_never_change_the_fingerprint(self, cls: type) -> None:
        extra = {"extra_headers": {"x-team": "a"}} if cls is OpenAIChatProvider else {}
        assert self.http(cls, **extra) == self.http(cls, api_key="key-2", **extra)
        assert "key-1" not in self.http(cls) and len(self.http(cls)) == 64
        if cls is OpenAIChatProvider:
            assert self.http(cls) == self.http(cls, extra_headers={"x-team": "b"})

    def test_what_shapes_an_answer_changes_the_fingerprint(self) -> None:
        base = self.http(OpenAIChatProvider)
        for change in (dict(model="n"), dict(base_url="https://example.test/v1"), dict(strategy="text"),
                       dict(max_answer_tokens=9), dict(extra_body={"seed": 1}), dict(top_logprobs=3), dict(name="other")):
            assert self.http(OpenAIChatProvider, **change) != base, change
        packed = self.http(OpenAIDecisionsProvider)
        for change in (dict(rows_per_request=4), dict(model="n"), dict(max_choice_options=9)):
            assert self.http(OpenAIDecisionsProvider, **change) != packed, change
        assert self.http(OpenAIChatProvider) != self.http(OpenAIDecisionsProvider)

    def test_a_chat_provider_that_settles_its_strategy_by_probe_has_no_fingerprint(self) -> None:
        assert OpenAIChatProvider("k", "m", strategy="auto").cache_fingerprint() is None
        assert OpenAIChatProvider("k", "m", strategy="text").cache_fingerprint() is not None
        with bind_providers({}, cache=MemorySemanticCache()), pytest.raises(ValueError, match="explicit strategy"):
            decide(repeated(["a"]), OpenAIChatProvider("k", "m", strategy="auto"))

    def test_callable_providers_have_one_only_when_given(self) -> None:
        assert CallableDecisionProvider(lambda e, q: True).cache_fingerprint() is None
        one = CallableDecisionProvider(lambda e, q: True, fingerprint="a").cache_fingerprint()
        assert one and one != CallableDecisionProvider(lambda e, q: True, fingerprint="b").cache_fingerprint()
        assert one != CallableDecisionProvider(lambda e, q: True, fingerprint="a", probability_kind="verbalized").cache_fingerprint()
        assert cache_key("f", "h") != cache_key("f", "h", "salt") != cache_key("g", "h")


class TestNoGlobalState:

    def test_the_semantic_modules_keep_no_process_wide_cache(self) -> None:
        for module in (cache_module, binding):
            state = [name for name, value in vars(module).items()
                     if not name.startswith("__") and isinstance(value, (dict, list, set, OrderedDict))]
            assert state == [], (module.__name__, state)

    def test_clearing_the_gfql_caches_leaves_a_caller_owned_cache_alone(self) -> None:
        from graphistry.compute.gfql import cache_registry
        cache = MemorySemanticCache()
        cache.put_many({"a": "1"})
        cache_registry.clear_all()
        assert cache.get_many(["a"]) == {"a": "1"}
