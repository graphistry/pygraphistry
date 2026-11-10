import json
from typing import Dict, List, Optional

import pandas as pd
import pytest
import requests

from graphistry.compute.semantic.contract import DecisionQuestion, DecisionRequest, check_decision_results
from graphistry.compute.semantic.ops import sem_decide
from graphistry.compute.semantic.providers.openai_decisions import OpenAIDecisionsProvider


PRED = DecisionQuestion(id="p", kind="predicate", text="Is it damaged?")
CHOICE = DecisionQuestion(id="c", kind="choice", text="Which team?", options=("billing", "tech", "other"))
SCORE = DecisionQuestion(id="s", kind="score", text="How severe?", levels=("low", "mid", "high"))
USAGE = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 100},
         "output_tokens": 50, "total_tokens": 1050}


class FakeResponse:
    def __init__(self, status: int, body: object = None, headers: Optional[Dict[str, str]] = None,
                 text: Optional[str] = None) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self._text = text

    def json(self) -> object:
        if self._text is not None:
            return json.loads(self._text)
        return self._body


class FakeSession:
    """Replays queued responses (or exceptions) and records every POST."""

    def __init__(self, responses: List[object]) -> None:
        self.responses = list(responses)
        self.posts: List[Dict[str, object]] = []

    def post(self, url: str, *, json: Dict[str, object], headers: Dict[str, str], timeout: float) -> object:
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(json)
        return item


def answers_for(body: Dict[str, object]) -> FakeResponse:
    out = []
    for q in body["questions"]:  # type: ignore[union-attr]
        if q["type"] == "predicate":
            out.append({"type": "predicate", "name": q["name"], "probability": 0.7})
        elif q["type"] == "choice":
            out.append({"type": "choice", "name": q["name"], "choice": q["choices"][0]["value"],
                        "probabilities": [{"value": c["value"], "probability": p}
                                          for c, p in zip(q["choices"], (0.95, 0.03, 0.02))], "confidence": 0.93})
        else:
            out.append({"type": "score", "name": q["name"], "score": 1.1, "confidence": 0.55,
                        "probabilities": [{"value": i, "label": lv["label"], "probability": p}
                                          for i, (lv, p) in enumerate(zip(q["levels"], (0.1, 0.7, 0.2)))]})
    return FakeResponse(200, {"model": "m-1", "answers": out, "usage": USAGE})


def provider(session: FakeSession, **kwargs: object) -> OpenAIDecisionsProvider:
    params = dict(api_key="k-secret", model="m", session=session, input_usd_per_mtok=1.0,
                  cached_input_usd_per_mtok=0.1, retry_backoff_s=0.0)
    params.update(kwargs)
    return OpenAIDecisionsProvider(**params)  # type: ignore[arg-type]


def request(questions=(PRED, CHOICE, SCORE), rid: str = "r0", evidence: object = "a box arrived crushed"
            ) -> DecisionRequest:
    return DecisionRequest(request_id=rid, evidence=evidence, questions=tuple(questions),  # type: ignore[arg-type]
                           task_id="t", task_version="1")


def decide(p: OpenAIDecisionsProvider, req: DecisionRequest, deadline_s: Optional[float] = None) -> list:
    results = p.decide_batch([req], deadline_s=deadline_s, budget=None)
    assert check_decision_results([req], results, p.capabilities()) == []
    return results


class TestMapping:

    def test_kinds_map_to_results(self) -> None:
        session = FakeSession([answers_for])
        results = decide(provider(session), request())
        by_q = {r.question_id: r for r in results}
        assert by_q["p"].value is True and by_q["p"].distribution == pytest.approx({"true": 0.7, "false": 0.3})
        assert by_q["p"].confidence is None and by_q["p"].probability_kind == "native_decision"
        assert by_q["c"].value == "billing" and by_q["c"].confidence == 0.93
        assert by_q["s"].value == "mid" and by_q["s"].extra["score_index"] == 1.1  # type: ignore[index]
        assert all(r.model == "m-1" and r.retry_count == 0 for r in results)
        sent = session.posts[0]["json"]
        assert sent["model"] == "m" and sent["input"] == "a box arrived crushed"  # type: ignore[index]
        assert [q["name"] for q in sent["questions"]] == ["p", "c", "s"]  # type: ignore[index]
        assert sent["questions"][1]["choices"] == [{"value": "billing"}, {"value": "tech"}, {"value": "other"}]  # type: ignore[index]

    def test_predicate_tie_and_refusal(self) -> None:
        def body(json_body: Dict[str, object]) -> FakeResponse:
            return FakeResponse(200, {"model": "m", "answers": [
                {"type": "predicate", "name": "p", "probability": 0.5},
                {"type": "refusal", "name": "c"},
            ], "usage": USAGE})

        results = decide(provider(FakeSession([body])), request((PRED, CHOICE)))
        assert results[0].value is None and results[0].status == "uncertain" and results[0].status_reason == "tie"
        assert results[1].status == "uncertain" and results[1].status_reason == "refusal"
        assert results[1].probability_kind == "none" and results[1].distribution is None

    def test_mapping_evidence_is_compact_sorted_json(self) -> None:
        session = FakeSession([answers_for])
        decide(provider(session), request((PRED,), evidence={"b": 2, "a": "x"}))
        assert session.posts[0]["json"]["input"] == '{"a":"x","b":2}'  # type: ignore[index]

    def test_score_without_every_level_is_error(self) -> None:
        for probs in ([], [{"value": 0, "label": "low", "probability": 1.0}],
                      [{"value": 0, "label": "WRONG", "probability": 0.2}, {"value": 1, "label": "mid", "probability": 0.5},
                       {"value": 2, "label": "high", "probability": 0.3}]):
            session = FakeSession([FakeResponse(200, {"model": "m", "usage": USAGE, "answers": [
                {"type": "score", "name": "s", "score": 1.0, "probabilities": probs, "confidence": 0.5}]})])
            result = decide(provider(session), request((SCORE,)))[0]
            assert result.status == "error" and result.error_code == "invalid_provider_result"
            assert result.distribution is None

    def test_choice_with_empty_probabilities_keeps_value(self) -> None:
        session = FakeSession([FakeResponse(200, {"model": "m", "usage": USAGE, "answers": [
            {"type": "choice", "name": "c", "choice": "tech", "probabilities": [], "confidence": 0.4}]})])
        result = decide(provider(session), request((CHOICE,)))[0]
        assert result.value == "tech" and result.distribution is None and result.probability_kind == "none"
        assert result.confidence == 0.4

    @pytest.mark.parametrize("answers", [
        [{"type": "predicate", "name": "p", "probability": 0.7}],
        [{"type": "choice", "name": "c", "choice": "billing", "probabilities": [], "confidence": 0.9},
         {"type": "predicate", "name": "p", "probability": 0.7}],
    ])
    def test_misaligned_answers_error_the_request(self, answers: list) -> None:
        session = FakeSession([FakeResponse(200, {"model": "m", "usage": USAGE, "answers": answers})])
        results = decide(provider(session), request((PRED, CHOICE)))
        assert {r.error_code for r in results} == {"invalid_provider_result"}

    def test_swapped_same_kind_answers_error_the_request(self) -> None:
        q2 = DecisionQuestion(id="p2", kind="predicate", text="Second?")
        session = FakeSession([FakeResponse(200, {"model": "m", "usage": USAGE, "answers": [
            {"type": "predicate", "name": "p2", "probability": 0.9},
            {"type": "predicate", "name": "p", "probability": 0.1}]})])
        results = decide(provider(session), request((PRED, q2)))
        assert {r.error_code for r in results} == {"invalid_provider_result"}

    def test_score_tie_is_uncertain(self) -> None:
        session = FakeSession([FakeResponse(200, {"model": "m", "usage": USAGE, "answers": [
            {"type": "score", "name": "s", "score": 1.0, "confidence": 0.4, "probabilities": [
                {"value": 0, "label": "low", "probability": 0.4}, {"value": 1, "label": "mid", "probability": 0.2},
                {"value": 2, "label": "high", "probability": 0.4}]}]})])
        result = decide(provider(session), request((SCORE,)))[0]
        assert result.value is None and result.status == "uncertain" and result.status_reason == "tie"

    def test_predicate_ignores_confidence_and_cache_and_latency_rules(self) -> None:
        no_cache = dict(USAGE, input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0})
        session = FakeSession([FakeResponse(200, {"model": "m", "usage": no_cache, "answers": [
            {"type": "predicate", "name": "p", "probability": 0.8, "confidence": 0.99},
            {"type": "choice", "name": "c", "choice": "tech", "probabilities": [], "confidence": 0.5}]})])
        results = decide(provider(session), request((PRED, CHOICE)))
        assert results[0].confidence is None and results[0].cache_hit is False
        assert results[0].latency_ms == results[1].latency_ms

    def test_out_of_set_choice_is_error(self) -> None:
        session = FakeSession([FakeResponse(200, {"model": "m", "usage": USAGE, "answers": [
            {"type": "choice", "name": "c", "choice": "nope", "probabilities": [], "confidence": 0.9}]})])
        assert decide(provider(session), request((CHOICE,)))[0].error_code == "invalid_provider_result"


class TestSplitting:

    def test_401_questions_split_into_two_requests(self) -> None:
        questions = tuple(DecisionQuestion(id=f"q{i}", kind="predicate", text="t") for i in range(401))
        session = FakeSession([answers_for])
        results = decide(provider(session), request(questions))
        assert len(session.posts) == 2
        assert [len(p["json"]["questions"]) for p in session.posts] == [400, 1]  # type: ignore[index,arg-type]
        assert len(results) == 401


class TestRetries:

    def test_429_then_200(self) -> None:
        session = FakeSession([FakeResponse(429), answers_for])
        results = decide(provider(session), request((PRED,)))
        assert results[0].status == "ok" and results[0].retry_count == 1

    def test_503_every_attempt(self) -> None:
        session = FakeSession([FakeResponse(503)])
        results = decide(provider(session, max_retries=2), request((PRED,)))
        assert results[0].error_code == "http_503" and results[0].retry_count == 2
        assert len(session.posts) == 3

    @pytest.mark.parametrize("status", [408, 409, 500])
    def test_transient_statuses_are_retried(self, status: int) -> None:
        session = FakeSession([FakeResponse(status), answers_for])
        results = decide(provider(session), request((PRED,)))
        assert results[0].status == "ok" and results[0].retry_count == 1

    def test_hook_raised_http_errors_follow_status_rules(self) -> None:
        retry_then_ok = requests.HTTPError("503", response=FakeResponse(503))  # type: ignore[arg-type]
        session = FakeSession([retry_then_ok, answers_for])
        assert decide(provider(session), request((PRED,)))[0].retry_count == 1
        terminal = requests.HTTPError("404", response=FakeResponse(404))  # type: ignore[arg-type]
        session = FakeSession([terminal])
        result = decide(provider(session), request((PRED,)))[0]
        assert result.error_code == "http_404" and len(session.posts) == 1
        session = FakeSession([requests.exceptions.RetryError("exhausted")])
        assert decide(provider(session, max_retries=1), request((PRED,)))[0].error_code == "transport_error"

    def test_missing_usage_leaves_cache_unknown(self) -> None:
        session = FakeSession([FakeResponse(200, {"model": "m", "answers": [
            {"type": "predicate", "name": "p", "probability": 0.8}]})])
        result = decide(provider(session), request((PRED,)))[0]
        assert result.cache_hit is None and result.cost_usd is None

    def test_estimate_rounds_each_text_up(self) -> None:
        p = provider(FakeSession([answers_for]))
        req = request((DecisionQuestion(id="q", kind="predicate", text="abcde"),), evidence="x")
        assert p.estimate_cost_usd([req]) == pytest.approx((1 + 2) * 1.0 / 1e6)

    def test_400_is_terminal(self) -> None:
        session = FakeSession([FakeResponse(400)])
        results = decide(provider(session), request((PRED,)))
        assert results[0].error_code == "http_400" and len(session.posts) == 1

    def test_connection_error_then_transport_error(self) -> None:
        session = FakeSession([requests.ConnectionError("down")])
        results = decide(provider(session, max_retries=1), request((PRED,)))
        assert results[0].error_code == "transport_error" and results[0].retry_count == 1

    @pytest.mark.parametrize("error", [requests.exceptions.ChunkedEncodingError("x"), requests.Timeout("x"),
                                       requests.exceptions.ContentDecodingError("x")])
    def test_other_transport_errors(self, error: Exception) -> None:
        session = FakeSession([error])
        results = decide(provider(session, max_retries=1), request((PRED,)))
        assert results[0].error_code == "transport_error" and results[0].retry_count == 1

    @pytest.mark.parametrize("error", [requests.exceptions.MissingSchema("x"), requests.exceptions.InvalidURL("x"),
                                       requests.exceptions.TooManyRedirects("x")])
    def test_unsendable_requests_are_not_retried(self, error: Exception) -> None:
        session = FakeSession([error])
        results = decide(provider(session, max_retries=3), request((PRED,)))
        assert results[0].error_code == "request_error" and results[0].retry_count == 0
        assert len(session.posts) == 1

    def test_huge_number_is_invalid_not_a_crash(self) -> None:
        session = FakeSession([FakeResponse(200, {"model": "m", "usage": USAGE, "answers": [
            {"type": "predicate", "name": "p", "probability": 10 ** 400}]})])
        assert decide(provider(session), request((PRED,)))[0].error_code == "invalid_provider_result"

    def test_retry_after_is_capped_without_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept: List[float] = []
        monkeypatch.setattr("graphistry.compute.semantic.providers.openai_decisions.time.sleep", slept.append)
        session = FakeSession([FakeResponse(429, headers={"Retry-After": "86400"}), answers_for])
        decide(provider(session), request((PRED,)))
        assert slept == [60.0]

    def test_malformed_json(self) -> None:
        session = FakeSession([FakeResponse(200, text="{not json")])
        assert decide(provider(session), request((PRED,)))[0].error_code == "invalid_provider_result"

    def test_retry_after_beyond_deadline_stops(self) -> None:
        session = FakeSession([FakeResponse(429, headers={"Retry-After": "30"})])
        results = decide(provider(session), request((PRED,)), deadline_s=1.0)
        assert results[0].error_code == "http_429" and len(session.posts) == 1

    def test_timeout_capped_by_deadline(self) -> None:
        session = FakeSession([answers_for])
        decide(provider(session, timeout_s=60), request((PRED,)), deadline_s=2.0)
        assert session.posts[0]["timeout"] <= 2.0


class TestCost:

    def test_cost_from_usage(self) -> None:
        session = FakeSession([answers_for])
        results = decide(provider(session), request((PRED, CHOICE)))
        expected = ((1000 - 200 - 100) * 1.0 + 200 * 0.1 + 100 * 1.0) / 1e6
        assert sum(r.cost_usd for r in results) == pytest.approx(expected)  # type: ignore[misc]
        assert all(r.cache_hit is True and r.extra["cached_input_tokens"] == 200 for r in results)  # type: ignore[index]

    def test_output_and_cache_write_rates(self) -> None:
        session = FakeSession([answers_for])
        p = provider(session, output_usd_per_mtok=2.0, cache_write_usd_per_mtok=5.0)
        cost = decide(p, request((PRED,)))[0].cost_usd
        assert cost == pytest.approx(((700 * 1.0) + 200 * 0.1 + 100 * 5.0 + 50 * 2.0) / 1e6)

    def test_large_counts_priced_linearly(self) -> None:
        big = dict(USAGE, input_tokens=10_000_000, input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0})
        session = FakeSession([FakeResponse(200, {"model": "m", "usage": big, "answers": [
            {"type": "predicate", "name": "p", "probability": 0.9}]})])
        assert decide(provider(session), request((PRED,)))[0].cost_usd == pytest.approx(10.0)

    def test_missing_rates_mean_no_cost(self) -> None:
        session = FakeSession([answers_for])
        p = provider(session, cached_input_usd_per_mtok=None)
        assert decide(p, request((PRED,)))[0].cost_usd is None
        assert p.estimate_cost_usd([request((PRED,))]) is None

    def test_estimate_scale(self) -> None:
        p = provider(FakeSession([answers_for]))
        req = request((DecisionQuestion(id="q", kind="predicate", text="abcd"),), evidence="x" * (4 * 10**6 - 4))
        assert p.estimate_cost_usd([req]) == pytest.approx(1.0)

    def test_price_per_question_wins_for_estimate(self) -> None:
        for rates in (dict(), dict(input_usd_per_mtok=None, cached_input_usd_per_mtok=None)):
            p = provider(FakeSession([answers_for]), price_per_question_usd=0.002, **rates)
            assert p.estimate_cost_usd([request(), request(rid="r1")]) == pytest.approx(0.012)
        p = provider(FakeSession([answers_for]), price_per_question_usd=0.002, input_usd_per_mtok=None)
        assert decide(p, request((PRED,)))[0].cost_usd is None


class TestCapabilitiesAndKeys:

    def test_declared_capabilities(self) -> None:
        caps = provider(FakeSession([answers_for])).capabilities()
        assert (caps.max_rows_per_request, caps.max_questions_per_request, caps.max_choice_options, caps.max_levels,
                caps.supports_native_probabilities, caps.supports_shared_prefix_cache) == (1, 400, 255, 10, True, True)

    def test_one_row_per_call_through_the_kernel(self) -> None:
        session = FakeSession([answers_for])
        df = pd.DataFrame({"text": list("abcde")})
        out = sem_decide(df, questions=[{"id": "p", "kind": "predicate", "text": "t"}], provider=provider(session),
                         evidence_cols=["text"], task_id="t", budget={"max_requests": 2}, max_concurrency=1)
        assert out["p__status"].tolist() == ["ok", "ok", "skipped", "skipped", "skipped"]

    def test_no_environment_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        with pytest.raises(ValueError):
            OpenAIDecisionsProvider(api_key="", model="m")


class TestPrivacy:

    def test_body_has_no_key_or_cache_key(self) -> None:
        session = FakeSession([answers_for])
        decide(provider(session), request())
        body = json.dumps(session.posts[0]["json"])
        assert "k-secret" not in body
        assert "prompt_cache_key" not in body and "safety_identifier" not in body
        assert session.posts[0]["headers"]["Authorization"] == "Bearer k-secret"  # type: ignore[index]
        assert "k-secret" not in repr(provider(session))


class TestThroughKernel:

    def test_one_row_per_request(self) -> None:
        session = FakeSession([answers_for])
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        out = sem_decide(df, questions=[{"id": "p", "kind": "predicate", "text": "t"}], provider=provider(session),
                         evidence_cols=["text"], task_id="t")
        assert len(session.posts) == 3
        assert out["p__p"].tolist() == pytest.approx([0.7, 0.7, 0.7])
        assert set(out["p__status"]) == {"ok"}
