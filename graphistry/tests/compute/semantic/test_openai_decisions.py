import json
from typing import Dict, List, Optional

import pandas as pd
import pytest

from graphistry.compute.semantic.contract import DecisionQuestion, DecisionRequest, DecisionResult, check_decision_result
from graphistry.compute.semantic.ops import sem_decide
from graphistry.compute.semantic.providers.http import TokenPrices
from graphistry.compute.semantic.providers.openai_decisions import PACKED_INPUT, OpenAIDecisionsProvider
from graphistry.tests.compute.semantic.http_test_utils import CHOICE, PRED, SCORE, FakeResponse, FakeSession, request


USAGE = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 100},
         "output_tokens": 50}


def answers_for(body: Dict[str, object]) -> FakeResponse:
    out = []
    for q in body["questions"]:  # type: ignore[union-attr]
        if q["type"] == "predicate":
            out.append({"type": "predicate", "name": q["name"], "probability": 0.7})
        elif q["type"] == "choice":
            out.append({"type": "choice", "name": q["name"], "choice": q["choices"][0]["value"], "confidence": 0.93,
                        "probabilities": [{"value": c["value"], "probability": p}
                                          for c, p in zip(q["choices"], (0.95, 0.03, 0.02))]})
        else:
            out.append({"type": "score", "name": q["name"], "score": 1.1, "confidence": 0.55,
                        "probabilities": [{"value": i, "label": lv["label"], "probability": p}
                                          for i, (lv, p) in enumerate(zip(q["levels"], (0.1, 0.7, 0.2)))]})
    return FakeResponse(200, {"model": "m-1", "answers": out, "usage": USAGE})


def reply(*answers: Dict[str, object], usage: Optional[Dict[str, object]] = None) -> FakeResponse:
    return FakeResponse(200, {"model": "m", "usage": USAGE if usage is None else usage, "answers": list(answers)})


def provider(session: FakeSession, **options: object) -> OpenAIDecisionsProvider:
    params: Dict[str, object] = dict(session=session, prices=TokenPrices(input=1.0, cached_input=0.1),
                                     retry_backoff_s=0.0)
    params.update(options)
    return OpenAIDecisionsProvider("k-secret", "m", **params)  # type: ignore[arg-type]


def decide(p: OpenAIDecisionsProvider, req: DecisionRequest) -> List[DecisionResult]:
    results = p.decide_batch([req], deadline_s=None, budget=None)
    by_id = {question.id: question for question in req.questions}
    assert [check_decision_result(req, by_id[r.question_id], r, p.capabilities()) for r in results] == [[]] * len(results)
    return results


class TestMapping:

    def test_kinds_map_to_results_and_the_body_is_built_once_per_row(self) -> None:
        session = FakeSession([answers_for])
        p_result, c_result, s_result = decide(provider(session), request())
        assert p_result.value is True and p_result.distribution == pytest.approx({"true": 0.7, "false": 0.3})
        assert p_result.confidence is None and p_result.probability_kind == "native_decision"
        assert (c_result.value, c_result.confidence, s_result.value) == ("billing", 0.93, "mid")
        assert s_result.extra == {"cached_input_tokens": 200, "score_index": 1.1}
        assert all(r.model == "m-1" and r.retry_count == 0 and r.cache_hit for r in (p_result, c_result, s_result))
        assert p_result.latency_ms == c_result.latency_ms
        sent = session.posts[0]["json"]
        assert sent["model"] == "m" and sent["input"] == "a box arrived crushed"  # type: ignore[index]
        assert [q["name"] for q in sent["questions"]] == ["p", "c", "s"]  # type: ignore[index]
        assert sent["questions"][1]["choices"] == [{"value": "billing"}, {"value": "tech"}, {"value": "other"}]  # type: ignore[index]
        assert session.posts[0]["url"] == "https://api.openai.com/v1/decisions"

    @pytest.mark.parametrize("answer,question,expected", [
        ({"type": "predicate", "name": "p", "probability": 0.5}, PRED, (None, "uncertain", "tie")),
        ({"type": "refusal", "name": "c"}, CHOICE, (None, "uncertain", "refusal")),
        ({"type": "refusal", "name": "s", "score": 1.0}, SCORE, (None, "uncertain", "refusal")),
        ({"type": "score", "name": "s", "score": 1.0, "probabilities": [
            {"value": 0, "label": "low", "probability": 0.4}, {"value": 1, "label": "mid", "probability": 0.2},
            {"value": 2, "label": "high", "probability": 0.4}]}, SCORE, (None, "uncertain", "tie")),
        ({"type": "choice", "name": "c", "choice": "tech", "probabilities": [], "confidence": 0.4}, CHOICE,
         ("tech", "ok", None)),
        ({"type": "predicate", "name": "p", "probability": 0.8, "confidence": 0.99}, PRED, (True, "ok", None)),
    ])
    def test_abstentions_and_partial_answers(self, answer: dict, question: DecisionQuestion, expected: tuple) -> None:
        result = decide(provider(FakeSession([reply(answer)])), request((question,)))[0]
        assert (result.value, result.status, result.status_reason) == expected
        if result.distribution is None:
            assert result.probability_kind == "none"
        if question.kind == "predicate":
            assert result.confidence is None
        if result.status_reason == "refusal":
            assert "score_index" not in (result.extra or {})

    @pytest.mark.parametrize("answer,question", [
        ({"type": "choice", "name": "c", "choice": "nope", "probabilities": []}, CHOICE),
        ({"type": "choice", "name": "c", "choice": "tech", "probabilities": [{"value": "tech", "probability": 2}]}, CHOICE),
        ({"type": "choice", "name": "c", "choice": "tech",
          "probabilities": [{"value": "tech", "probability": 0.5}, {"value": "tech", "probability": 0.5}]}, CHOICE),
        ({"type": "predicate", "name": "p", "probability": 10 ** 400}, PRED),
        ({"type": "score", "name": "s", "score": 1.0, "probabilities": []}, SCORE),
        ({"type": "score", "name": "s", "probabilities": [{"value": 0, "label": "low", "probability": 1.0}]}, SCORE),
        ({"type": "score", "name": "s", "probabilities": [
            {"value": 0, "label": "WRONG", "probability": 0.2}, {"value": 1, "label": "mid", "probability": 0.5},
            {"value": 2, "label": "high", "probability": 0.3}]}, SCORE),
        ({"type": "choice", "name": "p", "choice": "tech"}, PRED),
        ({"type": "choice", "name": "p", "probability": 0.8}, PRED),
    ])
    def test_malformed_answers_are_errors_never_defaults(self, answer: dict, question: DecisionQuestion) -> None:
        result = decide(provider(FakeSession([reply(answer)])), request((question,)))[0]
        assert (result.status, result.error_code, result.distribution) == ("error", "invalid_provider_result", None)

    @pytest.mark.parametrize("answers,second", [
        ([{"type": "predicate", "name": "p", "probability": 0.7}], CHOICE),
        ([{"type": "choice", "name": "c", "choice": "billing"}, {"type": "predicate", "name": "p", "probability": 0.7}],
         CHOICE),
        ([{"type": "predicate", "name": "p2", "probability": 0.9}, {"type": "predicate", "name": "p", "probability": 0.1}],
         DecisionQuestion(id="p2", kind="predicate", text="Second?")),
    ])
    def test_misaligned_answers_error_the_whole_request(self, answers: list, second: DecisionQuestion) -> None:
        results = decide(provider(FakeSession([reply(*answers)])), request((PRED, second)))
        assert {r.error_code for r in results} == {"invalid_provider_result"}

    def test_an_unread_reply_is_sent_once_more_and_a_partly_read_one_is_not(self) -> None:
        good = {"type": "predicate", "name": "p", "probability": 0.7}
        session = FakeSession([reply({"type": "choice", "name": "p", "choice": "tech"}), reply(good)])
        result = decide(provider(session), request((PRED,)))[0]
        assert (result.value, result.retry_count, len(session.posts)) == (True, 1, 2)
        assert result.cost_usd == pytest.approx(2 * decide(provider(FakeSession([reply(good)])), request((PRED,)))[0].cost_usd)
        once = FakeSession([reply({"type": "choice", "name": "p", "choice": "tech"}), reply(good)])
        assert decide(provider(once, reply_retries=0), request((PRED,)))[0].error_code == "invalid_provider_result"
        partly = FakeSession([reply(good, {"type": "choice", "name": "c", "choice": "nowhere"})])
        results = decide(provider(partly), request((PRED, CHOICE)))
        assert [r.error_code for r in results] == [None, "invalid_provider_result"] and len(once.posts) == len(partly.posts) == 1

    def test_401_questions_split_into_two_requests(self) -> None:
        questions = tuple(DecisionQuestion(id=f"q{i}", kind="predicate", text="t") for i in range(401))
        session = FakeSession([answers_for])
        assert len(decide(provider(session), request(questions))) == 401
        assert [len(p["json"]["questions"]) for p in session.posts] == [400, 1]  # type: ignore[index,arg-type]

    def test_a_retried_success_reports_its_retries(self) -> None:
        result = decide(provider(FakeSession([FakeResponse(429), answers_for])), request((PRED,)))[0]
        assert (result.status, result.retry_count) == ("ok", 1)

    def test_evidence_mapping_is_sent_as_compact_sorted_json_and_the_deadline_caps_the_timeout(self) -> None:
        session = FakeSession([answers_for])
        req = request((PRED,), evidence={"b": 2, "a": "x"})
        provider(session).decide_batch([req], deadline_s=2.0, budget=None)
        assert session.posts[0]["json"]["input"] == '{"a":"x","b":2}' and session.posts[0]["timeout"] <= 2.0  # type: ignore[index,operator]

    @pytest.mark.parametrize("options", [dict(timeout_s=0), dict(timeout_s=-1), dict(timeout_s=float("inf")),
                                         dict(max_retries=None), dict(max_retries=1.5), dict(max_retries=True),
                                         dict(retry_backoff_s=None), dict(retry_backoff_s=-1),
                                         dict(prices={"input": 1.0}), dict(session=object()), dict(api_key=5),
                                         dict(model=""), dict(model=b"m")])
    def test_constructor_rejects_unusable_settings(self, options: dict) -> None:
        params: Dict[str, object] = dict(api_key="k", model="m")
        params.update(options)
        with pytest.raises(ValueError, match=next(iter(options))):
            OpenAIDecisionsProvider(**params)  # type: ignore[arg-type]

    def test_a_provider_without_prices_answers_with_unknown_cost(self) -> None:
        result = decide(OpenAIDecisionsProvider("k", "m", session=FakeSession([answers_for])), request((PRED,)))[0]
        assert (result.status, result.cost_usd) == ("ok", None)

    def test_each_row_is_its_own_call_so_request_limits_count_http_requests(self) -> None:
        session = FakeSession([answers_for])
        out = sem_decide(pd.DataFrame({"t": ["a", "b", "c"]}), questions=[PRED], provider=provider(session),
                         evidence_cols=["t"], task_id="t", batch_size=3, budget={"max_requests": 2})
        assert out["p__status"].tolist() == ["ok", "ok", "skipped"] and len(session.posts) == 2

    def test_limits_for_servers_with_the_same_api(self) -> None:
        p = OpenAIDecisionsProvider("k", "m", base_url="http://engine.test/v1", max_questions_per_request=2,
                                    max_choice_options=26, max_levels=10, session=FakeSession([answers_for]))
        caps = p.capabilities()
        assert (caps.max_questions_per_request, caps.max_choice_options, caps.max_levels) == (2, 26, 10)
        assert [len(group) for group in p.question_groups(request((PRED, CHOICE, SCORE)))] == [2, 1]

    @pytest.mark.parametrize("options", [dict(max_questions_per_request=0), dict(max_questions_per_request=True),
                                         dict(max_choice_options=1), dict(max_choice_options=2.5), dict(max_levels=1),
                                         dict(max_levels=True), dict(max_levels=2.5)])
    def test_limit_settings_are_checked(self, options: dict) -> None:
        with pytest.raises(ValueError, match=next(iter(options))):
            OpenAIDecisionsProvider("k", "m", **options)  # type: ignore[arg-type]

    def test_a_bad_key_never_appears_in_the_error(self) -> None:
        with pytest.raises(ValueError) as info:
            OpenAIDecisionsProvider(b"sk-secret", "m")  # type: ignore[arg-type]
        assert "secret" not in str(info.value)

    def test_the_http_base_is_abstract(self) -> None:
        from graphistry.compute.semantic.providers.http import HttpDecisionProvider
        with pytest.raises(TypeError):
            HttpDecisionProvider("k", "m")  # type: ignore[abstract]

    def test_transport_failures_become_error_results(self) -> None:
        session = FakeSession([FakeResponse(429), FakeResponse(503)])
        results = decide(provider(session, max_retries=1), request((PRED, CHOICE)))
        assert {(r.error_code, r.retry_count) for r in results} == {("http_503", 1)}


class TestCostAndPrivacy:

    def test_cost_is_split_over_the_request_questions(self) -> None:
        results = decide(provider(FakeSession([answers_for])), request((PRED, CHOICE)))
        assert sum(r.cost_usd or 0 for r in results) == pytest.approx(((700 * 1.0) + 200 * 0.1 + 100 * 1.0) / 1e6)

    def test_missing_usage_or_rates_mean_unknown_cost_and_cache(self) -> None:
        result = decide(provider(FakeSession([FakeResponse(200, {"model": "m", "answers": [
            {"type": "predicate", "name": "p", "probability": 0.8}]})])), request((PRED,)))[0]
        assert (result.cache_hit, result.cost_usd, result.extra) == (None, None, {})
        unpriced = provider(FakeSession([answers_for]), prices=TokenPrices(input=1.0))
        assert decide(unpriced, request((PRED,)))[0].cost_usd is None and unpriced.estimate_cost_usd([request()]) is None
        flat = provider(FakeSession([answers_for]), prices=TokenPrices(input=1.0, cached_input=0.1, per_question=0.002))
        assert flat.estimate_cost_usd([request(), request(request_id="r1")]) == pytest.approx(0.012)
        assert decide(flat, request((PRED,)))[0].cost_usd == pytest.approx((700 + 20 + 100) / 1e6)
        only_flat = provider(FakeSession([answers_for]), prices=TokenPrices(per_question=0.002))
        assert decide(only_flat, request((PRED,)))[0].cost_usd is None

    def test_estimate_prices_evidence_and_question_texts_per_request(self) -> None:
        p = provider(FakeSession([answers_for]))
        req = request((DecisionQuestion(id="q", kind="predicate", text="abcde"),), evidence="x")
        assert p.estimate_cost_usd([req]) == pytest.approx((1 + 2) * 1.0 / 1e6)
        many = request(tuple(DecisionQuestion(id=f"q{i}", kind="predicate", text="abcd") for i in range(401)),
                       evidence="y" * 400)
        assert p.estimate_cost_usd([many]) == pytest.approx((2 * 100 + 401) * 1.0 / 1e6)

    @pytest.mark.parametrize("details,cache_hit,cost", [
        ({"cached_tokens": 200, "cache_write_tokens": 100}, True, (700 + 20 + 100) / 1e6),
        ({"cached_tokens": 0, "cache_write_tokens": 0}, False, 1000 / 1e6),
        ({"cached_tokens": 5000, "cache_write_tokens": 0}, True, 5000 * 0.1 / 1e6),
    ])
    def test_cache_hit_and_cached_tokens_beyond_the_input_count(self, details: dict, cache_hit: bool, cost: float) -> None:
        usage = dict(USAGE, input_tokens_details=details)
        p = provider(FakeSession([FakeResponse(200, {"model": "m", "usage": usage, "answers": [
            {"type": "predicate", "name": "p", "probability": 0.8}]})]))
        result = decide(p, request((PRED,)))[0]
        assert (result.cache_hit, result.cost_usd) == (cache_hit, pytest.approx(cost))

    def test_key_only_in_the_header(self) -> None:
        session = FakeSession([answers_for])
        p = provider(session)
        decide(p, request())
        assert "k-secret" not in json.dumps(session.posts[0]["json"]) and "k-secret" not in repr(p)
        assert session.posts[0]["headers"]["Authorization"] == "Bearer k-secret"  # type: ignore[index]
        with pytest.raises(ValueError):
            OpenAIDecisionsProvider("", "m")

    def test_capabilities_and_one_row_per_request_through_the_kernel(self) -> None:
        session = FakeSession([answers_for])
        p = provider(session)
        caps = p.capabilities()
        assert (caps.max_rows_per_request, caps.max_questions_per_request, caps.max_choice_options, caps.max_levels,
                caps.supports_native_probabilities) == (1, 400, 255, 10, True)
        out = sem_decide(pd.DataFrame({"text": list("abcde")}), questions=[{"id": "p", "kind": "predicate", "text": "t"}],
                         provider=p, evidence_cols=["text"], task_id="t", budget={"max_requests": 3}, max_concurrency=1)
        assert out["p__status"].tolist() == ["ok"] * 3 + ["skipped"] * 2 and len(session.posts) == 3
        assert out["p__p"].iloc[:3].tolist() == pytest.approx([0.7] * 3)


class TestRowPacking:

    def rows(self, n: int, questions=(PRED, CHOICE)) -> List[DecisionRequest]:
        return [request(questions, request_id=f"r{i}", evidence=f"text {i}") for i in range(n)]

    def test_rows_share_a_request_and_each_question_carries_its_own_record(self) -> None:
        session = FakeSession([answers_for])
        p = provider(session, rows_per_request=4)
        reqs = self.rows(2, (PRED, CHOICE)) + [request((PRED,), request_id="m", evidence={"b": 2, "a": "x"})]
        results = p.decide_batch(reqs, deadline_s=None, budget=None)
        assert len(session.posts) == 1
        sent = session.posts[0]["json"]
        assert sent["input"] == PACKED_INPUT  # type: ignore[index]
        assert [(q["name"], q["instructions"]) for q in sent["questions"]] == [  # type: ignore[index]
            ("r0.p", "Is it damaged?\n\nRecord:\ntext 0"), ("r0.c", "Which team?\n\nRecord:\ntext 0"),
            ("r1.p", "Is it damaged?\n\nRecord:\ntext 1"), ("r1.c", "Which team?\n\nRecord:\ntext 1"),
            ("r2.p", 'Is it damaged?\n\nRecord:\n{"a":"x","b":2}')]
        assert sent["questions"][1]["choices"] == [{"value": "billing"}, {"value": "tech"}, {"value": "other"}]  # type: ignore[index]
        assert [(r.request_id, r.question_id, r.value) for r in results] == [
            ("r0", "p", True), ("r0", "c", "billing"), ("r1", "p", True), ("r1", "c", "billing"), ("m", "p", True)]
        by_request = {r.request_id: r for r in reqs}
        caps = p.capabilities()
        assert all(check_decision_result(by_request[r.request_id], q, r, caps) == []
                   for r in results for q in by_request[r.request_id].questions if q.id == r.question_id)
        assert all(r.extra is not None and r.extra["packed_rows"] == 3 for r in results)
        assert results[0].input_hash == decide(provider(FakeSession([answers_for])), reqs[0])[0].input_hash
        total = (700 * 1.0 + 200 * 0.1 + 100 * 1.0) / 1e6
        assert [r.cost_usd for r in results] == pytest.approx([total / 5] * 5)

    def test_rows_and_questions_split_at_both_limits(self) -> None:
        session = FakeSession([answers_for])
        p = provider(session, rows_per_request=2, max_questions_per_request=3)
        results = p.decide_batch(self.rows(5, (PRED, CHOICE)), deadline_s=None, budget=None)
        assert [[q["name"] for q in post["json"]["questions"]] for post in session.posts] == [  # type: ignore[index]
            ["r0.p", "r0.c", "r1.p"], ["r0.c"], ["r0.p", "r0.c", "r1.p"], ["r0.c"], ["r0.p", "r0.c"]]
        assert [(r.request_id, r.question_id) for r in results] == [
            (f"r{i}", q) for i in range(5) for q in ("p", "c")] and {r.status for r in results} == {"ok"}
        assert [r.extra["packed_rows"] for r in results] == [2, 2, 2, 1, 2, 2, 2, 1, 1, 1]  # type: ignore[index]

    @pytest.mark.parametrize("answers", [
        [{"type": "predicate", "name": "r0.p", "probability": 0.7}],
        [{"type": "predicate", "name": "r1.p", "probability": 0.7}, {"type": "predicate", "name": "r0.p", "probability": 0.7}],
        [{"type": "predicate", "name": "p", "probability": 0.7}, {"type": "predicate", "name": "p", "probability": 0.7}],
    ])
    def test_misaligned_answers_error_the_whole_pack(self, answers: list) -> None:
        results = provider(FakeSession([reply(*answers)]), rows_per_request=2).decide_batch(
            self.rows(2, (PRED,)), deadline_s=None, budget=None)
        assert [(r.request_id, r.error_code) for r in results] == [("r0", "invalid_provider_result"),
                                                                  ("r1", "invalid_provider_result")]

    def test_a_malformed_or_refused_answer_affects_only_its_own_row(self) -> None:
        session = FakeSession([reply({"type": "predicate", "name": "r0.p", "probability": 7},
                                     {"type": "refusal", "name": "r1.p"},
                                     {"type": "predicate", "name": "r2.p", "probability": 0.2})])
        results = provider(session, rows_per_request=3).decide_batch(self.rows(3, (PRED,)), deadline_s=None, budget=None)
        assert [(r.status, r.error_code, r.status_reason, r.value) for r in results] == [
            ("error", "invalid_provider_result", None, None), ("uncertain", None, "refusal", None),
            ("ok", None, None, False)]
        assert results[1].extra == {"cached_input_tokens": 200, "packed_rows": 3}

    def test_a_transport_failure_errors_every_row_of_the_pack(self) -> None:
        session = FakeSession([FakeResponse(503)])
        results = provider(session, rows_per_request=2, max_retries=0).decide_batch(
            self.rows(2), deadline_s=2.0, budget=None)
        assert {(r.error_code, r.retry_count) for r in results} == {("http_503", 0)} and len(results) == 4
        assert session.posts[0]["timeout"] <= 2.0  # type: ignore[operator]

    def test_the_estimate_prices_the_packed_texts(self) -> None:
        p = provider(FakeSession([answers_for]), rows_per_request=2)
        reqs = [request((DecisionQuestion(id="q", kind="predicate", text="abcd"),), request_id=f"r{i}", evidence="x")
                for i in range(3)]
        tokens = -(-len(PACKED_INPUT) // 4)
        each = -(-len("abcd\n\nRecord:\nx") // 4)
        assert p.estimate_cost_usd(reqs) == pytest.approx((2 * tokens + 3 * each) * 1.0 / 1e6)

    @pytest.mark.parametrize("value", [0, True, 2.0, "2"])
    def test_rows_per_request_is_checked(self, value: object) -> None:
        with pytest.raises(ValueError, match="rows_per_request"):
            OpenAIDecisionsProvider("k", "m", rows_per_request=value)  # type: ignore[arg-type]

    def test_the_kernel_batches_rows_to_the_opt_in_and_request_limits_count_packs(self) -> None:
        session = FakeSession([answers_for])
        p = provider(session, rows_per_request=4)
        assert p.capabilities().max_rows_per_request == 4
        frame = pd.DataFrame({"text": list("abcdefghij")})
        out = sem_decide(frame, questions=[PRED], provider=p, evidence_cols=["text"], task_id="t", max_concurrency=1)
        assert [len(post["json"]["questions"]) for post in session.posts] == [4, 4, 2]  # type: ignore[arg-type]
        assert out["p__status"].tolist() == ["ok"] * 10 and out["p__p"].tolist() == pytest.approx([0.7] * 10)
        assert json.loads(out["p__meta"].iloc[0])["extra"]["packed_rows"] == 4
        session.posts.clear()
        out = sem_decide(frame, questions=[PRED], provider=p, evidence_cols=["text"], task_id="t", batch_size=3,
                         budget={"max_requests": 2}, max_concurrency=1)
        assert out["p__status"].tolist() == ["ok"] * 6 + ["skipped"] * 4 and len(session.posts) == 2

    @pytest.mark.parametrize("rows_per_request,max_questions", [(4, 3), (2, 3), (4, 5), (3, 2)])
    def test_one_planned_call_is_one_http_request_so_the_request_budget_holds(self, rows_per_request: int,
                                                                            max_questions: int) -> None:
        session = FakeSession([answers_for])
        p = provider(session, rows_per_request=rows_per_request, max_questions_per_request=max_questions)
        caps = p.capabilities()
        assert caps.max_rows_per_request * caps.max_questions_per_request <= max_questions  # type: ignore[operator]
        frame = pd.DataFrame({"text": list("abcdefgh")})
        out = sem_decide(frame, questions=[PRED, CHOICE], provider=p, evidence_cols=["text"], task_id="t",
                         budget={"max_requests": 1}, max_concurrency=1)
        assert len(session.posts) == 1
        assert all(len(post["json"]["questions"]) <= max_questions for post in session.posts)  # type: ignore[arg-type,index]
        assert out["p__status"].tolist().count("skipped") > 0

    def test_every_request_of_a_budgeted_packed_run_is_counted(self) -> None:
        session = FakeSession([answers_for])
        p = provider(session, rows_per_request=4, max_questions_per_request=3)
        frame = pd.DataFrame({"text": list("abcdefgh")})
        out = sem_decide(frame, questions=[PRED, CHOICE], provider=p, evidence_cols=["text"], task_id="t",
                         budget={"max_requests": 5}, max_concurrency=1)
        assert len(session.posts) == 5 and "skipped" in out["c__status"].tolist()
