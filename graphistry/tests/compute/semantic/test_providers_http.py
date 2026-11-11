from typing import List

import pytest
import requests

from graphistry.compute.semantic.providers.http import RequestFailure, TokenPrices, TokenUsage, evidence_text, post_json
from graphistry.tests.compute.semantic.http_test_utils import FakeResponse, FakeSession


OK = FakeResponse(200, {"ok": True})


def post(session: FakeSession, deadline_at: float = None, **options: object):  # type: ignore[no-untyped-def]
    params = dict(timeout_s=60.0, max_retries=3, backoff_s=0.0)
    params.update(options)
    return post_json("https://api.test/v1/x", {"q": 1}, {"Authorization": "Bearer k"}, deadline_at=deadline_at,
                     session=session, **params)  # type: ignore[arg-type]


def failure(session: FakeSession, **options: object) -> RequestFailure:
    with pytest.raises(RequestFailure) as info:
        post(session, **options)
    return info.value


class TestRetries:

    @pytest.mark.parametrize("first", [FakeResponse(429), FakeResponse(408), FakeResponse(409), FakeResponse(500),
                                       requests.ConnectionError("down"), requests.Timeout("slow"),
                                       requests.exceptions.ChunkedEncodingError("x"),
                                       requests.exceptions.ContentDecodingError("x"),
                                       requests.HTTPError("503", response=FakeResponse(503))])  # type: ignore[arg-type]
    def test_transient_failures_are_retried(self, first: object) -> None:
        payload, retries, latency_ms = post(FakeSession([first, OK]))  # type: ignore[list-item]
        assert payload == {"ok": True} and retries == 1 and latency_ms >= 0

    @pytest.mark.parametrize("reply,code", [
        (FakeResponse(400), "http_400"), (requests.HTTPError("404", response=FakeResponse(404)), "http_404"),  # type: ignore[arg-type]
        (requests.exceptions.MissingSchema("x"), "request_error"), (requests.exceptions.InvalidURL("x"), "request_error"),
        (requests.exceptions.TooManyRedirects("x"), "request_error"), (requests.HTTPError("no response"), "request_error"),
        (FakeResponse(200, text="{not json"), "invalid_provider_result"), (FakeResponse(200, [1, 2]), "invalid_provider_result"),
    ])
    def test_terminal_failures_are_not_retried(self, reply: object, code: str) -> None:
        session = FakeSession([reply])  # type: ignore[list-item]
        error = failure(session)
        assert (error.error_code, error.retry_count, len(session.posts)) == (code, 0, 1)

    def test_retries_stop_at_the_limit_with_the_last_code(self) -> None:
        session = FakeSession([FakeResponse(503)])
        error = failure(session, max_retries=2)
        assert (error.error_code, error.retry_count, len(session.posts)) == ("http_503", 2, 3)
        assert failure(FakeSession([requests.exceptions.RetryError("x")]), max_retries=1).error_code == "transport_error"

    def test_retry_after_is_honoured_and_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept: List[float] = []
        monkeypatch.setattr("graphistry.compute.semantic.providers.http.time.sleep", slept.append)
        post(FakeSession([FakeResponse(429, headers={"Retry-After": "86400"}), FakeResponse(429, headers={"Retry-After": "2"}),
                          FakeResponse(429, headers={"Retry-After": "soon"}), OK]), backoff_s=0.5)
        assert slept == [60.0, 2.0, 2.0]

    def test_deadline_caps_timeouts_and_stops_retries(self) -> None:
        import time
        session = FakeSession([OK])
        post(session, deadline_at=time.monotonic() + 2.0)
        assert session.posts[0]["timeout"] <= 2.0
        session = FakeSession([FakeResponse(429, headers={"Retry-After": "30"})])
        assert failure(session, deadline_at=time.monotonic() + 1.0).error_code == "http_429" and len(session.posts) == 1
        assert failure(FakeSession([OK]), deadline_at=time.monotonic() - 1.0).error_code == "deadline_exceeded"


class TestTokenPrices:

    USAGE = TokenUsage(uncached_input=700, cached_input=200, cache_write=100, output=50)

    def test_cost_from_usage(self) -> None:
        assert TokenPrices(input=1.0, cached_input=0.1).cost_usd(self.USAGE) == pytest.approx((700 + 20 + 100) / 1e6)
        full = TokenPrices(input=1.0, cached_input=0.1, cache_write=5.0, output=2.0)
        assert full.cost_usd(self.USAGE) == pytest.approx((700 + 20 + 500 + 100) / 1e6)
        assert TokenPrices(input=1.0, cached_input=0.0).cost_usd(TokenUsage(10_000_000, 0, 0, 0)) == pytest.approx(10.0)

    @pytest.mark.parametrize("prices", [TokenPrices(), TokenPrices(input=1.0), TokenPrices(cached_input=1.0)])
    def test_missing_rates_mean_unknown_cost(self, prices: TokenPrices) -> None:
        assert prices.cost_usd(self.USAGE) is None and prices.estimate_usd(["text"], 1) is None
        assert TokenPrices(input=1.0, cached_input=0.1).cost_usd(None) is None

    def test_estimate(self) -> None:
        prices = TokenPrices(input=1.0, cached_input=0.1)
        assert prices.estimate_usd(["x", "abcde"], 1) == pytest.approx(3 / 1e6)
        assert prices.estimate_usd(["x" * (4 * 10 ** 6)], 1) == pytest.approx(1.0)
        assert TokenPrices(per_question=0.002).estimate_usd(["anything"], 6) == pytest.approx(0.012)

    @pytest.mark.parametrize("bad", [dict(input=-1.0), dict(output=float("nan")), dict(per_question=True),
                                     dict(cached_input=float("inf"))])
    def test_rates_must_be_finite_and_non_negative(self, bad: dict) -> None:
        with pytest.raises(ValueError):
            TokenPrices(**bad)


def test_mapping_evidence_is_compact_sorted_json() -> None:
    assert evidence_text({"b": 2, "a": "x"}) == '{"a":"x","b":2}' and evidence_text("as is ") == "as is "
