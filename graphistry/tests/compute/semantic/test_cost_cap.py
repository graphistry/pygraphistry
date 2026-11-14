from typing import Optional

import pytest

from graphistry import call
from graphistry.compute.semantic.contract import Budget
from graphistry.tests.compute.semantic.semantic_test_utils import Recording, frame
from graphistry.tests.compute.semantic.test_gfql_bindings import SEL, counting, decide_params, polars_graph, run
from graphistry.tests.compute.semantic.test_sem_decide import decide, reasons


def priced(estimate: object, **kwargs: object) -> Recording:
    provider = Recording(**kwargs)
    provider.estimate_cost_usd = estimate  # type: ignore[method-assign]
    return provider


def per_request(cost: Optional[float]):  # type: ignore[no-untyped-def]
    return lambda requests: None if cost is None else cost * len(requests)


class TestCostCap:

    @pytest.mark.parametrize("cap,rows,ok", [(0.025, 10, 2), (0.3, 4, 3), (0.0195, 3, 1), (24.31, 2432, 2431)])
    def test_reservations_fit_the_cap_exactly(self, cap: float, rows: int, ok: int) -> None:
        cost = 0.1 if cap == 0.3 else 0.01
        out = decide(frame(rows), priced(per_request(cost)), batch_size=1, max_concurrency=1,
                     budget=Budget(max_cost_usd=cap))
        assert out["p__status"].tolist() == ["ok"] * ok + ["skipped"] * (rows - ok)
        assert set(reasons(out)[ok:]) <= {"budget"}

    def test_remaining_cost_and_requests_reach_the_provider(self) -> None:
        provider = priced(per_request(0.1))
        decide(frame(3), provider, batch_size=1, max_concurrency=1, budget={"max_cost_usd": 1.0, "max_requests": 5})
        assert [(b.max_requests, round(b.max_cost_usd, 6)) for b in provider.budgets if b and b.max_cost_usd is not None] \
            == [(5, 1.0), (4, 0.9), (3, 0.8)]

    @pytest.mark.parametrize("estimate", [per_request(None), per_request(float("nan")), per_request(-1.0),
                                          lambda requests: 1 / 0, lambda requests: "0.1"])
    def test_cap_without_a_valid_first_estimate_raises_before_any_call(self, estimate: object) -> None:
        provider = priced(estimate)
        with pytest.raises(ValueError, match="cannot estimate"):
            decide(frame(3), provider, budget={"max_cost_usd": 1.0})
        assert provider.calls == []
        assert set(decide(frame(3), provider, budget={"max_requests": 5})["p__status"]) == {"ok"}

    def test_invalid_estimate_mid_run_stops_new_calls(self) -> None:
        provider = priced(lambda requests: float("nan") if requests[0].request_id == "1" else 0.01)
        out = decide(frame(4), provider, batch_size=1, max_concurrency=1, budget={"max_cost_usd": 1.0})
        assert out["p__status"].tolist() == ["ok", "skipped", "skipped", "skipped"]

    def test_request_limit_still_binds_under_a_cost_cap(self) -> None:
        out = decide(frame(4), priced(per_request(0.01)), batch_size=1, max_concurrency=1,
                     budget={"max_cost_usd": 1.0, "max_requests": 2})
        assert out["p__status"].tolist() == ["ok", "ok", "skipped", "skipped"]

    @pytest.mark.parametrize("cap", [float("nan"), float("inf"), -1.0, True, "1"])
    def test_a_budget_built_directly_rejects_caps_that_mean_no_cap(self, cap: object) -> None:
        with pytest.raises(ValueError):
            Budget(max_cost_usd=cap)  # type: ignore[arg-type]

    def test_a_free_call_after_the_cap_is_reached_still_runs(self) -> None:
        provider = priced(lambda requests: 0.0 if requests[0].request_id == "3" else 0.1)
        out = decide(frame(4), provider, batch_size=1, max_concurrency=1, budget={"max_cost_usd": 0.3})
        assert out["p__status"].tolist() == ["ok"] * 4

    def test_a_cap_on_a_table_with_nothing_to_send_is_not_an_error(self) -> None:
        provider = priced(per_request(None))
        assert decide(frame(0), provider, budget={"max_cost_usd": 1.0}).empty
        nulls = decide(frame(2).assign(name=None), provider, budget={"max_cost_usd": 1.0})
        assert set(reasons(nulls)) == {"null_evidence"} and provider.calls == []

    def test_callable_provider_estimates_zero(self) -> None:
        assert set(decide(frame(2), Recording(), budget={"max_cost_usd": 0.0})["p__status"]) == {"ok"}


def test_no_provider_call_means_plain_auto_fallback() -> None:
    provider = counting()
    provider.estimate_cost_usd = lambda requests: 1.0  # type: ignore[method-assign]
    out = run(provider, [call("sem_decide", decide_params(budget={"max_cost_usd": 0.5}))] + SEL, g=polars_graph())
    assert provider.answers == 0 and type(out._nodes).__module__.startswith("pandas")
