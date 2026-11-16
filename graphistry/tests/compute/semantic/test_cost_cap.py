from typing import List, Optional, Sequence

import pandas as pd
import pytest

from graphistry import call
from graphistry.compute.semantic.contract import Budget, DecisionRequest
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


class Probing(Recording):
    """Recording provider whose ``prepare_run`` sends setup requests costing ``costs`` through ``admit``."""

    def __init__(self, costs: Sequence[object] = (), outcome: object = "self", **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.costs, self.outcome = list(costs), outcome
        self.admitted: List[bool] = []
        self.prepared: List[List[str]] = []

    def prepare_run(self, requests: Sequence[DecisionRequest], admit, *, deadline_s):  # type: ignore[no-untyped-def]
        self.prepared.append([request.request_id for request in requests])
        for cost in self.costs:
            self.admitted.append(admit(cost))
            if not self.admitted[-1]:
                return None
        if self.outcome == "raise":
            raise RuntimeError("setup failed")
        return self if self.outcome == "self" else self.outcome


class TestSetupRequests:

    def test_setup_requests_count_against_max_requests(self) -> None:
        provider = Probing([0.0, 0.0])
        out = decide(frame(5), provider, batch_size=1, max_concurrency=1, budget={"max_requests": 4})
        assert out["p__status"].tolist() == ["ok"] * 2 + ["skipped"] * 3 and set(reasons(out)[2:]) == {"budget"}
        assert [b.max_requests for b in provider.budgets if b] == [2, 1]
        assert provider.prepared == [["0", "1", "2", "3", "4"]]

    def test_setup_cost_is_reserved_before_row_calls(self) -> None:
        provider = Probing([0.25])
        provider.estimate_cost_usd = per_request(0.1)  # type: ignore[method-assign]
        out = decide(frame(4), provider, batch_size=1, max_concurrency=1, budget={"max_cost_usd": 0.5})
        assert out["p__status"].tolist() == ["ok"] * 2 + ["skipped"] * 2
        assert [round(b.max_cost_usd, 6) for b in provider.budgets if b and b.max_cost_usd is not None] == [0.25, 0.15]

    @pytest.mark.parametrize("costs,budget,admitted", [
        ([0.0, 0.0], {"max_requests": 1}, [True, False]), ([0.6], {"max_cost_usd": 0.5}, [False]),
        ([None], {"max_cost_usd": 0.5}, [False]), ([float("nan")], {"max_cost_usd": 0.5}, [False]),
        ([0.3, 0.3], {"max_cost_usd": 0.5}, [True, False])])
    def test_setup_the_budget_refuses_skips_every_row(self, costs: List[object], budget: object,
                                                      admitted: List[bool]) -> None:
        provider = Probing(costs)
        provider.estimate_cost_usd = per_request(0.0)  # type: ignore[method-assign]
        out = decide(frame(3), provider, budget=budget)
        assert provider.admitted == admitted and provider.calls == []
        assert out["p__status"].tolist() == ["skipped"] * 3 and set(reasons(out)) == {"budget"}

    def test_without_a_budget_setup_runs_and_its_provider_answers(self) -> None:
        answering = Recording()
        provider = Probing([None, 0.1], outcome=answering)
        out = decide(frame(3), provider)
        assert provider.admitted == [True, True] and set(out["p__status"]) == {"ok"}
        assert provider.calls == [] and sorted(answering.request_ids) == ["0", "1", "2"]

    @pytest.mark.parametrize("budget", [None, {"max_cost_usd": 1.0}])
    @pytest.mark.parametrize("outcome", ["raise", object()])
    def test_a_failing_or_invalid_setup_is_a_provider_exception(self, outcome: object, budget: object) -> None:
        provider = Probing(outcome=outcome)
        out = decide(frame(3), provider, budget=budget)
        assert provider.calls == [] and out["p__error_code"].tolist() == ["provider_exception"] * 3

    def test_no_setup_when_no_row_is_sent(self) -> None:
        provider = Probing([0.0])
        out = decide(pd.DataFrame({"name": [None, None]}), provider)
        assert provider.prepared == [] and provider.admitted == [] and set(reasons(out)) == {"null_evidence"}

    def test_setup_requests_alone_are_a_side_effect(self) -> None:
        from graphistry.compute.gfql.effects import effect_scope
        with effect_scope() as probed:
            decide(frame(2), Probing([0.0]), budget={"max_requests": 1})
        with effect_scope() as refused:
            decide(frame(2), Probing([0.6]), budget={"max_cost_usd": 0.5})
        assert (probed, refused) == (["sem_decide"], [])
