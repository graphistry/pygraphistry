import contextvars
import copy
import json
import sys
import threading
from typing import Dict, List, Optional
from unittest import mock

import pandas as pd
import pytest

import graphistry
from graphistry import call, e_forward, let, n, ref
from graphistry.compute.ast import ASTCall
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import (
    ErrorCode,
    GFQLCallExecutionError,
    GFQLSchemaError,
    GFQLTypeError,
    GFQLUnsupportedError,
)
from graphistry.compute.gfql.call.validation import SAFELIST_V1
from graphistry.compute.gfql.policy import PolicyContext, PolicyException
from graphistry.compute.semantic import (
    CallableDecisionProvider,
    CallableGenerationProvider,
    ProviderCapabilities,
    bind_providers,
)
from graphistry.compute.semantic.binding import bound_providers


Q_PRED = {"id": "co", "kind": "predicate", "text": "Is this a company?"}
NODES = pd.DataFrame({"id": [1, 2, 3], "name": ["acme inc", "bob", "zeta inc"], "note": ["a", "b", "c"]})
EDGES = pd.DataFrame({"s": [1, 2], "d": [2, 3], "w": ["x", "y"]})


class Counting(CallableDecisionProvider):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(lambda evidence, q: "inc" in str(evidence), **kwargs)  # type: ignore[arg-type]
        self.items = 0
        self.lock = threading.Lock()

    def decide_batch(self, requests, *, deadline_s, budget):  # type: ignore[no-untyped-def]
        with self.lock:
            self.items += sum(len(r.questions) for r in requests)
        return super().decide_batch(requests, deadline_s=deadline_s, budget=budget)


def graph() -> graphistry.Plottable:
    return graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")


def decide_params(**overrides: object) -> Dict[str, object]:
    params: Dict[str, object] = {"questions": [Q_PRED], "provider": "p", "task_id": "t", "evidence_cols": ["name"]}
    params.update(overrides)
    return params


def run(provider: object, query: object, g: Optional[graphistry.Plottable] = None, **kwargs: object):
    with bind_providers({"p": provider}):  # type: ignore[dict-item]
        return (g or graph()).gfql(query, **kwargs)  # type: ignore[arg-type]


class TestCallMethodParity:

    def test_gfql_equals_method(self) -> None:
        provider = Counting()
        out = run(provider, [call("sem_decide", decide_params())])
        direct = graph().sem_decide([Q_PRED], provider, task_id="t", evidence_cols=["name"])
        cols = ["co", "co__status", "co__p", "co__probability_kind"]
        pd.testing.assert_frame_equal(out._nodes[cols].reset_index(drop=True),
                                      direct._nodes[cols].reset_index(drop=True))
        assert out._nodes["co"].tolist() == [True, False, True]
        pd.testing.assert_frame_equal(out._edges.reset_index(drop=True), EDGES)

    def test_edges_kind_without_edge_binding(self) -> None:
        provider = CallableDecisionProvider(lambda evidence, q: evidence["w"] == "x")  # type: ignore[index]
        params = decide_params(kind="edges", evidence_cols=["w"])
        out = run(provider, [call("sem_decide", params)])
        direct = graph().sem_decide([Q_PRED], provider, kind="edges", task_id="t", evidence_cols=["w"])
        assert out._edges["co"].tolist() == [True, False] == direct._edges["co"].tolist()

    def test_nodes_only_graph_without_id_binding(self) -> None:
        g = graphistry.nodes(NODES[["name"]])
        out = run(Counting(), [call("sem_decide", decide_params())], g=g)
        assert out._nodes["co"].tolist() == [True, False, True]

    def test_nodes_kind_on_edges_only_graph(self) -> None:
        g = graphistry.edges(EDGES, "s", "d")
        provider = Counting()
        with pytest.raises(ValueError, match="needs a nodes table"):
            g.sem_decide([Q_PRED], provider, task_id="t", evidence_cols=["w"])
        with pytest.raises(GFQLSchemaError):
            run(provider, [call("sem_decide", decide_params(evidence_cols=["w"]))], g=g)
        assert provider.items == 0

    def test_sem_map_parity(self) -> None:
        gen = CallableGenerationProvider(lambda evidence, instruction: str(evidence["name"]).upper())  # type: ignore[index]
        params = {"instruction": "Shout it", "out_col": "loud", "provider": "p", "task_id": "t",
                  "evidence_cols": ["name"]}
        out = run(gen, [call("sem_map", params)])
        assert out._nodes["loud"].tolist() == ["ACME INC", "BOB", "ZETA INC"]


class TestJsonRoundTrip:

    def test_sem_map_round_trip(self) -> None:
        params = {"instruction": "i", "out_col": "o", "provider": "p", "task_id": "t", "evidence_cols": ["name"],
                  "max_output_tokens": 20, "budget": {"max_requests": 3}}
        payload = call("sem_map", params).to_json(validate=True)
        back = ASTCall.from_json(json.loads(json.dumps(payload)), validate=True)
        assert back.function == "sem_map" and back.params == params

    def test_round_trip_and_params_untouched(self) -> None:
        params = decide_params(
            questions=[Q_PRED, {"id": "grp", "kind": "choice", "text": "Which?", "options": ["a", "b"]},
                       {"id": "lvl", "kind": "score", "text": "How?", "levels": ["lo", "hi"]}],
            kind="nodes", id_col="id", task_version="2", batch_size=2, max_concurrency=1, deadline_s=30.0,
            budget={"max_requests": 10}, max_rows=100, min_p=0.5, column_max_chars=50, overwrite=False,
        )
        frozen = copy.deepcopy(params)
        payload = call("sem_decide", params).to_json(validate=True)
        back = ASTCall.from_json(json.loads(json.dumps(payload)), validate=True)
        assert back.function == "sem_decide" and back.params == params
        run(Counting(), [call("sem_decide", params)])
        assert params == frozen

    @pytest.mark.parametrize("params,code", [
        (decide_params(bogus=1), ErrorCode.E303),
        ({"questions": [Q_PRED], "provider": "p", "evidence_cols": ["name"]}, ErrorCode.E105),
        (decide_params(kind="both"), ErrorCode.E201),
    ])
    def test_validation_codes(self, params: Dict[str, object], code: str) -> None:
        with pytest.raises(GFQLTypeError) as info:
            ASTCall("sem_decide", params).validate()
        assert info.value.code == code


class TestGfqlValidation:

    @pytest.mark.parametrize("overrides", [
        dict(evidence_cols="name"),
        dict(provider=5),
        dict(max_rows=True),
        dict(min_p=1.5),
        dict(questions=[{"id": "q", "kind": "choice", "text": "t", "options": ["a"]}]),
        dict(questions=[{"id": "q", "kind": "score", "text": "t", "levels": ["x", "y"], "options": ["a", "b"]}]),
        dict(questions=[{"id": "__gfql_x__", "kind": "predicate", "text": "t"}]),
        dict(budget={"max_cost_usd": -1}),
        dict(evidence_cols=["name", "name"]),
        dict(deadline_s=0),
    ])
    def test_e201_before_kernel(self, overrides: Dict[str, object]) -> None:
        provider = Counting()
        with pytest.raises(GFQLTypeError) as info:
            run(provider, [call("sem_decide", decide_params(**overrides))])
        assert info.value.code == ErrorCode.E201
        assert provider.items == 0

    @pytest.mark.parametrize("overrides", [
        dict(evidence_cols=["missing"]),
        dict(id_col="missing"),
        dict(evidence_cols=None, template="{missing}"),
    ])
    def test_e301_missing_columns(self, overrides: Dict[str, object]) -> None:
        provider = Counting()
        params = decide_params(**overrides)
        params = {k: v for k, v in params.items() if v is not None}
        with pytest.raises(GFQLSchemaError):
            run(provider, [call("sem_decide", params)])
        assert provider.items == 0

    @pytest.mark.parametrize("overrides", [
        dict(template="{name}"),
        dict(evidence_cols=None),
        dict(evidence_cols=None, template="{0}"),
        dict(evidence_cols=None, template="{name.upper}"),
        dict(evidence_cols=None, template="{name!r}"),
        dict(evidence_cols=None, template="{name:>5}"),
        dict(questions=[{"id": "a", "kind": "predicate", "text": "t"}, {"id": "a__p", "kind": "predicate", "text": "t"}]),
    ])
    def test_e306_kernel_value_errors(self, overrides: Dict[str, object]) -> None:
        provider = Counting()
        params = {k: v for k, v in decide_params(**overrides).items() if v is not None}
        with pytest.raises(GFQLCallExecutionError) as info:
            run(provider, [call("sem_decide", params)])
        assert isinstance(info.value.__cause__, ValueError)
        assert provider.items == 0

    def test_e306_too_many_options(self) -> None:
        limited = Counting(capabilities=ProviderCapabilities(True, 10, None, False, 1, max_choice_options=2))
        question = {"id": "c", "kind": "choice", "text": "t", "options": ["a", "b", "c"]}
        with pytest.raises(GFQLCallExecutionError):
            run(limited, [call("sem_decide", decide_params(questions=[question]))])
        assert limited.items == 0

    def test_e306_capabilities_and_existing_column(self) -> None:
        limited = Counting(capabilities=ProviderCapabilities(True, 10, None, False, 1, supported_kinds=("choice",)))
        with pytest.raises(GFQLCallExecutionError):
            run(limited, [call("sem_decide", decide_params())])
        g = graphistry.nodes(NODES.assign(co__status="old"), "id")
        provider = Counting()
        with pytest.raises(GFQLCallExecutionError):
            run(provider, [call("sem_decide", decide_params())], g=g)
        assert limited.items == 0 and provider.items == 0

    def test_polars_missing_column_is_e306(self) -> None:
        pl = pytest.importorskip("polars")
        g = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
        provider = Counting()
        with pytest.raises(GFQLCallExecutionError):
            run(provider, [call("sem_decide", decide_params(evidence_cols=["missing"]))], g=g, engine="polars")
        assert provider.items == 0


class TestBindingScope:

    def test_unbound_alias(self) -> None:
        provider = Counting()
        with pytest.raises(ValueError, match="alias 'x' is not bound"):
            graph().sem_decide([Q_PRED], "x", task_id="t", evidence_cols=["name"])
        with pytest.raises(GFQLCallExecutionError) as info:
            graph().gfql([call("sem_decide", decide_params(provider="x"))])
        assert isinstance(info.value.__cause__, ValueError) and "'x'" in str(info.value.__cause__)
        assert provider.items == 0

    def test_nesting_copies_and_threads(self) -> None:
        a, b = Counting(), Counting()
        mapping = {"x": a}
        with bind_providers(mapping):
            mapping["x"] = b
            assert bound_providers()["x"] is a
            with bind_providers({"x": b}):
                assert bound_providers()["x"] is b
            assert bound_providers()["x"] is a
            assert contextvars.copy_context().run(lambda: bound_providers()["x"]) is a
            seen: List[object] = []
            thread = threading.Thread(target=lambda: seen.append(bound_providers().get("x")))
            thread.start()
            thread.join()
            inherits = bool(getattr(sys.flags, "thread_inherit_context", 0))
            assert seen == ([a] if inherits else [None])
        assert "x" not in bound_providers()

    def test_generation_provider_under_decide_alias(self) -> None:
        gen = CallableGenerationProvider(lambda e, i: "x")
        with bind_providers({"p": gen}):
            with pytest.raises(ValueError, match="not a DecisionProvider"):
                graph().sem_decide([Q_PRED], "p", task_id="t", evidence_cols=["name"])


class TestPolicyHooks:

    def _hooks(self, events: List[Dict[str, object]], veto: bool = False) -> Dict[str, object]:
        def precall(context: PolicyContext) -> None:
            events.append({"phase": "precall", "op": context.get("call_op"), "params": context.get("call_params")})
            if veto and context.get("call_op") == "sem_decide":
                raise PolicyException(phase="precall", reason="denied")

        def postcall(context: PolicyContext) -> None:
            events.append({"phase": "postcall", "op": context.get("call_op"), "success": context.get("success"),
                           "error": context.get("error")})

        return {"precall": precall, "postcall": postcall}

    def test_hooks_fire_with_alias_only(self) -> None:
        events: List[Dict[str, object]] = []
        provider = Counting()
        run(provider, [call("sem_decide", decide_params())], policy=self._hooks(events))
        sem = [e for e in events if e["op"] == "sem_decide"]
        assert [e["phase"] for e in sem] == ["precall", "postcall"] and sem[1]["success"] is True
        params = sem[0]["params"]
        assert params["provider"] == "p"  # type: ignore[index]
        assert "acme" not in json.dumps(params)

    def test_veto_stops_before_provider(self) -> None:
        events: List[Dict[str, object]] = []
        provider = Counting()
        with pytest.raises(PolicyException):
            run(provider, [call("sem_decide", decide_params())], policy=self._hooks(events, veto=True))
        assert provider.items == 0
        assert not [e for e in events if e["phase"] == "postcall" and e["op"] == "sem_decide"]

    def test_unbound_alias_postcall_reports_failure(self) -> None:
        events: List[Dict[str, object]] = []
        with pytest.raises(GFQLCallExecutionError):
            graph().gfql([call("sem_decide", decide_params(provider="nope"))], policy=self._hooks(events))
        post = [e for e in events if e["phase"] == "postcall" and e["op"] == "sem_decide"]
        assert post and post[0]["success"] is False and "not bound" in str(post[0]["error"])

    def test_direct_method_bypasses_execute_call(self) -> None:
        with mock.patch("graphistry.compute.gfql.call.executor.execute_call") as execute:
            graph().sem_decide([Q_PRED], Counting(), task_id="t", evidence_cols=["name"])
        assert execute.call_count == 0

    def test_early_errors_fire_no_hooks(self) -> None:
        events: List[Dict[str, object]] = []
        hooks = self._hooks(events)
        with pytest.raises(GFQLTypeError):
            run(Counting(), [call("sem_decide", decide_params(min_p=1.5))], policy=hooks)
        with pytest.raises(GFQLSchemaError):
            run(Counting(), [call("sem_decide", decide_params(evidence_cols=["missing"]))], policy=hooks)
        assert not [e for e in events if e["op"] == "sem_decide"]


class TestRemoteRefusal:

    QUERIES = [
        [call("sem_decide", decide_params())],
        Chain([call("sem_decide", decide_params())]),
        let({"a": call("sem_map", {"instruction": "i", "out_col": "o", "provider": "p", "task_id": "t",
                                     "evidence_cols": ["name"]})}),
        let({"a": [n()], "b": ref("a", [call("sem_decide", decide_params())])}),
        {"type": "Chain", "chain": [{"type": "Call", "function": "sem_decide", "params": decide_params()}]},
        {"chain": [{"type": "Call", "function": "sem_decide", "params": decide_params()},
                   {"type": "Call", "function": "get_degrees", "params": {"bogus": 1}}]},
    ]

    @pytest.mark.parametrize("query", QUERIES)
    @pytest.mark.parametrize("method", ["gfql_remote", "gfql_remote_shape", "chain_remote", "chain_remote_shape"])
    def test_refused_before_any_network(self, query: object, method: str) -> None:
        g = graph()
        with mock.patch("graphistry.compute.chain_remote.requests.post") as post, \
                mock.patch.object(type(g), "upload") as upload:
            with pytest.raises(GFQLUnsupportedError, match="run locally"):
                getattr(g, method)(query, api_token="t", dataset_id="ds")
            assert post.call_count == 0 and upload.call_count == 0

    @pytest.mark.parametrize("validate", [True, False])
    def test_refused_before_refresh_and_upload(self, validate: bool) -> None:
        g = graph()
        with mock.patch("graphistry.compute.chain_remote.requests.post") as post, \
                mock.patch.object(type(g), "upload") as upload, \
                mock.patch.object(g._pygraphistry, "refresh") as refresh:
            with pytest.raises(GFQLUnsupportedError):
                g.gfql_remote([call("sem_decide", decide_params())], validate=validate)
            assert post.call_count == 0 and upload.call_count == 0 and refresh.call_count == 0

    @pytest.mark.parametrize("query", [
        {"type": "Chain", "chain": ({"type": "Call", "function": "sem_decide", "params": decide_params()},)},
        {"type": "Chain", "chain": [call("sem_decide", decide_params())]},
        {"type": "Let", "bindings": {"a": {"type": "Chain", "chain": ({"type": "Call", "function": "sem_map",
                                                                        "params": {}},)}}},
    ])
    def test_walker_sees_tuples_and_ast_objects(self, query: object) -> None:
        g = graph()
        with mock.patch("graphistry.compute.chain_remote.requests.post") as post, \
                mock.patch.object(type(g), "upload") as upload, \
                mock.patch.object(g._pygraphistry, "refresh") as refresh:
            with pytest.raises(GFQLUnsupportedError):
                g.gfql_remote(query, validate=False)
            assert post.call_count == 0 and upload.call_count == 0 and refresh.call_count == 0

    def test_sem_map_late_decline_raises_after_one_pass(self) -> None:
        pl = pytest.importorskip("polars")
        g = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
        gen = CallableGenerationProvider(lambda e, i: "x")
        calls: List[int] = []
        original = gen.generate_batch

        def counting(requests, *, deadline_s, budget):  # type: ignore[no-untyped-def]
            calls.append(len(requests))
            return original(requests, deadline_s=deadline_s, budget=budget)

        gen.generate_batch = counting  # type: ignore[method-assign]
        params = {"instruction": "i", "out_col": "o", "provider": "p", "task_id": "t", "evidence_cols": ["name"]}
        with pytest.raises(GFQLCallExecutionError):
            run(gen, [call("sem_map", params)] + TestAutoEffectBarrier.SEL, g=g)
        assert sum(calls) == 3

    def test_malformed_non_semantic_call_keeps_its_own_errors(self) -> None:
        g = graph()
        query = {"type": "Chain", "chain": [{"type": "Call", "function": ["get_degrees"], "params": {}}]}
        with mock.patch("graphistry.compute.chain_remote.requests.post", side_effect=RuntimeError("sent")) as post:
            with pytest.raises(GFQLTypeError, match="function must be a string"):
                g.gfql_remote(query, api_token="t", dataset_id="ds")
            with pytest.raises(RuntimeError, match="sent"):
                g.gfql_remote(query, api_token="t", dataset_id="ds", validate=False)
            assert post.call_count == 1

    def test_control_query_reaches_post(self) -> None:
        g = graph()
        with mock.patch("graphistry.compute.chain_remote.requests.post", side_effect=RuntimeError("sent")) as post:
            with pytest.raises(RuntimeError, match="sent"):
                g.gfql_remote([n()], api_token="t", dataset_id="ds")
            assert post.call_count == 1


class TestAutoEffectBarrier:

    SEL = [call("rows", {"table": "nodes"}), call("select", {"items": [["out", "reverse(name)"]]})]

    def _polars_graph(self) -> graphistry.Plottable:
        pl = pytest.importorskip("polars")
        return graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")

    def test_call_alone_runs_once_on_polars(self) -> None:
        provider = Counting()
        for engine in ("auto", "polars"):
            out = run(provider, [call("sem_decide", decide_params())], g=self._polars_graph(), engine=engine)
            assert type(out._nodes).__module__.startswith("polars")
            assert list(out._nodes["co"]) == [True, False, True]
        assert provider.items == 6

    @pytest.mark.parametrize("shape", ["late", "traversal", "let"])
    def test_late_decline_raises_after_one_pass(self, shape: str) -> None:
        sd = call("sem_decide", decide_params())
        query = {"late": [sd] + self.SEL, "traversal": [n(), e_forward(), n(), sd] + self.SEL,
                 "let": let({"a": [sd], "b": ref("a", self.SEL)})}[shape]
        provider = Counting()
        with pytest.raises(GFQLCallExecutionError) as info:
            run(provider, query, g=self._polars_graph())
        assert not isinstance(info.value, NotImplementedError)
        assert isinstance(info.value.__cause__, NotImplementedError)
        assert "engine='pandas'" in str(info.value)
        assert provider.items == 3
        provider_again = Counting()
        out = run(provider_again, [call("sem_decide", decide_params())], g=self._polars_graph())
        assert provider_again.items == 3 and list(out._nodes["co"]) == [True, False, True]

    def test_barrier_code_ledger_reset_and_raising_provider(self) -> None:
        from graphistry.compute.gfql import effects
        provider = Counting()
        with pytest.raises(GFQLCallExecutionError) as info:
            run(provider, [call("sem_decide", decide_params())] + self.SEL, g=self._polars_graph())
        assert info.value.code == ErrorCode.E306 and "(sem_decide)" in str(info.value)
        assert effects._LEDGER.get() is None

        class Raising(Counting):
            def decide_batch(self, requests, *, deadline_s, budget):  # type: ignore[no-untyped-def]
                super().decide_batch(requests, deadline_s=deadline_s, budget=budget)
                raise RuntimeError("provider down after billing")

        raising = Raising()
        with pytest.raises(GFQLCallExecutionError):
            run(raising, [call("sem_decide", decide_params())] + self.SEL, g=self._polars_graph())
        assert raising.items == 3

    def test_explicit_polars_late_decline_is_plain_nie(self) -> None:
        provider = Counting()
        with pytest.raises(NotImplementedError):
            run(provider, [call("sem_decide", decide_params())] + self.SEL, g=self._polars_graph(), engine="polars")
        assert provider.items == 3

    def test_early_decline_falls_back_once(self) -> None:
        provider = Counting()
        out = run(provider, [call("sem_decide", decide_params()), n(), e_forward(), n()], g=self._polars_graph())
        assert provider.items == 3
        assert type(out._nodes).__module__.startswith("pandas")

    def test_queries_without_effects_still_fall_back(self) -> None:
        out = self._polars_graph().gfql([call("prune_self_edges")] + self.SEL)
        assert type(out._nodes).__module__.startswith("pandas")

    def test_deadline_before_any_call_means_plain_fallback(self) -> None:
        provider = Counting()
        out = run(provider, [call("sem_decide", decide_params(deadline_s=1e-9))] + self.SEL, g=self._polars_graph())
        assert provider.items == 0
        assert type(out._nodes).__module__.startswith("pandas")

    def test_no_provider_call_means_plain_fallback(self) -> None:
        provider = Counting()
        provider.estimate_cost_usd = lambda requests: 1.0  # type: ignore[method-assign]
        out = run(provider, [call("sem_decide", decide_params(budget={"max_cost_usd": 0.5}))] + self.SEL,
                  g=self._polars_graph())
        assert provider.items == 0
        assert type(out._nodes).__module__.startswith("pandas")

    def test_cudf_auto_arm_refuses_rerun_after_effect(self) -> None:
        import graphistry.compute.gfql_unified as gu

        def fake_route(self, query, **kwargs):  # type: ignore[no-untyped-def]
            self.gfql(query, engine="pandas")
            raise NotImplementedError("declined after the call")

        provider = Counting()
        with mock.patch.object(gu, "_is_cudf_frame", return_value=True), \
                mock.patch.object(gu, "_polars_gpu_probe", return_value=True), \
                mock.patch.object(gu, "_auto_cudf_polars_gpu_route", side_effect=fake_route):
            with pytest.raises(GFQLCallExecutionError) as info:
                run(provider, [call("sem_decide", decide_params())])
        assert isinstance(info.value.__cause__, NotImplementedError) and provider.items == 3

    def test_polars_both_evidence_params_is_e306(self) -> None:
        provider = Counting()
        with pytest.raises(GFQLCallExecutionError):
            run(provider, [call("sem_decide", decide_params(template="{name}"))], g=self._polars_graph(),
                engine="polars")
        assert provider.items == 0

    def test_concurrent_auto_queries_keep_separate_ledgers(self) -> None:
        results: List[object] = []
        graphs = [self._polars_graph() for _ in range(4)]

        def worker(fail: bool, g: graphistry.Plottable) -> None:
            provider = Counting()
            query = [call("sem_decide", decide_params())] + (self.SEL if fail else [])
            try:
                run(provider, query, g=g)
                results.append(("ok", provider.items))
            except GFQLCallExecutionError:
                results.append(("e306", provider.items))

        threads = [threading.Thread(target=worker, args=(i % 2 == 0, graphs[i])) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(results) == [("e306", 3), ("e306", 3), ("ok", 3), ("ok", 3)]


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_declared_schema_effects_match_written_columns(kind: str) -> None:
    from graphistry.compute.gfql.schema_effects import schema_effect_for_call
    questions = [Q_PRED, {"id": "grp", "kind": "choice", "text": "Which?", "options": ["a", "b"]}]
    cols = ["w"] if kind == "edges" else ["name"]
    decide = decide_params(questions=questions, kind=kind, evidence_cols=cols)
    gen = {"instruction": "i", "out_col": "o", "provider": "p", "task_id": "t", "kind": kind, "evidence_cols": cols}
    provider = CallableDecisionProvider(lambda e, q: True if q.kind == "predicate" else "a")
    generator = CallableGenerationProvider(lambda e, i: "x")
    for function, params, bound in (("sem_decide", decide, provider), ("sem_map", gen, generator)):
        out = run(bound, [call(function, params)])
        table = out._edges if kind == "edges" else out._nodes
        base = EDGES if kind == "edges" else NODES
        written = [col for col in table.columns if col not in base.columns]
        entry = SAFELIST_V1[function]["schema_effects"]
        adds = entry["adds_edge_cols" if kind == "edges" else "adds_node_cols"](params)
        assert adds == written
        effect = schema_effect_for_call(function, params)
        typed = effect.adds_edge_properties if kind == "edges" else effect.adds_node_properties
        assert list(typed) == written
        for name, logical in typed.items():
            dtype = str(table[name].dtype)
            assert {"bool": "boolean", "float64": "float64", "string": "object"}[logical.kind] == dtype


def test_safelist_entries_registered() -> None:
    assert {"sem_decide", "sem_map"} <= set(SAFELIST_V1)
    assert SAFELIST_V1["sem_decide"]["required_params"] == {"questions", "provider", "task_id"}
    assert SAFELIST_V1["sem_map"]["required_params"] == {"instruction", "out_col", "provider", "task_id"}
