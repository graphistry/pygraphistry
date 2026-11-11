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
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import (
    ErrorCode, GFQLCallExecutionError, GFQLSchemaError, GFQLTypeError, GFQLUnsupportedError)
from graphistry.compute.gfql.call.validation import SAFELIST_V1
from graphistry.compute.gfql.policy import PolicyContext, PolicyException
from graphistry.compute.semantic import CallableDecisionProvider, bind_providers
from graphistry.compute.semantic.binding import resolve_provider
from graphistry.tests.compute.semantic.semantic_test_utils import Recording, caps


Q_PRED = {"id": "co", "kind": "predicate", "text": "Is this a company?"}
NODES = pd.DataFrame({"id": [1, 2, 3], "name": ["acme inc", "bob", "zeta inc"], "note": ["a", "b", "c"]})
EDGES = pd.DataFrame({"s": [1, 2], "d": [2, 3], "w": ["x", "y"]})
SEL = [call("rows", {"table": "nodes"}), call("select", {"items": [["out", "reverse(name)"]]})]


def counting(**kwargs: object) -> Recording:
    return Recording(lambda evidence, q: "inc" in str(evidence), **kwargs)


def graph() -> Plottable:
    return graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")


def polars_graph() -> Plottable:
    pl = pytest.importorskip("polars")
    return graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")


def decide_params(**overrides: object) -> Dict[str, object]:
    params: Dict[str, object] = {"questions": [Q_PRED], "provider": "p", "task_id": "t", "evidence_cols": ["name"]}
    params.update(overrides)
    return {key: value for key, value in params.items() if value is not None}


def run(provider: object, query: object, g: Optional[Plottable] = None, **kwargs: object) -> Plottable:
    with bind_providers({"p": provider}):  # type: ignore[dict-item]
        return (g or graph()).gfql(query, **kwargs)  # type: ignore[arg-type]


class TestCallMethodParity:

    def test_gfql_equals_method(self) -> None:
        provider = counting()
        out = run(provider, [call("sem_decide", decide_params())])
        direct = graph().sem_decide([Q_PRED], provider, task_id="t", evidence_cols=["name"])
        cols = ["co", "co__status", "co__p", "co__probability_kind"]
        pd.testing.assert_frame_equal(out._nodes[cols].reset_index(drop=True), direct._nodes[cols].reset_index(drop=True))
        assert out._nodes["co"].tolist() == [True, False, True]
        pd.testing.assert_frame_equal(out._edges.reset_index(drop=True), EDGES)

    def test_edges_and_graphs_without_bindings(self) -> None:
        provider = CallableDecisionProvider(lambda evidence, q: evidence["w"] == "x")  # type: ignore[index]
        out = run(provider, [call("sem_decide", decide_params(kind="edges", evidence_cols=["w"]))])
        direct = graph().sem_decide([Q_PRED], provider, kind="edges", task_id="t", evidence_cols=["w"])
        assert out._edges["co"].tolist() == [True, False] == direct._edges["co"].tolist()
        out = run(counting(), [call("sem_decide", decide_params())], g=graphistry.nodes(NODES[["name"]]))
        assert out._nodes["co"].tolist() == [True, False, True]

    def test_kind_must_name_a_table(self) -> None:
        provider = counting()
        for kind in ("both", pd.NA):
            with pytest.raises(ValueError, match="kind"):
                graph().sem_decide([Q_PRED], provider, kind=kind, task_id="t", evidence_cols=["name"])  # type: ignore[arg-type]
        for params in (decide_params(kind=pd.NA), decide_params(questions=[dict(Q_PRED, kind=pd.NA)])):
            with pytest.raises(GFQLTypeError) as info:
                ASTCall("sem_decide", params).validate()
            assert info.value.code == ErrorCode.E201
        assert provider.answers == 0

    def test_direct_method_on_a_polars_graph_returns_pandas_tables(self) -> None:
        out = polars_graph().sem_decide([Q_PRED], counting(), task_id="t", evidence_cols=["name"])
        assert type(out._nodes).__module__.startswith("pandas") and out._nodes["co"].tolist() == [True, False, True]

    def test_edge_id_binding_is_the_default_row_id(self) -> None:
        provider = counting()
        edges = EDGES.assign(eid=["e1", "e2"])
        graphistry.edges(edges, "s", "d", edge="eid").sem_decide([Q_PRED], provider, kind="edges", task_id="t",
                                                                evidence_cols=["w"])
        assert sorted(provider.request_ids) == ["e1", "e2"]

    @pytest.mark.parametrize("surface", ["method", "gfql"])
    def test_limits_and_the_id_binding_reach_the_kernel(self, surface: str) -> None:
        provider = counting()
        limits: Dict[str, object] = dict(max_rows=2, budget={"max_requests": 1}, deadline_s=30.0, batch_size=1,
                                         min_p=0.5, task_version="7")
        if surface == "method":
            out = graph().sem_decide([Q_PRED], provider, task_id="t", evidence_cols=["name"],
                                     **limits)  # type: ignore[arg-type]
        else:
            out = run(provider, [call("sem_decide", decide_params(**limits))])
        metas = [json.loads(meta) for meta in out._nodes["co__meta"]]
        assert out._nodes["co__status"].tolist() == ["uncertain", "skipped", "skipped"]
        assert [(m["status_reason"], m["task_version"]) for m in metas] == [
            ("min_p", "7"), ("budget", "7"), ("max_rows", "7")]
        assert provider.request_ids == ["1"] and provider.budgets[0] is not None and provider.budgets[0].max_requests == 1
        assert provider.task_ids == ["t"]
        assert provider.deadlines[0] is not None and 0 < provider.deadlines[0] <= 30.0

    @pytest.mark.parametrize("surface", ["method", "gfql"])
    def test_concurrency_and_evidence_truncation_reach_the_kernel(self, surface: str) -> None:
        seen: List[object] = []
        provider = Recording(lambda evidence, q: seen.append(evidence) or True, delay=0.05)
        limits: Dict[str, object] = dict(max_concurrency=1, batch_size=1, column_max_chars=3)
        if surface == "method":
            graph().sem_decide([Q_PRED], provider, task_id="t", evidence_cols=["name"], **limits)  # type: ignore[arg-type]
        else:
            run(provider, [call("sem_decide", decide_params(**limits))])
        assert provider.peak == 1 and sorted(e["name"] for e in seen) == ["acm", "bob", "zet"]  # type: ignore[index]

    def test_parameter_lists_agree_across_surfaces(self) -> None:
        import inspect
        from graphistry.compute.ComputeMixin import ComputeMixin
        from graphistry.compute.semantic.ops import sem_decide
        from graphistry.compute.semantic.rules import RULES
        from graphistry.models.gfql.types.call import SemDecideParams
        kernel = set(inspect.signature(sem_decide).parameters) - {"df"}
        method = set(inspect.signature(ComputeMixin.sem_decide).parameters) - {"self", "kind"}
        plottable = set(inspect.signature(Plottable.sem_decide).parameters) - {"self", "kind"}
        assert kernel == method == plottable == set(RULES) - {"kind"} == set(SemDecideParams.__annotations__) - {"kind"}

    def test_nodes_kind_on_edges_only_graph(self) -> None:
        g = graphistry.edges(EDGES, "s", "d")
        provider = counting()
        with pytest.raises(ValueError, match="needs a nodes table"):
            g.sem_decide([Q_PRED], provider, task_id="t", evidence_cols=["w"])
        with pytest.raises(GFQLSchemaError):
            run(provider, [call("sem_decide", decide_params(evidence_cols=["w"]))], g=g)
        assert provider.answers == 0

    def test_id_defaults_to_the_node_binding_and_internal_edge_binding_is_unbound(self) -> None:
        provider = counting()
        g = graphistry.nodes(pd.DataFrame({"nid": ["x", "y"], "name": ["a", "b"]}, index=[5, 5]), "nid")
        g.sem_decide([Q_PRED], provider, task_id="t", evidence_cols=["name"])
        assert sorted(provider.request_ids) == ["x", "y"]
        edges = pd.DataFrame({"s": [1, 2, 1], "d": [2, 3, 3], "w": ["a", "b", "c"]}, index=[7, 7, 9])
        g = graphistry.nodes(pd.DataFrame({"id": [1, 2, 3]}), "id").edges(edges, "s", "d")
        provider = counting()
        out = run(provider, [n(), e_forward(), n(), call("sem_decide", decide_params(kind="edges", evidence_cols=["w"]))], g=g)
        assert sorted(provider.request_ids) == ["0", "1", "2"] and set(out._edges["co__status"]) == {"ok"}


class TestJsonAndValidation:

    def test_round_trip_and_params_untouched(self) -> None:
        params = decide_params(
            questions=[Q_PRED, {"id": "grp", "kind": "choice", "text": "Which?", "options": ["a", "b"]},
                       {"id": "lvl", "kind": "score", "text": "How?", "levels": ["lo", "hi"]}],
            kind="nodes", id_col="id", task_version="2", batch_size=2, max_concurrency=1, deadline_s=30.0,
            budget={"max_requests": 10}, max_rows=100, min_p=0.5, column_max_chars=50, overwrite=False)
        frozen = copy.deepcopy(params)
        back = ASTCall.from_json(json.loads(json.dumps(call("sem_decide", params).to_json(validate=True))), validate=True)
        assert back.function == "sem_decide" and back.params == params
        run(counting(), [call("sem_decide", params)])
        assert params == frozen

    @pytest.mark.parametrize("params,code", [
        (decide_params(bogus=1), ErrorCode.E303), (decide_params(task_id=None), ErrorCode.E105),
        (decide_params(kind="both"), ErrorCode.E201),
    ])
    def test_validation_codes(self, params: Dict[str, object], code: str) -> None:
        with pytest.raises(GFQLTypeError) as info:
            ASTCall("sem_decide", params).validate()
        assert info.value.code == code

    @pytest.mark.parametrize("overrides", [
        dict(evidence_cols="name"), dict(provider=5), dict(max_rows=True), dict(min_p=1.5), dict(deadline_s=0),
        dict(questions=[{"id": "q", "kind": "choice", "text": "t", "options": ["a"]}]),
        dict(questions=[{"id": "q", "kind": "score", "text": "t", "levels": ["x", "y"], "options": ["a", "b"]}]),
        dict(questions=[{"id": "__gfql_x__", "kind": "predicate", "text": "t"}]), dict(questions=[Q_PRED, Q_PRED]),
        dict(budget={"max_requests": 0}), dict(budget={"max_cost_usd": 1.0}), dict(evidence_cols=["name", "name"]),
        dict(questions=[{"id": "a", "kind": "predicate", "text": "t"}, {"id": "a__p", "kind": "predicate", "text": "t"}]),
    ])
    def test_e201_before_the_kernel(self, overrides: Dict[str, object]) -> None:
        provider = counting()
        with pytest.raises(GFQLTypeError) as info:
            run(provider, [call("sem_decide", decide_params(**overrides))])
        assert info.value.code == ErrorCode.E201 and provider.answers == 0

    @pytest.mark.parametrize("overrides", [dict(evidence_cols=["missing"]), dict(evidence_cols=("missing",)),
                                           dict(id_col="missing"), dict(evidence_cols=None, template="{missing}")])
    def test_e301_missing_columns(self, overrides: Dict[str, object]) -> None:
        provider = counting()
        with pytest.raises(GFQLSchemaError):
            run(provider, [call("sem_decide", decide_params(**overrides))])
        assert provider.answers == 0

    @pytest.mark.parametrize("overrides,limits,graph_nodes", [
        (dict(template="{name}"), {}, NODES), (dict(evidence_cols=None), {}, NODES),
        (dict(evidence_cols=None, template="{0}"), {}, NODES), (dict(evidence_cols=None, template="{name!r}"), {}, NODES),
        (dict(questions=[{"id": "c", "kind": "choice", "text": "t", "options": ["a", "b", "c"]}]),
         dict(max_choice_options=2), NODES),
        ({}, dict(supported_kinds=("choice",)), NODES), ({}, {}, NODES.assign(co__status="old")),
    ])
    def test_e306_kernel_value_errors(self, overrides: Dict[str, object], limits: dict, graph_nodes: pd.DataFrame) -> None:
        provider = counting(capabilities=caps(**limits))
        with pytest.raises(GFQLCallExecutionError) as info:
            run(provider, [call("sem_decide", decide_params(**overrides))], g=graphistry.nodes(graph_nodes, "id"))
        assert isinstance(info.value.__cause__, ValueError) and provider.answers == 0

    @pytest.mark.parametrize("overrides", [dict(evidence_cols=["missing"]), dict(template="{name}")])
    def test_polars_kernel_errors_are_e306(self, overrides: Dict[str, object]) -> None:
        provider = counting()
        with pytest.raises(GFQLCallExecutionError):
            run(provider, [call("sem_decide", decide_params(**overrides))], g=polars_graph(), engine="polars")
        assert provider.answers == 0


class TestBindingScope:

    def test_unbound_alias(self) -> None:
        with pytest.raises(ValueError, match="alias 'x' is not bound"):
            graph().sem_decide([Q_PRED], "x", task_id="t", evidence_cols=["name"])
        with pytest.raises(GFQLCallExecutionError) as info:
            graph().gfql([call("sem_decide", decide_params(provider="x"))])
        assert isinstance(info.value.__cause__, ValueError) and "'x'" in str(info.value.__cause__)

    def test_overlay_copies_and_threads(self) -> None:
        a, b, c = counting(), counting(), counting()
        mapping = {"x": a, "y": b}
        with bind_providers(mapping):
            mapping["x"] = c
            assert resolve_provider("x") is a
            with bind_providers({"x": c}):
                assert resolve_provider("x") is c and resolve_provider("y") is b
            assert resolve_provider("x") is a
            assert contextvars.copy_context().run(lambda: resolve_provider("x")) is a
            seen: List[object] = []

            def probe() -> None:
                try:
                    seen.append(resolve_provider("x"))
                except ValueError:
                    seen.append(None)

            thread = threading.Thread(target=probe)
            thread.start()
            thread.join()
            assert seen == ([a] if getattr(sys.flags, "thread_inherit_context", 0) else [None])
        with pytest.raises(ValueError, match="not bound"):
            resolve_provider("x")

    @pytest.mark.parametrize("bad", [{"": counting()}, {"p": object()}, [("p", counting())]])
    def test_bind_rejects_bad_aliases_and_providers(self, bad: object) -> None:
        with pytest.raises(ValueError):
            with bind_providers(bad):  # type: ignore[arg-type]
                pass


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
        run(counting(), [call("sem_decide", decide_params())], policy=self._hooks(events))
        sem = [e for e in events if e["op"] == "sem_decide"]
        assert [e["phase"] for e in sem] == ["precall", "postcall"] and sem[1]["success"] is True
        assert sem[0]["params"]["provider"] == "p" and "acme" not in json.dumps(sem[0]["params"])  # type: ignore[index]

    def test_veto_stops_before_the_provider(self) -> None:
        events: List[Dict[str, object]] = []
        provider = counting()
        with pytest.raises(PolicyException):
            run(provider, [call("sem_decide", decide_params())], policy=self._hooks(events, veto=True))
        assert provider.answers == 0 and not [e for e in events if e["phase"] == "postcall" and e["op"] == "sem_decide"]

    def test_unbound_alias_postcall_reports_failure(self) -> None:
        events: List[Dict[str, object]] = []
        with pytest.raises(GFQLCallExecutionError):
            graph().gfql([call("sem_decide", decide_params(provider="nope"))], policy=self._hooks(events))
        post = [e for e in events if e["phase"] == "postcall" and e["op"] == "sem_decide"]
        assert post and post[0]["success"] is False and "not bound" in str(post[0]["error"])

    def test_early_errors_and_direct_methods_fire_no_hooks(self) -> None:
        events: List[Dict[str, object]] = []
        with pytest.raises(GFQLTypeError):
            run(counting(), [call("sem_decide", decide_params(min_p=1.5))], policy=self._hooks(events))
        with pytest.raises(GFQLSchemaError):
            run(counting(), [call("sem_decide", decide_params(evidence_cols=["missing"]))], policy=self._hooks(events))
        assert not [e for e in events if e["op"] == "sem_decide"]


class TestRemoteRefusal:

    SEM = {"type": "Call", "function": "sem_decide", "params": decide_params()}

    @pytest.mark.parametrize("query", [
        [call("sem_decide", decide_params())], Chain([call("sem_decide", decide_params())]),
        let({"a": [n()], "b": ref("a", [call("sem_decide", decide_params())])}), {"type": "Chain", "chain": [SEM]},
        {"chain": [SEM, {"type": "Call", "function": "get_degrees", "params": {"bogus": 1}}]},
        {"type": "Chain", "chain": (SEM,)}, {"type": "Chain", "chain": [call("sem_decide", decide_params())]},
    ])
    @pytest.mark.parametrize("method", ["gfql_remote", "gfql_remote_shape", "chain_remote", "chain_remote_shape"])
    def test_refused_before_any_network(self, query: object, method: str) -> None:
        g = graph()
        with mock.patch("graphistry.compute.chain_remote.requests.post") as post, \
                mock.patch.object(type(g), "upload") as upload, mock.patch.object(g._pygraphistry, "refresh") as refresh:
            with pytest.raises(GFQLUnsupportedError) as info:
                getattr(g, method)(query, api_token="t", dataset_id="ds")
            assert info.value.code == ErrorCode.E110 and info.value.context["value"] == ["sem_decide"]
            with pytest.raises(GFQLUnsupportedError):
                getattr(g, method)(query, validate=False)
            assert post.call_count == upload.call_count == refresh.call_count == 0

    def test_other_queries_reach_post_with_their_own_errors(self) -> None:
        g = graph()
        malformed = {"type": "Chain", "chain": [{"type": "Call", "function": ["get_degrees"], "params": {}}]}
        with mock.patch("graphistry.compute.chain_remote.requests.post", side_effect=RuntimeError("sent")) as post:
            with pytest.raises(GFQLTypeError) as info:
                g.gfql_remote(malformed, api_token="t", dataset_id="ds")
            assert info.value.context.get("field") == "function"
            for query in ([n()], malformed):
                with pytest.raises(RuntimeError, match="sent"):
                    g.gfql_remote(query, api_token="t", dataset_id="ds", validate=False)
            assert post.call_count == 2


class TestAutoEffectBarrier:

    @pytest.mark.parametrize("shape", ["late", "traversal", "let"])
    def test_late_decline_raises_after_one_pass(self, shape: str) -> None:
        sd = call("sem_decide", decide_params())
        query = {"late": [sd] + SEL, "traversal": [n(), e_forward(), n(), sd] + SEL,
                 "let": let({"a": [sd], "b": ref("a", SEL)})}[shape]
        provider = counting()
        with pytest.raises(GFQLCallExecutionError) as info:
            run(provider, query, g=polars_graph())
        assert info.value.code == ErrorCode.E306 and info.value.context["field"] == "engine"
        assert isinstance(info.value.__cause__, NotImplementedError)
        assert provider.answers == 3

    def test_ledger_resets_and_a_raising_provider_still_counts(self) -> None:
        from graphistry.compute.gfql import effects

        class Raising(Recording):
            def decide_batch(self, requests, *, deadline_s, budget):  # type: ignore[no-untyped-def]
                super().decide_batch(requests, deadline_s=deadline_s, budget=budget)
                raise RuntimeError("provider down after billing")

        raising = Raising()
        with pytest.raises(GFQLCallExecutionError):
            run(raising, [call("sem_decide", decide_params())] + SEL, g=polars_graph())
        assert raising.answers == 3 and effects._LEDGER.get() is None

    @pytest.mark.parametrize("query,engine,answers,result", [
        ([call("sem_decide", decide_params())], "auto", 3, "polars"),
        ([call("sem_decide", decide_params())], "polars", 3, "polars"),
        ([call("sem_decide", decide_params()), n(), e_forward(), n()], "auto", 3, "pandas"),
        ([call("sem_decide", decide_params(deadline_s=1e-9))] + SEL, "auto", 0, "pandas"),
        ([call("prune_self_edges")] + SEL, "auto", 0, "pandas"),
        ([call("sem_decide", decide_params())] + SEL, "polars", 3, NotImplementedError),
    ])
    def test_runs_once_unless_a_decline_follows_a_call(self, query: list, engine: str, answers: int,
                                                       result: object) -> None:
        provider = counting()
        if isinstance(result, type):
            with pytest.raises(result):
                run(provider, query, g=polars_graph(), engine=engine)
        else:
            out = run(provider, query, g=polars_graph(), engine=engine)
            assert type(out._nodes).__module__.startswith(result)
        assert provider.answers == answers

    def test_cudf_auto_arm_refuses_rerun_after_effect(self) -> None:
        import graphistry.compute.gfql_unified as gu

        def fake_route(self, query, **kwargs):  # type: ignore[no-untyped-def]
            self.gfql(query, engine="pandas")
            raise NotImplementedError("declined after the call")

        provider = counting()
        with mock.patch.object(gu, "_is_cudf_frame", return_value=True), \
                mock.patch.object(gu, "_polars_gpu_probe", return_value=True), \
                mock.patch.object(gu, "_auto_cudf_polars_gpu_route", side_effect=fake_route):
            with pytest.raises(GFQLCallExecutionError) as info:
                run(provider, [call("sem_decide", decide_params())])
        assert isinstance(info.value.__cause__, NotImplementedError) and provider.answers == 3

    def test_concurrent_auto_queries_keep_separate_ledgers(self) -> None:
        results: List[object] = []

        def worker(fail: bool, g: Plottable) -> None:
            provider = counting()
            try:
                run(provider, [call("sem_decide", decide_params())] + (SEL if fail else []), g=g)
                results.append(("ok", provider.answers))
            except GFQLCallExecutionError:
                results.append(("e306", provider.answers))

        threads = [threading.Thread(target=worker, args=(i % 2 == 0, polars_graph())) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(results) == [("e306", 3), ("e306", 3), ("ok", 3), ("ok", 3)]


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_declared_schema_effects_match_written_columns(kind: str) -> None:
    from graphistry.compute.gfql.schema_effects import schema_effect_for_call
    questions = [Q_PRED, {"id": "grp", "kind": "choice", "text": "Which?", "options": ["a", "b"]}]
    params = decide_params(questions=questions, kind=kind, evidence_cols=["w"] if kind == "edges" else ["name"])
    out = run(CallableDecisionProvider(lambda e, q: True if q.kind == "predicate" else "a"), [call("sem_decide", params)])
    table, base = (out._edges, EDGES) if kind == "edges" else (out._nodes, NODES)
    written = [col for col in table.columns if col not in base.columns]
    assert SAFELIST_V1["sem_decide"]["schema_effects"][f"adds_{kind[:-1]}_cols"](params) == written
    effect = schema_effect_for_call("sem_decide", params)
    typed = effect.adds_edge_properties if kind == "edges" else effect.adds_node_properties
    assert list(typed) == written
    for name, logical in typed.items():
        assert str(table[name].dtype) in {"bool": ("boolean",), "float64": ("float64",), "string": ("object", "str")}[logical.kind]
    assert SAFELIST_V1["sem_decide"]["required_params"] == {"questions", "provider", "task_id"}


def test_effect_ledgers_are_per_context_and_nest() -> None:
    from graphistry.compute.gfql.effects import effect_scope, record_effect
    meet = threading.Barrier(2, timeout=10)
    seen: Dict[str, List[str]] = {}

    def worker(name: str) -> None:
        with effect_scope() as ledger:
            record_effect(name)
            meet.wait()
            with effect_scope() as inner:
                assert inner is ledger
            seen[name] = list(ledger)

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == {"a": ["a"], "b": ["b"]}


def test_a_decline_after_an_earlier_effect_in_the_same_scope_still_falls_back() -> None:
    from graphistry.compute.gfql.effects import effect_scope, record_effect
    with effect_scope():
        record_effect("earlier")
        out = polars_graph().gfql([call("prune_self_edges")] + SEL)
    assert type(out._nodes).__module__.startswith("pandas")
