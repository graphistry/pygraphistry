"""Cypher-native sem_decide / sem_map: ``CALL graphistry.sem_*`` and the lifted ``WHERE sem_decide(...)``.

Everything runs on a $0 callable provider. pandas runs in the core lane; Polars runs in the Polars lane.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLSyntaxError, GFQLUnsupportedError, GFQLValidationError
from graphistry.compute.gfql.cypher import compile_cypher, cypher_to_gfql
from graphistry.compute.semantic import MemorySemanticCache, bind_providers, semantic_trace
from graphistry.Plottable import Plottable
from graphistry.tests.compute.semantic.semantic_test_utils import Recording, RecordingGenerator

NODES = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "text": ["web inc", "db", "lib inc", "tool", "lib inc"],
    "tier": ["prod", "prod", "dev", "dev", "prod"],
})
EDGES = pd.DataFrame({"s": [1, 2, 1, 2], "d": [3, 4, 5, 4]})
ENGINES = ["pandas", "polars"]


def graph(engine: str = "pandas") -> Plottable:
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    return graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")


def inc(**kwargs: object) -> Recording:
    return Recording(lambda evidence, q: "inc" in str(evidence), **kwargs)  # type: ignore[arg-type]


def rows(result: Plottable) -> pd.DataFrame:
    frame = result._nodes
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame  # hygiene-ok: dynamic-attr -- test helper


def run(query: str, engine: str = "pandas", provider: Optional[Recording] = None, **kwargs: object) -> pd.DataFrame:
    with bind_providers({"default": provider or inc()}):
        return rows(graph(engine).gfql(query, engine=engine, **kwargs))  # type: ignore[arg-type]


HERO = """
MATCH (a)-[]->(b)
WITH a.id AS aid, b.id AS bid, b.text AS text
CALL graphistry.sem_decide({question: $question, evidence: ['text']})
YIELD verdict, score, status
RETURN aid, bid, text, verdict, status
ORDER BY bid
"""
PARAMS: Dict[str, object] = {"question": "Is it an inc?"}


@pytest.mark.parametrize("engine", ENGINES)
class TestCallAfterWith:

    def test_keeps_upstream_columns_and_adds_the_yielded_ones(self, engine: str) -> None:
        out = run(HERO, engine, params=PARAMS)
        assert list(out.columns) == ["aid", "bid", "text", "verdict", "status"]
        assert out["bid"].tolist() == [3, 4, 4, 5]
        assert out["verdict"].tolist() == [True, False, False, True]
        assert set(out["status"]) == {"ok"}

    def test_matches_the_method_on_the_same_rows(self, engine: str) -> None:
        cypher = run(HERO, engine, params=PARAMS).drop(columns=["status"])
        candidates = graph(engine).gfql("MATCH (a)-[]->(b) RETURN a.id AS aid, b.id AS bid, b.text AS text", engine=engine)
        with bind_providers({"default": inc()}):
            method = rows(candidates.sem_decide(
                [{"id": "verdict", "kind": "predicate", "text": "Is it an inc?"}], "default", task_id="t", evidence_cols=["text"]))
        method = method.rename(columns={"verdict": "verdict"})[["aid", "bid", "text", "verdict"]].sort_values("bid")
        pd.testing.assert_frame_equal(cypher.reset_index(drop=True), method.reset_index(drop=True), check_dtype=False)

    def test_default_yield_is_verdict_score_status(self, engine: str) -> None:
        out = run("MATCH (a) WITH a.id AS id, a.text AS text "
                  "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']})", engine)
        assert list(out.columns) == ["id", "text", "verdict", "score", "status"]

    def test_yield_alias_and_subset(self, engine: str) -> None:
        out = run("MATCH (a) WITH a.id AS id, a.text AS text "
                  "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict AS is_inc, status AS st "
                  "RETURN id, is_inc, st ORDER BY id", engine)
        assert list(out.columns) == ["id", "is_inc", "st"]
        assert out["is_inc"].tolist() == [True, False, True, False, True]

    def test_downstream_where_with_return_order_skip_limit_distinct(self, engine: str) -> None:
        out = run("MATCH (a) WITH a.id AS id, a.text AS text "
                  "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict "
                  "WHERE verdict WITH id, text, verdict WHERE id > 1 "
                  "RETURN DISTINCT text ORDER BY text DESC SKIP 0 LIMIT 5", engine)
        assert out["text"].tolist() == ["lib inc"]

    def test_yield_where_filters_before_the_return(self, engine: str) -> None:
        out = run("MATCH (a) WITH a.id AS id, a.text AS text "
                  "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict WHERE verdict "
                  "RETURN id ORDER BY id", engine)
        assert out["id"].tolist() == [1, 3, 5]

    def test_aggregates_after_the_call(self, engine: str) -> None:
        out = run("MATCH (a) WITH a.id AS id, a.text AS text "
                  "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict "
                  "RETURN verdict, count(*) AS n ORDER BY verdict", engine)
        assert out["n"].tolist() == [2, 3]

    def test_whole_options_map_as_one_param(self, engine: str) -> None:
        out = run("MATCH (a) WITH a.id AS id, a.text AS text CALL graphistry.sem_decide($opts) YIELD verdict RETURN id, verdict",
                  engine, params={"opts": {"question": "inc?", "evidence": ["text"]}})
        assert out.sort_values("id")["verdict"].tolist() == [True, False, True, False, True]

    def test_nested_params_in_map_and_list(self, engine: str) -> None:
        out = run("MATCH (a) WITH a.id AS id, a.text AS text "
                  "CALL graphistry.sem_decide({question: $q, evidence: $cols, max_usd: $cap}) YIELD verdict RETURN id, verdict",
                  engine, params={"q": "inc?", "cols": ["text"], "cap": 5.0})
        assert out.sort_values("id")["verdict"].tolist() == [True, False, True, False, True]

    def test_raw_questions_list_yields_per_question_columns(self, engine: str) -> None:
        out = run("MATCH (a) WITH a.id AS id, a.text AS text "
                  "CALL graphistry.sem_decide({questions: [{id: 'q1', kind: 'predicate', text: 'inc?'}], evidence: ['text']}) "
                  "YIELD q1, q1__status RETURN id, q1, q1__status ORDER BY id", engine)
        assert out["q1"].tolist() == [True, False, True, False, True]
        assert set(out["q1__status"]) == {"ok"}

    def test_choice_question(self, engine: str) -> None:
        provider = Recording(lambda evidence, q: "a" if "inc" in str(evidence) else "b")
        out = run("MATCH (a) WITH a.id AS id, a.text AS text "
                  "CALL graphistry.sem_decide({question: 'which?', options: ['a', 'b'], evidence: ['text']}) YIELD verdict AS grp "
                  "RETURN id, grp ORDER BY id", engine, provider)
        assert out["grp"].tolist() == ["a", "b", "a", "b", "a"]

    def test_template_instead_of_evidence(self, engine: str) -> None:
        provider = inc()
        run("MATCH (a) WITH a.id AS id, a.text AS text "
            "CALL graphistry.sem_decide({question: 'inc?', template: 'Name: {text}'}) YIELD verdict RETURN id", engine, provider)
        assert provider.answers == 4  # five rows, one "lib inc" repeat answered once

    def test_provider_alias_selects_a_bound_provider(self, engine: str) -> None:
        other = Recording(lambda evidence, q: True)
        with bind_providers({"default": inc(), "other": other}):
            out = rows(graph(engine).gfql(
                "MATCH (a) WITH a.id AS id, a.text AS text "
                "CALL graphistry.sem_decide({question: 'x', evidence: ['text'], provider: 'other'}) YIELD verdict RETURN id, verdict",
                engine=engine))
        assert out["verdict"].all() and other.answers > 0


@pytest.mark.parametrize("engine", ENGINES)
class TestObservability:

    def test_semantic_trace_records_the_call_with_dedupe(self, engine: str) -> None:
        with semantic_trace() as runs:
            run(HERO, engine, params=PARAMS)
        assert [(r["op"], r["total_rows"], r["deduped_rows"]) for r in runs] == [("sem_decide", 4, 2)]

    def test_one_stage_asks_once_per_distinct_evidence(self, engine: str) -> None:
        provider = inc()
        with semantic_trace() as runs:
            run(HERO, engine, provider, params=PARAMS)
        assert len(runs) == 1 and runs[0]["provider_rows"] == 2 and provider.answers == 2

    def test_cache_bound_with_the_providers_serves_a_repeat(self, engine: str) -> None:
        provider, cache = inc(fingerprint="inc-v1"), MemorySemanticCache()
        for _ in range(2):
            with bind_providers({"default": provider}, cache=cache):
                graph(engine).gfql(HERO, params=PARAMS, engine=engine)
        assert provider.answers == 2

    def test_max_usd_reaches_the_kernel_budget(self, engine: str) -> None:
        compiled = compile_cypher(
            "MATCH (a) WITH a.text AS text CALL graphistry.sem_decide({question: 'x', evidence: ['text'], max_usd: $cap, max_requests: 3})",
            params={"cap": 0.25}, _warn_deprecated=False)
        assert compiled.procedure_call.call_params["budget"] == {"max_cost_usd": 0.25, "max_requests": 3}  # type: ignore[union-attr]


@pytest.mark.parametrize("engine", ENGINES)
class TestSemMapAndStandalone:

    def test_sem_map_yields_the_generated_column(self, engine: str) -> None:
        generator = RecordingGenerator(lambda evidence, instruction: "LOUD")
        with bind_providers({"gen": generator}):
            out = rows(graph(engine).gfql(
                "MATCH (a) WITH a.id AS id, a.text AS text "
                "CALL graphistry.sem_map({instruction: 'shout', out_col: 'loud', evidence: ['text'], provider: 'gen'}) "
                "YIELD loud RETURN id, loud ORDER BY id", engine=engine))
        assert out["loud"].tolist() == ["LOUD"] * 5

    def test_standalone_reads_the_node_table(self, engine: str) -> None:
        out = run("CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict "
                  "RETURN id, text, verdict ORDER BY id", engine)
        assert out["verdict"].tolist() == [True, False, True, False, True]

    def test_standalone_yield_where(self, engine: str) -> None:
        out = run("CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict WHERE verdict "
                  "RETURN id ORDER BY id", engine)
        assert out["id"].tolist() == [1, 3, 5]


@pytest.mark.parametrize("engine", ENGINES)
class TestLiftedWhere:

    LIFTED = ("MATCH (a)-[]->(b) WHERE a.tier = 'prod' AND sem_decide($question, {text: b.text}) "
              "RETURN a.id AS aid, b.id AS bid ORDER BY bid")

    def test_structural_predicates_run_first_and_one_batch_follows(self, engine: str) -> None:
        provider = inc()
        with semantic_trace() as runs:
            out = run(self.LIFTED, engine, provider, params=PARAMS)
        assert out["bid"].tolist() == [3, 5]
        assert len(runs) == 1 and runs[0]["total_rows"] == 4
        assert provider.answers == 2  # four prod edges reach two distinct texts

    def test_the_model_sees_only_rows_that_pass_the_structural_term(self, engine: str) -> None:
        provider = inc()
        with semantic_trace() as runs:
            out = run("MATCH (a)-[]->(b) WHERE b.tier = 'dev' AND sem_decide('inc?', {text: b.text}) RETURN b.id AS bid ORDER BY bid",
                      engine, provider)
        assert out["bid"].tolist() == [3]
        assert runs[0]["total_rows"] == 3 and provider.answers == 2  # the prod-tier edge never reaches the model

    def test_equals_the_call_form(self, engine: str) -> None:
        call_form = run(
            "MATCH (a)-[]->(b) WHERE a.tier = 'prod' WITH a.id AS aid, b.id AS bid, b.text AS text "
            "CALL graphistry.sem_decide({question: $question, evidence: ['text']}) YIELD verdict WHERE verdict "
            "RETURN aid, bid ORDER BY bid", engine, params=PARAMS)
        pd.testing.assert_frame_equal(run(self.LIFTED, engine, params=PARAMS), call_form)

    def test_third_argument_options_reach_the_call(self, engine: str) -> None:
        compiled = compile_cypher(
            "MATCH (a)-[]->(b) WHERE sem_decide('x', {t: b.text}, {max_usd: $cap, min_p: 0.5}) RETURN b.id AS id",
            params={"cap": 1.0}, _warn_deprecated=False)
        params = compiled.procedure_call.call_params  # type: ignore[union-attr]
        assert params["budget"] == {"max_cost_usd": 1.0} and params["min_p"] == 0.5

    def test_a_string_that_spells_sem_decide_is_only_a_string(self, engine: str) -> None:
        out = run("MATCH (a) WHERE a.text CONTAINS 'sem_decide(' RETURN a.id AS id", engine)
        assert out.empty

    def test_unaliased_return_keeps_its_name(self, engine: str) -> None:
        out = run("MATCH (a)-[]->(b) WHERE sem_decide('x', {t: b.text}) RETURN b.id", engine)
        assert list(out.columns) == ["b.id"]


class TestWhatTheModelReceives:

    def test_the_question_text_and_every_evidence_column_reach_the_provider(self) -> None:
        seen: List[tuple] = []
        provider = Recording(lambda evidence, q: seen.append((dict(evidence), q.text)) or True)  # type: ignore[arg-type,func-returns-value]
        run("MATCH (a) WITH a.id AS id, a.text AS text, a.tier AS tier "
            "CALL graphistry.sem_decide({question: $q, evidence: ['text', 'tier']}) YIELD verdict RETURN id",
            provider=provider, params={"q": "Is it here?"})
        assert {text for _, text in seen} == {"Is it here?"}
        assert all(set(evidence) == {"text", "tier"} for evidence, _ in seen)

    def test_lifted_question_and_evidence_reach_the_provider(self) -> None:
        seen: List[tuple] = []
        provider = Recording(lambda evidence, q: seen.append((dict(evidence), q.text)) or True)  # type: ignore[arg-type,func-returns-value]
        run("MATCH (a) WHERE sem_decide('Is it here?', {text: a.text, tier: a.tier}) RETURN a.id AS id", provider=provider)
        assert {text for _, text in seen} == {"Is it here?"} and all(set(e) == {"text", "tier"} for e, _ in seen)


class TestEdgeCases:

    @pytest.mark.parametrize("engine", ENGINES)
    def test_a_duplicate_id_column_is_not_the_row_id(self, engine: str) -> None:
        out = run("MATCH (a)-[]->(b) WITH a.id AS id, b.text AS text "
                  "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict RETURN id, text, verdict", engine)
        assert len(out) == 4

    @pytest.mark.parametrize("engine", ENGINES)
    def test_zero_rows_from_the_prefix(self, engine: str) -> None:
        out = run("MATCH (a {id: 99}) WITH a.id AS id, a.text AS text "
                  "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict RETURN id, verdict", engine)
        assert out.empty and list(out.columns) == ["id", "verdict"]

    def test_standalone_keeps_columns_that_hold_mixed_values(self) -> None:
        table = NODES.assign(tags=[["a"], ["b"], [], ["c"], ["d"]], mixed=[1, "a", None, 2.5, "b"])
        with bind_providers({"default": inc()}):
            out = rows(graphistry.nodes(table, "id").gfql(
                "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict RETURN id, mixed, tags, verdict ORDER BY id"))
        assert out["verdict"].tolist() == [True, False, True, False, True] and out["mixed"].tolist()[1] == "a"

    def test_standalone_reads_columns_from_the_table_not_the_cache(self) -> None:
        query = "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict RETURN id, tier, verdict"
        with bind_providers({"default": inc()}):
            graphistry.nodes(NODES, "id").gfql(query)
            with pytest.raises(GFQLValidationError):
                graphistry.nodes(NODES.drop(columns=["tier"]), "id").gfql(query)

    def test_a_string_that_spells_sem_decide_in_return_is_only_a_string(self) -> None:
        out = run("MATCH (a) RETURN a.id AS id, 'sem_decide(' AS s ORDER BY id")
        assert out["s"].tolist() == ["sem_decide("] * 5

    def test_lifted_order_by_the_return_expression(self) -> None:
        out = run("MATCH (a)-[]->(b) WHERE sem_decide('inc?', {t: b.text}) RETURN b.id AS bid ORDER BY b.id DESC")
        assert out["bid"].tolist() == [5, 3]

    def test_lifted_term_beside_structural_or_and_not(self) -> None:
        out = run("MATCH (a)-[]->(b) WHERE (b.id = 3 OR b.id = 5) AND NOT b.id = 99 AND sem_decide('inc?', {t: b.text}) "
                  "RETURN b.id AS bid ORDER BY bid")
        assert out["bid"].tolist() == [3, 5]

    def test_lifted_evidence_with_commas_colons_quotes_and_backslashes(self) -> None:
        out = run("MATCH (a) WHERE sem_decide('Is it: \\'a, b\\'?', {t: a.text + ', ' + 'x:y', u: a.id}, {max_usd: 1}) RETURN a.id AS id ORDER BY id")
        assert out["id"].tolist() == [1, 3, 5]


class TestWaveTwoFixes:

    def native(self, fn_p: Dict[bool, float]) -> Recording:
        """True for every row, with probability 0.6 for an 'inc' text and 0.9 for another."""
        def answer(evidence: object, question: object) -> object:
            p = fn_p[("inc" in str(evidence))]
            return True, {"true": p, "false": 1 - p}
        return Recording(answer, probability_kind="native_decision")  # type: ignore[arg-type]

    @pytest.mark.parametrize("engine", ENGINES)
    def test_min_p_removes_unsure_rows_from_the_lifted_where(self, engine: str) -> None:
        provider = self.native({True: 0.6, False: 0.9})
        out = run("MATCH (a) WHERE sem_decide('x', {t: a.text}, {min_p: 0.8}) RETURN a.id AS id ORDER BY id", engine, provider)
        assert out["id"].tolist() == [2, 4]

    def test_min_p_in_the_call_form_leaves_the_value_and_marks_uncertain(self) -> None:
        provider = self.native({True: 0.6, False: 0.9})
        out = run("MATCH (a) WITH a.id AS id, a.text AS text CALL graphistry.sem_decide({question: 'x', evidence: ['text'], min_p: 0.8}) "
                  "YIELD verdict, status RETURN id, verdict, status ORDER BY id", provider=provider)
        assert out["status"].tolist() == ["uncertain", "ok", "uncertain", "ok", "uncertain"]
        assert out["verdict"].tolist() == [True] * 5

    @pytest.mark.parametrize("structural", ["(b)-[]->(a)", "NOT (a)-[]->(b)", "EXISTS { (b)-[]->() }", "b.id > 0 AND (b)-[]->(a)"])
    def test_a_pattern_term_with_the_lifted_where_fails_closed_not_silently(self, structural: str) -> None:
        message = TestRejected.error(f"MATCH (a)-[]->(b) WHERE {structural} AND sem_decide('x', {{t: b.text}}) RETURN b.id AS id")
        assert "pattern or EXISTS" in message

    def test_validate_and_union_accept_a_standalone_call(self) -> None:
        query = "CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD verdict RETURN id, verdict"
        graph().gfql_validate(query)
        out = run(f"{query} UNION ALL {query}")
        assert len(out) == 10

    def test_compile_cypher_says_what_a_standalone_yield_where_needs(self) -> None:
        with pytest.raises(GFQLValidationError, match="needs the node table"):
            compile_cypher("CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD verdict WHERE verdict RETURN verdict",
                           _warn_deprecated=False)

    def test_overwrite_is_not_an_option_of_the_call(self) -> None:
        assert "overwrite" in TestRejected.error(
            "MATCH (a) WITH a.id AS id, a.text AS text CALL graphistry.sem_decide({question: 'x', evidence: ['text'], overwrite: true}) YIELD verdict RETURN id")

    @pytest.mark.parametrize("carried, call, needle", [
        ("1 AS verdict__meta", "{question: 'x', evidence: ['text']}", "verdict__meta"),
        ("1 AS q2", "{questions: [{id: 'q1', kind: 'predicate', text: 'a'}, {id: 'q2', kind: 'predicate', text: 'b'}], evidence: ['text']}", "q2"),
    ])
    def test_every_written_column_is_checked_against_the_carried_ones(self, carried: str, call: str, needle: str) -> None:
        message = TestRejected.error(
            f"MATCH (a) WITH a.id AS id, a.text AS text, {carried} CALL graphistry.sem_decide({call}) RETURN id")
        assert f"writes the column '{needle}'" in message

    def test_sem_map_status_column_clash_is_named(self) -> None:
        assert "writes the column 'result__status'" in TestRejected.error(
            "MATCH (a) WITH a.id AS id, a.text AS text, 1 AS result__status CALL graphistry.sem_map({instruction: 'x', evidence: ['text']}) YIELD result RETURN id")

    def test_the_compile_cache_does_not_reuse_a_plan_across_tables_with_other_columns(self) -> None:
        query = "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict WHERE verdict"
        plain = NODES[["id", "text"]]
        extra = plain.assign(extra=[["a"], ["b"], [], ["c"], ["d"]])
        with bind_providers({"default": inc()}):
            first = rows(graphistry.nodes(plain, "id").gfql(query))
            second = rows(graphistry.nodes(extra, "id").gfql(query))
        assert "extra" not in first.columns and "extra" in second.columns


class TestWaveThreeFixes:

    def flaky(self) -> Recording:
        def answer(evidence: object, question: object) -> object:
            if "web" in str(evidence):
                raise RuntimeError("provider down")
            return "inc" in str(evidence)
        return Recording(answer)

    LIFTED = "MATCH (a) WHERE sem_decide('x', {t: a.text}) RETURN a.id AS id ORDER BY id"

    @pytest.mark.parametrize("engine", ENGINES)
    def test_a_provider_error_is_not_a_silent_no_match_in_the_lifted_where(self, engine: str) -> None:
        with pytest.raises(GFQLValidationError, match="without an answer"):
            run(self.LIFTED, engine, self.flaky())

    def test_the_lifted_where_can_opt_out_and_the_call_form_shows_the_status(self) -> None:
        out = run("MATCH (a) WHERE sem_decide('x', {t: a.text}, {fail_on_incomplete: false}) RETURN a.id AS id ORDER BY id", provider=self.flaky())
        assert out["id"].tolist() == [3, 5]
        call = run("MATCH (a) WITH a.id AS id, a.text AS text CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) "
                   "YIELD verdict, status RETURN id, status ORDER BY id", provider=self.flaky())
        assert call["status"].tolist()[0] == "error"

    def test_a_budget_stop_is_not_a_silent_no_match_either(self) -> None:
        with pytest.raises(GFQLValidationError, match="without an answer"):
            run("MATCH (a) WHERE sem_decide('x', {t: a.text}, {max_requests: 1}) RETURN a.id AS id")

    def test_empty_evidence_and_a_requested_min_p_are_not_failures(self) -> None:
        table = NODES.assign(text=[None, "db", "lib inc", "tool", "lib inc"])
        with bind_providers({"default": TestWaveTwoFixes.native(TestWaveTwoFixes(), {True: 0.6, False: 0.9})}):
            out = rows(graphistry.nodes(table, "id").gfql(
                "MATCH (a) WHERE sem_decide('x', {t: a.text}, {min_p: 0.8}) RETURN a.id AS id ORDER BY id"))
        assert out["id"].tolist() == [2, 4]

    def test_fail_on_incomplete_must_be_a_boolean(self) -> None:
        assert "true or false" in TestRejected.error(
            "MATCH (a) WITH a.id AS id, a.text AS text CALL graphistry.sem_decide({question: 'x', evidence: ['text'], fail_on_incomplete: 'yes'}) YIELD verdict RETURN id")

    @pytest.mark.parametrize("extra", ["out_col: 'v', score_col: 'v'", "status_col: 'verdict'"])
    def test_colliding_output_names_are_rejected(self, extra: str) -> None:
        assert "different columns" in TestRejected.error(
            f"MATCH (a) WITH a.id AS id, a.text AS text CALL graphistry.sem_decide({{question: 'x', evidence: ['text'], {extra}}}) YIELD verdict RETURN id")

    def test_levels_and_options_together_are_rejected(self) -> None:
        assert "cannot be given together" in TestRejected.error(
            "MATCH (a) WITH a.id AS id, a.text AS text CALL graphistry.sem_decide({question: 'x', evidence: ['text'], levels: ['a', 'b'], options: ['p', 'q']}) YIELD verdict RETURN id")

    def test_standalone_yield_where_with_non_identifier_column_names(self) -> None:
        table = NODES.rename(columns={"tier": "my col"})
        with bind_providers({"default": inc()}):
            out = rows(graphistry.nodes(table, "id").gfql(
                "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict WHERE verdict RETURN id ORDER BY id"))
        assert out["id"].tolist() == [1, 3, 5]

    def test_standalone_with_a_non_string_column_label(self) -> None:
        table = NODES.copy()
        table[5] = 1
        with bind_providers({"default": inc()}):
            out = rows(graphistry.nodes(table, "id").gfql(
                "CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict RETURN id, verdict ORDER BY id"))
        assert out["verdict"].tolist() == [True, False, True, False, True]

    def test_the_procedure_name_may_have_spaces_around_the_dot(self) -> None:
        out = run("CALL graphistry . sem_decide({question: 'inc?', evidence: ['text']}) YIELD verdict RETURN id, verdict ORDER BY id")
        assert out["verdict"].tolist() == [True, False, True, False, True]

    def test_lifted_order_by_an_unaliased_expression_is_rejected_clearly(self) -> None:
        assert "named with AS" in TestRejected.error("MATCH (a) WHERE sem_decide('x', {t: a.text}) RETURN a.id ORDER BY a.id DESC")


class TestRejected:

    @staticmethod
    def error(query: str, **kwargs: object) -> str:
        with pytest.raises((GFQLValidationError, GFQLSyntaxError)) as info:
            run(query, **kwargs)  # type: ignore[arg-type]
        return str(info.value)

    HEAD = "MATCH (a) WITH a.id AS id, a.text AS text "

    @pytest.mark.parametrize("where", [
        "a.id < 3 OR sem_decide('x', {t: b.text})",
        "NOT sem_decide('x', {t: b.text})",
        "sem_decide('x', {t: b.text}) = true",
        "(sem_decide('x', {t: b.text}) AND a.id > 1) OR a.id = 2",
        "sem_decide('x', {t: b.text}) AND sem_decide('y', {t: b.text})",
    ])
    def test_lifted_where_fails_closed_outside_a_top_level_and(self, where: str) -> None:
        message = self.error(f"MATCH (a)-[]->(b) WHERE {where} RETURN b.id AS id")
        assert "CALL graphistry.sem_decide" in message

    @pytest.mark.parametrize("tail", [
        "RETURN count(*) AS n", "RETURN b.id AS id ORDER BY a.id", "RETURN *",
        "WITH b.id AS id RETURN id",
    ])
    def test_lifted_where_rejects_shapes_it_cannot_carry(self, tail: str) -> None:
        assert "sem_decide" in self.error(f"MATCH (a)-[]->(b) WHERE sem_decide('x', {{t: b.text}}) {tail}")

    def test_sem_decide_outside_where_is_rejected(self) -> None:
        assert "term of the WHERE" in self.error("MATCH (a) WITH a AS n RETURN sem_decide('x', {t: n.text}) AS v")

    @pytest.mark.parametrize("call, needle", [
        ("CALL graphistry.sem_decide({question: 'x', evidence: ['text'], bogus: 1}) YIELD verdict", "bogus"),
        ("CALL graphistry.sem_decide('x') YIELD verdict", "must be a map"),
        ("CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD nope", "nope"),
        ("CALL graphistry.sem_decide({question: 'x', evidence: ['missing']}) YIELD verdict", "missing"),
        ("CALL graphistry.sem_decide({question: 'x'}) YIELD verdict", "evidence"),
        ("CALL graphistry.sem_decide({evidence: ['text']}) YIELD verdict", "question"),
        ("CALL graphistry.sem_decide({question: 'x', questions: [], evidence: ['text']}) YIELD verdict", "question"),
        ("CALL graphistry.sem_decide({question: 'x', evidence: ['text'], budget: 3}) YIELD verdict", "budget"),
        ("CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD verdict, verdict", "twice"),
        ("CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD verdict AS text", "already a column"),
        ("CALL graphistry.sem_map({instruction: 'x', schema: {a: 'b'}, evidence: ['text']}) YIELD result", "schema"),
        ("CALL graphistry.sem_map({evidence: ['text']}) YIELD result", "instruction"),
    ])
    def test_bad_options_and_yield_fail_closed(self, call: str, needle: str) -> None:
        assert needle in self.error(f"{self.HEAD}{call} RETURN id")

    def test_clauses_after_the_call_must_be_row_clauses(self) -> None:
        call = "CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD verdict"
        assert "MATCH before the CALL" in self.error(f"{self.HEAD}{call} MATCH (b) RETURN count(*) AS n")
        assert "MATCH before the CALL" in self.error(f"{call.replace('{', '{', 1)} MATCH (b) RETURN count(*) AS n")

    def test_use_and_graph_bindings_are_rejected_not_ignored(self) -> None:
        call = "WITH a.id AS id, a.text AS text CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD verdict RETURN id"
        assert "USE or GRAPH" in self.error(f"USE nope MATCH (a) {call}")
        assert "USE or GRAPH" in self.error(f"GRAPH g = GRAPH {{ MATCH (a) WHERE a.id > 3 }} USE g MATCH (a) {call}")

    def test_yield_where_is_semantic_only(self) -> None:
        assert "semantic" in self.error("CALL graphistry.degree() YIELD nodeId, degree WHERE degree > 99 RETURN nodeId").lower() \
            or "sem_decide" in self.error("CALL graphistry.degree() YIELD nodeId, degree WHERE degree > 99 RETURN nodeId")

    def test_an_output_column_clash_is_named(self) -> None:
        assert "writes the column 'verdict'" in self.error("MATCH (a) WHERE sem_decide('x', {verdict: a.text}) RETURN a.id AS id")

    def test_duplicate_option_keys_are_rejected(self) -> None:
        assert "twice" in self.error(f"{self.HEAD}CALL graphistry.sem_decide({{question: 'a', evidence: ['text'], evidence: ['id']}}) YIELD verdict RETURN id")

    def test_returning_a_whole_node_is_rejected_with_a_clear_message(self) -> None:
        assert "whole node" in self.error("MATCH (a) WHERE sem_decide('x', {t: a.text}) RETURN a")

    def test_missing_param_names_the_param(self) -> None:
        assert "'q'" in self.error(f"{self.HEAD}CALL graphistry.sem_decide({{question: $q, evidence: ['text']}}) YIELD verdict RETURN id")

    def test_call_needs_a_projection_before_it(self) -> None:
        assert "WITH or RETURN" in self.error("MATCH (a) CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD verdict RETURN verdict")

    def test_projection_before_the_call_must_name_columns(self) -> None:
        assert "plain name" in self.error("MATCH (a) WITH a.id CALL graphistry.sem_decide({question: 'x', evidence: ['id']}) YIELD verdict RETURN verdict")

    def test_only_one_semantic_call_per_query(self) -> None:
        call = "CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD verdict"
        assert self.error(f"{self.HEAD}{call} WITH id, text, verdict AS v {call.replace('verdict', 'verdict AS w')} RETURN id")

    def test_one_chain_translation_declines_a_semantic_call(self) -> None:
        with pytest.raises(GFQLValidationError):
            cypher_to_gfql("MATCH (a) WITH a.text AS text CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD verdict RETURN verdict")

    @pytest.mark.parametrize("query", [
        "MATCH (a) WITH a.text AS text CALL graphistry.sem_decide({question: 'x', evidence: ['text']}) YIELD verdict RETURN verdict",
        "MATCH (a)-[]->(b) WHERE sem_decide('x', {t: b.text}) RETURN b.id AS id",
    ])
    def test_remote_refuses_a_semantic_cypher_query(self, query: str) -> None:
        with pytest.raises(GFQLUnsupportedError, match="run locally only"):
            graph().gfql_remote(query)

    def test_graph_algorithm_calls_keep_their_clause_rules(self) -> None:
        with pytest.raises(GFQLValidationError, match="first clause"):
            graph().gfql("MATCH (a) WITH a.id AS id CALL graphistry.degree() YIELD degree RETURN id")


class TestPolarsAgree:

    @pytest.mark.parametrize("query", [HERO, TestLiftedWhere.LIFTED])
    def test_polars_equals_pandas(self, query: str) -> None:
        pytest.importorskip("polars")
        left, right = run(query, "pandas", params=PARAMS), run(query, "polars", params=PARAMS)
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_dtype=False)

    def test_strict_call_mode_declines_the_off_engine_bridge_as_call_does(self) -> None:
        pytest.importorskip("polars")
        from graphistry.compute.gfql.lazy import set_call_mode

        set_call_mode("strict")
        try:
            with pytest.raises(Exception, match="call_mode='strict'"):
                run(HERO, "polars", params=PARAMS)
            assert run(HERO, "pandas", params=PARAMS)["verdict"].tolist() == [True, False, False, True]
        finally:
            set_call_mode(None)

    def test_a_polars_graph_returns_a_polars_table(self) -> None:
        pl = pytest.importorskip("polars")
        with bind_providers({"default": inc()}):
            result = graph("polars").gfql(HERO, params=PARAMS, engine="polars")
            bare = graph("polars").gfql("CALL graphistry.sem_decide({question: 'inc?', evidence: ['text']})", engine="polars")
        assert isinstance(result._nodes, pl.DataFrame) and isinstance(bare._nodes, pl.DataFrame)
