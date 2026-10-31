"""Stage1 row-token correlation and bag semantics for bounded reentry."""
from collections import Counter

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.api import compile_cypher
from graphistry.compute.gfql.cypher.reentry.naming import REENTRY_ROW_TOKEN_OUTPUT


NODES = pd.DataFrame(
    {
        "id": ["a", "x-red", "x-blue", "x-green", "b-red", "b-blue", "c-red", "c-blue", "q-mid"],
        "payload": [None, "red", "blue", "green", "red", "blue", "red", "blue", "neutral"],
    }
)
EDGES = pd.DataFrame(
    {
        "s": ["a", "a", "a", "a", "a", "a", "a", "a", "a", "q-mid", "q-mid"],
        "d": ["x-red", "x-red", "x-blue", "x-green", "b-red", "b-blue", "c-red", "c-blue", "q-mid", "c-red", "c-blue"],
        "type": ["P", "P", "P", "P", "Q", "Q", "R", "R", "S", "T", "T"],
    }
)
CYCLE_NODES = pd.DataFrame(
    {
        "id": ["a", "x-red", "x-blue", "m", "c-red", "c-blue"],
        "payload": [None, "red", "blue", "neutral", "red", "blue"],
    }
)
CYCLE_EDGES = pd.DataFrame(
    {
        "s": ["a", "a", "a", "m", "m", "m"],
        "d": ["x-red", "x-blue", "m", "a", "c-red", "c-blue"],
        "type": ["P", "P", "S", "T", "T", "T"],
    }
)
CONVERGENT_NODES = pd.DataFrame(
    {
        "id": ["a", "x-red", "x-blue", "m1", "m2", "z"],
        "payload": [None, "red", "blue", "neutral", "neutral", "terminal"],
    }
)
CONVERGENT_EDGES = pd.DataFrame(
    {
        "s": ["a", "a", "a", "a", "m1", "m2"],
        "d": ["x-red", "x-blue", "m1", "m2", "z", "z"],
        "type": ["P", "P", "S", "S", "T", "T"],
    }
)


def _graph(engine: str):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(NODES), pl.from_pandas(EDGES)
    else:
        nodes, edges = NODES.copy(), EDGES.copy()
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _graph_with_two_q_matches(engine: str):
    edges = pd.concat([EDGES, EDGES[EDGES["type"] == "Q"]], ignore_index=True)
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes_frame, edges_frame = pl.from_pandas(NODES), pl.from_pandas(edges)
    else:
        nodes_frame, edges_frame = NODES.copy(), edges
    return graphistry.nodes(nodes_frame, "id").edges(edges_frame, "s", "d")


def _cycle_graph(engine: str):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(CYCLE_NODES), pl.from_pandas(CYCLE_EDGES)
    else:
        nodes, edges = CYCLE_NODES.copy(), CYCLE_EDGES.copy()
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _convergent_graph(engine: str):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(CONVERGENT_NODES), pl.from_pandas(CONVERGENT_EDGES)
    else:
        nodes, edges = CONVERGENT_NODES.copy(), CONVERGENT_EDGES.copy()
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _records(frame, engine: str):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        assert isinstance(frame, pl.DataFrame)
        return frame.to_dicts()
    return frame.to_dict(orient="records")


def _assert_no_hidden_reentry_columns(rows) -> None:
    assert all(
        not any(str(column).startswith("__cypher_reentry_") for column in row)
        for row in rows
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_same_node_payload_pairing_and_repeated_input_bag(engine: str) -> None:
    query = (
        "MATCH (a {id: 'a'})-[:P]->(x) "
        "WITH a, x.payload AS carried "
        "MATCH (a)-[:Q]->(b) WHERE b.payload = carried "
        "RETURN carried, b.id AS bid"
    )
    rows = _records(_graph(engine).gfql(query, engine=engine)._nodes, engine)
    _assert_no_hidden_reentry_columns(rows)
    assert Counter((row["carried"], row["bid"]) for row in rows) == Counter(
        {("red", "b-red"): 2, ("blue", "b-blue"): 1}
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_whole_row_reentry_without_scalar_carry_stays_on_no_token_route(engine: str) -> None:
    query = (
        "MATCH (a {id: 'a'}) "
        "WITH a "
        "MATCH (a)-[:Q]->(b) "
        "RETURN b.id AS bid"
    )
    compiled = compile_cypher(query, _warn_deprecated=False)
    assert compiled.reentry_plan is not None
    assert not getattr(compiled.reentry_plan, "row_token_enabled", False)
    rows = _records(_graph(engine).gfql(query, engine=engine)._nodes, engine)
    _assert_no_hidden_reentry_columns(rows)
    assert Counter(row["bid"] for row in rows) == Counter({"b-red": 1, "b-blue": 1})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_mismatched_payload_and_empty_prefix(engine: str) -> None:
    mismatch = (
        "MATCH (a {id: 'a'})-[:P]->(x {payload: 'green'}) "
        "WITH a, x.payload AS carried "
        "MATCH (a)-[:Q]->(b) WHERE b.payload = carried "
        "RETURN carried, b.id AS bid"
    )
    missing = (
        "MATCH (a {id: 'missing'})-[:P]->(x) "
        "WITH a, x.payload AS carried "
        "MATCH (a)-[:Q]->(b) WHERE b.payload = carried "
        "RETURN carried, b.id AS bid"
    )
    assert _records(_graph(engine).gfql(mismatch, engine=engine)._nodes, engine) == []
    assert _records(_graph(engine).gfql(missing, engine=engine)._nodes, engine) == []


def test_reserved_row_token_alias_is_a_typed_decline() -> None:
    query = (
        "MATCH (a {id: 'a'})-[:P]->(x) "
        "WITH a, x.payload AS __cypher_reentry_row_token "
        "MATCH (a)-[:Q]->(b) RETURN b.id AS bid"
    )
    with pytest.raises(GFQLValidationError) as exc_info:
        _graph("pandas").gfql(query)
    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "with"
    assert exc_info.value.context["value"] == REENTRY_ROW_TOKEN_OUTPUT


def test_user_expression_cannot_spoof_internal_row_token_projection() -> None:
    hidden_name = "__cypher_reentry___cypher_reentry_row_token__"
    query = (
        "MATCH (a {id: 'a'})-[:P]->(x) "
        f"WITH a, a.{hidden_name} AS {REENTRY_ROW_TOKEN_OUTPUT} "
        "MATCH (a)-[:Q]->(b) RETURN b.id AS bid"
    )
    with pytest.raises(GFQLValidationError) as exc_info:
        _graph("pandas").gfql(query)
    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "with"
    assert exc_info.value.context["value"] == REENTRY_ROW_TOKEN_OUTPUT


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_row_token_survives_a_second_with_match_boundary(engine: str) -> None:
    query = (
        "MATCH (a {id: 'a'})-[:P]->(x) "
        "WITH a, x.payload AS carried "
        "MATCH (a)-[:S]->(m) "
        "WITH a, carried "
        "MATCH (a)-[:S]->(m)-[:T]->(c) WHERE c.payload = carried "
        "RETURN carried, c.id AS cid"
    )
    rows = _records(_graph(engine).gfql(query, engine=engine)._nodes, engine)
    _assert_no_hidden_reentry_columns(rows)
    assert Counter((row["carried"], row["cid"]) for row in rows) == Counter(
        {("red", "c-red"): 2, ("blue", "c-blue"): 1}
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_consecutive_with_projection_retains_structured_decline(engine: str) -> None:
    query = (
        "MATCH (a {id: 'a'})-[:P]->(x) "
        "WITH a, x.payload AS carried "
        "MATCH (a)-[:S]->(m) "
        "WITH a, carried "
        "WITH a, carried "
        "MATCH (a)-[:S]->(m)-[:T]->(c) WHERE c.payload = carried "
        "RETURN carried, c.id AS cid"
    )
    with pytest.raises(GFQLValidationError) as exc_info:
        _graph(engine).gfql(query, engine=engine)
    assert exc_info.value.code == "invalid-name-type"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_two_matching_intermediate_rows_multiply_carried_bag(engine: str) -> None:
    query = (
        "MATCH (a {id: 'a'})-[:P]->(x) "
        "WITH a, x.payload AS carried "
        "MATCH (a)-[:Q]->(b) WHERE b.payload = carried "
        "WITH a, carried "
        "MATCH (a)-[:S]->(m)-[:T]->(c) WHERE c.payload = carried "
        "RETURN carried, c.id AS cid"
    )
    rows = _records(_graph_with_two_q_matches(engine).gfql(query, engine=engine)._nodes, engine)
    _assert_no_hidden_reentry_columns(rows)
    assert Counter((row["carried"], row["cid"]) for row in rows) == Counter(
        {("red", "c-red"): 4, ("blue", "c-blue"): 2}
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_cycle_revisiting_seed_preserves_carried_bag(engine: str) -> None:
    query = (
        "MATCH (a {id: 'a'})-[:P]->(x) "
        "WITH a, x.payload AS carried "
        "MATCH (a)-[:S]->(m)-[:T]->(z) "
        "WHERE z.id = a.id "
        "RETURN carried"
    )
    rows = _records(_cycle_graph(engine).gfql(query, engine=engine)._nodes, engine)
    assert Counter(row["carried"] for row in rows) == Counter({"red": 1, "blue": 1})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_non_revisiting_path_does_not_change_carried_bag(engine: str) -> None:
    query = (
        "MATCH (a {id: 'a'})-[:P]->(x) "
        "WITH a, x.payload AS carried "
        "MATCH (a)-[:S]->(m)-[:T]->(z) "
        "WHERE z.payload = carried "
        "RETURN carried"
    )
    rows = _records(_cycle_graph(engine).gfql(query, engine=engine)._nodes, engine)
    assert Counter(row["carried"] for row in rows) == Counter({"red": 1, "blue": 1})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_convergent_paths_preserve_relationship_bag(engine: str) -> None:
    query = (
        "MATCH (a {id: 'a'})-[:P]->(x) "
        "WITH a, x.payload AS carried "
        "MATCH (a)-[:S]->(m)-[:T]->(z) WHERE z.id = 'z' "
        "RETURN carried"
    )
    rows = _records(_convergent_graph(engine).gfql(query, engine=engine)._nodes, engine)
    assert Counter(row["carried"] for row in rows) == Counter({"red": 2, "blue": 2})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_row_token_preserves_bag_until_grouped_return(engine: str) -> None:
    query = (
        "MATCH (a {id: 'a'})-[:P]->(x) "
        "WITH a, x.payload AS carried "
        "MATCH (a)-[:Q]->(b) WHERE b.payload = carried "
        "RETURN carried, count(*) AS n"
    )
    rows = _records(_graph(engine).gfql(query, engine=engine)._nodes, engine)
    _assert_no_hidden_reentry_columns(rows)
    assert Counter((row["carried"], row["n"]) for row in rows) == Counter(
        {("red", 2): 1, ("blue", 1): 1}
    )
