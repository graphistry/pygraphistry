"""#1712 residual: WITH->MATCH re-entry seeds must restrict EVERY execution route.

The bare-carry single-pattern shape was fixed earlier; these pin the routes that still
re-matched the trailing MATCH from the WHOLE graph, silently widening the carried set:

1. projection carry (``WITH p, p.x AS t``) + grouped aggregate: the single-hop
   grouped-aggregate fast path derived its seed from filter_dicts alone, leaking the
   un-carried rows as an extra NULL-keyed group (2 extra persons here);
2. comma-pattern trailing MATCH: the connected match-join re-ran each arm globally
   (bare carry AND projection carry both leaked);
3. two-hop trailing MATCH + count: the two-hop count fast path, same seed-blindness.

Hand-computed oracle throughout: persons 0,1,2; only person 0 has the Books interest,
so every carried re-entry answers 1 (the unrestricted answer is 3 — or a spurious
second group of 2 — so a vacuous pass is impossible).

polars runs the shapes it supports natively and typed-declines projection carry
(scalar WITH columns into trailing MATCH); those declines are pinned as declines.
"""
from typing import Any, List

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError

NODES = pd.DataFrame({
    "node_id": [0, 1, 2, 10, 20, 30, 40],
    "node_type": ["Person", "Person", "Person", "City", "Interest", "Pet", "Country"],
    # nickname is NULL for the carried person 0: the NULL must survive the carry
    # (the NULL-vs-membership class), never a sentinel like '0000-00-00'.
    "nickname": [None, "Bee", "Cee", None, None, None, None],
    "interest": [None, None, None, None, "Books", None, None],
})
EDGES = pd.DataFrame({
    "src": [0, 1, 2, 0, 0, 1, 10],
    "dst": [10, 10, 10, 20, 30, 30, 40],
    "rel": ["LIVES_IN", "LIVES_IN", "LIVES_IN", "HAS_INTEREST", "OWNS", "OWNS", "IN_COUNTRY"],
})

CARRY_PREFIX = (
    "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'})\n"
    "WHERE i.interest='Books'\n"
)

GROUPED_COUNT_PROJ = CARRY_PREFIX + (
    "WITH p, p.node_type AS t\n"
    "MATCH (p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'})\n"
    "RETURN t, count(p) AS numPersons"
)
GROUPED_COUNT_NULL_CARRY = CARRY_PREFIX + (
    "WITH p, p.nickname AS t\n"
    "MATCH (p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'})\n"
    "RETURN t, count(p) AS numPersons"
)
ROWS_PROJ = CARRY_PREFIX + (
    "WITH p, p.node_type AS t\n"
    "MATCH (p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'})\n"
    "RETURN t, p.node_id AS pid"
)
CONNECTED_JOIN_BARE = CARRY_PREFIX + (
    "WITH p\n"
    "MATCH (p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}), (p)-[{rel:'OWNS'}]->(d {node_type:'Pet'})\n"
    "RETURN p.node_id AS pid, count(d) AS pets"
)
CONNECTED_JOIN_PROJ = CARRY_PREFIX + (
    "WITH p, p.node_type AS t\n"
    "MATCH (p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}), (p)-[{rel:'OWNS'}]->(d {node_type:'Pet'})\n"
    "RETURN t, count(d) AS pets"
)
TWO_HOP_COUNT = CARRY_PREFIX + (
    "WITH p\n"
    "MATCH (p)-[{rel:'LIVES_IN'}]->(c)-[{rel:'IN_COUNTRY'}]->(x)\n"
    "RETURN count(*) AS n"
)

POLARS_SCALAR_CARRY_DECLINE = "carries scalar WITH columns into the trailing MATCH"


def _graph(engine: str) -> Any:
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return graphistry.nodes(pl.from_pandas(NODES), "node_id").edges(
            pl.from_pandas(EDGES), "src", "dst"
        )
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return graphistry.nodes(cudf.from_pandas(NODES), "node_id").edges(
            cudf.from_pandas(EDGES), "src", "dst"
        )
    return graphistry.nodes(NODES, "node_id").edges(EDGES, "src", "dst")


def _rows(engine: str, query: str) -> List[dict]:
    frame = _graph(engine).gfql(query, engine=engine)._nodes
    if hasattr(frame, "collect"):
        frame = frame.collect()
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    return frame.to_dict("records")


# ------------------------------ route 1: single-hop grouped-aggregate fast path

@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_projection_carry_grouped_count_restricts_to_carried_rows(engine: str) -> None:
    """Was ``[{'t': 'Person', 'numPersons': 1}, {'t': NaN, 'numPersons': 2}]`` — the
    fast path re-matched all 3 persons and parked the 2 un-carried ones in a NULL
    group. Exactly one group may remain."""
    assert _rows(engine, GROUPED_COUNT_PROJ) == [{"t": "Person", "numPersons": 1}]


def test_projection_carry_grouped_count_polars_declines_typed() -> None:
    pytest.importorskip("polars")
    with pytest.raises(NotImplementedError) as exc_info:
        _rows("polars", GROUPED_COUNT_PROJ)
    assert POLARS_SCALAR_CARRY_DECLINE in str(exc_info.value)


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_projection_carry_null_scalar_survives_the_carry(engine: str) -> None:
    """NULL cell in the carried column: person 0's nickname is NULL, and the carried
    group key must stay NULL (this rendered as the sentinel string '0000-00-00'
    via a vacuously-true all-null temporal-constructor probe)."""
    rows = _rows(engine, GROUPED_COUNT_NULL_CARRY)
    assert len(rows) == 1 and rows[0]["numPersons"] == 1
    assert pd.isna(rows[0]["t"])


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_projection_carry_row_form_control(engine: str) -> None:
    """CONTROL (discriminator): the non-aggregate row form of the same query was
    already restricted — the leak was fast-path-specific."""
    assert _rows(engine, ROWS_PROJ) == [{"t": "Person", "pid": 0}]


# ------------------------------------- route 2: connected comma-pattern join

@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
def test_bare_carry_connected_join_restricts_to_carried_rows(engine: str) -> None:
    """Was ``[{'pid': 0, ...}, {'pid': 1, ...}]`` on all three engines: person 1 owns a
    pet and lives in the city but was never carried."""
    assert _rows(engine, CONNECTED_JOIN_BARE) == [{"pid": 0, "pets": 1}]


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_projection_carry_connected_join_restricts_to_carried_rows(engine: str) -> None:
    assert _rows(engine, CONNECTED_JOIN_PROJ) == [{"t": "Person", "pets": 1}]


def test_projection_carry_connected_join_polars_declines_typed() -> None:
    pytest.importorskip("polars")
    with pytest.raises(NotImplementedError) as exc_info:
        _rows("polars", CONNECTED_JOIN_PROJ)
    assert POLARS_SCALAR_CARRY_DECLINE in str(exc_info.value)


# --------------------------------------------- route 3: two-hop count fast path

@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
def test_two_hop_count_after_carry_restricts_to_carried_rows(engine: str) -> None:
    """Was ``n=3`` (all persons re-matched). Only carried person 0's path counts."""
    assert _rows(engine, TWO_HOP_COUNT) == [{"n": 1}]


# ----------------------------------------------------- helper-level unit pins

def test_restrict_helper_filters_by_bare_alias_column() -> None:
    from graphistry.compute.gfql.cypher.reentry.execution import (
        restrict_connected_join_rows_to_reentry_seed,
    )

    joined = pd.DataFrame({"p": [0, 1, 2], "d": [30, 30, 31], "x": [None, "v", None]})
    seeds = pd.DataFrame({"node_id": [0, 2]})
    out = restrict_connected_join_rows_to_reentry_seed(
        joined, start_nodes=seeds, reentry_alias="p", node_col="node_id"
    )
    # NULL cells in non-key columns ride along untouched
    assert list(out["p"]) == [0, 2] and pd.isna(out["x"]).tolist() == [True, True]


def test_restrict_helper_declines_without_alias_or_seed_columns() -> None:
    from graphistry.compute.gfql.cypher.reentry.execution import (
        restrict_connected_join_rows_to_reentry_seed,
    )

    joined = pd.DataFrame({"q": [0, 1]})
    seeds = pd.DataFrame({"node_id": [0]})
    with pytest.raises(GFQLValidationError):
        restrict_connected_join_rows_to_reentry_seed(
            joined, start_nodes=seeds, reentry_alias=None, node_col="node_id"
        )
    with pytest.raises(GFQLValidationError):
        restrict_connected_join_rows_to_reentry_seed(
            joined, start_nodes=seeds, reentry_alias="p", node_col="node_id"
        )
    with pytest.raises(GFQLValidationError):
        restrict_connected_join_rows_to_reentry_seed(
            joined, start_nodes=pd.DataFrame({"other": [0]}), reentry_alias="q", node_col="node_id"
        )


def test_restrict_helper_polars_frames() -> None:
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql.cypher.reentry.execution import (
        restrict_connected_join_rows_to_reentry_seed,
    )

    joined = pl.DataFrame({"p": [0, 1, 2]})
    seeds = pl.DataFrame({"node_id": [2]})
    out = restrict_connected_join_rows_to_reentry_seed(
        joined, start_nodes=seeds, reentry_alias="p", node_col="node_id"
    )
    assert out["p"].to_list() == [2]
