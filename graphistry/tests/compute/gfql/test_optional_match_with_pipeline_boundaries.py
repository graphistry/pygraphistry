"""Round-1 amplification boundaries for the #1896 OPTIONAL MATCH -> WITH pipeline.

#1897 added terminal-WITH-over-OPTIONAL flattening plus a carried-output
null-fill, and pinned the happy shapes on pandas and polars. This module pins
the BOUNDARIES around those shapes -- degenerate frames, null-valued
predicates, name collisions, stage shapes the flatten declines -- on all three
engines, because the compiled route is chosen per engine and a shape that
serves on pandas can silently answer differently on cuDF with no value test
noticing.

Every expected value here is HAND-COMPUTED from the openCypher contract
(OPTIONAL MATCH keeps every incoming row, binding the optional aliases to NULL
when the pattern does not match; WITH without DISTINCT/ORDER BY/SKIP/LIMIT is a
row no-op; a WHERE on a stage filters binding ROWS with three-valued logic).
Cross-engine agreement is NOT the oracle: ``test_carried_prefix_row_multiplicity``
below is a shape where all three engines agree on the WRONG answer.

Anti-vacuity (measured at merge-base a7c9d6f, the tree #1897 branched from):
10 of these 15 shapes answer differently there -- silent inner-joins that drop
every unmatched seed, a wrong row count under a null-valued predicate, and two
declines whose message described a different limitation. The 5 that already
held at the merge base are marked ``deliberate control`` in their docstring and
each one kills a mutation of this PR's diff that nothing else catches.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError

ENGINES = ["pandas", "polars", "cudf"]

# ------------------------------------------------------------------ fixture
# Same graph as the #1896 pins in test_optional_match_semantics.py, so the
# oracles there and here are computed against one hand-checked shape.
# nodes: a1..a4 (:P, v=1..4), b1/b2 (:C, v=10/20), z (unlabelled, v=99)
# edges: a1-KNOWS->b1, a1-KNOWS->b2, a2-LIKES->b1, b1-KNOWS->a3


def _nodes() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["a1", "a2", "a3", "a4", "b1", "b2", "z"],
        "label__P": [True] * 4 + [False] * 3,
        "label__C": [False] * 4 + [True, True, False],
        "v": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 99.0],
        "name": ["a1", "a2", "a3", "a4", "b1", "b2", "z"],
    })


def _edges() -> pd.DataFrame:
    return pd.DataFrame({
        "s": ["a1", "a1", "a2", "b1"],
        "d": ["b1", "b2", "b1", "a3"],
        "eid": ["e1", "e2", "e3", "e4"],
        "type": ["KNOWS", "KNOWS", "LIKES", "KNOWS"],
        "w": [5, 6, 7, 8],
    })


def _run(query: str, engine: str, *, nodes=None, edges=None) -> pd.DataFrame:
    nodes = _nodes() if nodes is None else nodes
    edges = _edges() if edges is None else edges
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    out = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _scalar(x):
    """Engine-neutral scalar: NaN==None, whole floats == ints."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if hasattr(x, "item"):
        try:
            return _scalar(x.item())
        except (ValueError, AttributeError):
            pass
    return x


def _assert_rows(df: pd.DataFrame, expected) -> None:
    got = [{k: _scalar(v) for k, v in r.items()} for r in df.to_dict("records")]
    key = lambda r: repr(sorted(r.items(), key=str))  # noqa: E731
    assert sorted(got, key=key) == sorted(expected, key=key), f"got {got}, expected {expected}"


_PURE_CARRY = "MATCH (a:P) OPTIONAL MATCH (a)-->(b) WITH a, b RETURN a.id AS aid, b.id AS bid"
_ALL_SEEDS_NULL = [
    {"aid": "a1", "bid": None}, {"aid": "a2", "bid": None},
    {"aid": "a3", "bid": None}, {"aid": "a4", "bid": None},
]


# ===================================================== degenerate frames


@pytest.mark.parametrize("engine", ENGINES)
def test_pure_carry_over_empty_edge_table_null_extends_every_seed(engine):
    """No edge can match, so all four :P seeds survive null-extended -- the
    empty arm must not collapse the pipeline to an inner join."""
    _assert_rows(_run(_PURE_CARRY, engine, edges=_edges().iloc[0:0]), _ALL_SEEDS_NULL)


@pytest.mark.parametrize("engine", ENGINES)
def test_stage_aggregate_over_empty_edge_table_keeps_every_zero_group(engine):
    """count(b) over a fully-unmatched arm is the empty-group value 0 for every
    seed, not a missing group."""
    q = ("MATCH (a:P) OPTIONAL MATCH (a)-->(b) WITH a.id AS aid, count(b) AS cnt "
         "RETURN aid, cnt")
    _assert_rows(_run(q, engine, edges=_edges().iloc[0:0]),
                 [{"aid": "a1", "cnt": 0}, {"aid": "a2", "cnt": 0},
                  {"aid": "a3", "cnt": 0}, {"aid": "a4", "cnt": 0}])


@pytest.mark.parametrize("engine", ENGINES)
def test_pure_carry_over_empty_node_table_is_empty(engine):
    """Deliberate control (holds at the merge base too): with no rows to carry
    there is nothing to null-extend, and the null-fill must not fabricate a
    row. Kills the fill path that synthesises prefix_rows rows unconditionally."""
    _assert_rows(_run(_PURE_CARRY, engine, nodes=_nodes().iloc[0:0],
                      edges=_edges().iloc[0:0]), [])


@pytest.mark.parametrize("engine", ENGINES)
def test_pure_carry_single_binding_row_with_no_arm_match_null_extends(engine):
    """a3 is the only :P node with v=3 and it has no outgoing edge: exactly one
    null-extended row, never zero."""
    q = ("MATCH (a:P {v:3}) OPTIONAL MATCH (a)-->(b) WITH a, b "
         "RETURN a.id AS aid, b.id AS bid")
    _assert_rows(_run(q, engine), [{"aid": "a3", "bid": None}])


@pytest.mark.parametrize("engine", ENGINES)
def test_zero_matching_arm_null_extends_every_row(engine):
    """A typed arm that matches nothing is the same contract as an empty edge
    table, reached through the type predicate instead of the frame."""
    q = ("MATCH (a:P) OPTIONAL MATCH (a)-[:NOPE]->(b) WITH a, b "
         "RETURN a.id AS aid, b.id AS bid")
    _assert_rows(_run(q, engine), _ALL_SEEDS_NULL)


# ===================================================== nulls and collisions


@pytest.mark.parametrize("engine", ENGINES)
def test_stage_where_over_a_null_property_drops_the_row(engine):
    """Three-valued logic: a2.v is NULL, so `a.v <= 2` is NULL for a2's row and
    the row drops -- it is neither kept (as a true match) nor null-extended.
    Only a1 (v=1) passes, keeping BOTH of its arm matches."""
    nodes = _nodes()
    nodes.loc[nodes["id"] == "a2", "v"] = None
    q = ("MATCH (a:P) OPTIONAL MATCH (a)-->(b) WITH a, b WHERE a.v <= 2 "
         "RETURN a.id AS aid, b.id AS bid")
    _assert_rows(_run(q, engine, nodes=nodes),
                 [{"aid": "a1", "bid": "b1"}, {"aid": "a1", "bid": "b2"}])


@pytest.mark.parametrize("engine", ENGINES)
def test_carried_property_named_like_an_edge_endpoint_column_reads_the_node(engine):
    """The node table carries a column named `s`, which is also the edge SOURCE
    binding. `a.s` must read the node property and the output may be named `s`
    too: a1 appears twice (two arm matches), a3/a4 null-extended."""
    nodes = _nodes().assign(s=["n1", "n2", "n3", "n4", "n5", "n6", "n7"])
    q = ("MATCH (a:P) OPTIONAL MATCH (a)-->(b) WITH a, b RETURN a.s AS s, b.id AS bid")
    _assert_rows(_run(q, engine, nodes=nodes),
                 [{"s": "n1", "bid": "b1"}, {"s": "n1", "bid": "b2"},
                  {"s": "n2", "bid": "b1"}, {"s": "n3", "bid": None},
                  {"s": "n4", "bid": None}])


# ===================================================== stage/RETURN interplay


@pytest.mark.parametrize("engine", ENGINES)
def test_return_may_rename_a_folded_stage_output(engine):
    """When the terminal WITH folds into RETURN, the RETURN's own aliases win:
    the output columns are person/n, not the stage's aid/cnt. Values are the
    grouped counts, unmatched seeds keeping their zero group."""
    q = ("MATCH (a:P) OPTIONAL MATCH (a)-->(b) WITH a.id AS aid, count(b) AS cnt "
         "RETURN aid AS person, cnt AS n")
    _assert_rows(_run(q, engine),
                 [{"person": "a1", "n": 2}, {"person": "a2", "n": 1},
                  {"person": "a3", "n": 0}, {"person": "a4", "n": 0}])


@pytest.mark.parametrize("engine", ENGINES)
def test_stage_where_over_an_alias_the_stage_dropped_declines_typed(engine):
    """`WITH a` removes b from scope, so `WHERE b.v >= 20` reaches for a
    variable the stage does not carry. That must decline as a WITH-pipeline
    limitation, never silently apply the predicate to the still-joined frame
    (which would answer [a1] for a query openCypher rejects outright)."""
    q = "MATCH (a:P) OPTIONAL MATCH (a)-->(b) WITH a WHERE b.v >= 20 RETURN a.id AS aid"
    with pytest.raises(GFQLValidationError) as err:
        _run(q, engine)
    assert "WITH pipelines after OPTIONAL MATCH" in str(err.value), str(err.value)


@pytest.mark.parametrize("engine", ENGINES)
def test_duplicate_stage_output_names_decline_rather_than_silently_picking_one(engine):
    """Deliberate control (the merge base declined here too, with a different
    message): two stage outputs named `x` must not resolve to whichever one is
    written last. Kills the fold path that overwrites the earlier name and then
    answers b.v for a query that also asked for a.v."""
    q = "MATCH (a:P) OPTIONAL MATCH (a)-->(b) WITH a.v AS x, b.v AS x RETURN x"
    with pytest.raises(GFQLValidationError):
        _run(q, engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_renaming_carry_declines_without_claiming_the_alias_is_unknown(engine):
    """`WITH a AS a2` is a rename the flatten does not serve. The decline must
    describe THAT, not assert the falsehood that a2 is an unknown alias -- a2
    is declared, by the very stage being rejected (F-05 message contract)."""
    q = ("MATCH (a:P) OPTIONAL MATCH (a)-[:KNOWS]->(b) WITH a AS a2, b "
         "RETURN a2.id AS aid, b.id AS bid")
    with pytest.raises(GFQLValidationError) as err:
        _run(q, engine)
    msg = str(err.value)
    assert "WITH pipelines after OPTIONAL MATCH" in msg, msg
    assert "Unknown Cypher alias" not in msg, msg


@pytest.mark.parametrize("engine", ENGINES)
def test_with_pipeline_without_any_optional_match_still_answers(engine):
    """Deliberate control (holds at the merge base): the OPTIONAL-MATCH
    precondition on the flatten hook is load-bearing. Without it every plain
    WITH pipeline falls into the hook's unconditional decline, so this ordinary
    query must keep answering all four rows."""
    _assert_rows(_run("MATCH (a:P) WITH a.v AS av RETURN av", engine),
                 [{"av": 1}, {"av": 2}, {"av": 3}, {"av": 4}])


# ===================================================== reentry null-fill


@pytest.mark.parametrize("engine", ENGINES)
def test_reentry_prefix_without_an_identity_projection_declines_typed(engine):
    """LIMIT 2 carries {a1, a2}; a1's two KNOWS matches make the result exactly
    as long as the prefix, so a row-count comparison cannot tell that a2 never
    matched. With no carried-alias column projected there is nothing to
    anti-join on, so the fill must decline rather than return a1's two rows as
    if the pipeline were complete."""
    q = ("MATCH (a:P) WITH a AS p LIMIT 2 OPTIONAL MATCH (p)-[:KNOWS]->(b) "
         "RETURN b.id AS bid")
    with pytest.raises(GFQLValidationError) as err:
        _run(q, engine)
    assert "null-extend unmatched carried rows" in str(err.value), str(err.value)


@pytest.mark.parametrize("engine", ENGINES)
def test_reentry_single_prefix_row_with_zero_matches_null_extends(engine):
    """Deliberate control (holds at the merge base): one carried row that
    matches nothing still yields one null row. Kills the single-row
    short-circuit if it stops checking whether the reentry produced any result
    at all."""
    q = ("MATCH (a:P {v:3}) WITH a AS p OPTIONAL MATCH (p)-[:KNOWS]->(b) "
         "RETURN b.id AS bid")
    _assert_rows(_run(q, engine), [{"bid": None}])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.xfail(
    strict=True,
    reason="KNOWN WRONG (all three engines agree): a whole-row WITH carry "
           "deduplicates the prefix frame by node id, so a prefix row that "
           "appears twice contributes its arm matches once. #1897 fixed the "
           "unmatched-row half of this shape (a2 is now null-extended instead "
           "of dropped) but not the multiplicity half.",
)
def test_carried_prefix_row_multiplicity_survives_the_reentry(engine):
    """`MATCH (a:P)-[]->(c)` binds a1 twice (a1->b1, a1->b2) and a2 once, so
    `WITH a AS p` carries THREE rows. Each a1 row then expands to a1's two
    KNOWS matches and a2's row null-extends: 5 rows, with (a1,b1) and (a1,b2)
    each appearing twice. Cross-engine agreement is not evidence here -- all
    three engines return the 3-row deduplicated answer."""
    q = ("MATCH (a:P)-[]->(c) WITH a AS p OPTIONAL MATCH (p)-[:KNOWS]->(d) "
         "RETURN p.id AS pid, d.id AS did")
    _assert_rows(_run(q, engine), [
        {"pid": "a1", "did": "b1"}, {"pid": "a1", "did": "b2"},
        {"pid": "a1", "did": "b1"}, {"pid": "a1", "did": "b2"},
        {"pid": "a2", "did": None},
    ])
