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

Anti-vacuity (re-measured at this PR's base 21167e08, which reproduces the
figure first taken at a7c9d6f): 30 of the first 45 cells FAIL there -- 10 of
the 15 shapes answer differently, in silent inner-joins that drop every
unmatched seed, a wrong row count under a null-valued predicate, and two
declines whose message described a different limitation. Of the remaining 5
shapes, 4 are marked ``deliberate control`` in their docstring -- each kills a
mutation of this PR's diff, one of them (the single-prefix-row control) also
covered by an existing pandas-only pin named in its docstring -- and the 5th is
the strict xfail below, which fails as expected on both trees because the
defect it names predates this PR.
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
    query must keep answering all four rows. Round 2's cross-suite sweep found
    that mutation is caught 151 times over in the pre-existing cypher suite; the
    value here is that it is caught by NAME and on all three engines."""
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
    at all -- round 2's cross-suite sweep found that mutation is ALSO caught by
    ``test_lowering.py::test_issue_1461_optional_reentry_null_extension_does_not_leak_unprojected_scalar``,
    which is pandas-only; what this cell adds over it is polars and cuDF."""
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


# ===================================================== round 2
# Round 1 called ten surviving mutations "defensive branches with no Cypher
# spelling that reaches them", on the strength of a 30-query battery. Six do
# have one; two of those six were already caught elsewhere in the suite, and
# the four below had nothing. The shapes here are those spellings, plus the
# NULL axis (NULL ids, NULL carried values, a NULL key shared by a matched and
# an unmatched row) that the sibling PR's silent-wrong answer came from.
#
# Anti-vacuity, measured at this PR's base 21167e08: 27 of the 29 cells below
# FAIL there. The two that do not are the strict xfail, whose defect predates
# the PR and xfails on both trees. No cell here is base-passing.

# A PATH alias (`MATCH path = ...`) is the one carried bare identifier that no
# MATCH clause binds as a pattern variable.
_PATH_BASE = ("MATCH path = (a:P)-[:KNOWS]->(b) "
              "OPTIONAL MATCH (b)-[:KNOWS]->(c) ")


@pytest.mark.parametrize("engine", ENGINES)
def test_path_alias_carry_null_extends_the_unmatched_arm(engine):
    """The prefix binds path/a/b twice (a1->b1, a1->b2 are the only :P KNOWS
    edges). b1-[:KNOWS]->a3 matches the arm, b2 has no outgoing edge, so the
    second row null-extends. The merge base declined this shape outright."""
    q = _PATH_BASE + "WITH path, a, b, c RETURN a.id AS aid, c.id AS cid"
    _assert_rows(_run(q, engine),
                 [{"aid": "a1", "cid": "a3"}, {"aid": "a1", "cid": None}])


@pytest.mark.parametrize("engine", ENGINES)
def test_path_alias_carry_with_a_stage_where_declines_typed(engine):
    """openCypher answers the two rows above (a1.v is 1, so the predicate keeps
    both). We decline instead, and that decline is the only thing standing
    between this shape and the pure-carry rewrite, which would drop a WITH
    stage whose carried set is not a subset of the bound aliases. Pinned as the
    limitation it is, named by the WITH-pipeline message."""
    with pytest.raises(GFQLValidationError) as err:
        _run(_PATH_BASE + "WITH path, a, b, c WHERE a.v <= 1 "
                          "RETURN a.id AS aid, c.id AS cid", engine)
    assert "WITH pipelines after OPTIONAL MATCH" in str(err.value), str(err.value)


@pytest.mark.parametrize("engine", ENGINES)
def test_variable_length_arm_declines_on_row_synthesis_not_the_with_pipeline(engine):
    """The terminal WITH here IS supported -- it is the variable-length arm the
    left-join lowering cannot null-extend. The decline must say so. Naming the
    WITH pipeline instead would send a reader to rewrite the clause that is not
    the problem. The merge base answered this WRONG and differently per engine
    (pandas one row with a null, polars/cuDF a1 joined to itself)."""
    q = ("MATCH (a:P) OPTIONAL MATCH (a)-[:KNOWS*1..2]->(b) WITH a, b "
         "RETURN a.id AS aid, b.id AS bid")
    with pytest.raises(GFQLValidationError) as err:
        _run(q, engine)
    msg = str(err.value)
    assert "null-extension rows that the local compiler cannot synthesize" in msg, msg
    assert "WITH pipelines after OPTIONAL MATCH" not in msg, msg


# ------------------------------------------------- carried-identity keys

def _grouped_nodes(**extra) -> pd.DataFrame:
    """a1/a2 share a grp value, a3 has its own; only a1 has a KNOWS edge."""
    return pd.DataFrame({
        "id": ["a1", "a2", "a3", "b1"],
        "label__P": [True, True, True, False],
        "label__C": [False, False, False, True],
        "grp": ["g", "g", "h", "z"],
        "nul": [None, None, None, None],
        "v": [1.0, 2.0, 3.0, 10.0],
        **extra,
    })


_GROUPED_EDGES = pd.DataFrame({"s": ["a1"], "d": ["b1"], "type": ["KNOWS"]})


@pytest.mark.parametrize("engine", ENGINES)
def test_non_unique_carried_identity_projection_declines(engine):
    """openCypher answers [(g,b1),(g,None),(h,None)]. `p.grp` repeats across
    a1 and a2, so the anti-join that finds unmatched carried rows cannot tell
    a2's row from a1's. Declining is the only sound option: without the
    duplicate-key guard the fill silently emits [(g,b1),(h,None)] and a2's row
    disappears. The merge base answered three rows with two of the grp values
    replaced by null."""
    q = ("MATCH (a:P) WITH a AS p OPTIONAL MATCH (p)-[:KNOWS]->(b) "
         "RETURN p.grp AS g, b.id AS bid")
    with pytest.raises(GFQLValidationError) as err:
        _run(q, engine, nodes=_grouped_nodes(), edges=_GROUPED_EDGES)
    assert "no uniquely-identifying carried-alias columns" in str(err.value), str(err.value)


@pytest.mark.parametrize("engine", ENGINES)
def test_all_null_carried_identity_projection_declines(engine):
    """The NULL flavour of the same guard: every key is NULL, so every key
    collides. Dropping the guard here loses BOTH unmatched rows, not one --
    the fill returns a single row. openCypher answers [(None,b1),(None,None),
    (None,None)], which the merge base produced by accident (its fill padded
    with all-null rows and the projected column is null anyway); we decline,
    which is a lost query but never a wrong one."""
    q = ("MATCH (a:P) WITH a AS p OPTIONAL MATCH (p)-[:KNOWS]->(b) "
         "RETURN p.nul AS g, b.id AS bid")
    with pytest.raises(GFQLValidationError) as err:
        _run(q, engine, nodes=_grouped_nodes(), edges=_GROUPED_EDGES)
    assert "no uniquely-identifying carried-alias columns" in str(err.value), str(err.value)


# ------------------------------------------------- the NULL axis

def _null_valued_nodes() -> pd.DataFrame:
    """a2 and a3 both have a NULL v; a2 matches the arm and a3 does not."""
    return pd.DataFrame({
        "id": ["a1", "a2", "a3", "a4", "b1"],
        "label__P": [True, True, True, True, False],
        "label__C": [False, False, False, False, True],
        "v": [1.0, None, None, 4.0, 10.0],
    })


_NULL_EDGES = pd.DataFrame({"s": ["a1", "a2"], "d": ["b1", "b1"],
                            "type": ["KNOWS", "KNOWS"]})


@pytest.mark.parametrize("engine", ENGINES)
def test_null_carried_value_beside_an_identity_column_null_extends(engine):
    """A NULL carried value must not be treated as membership in the matched
    set. a2 (v NULL) matched and a3 (v NULL) did not; only `p.id` separates
    them. a3 must come back null-extended with its own NULL v, and a4 must keep
    v=4. The merge base returned NULL for both pid and pv on the two unmatched
    rows, losing a4's value and both identities."""
    q = ("MATCH (a:P) WITH a AS p OPTIONAL MATCH (p)-[:KNOWS]->(b) "
         "RETURN p.id AS pid, p.v AS pv, b.id AS bid")
    _assert_rows(_run(q, engine, nodes=_null_valued_nodes(), edges=_NULL_EDGES), [
        {"pid": "a1", "pv": 1, "bid": "b1"},
        {"pid": "a2", "pv": None, "bid": "b1"},
        {"pid": "a3", "pv": None, "bid": None},
        {"pid": "a4", "pv": 4, "bid": None},
    ])


@pytest.mark.parametrize("engine", ENGINES)
def test_null_valued_property_is_indistinguishable_from_an_unbound_alias(engine):
    """Same graph through the flatten path rather than the reentry path. Both
    (NULL, b1) -- a2 matched but has no v -- and (NULL, None) -- a3 unmatched --
    are legal openCypher rows and the projection cannot tell them apart. The
    merge base collapsed this to two rows on pandas/cuDF and answered b.id as
    a1/a2 on polars."""
    q = ("MATCH (a:P) OPTIONAL MATCH (a)-[:KNOWS]->(b) WITH a, b "
         "RETURN a.v AS av, b.id AS bid")
    _assert_rows(_run(q, engine, nodes=_null_valued_nodes(), edges=_NULL_EDGES), [
        {"av": 1, "bid": "b1"}, {"av": None, "bid": "b1"},
        {"av": None, "bid": None}, {"av": 4, "bid": None},
    ])


@pytest.mark.parametrize("engine", ENGINES)
def test_null_node_id_seed_still_null_extends(engine):
    """A node whose id property is NULL is still a node and still a carried
    row. Its identity key is NULL, distinct from a2's, so a2 keeps its own id
    on the fill row -- the merge base returned NULL for a2 as well."""
    nodes = pd.DataFrame({
        "id": ["a1", "a2", None, "b1"],
        "label__P": [True, True, True, False],
        "label__C": [False, False, False, True],
        "v": [1.0, 2.0, 3.0, 10.0],
    })
    edges = pd.DataFrame({"s": ["a1"], "d": ["b1"], "type": ["KNOWS"]})
    q = ("MATCH (a:P) WITH a AS p OPTIONAL MATCH (p)-[:KNOWS]->(b) "
         "RETURN p.id AS pid, b.id AS bid")
    _assert_rows(_run(q, engine, nodes=nodes, edges=edges),
                 [{"pid": "a1", "bid": "b1"}, {"pid": "a2", "bid": None},
                  {"pid": None, "bid": None}])


# ------------------------------------------------- irreproducible outputs

@pytest.mark.parametrize("engine", ENGINES)
def test_expression_over_a_carried_alias_declines_as_irreproducible(engine):
    """Round 1 reported that no Cypher query reaches the
    CARRIED_OUTPUTS_NOT_REPRODUCIBLE decline. This one does: `p.v + 1` is not a
    column the fill can copy off the prefix frame, and the RETURN names only
    `p`, so the multi-source residual does not intercept it first. openCypher
    answers [2, 2, 3]; the merge base answered [2, None]."""
    q = ("MATCH (a:P) WITH a AS p LIMIT 2 OPTIONAL MATCH (p)-[:KNOWS]->(b) "
         "RETURN p.v + 1 AS pv")
    with pytest.raises(GFQLValidationError) as err:
        _run(q, engine)
    assert "the null-extension cannot reproduce" in str(err.value), str(err.value)


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_whole_entity_carried_alias_keeps_its_values_on_the_null_extended_row(engine):
    """`WITH a AS p, a.id AS pid` binds p for every carried row. OPTIONAL MATCH
    cannot unbind it, so a2's null-extended row must still carry a2's own node
    columns; only `bid` is NULL. polars declines this shape outright with the
    typed scalar-carry NotImplementedError, so it is not parametrized here."""
    q = ("MATCH (a:P) WITH a AS p, a.id AS pid LIMIT 2 "
         "OPTIONAL MATCH (p)-[:KNOWS]->(b) RETURN p, pid, b.id AS bid")
    _assert_rows(_run(q, engine), [
        {"p.id": "a1", "p.v": 1, "p.name": "a1", "p.label__P": True,
         "p.label__C": False, "pid": "a1", "bid": "b1"},
        {"p.id": "a1", "p.v": 1, "p.name": "a1", "p.label__P": True,
         "p.label__C": False, "pid": "a1", "bid": "b2"},
        {"p.id": "a2", "p.v": 2, "p.name": "a2", "p.label__P": True,
         "p.label__C": False, "pid": "a2", "bid": None},
    ])
