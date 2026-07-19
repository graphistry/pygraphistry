"""Regression: connected OPTIONAL MATCH seed extraction must be engine-polymorphic.

_optional_arm_start_nodes (gfql_unified.py) applied pandas-only frame ops
(.dropna()/.drop_duplicates()/.rename(columns=)/boolean-mask __getitem__) to the joined
binding rows, so an IS7-shaped Cypher query (MATCH ... OPTIONAL MATCH ...) on
engine='polars' crashed with AttributeError before reaching the row pipeline
(LDBC SNB interactive-short-7 via the pyg-bench harness). The polars row pipeline may
still honestly decline the query (NotImplementedError, parity-or-error by design), but
it must never crash with a pandas-ism.
"""
from typing import Optional

import pandas as pd
import pytest

import graphistry

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


IS7_SHAPED = """
MATCH (m:Message {id: $messageId })<-[:REPLY_OF]-(c:Comment)-[:HAS_CREATOR]->(p:Person)
    OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:Person)-[r:KNOWS]-(p)
    RETURN c.id AS commentId,
        c.creationDate AS commentCreationDate,
        p.id AS replyAuthorId,
        CASE r
            WHEN null THEN false
            ELSE true
        END AS replyAuthorKnowsOriginalMessageAuthor
    ORDER BY commentCreationDate DESC, replyAuthorId
"""


def _nodes_pd() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 3, 10, 11, 12],
        "label__Message": [True, False, False, False, False, False],
        "label__Comment": [False, True, True, False, False, False],
        "label__Person": [False, False, False, True, True, True],
        "creationDate": [100, 200, 300, None, None, None],
    })


def _edges_pd() -> pd.DataFrame:
    # message creator (12) KNOWS alice (10) but not bob (11) -> discriminating flags
    return pd.DataFrame({
        "src": [2, 3, 2, 3, 1, 10, 11, 12],
        "dst": [1, 1, 10, 11, 12, 11, 10, 10],
        "type": ["REPLY_OF", "REPLY_OF", "HAS_CREATOR", "HAS_CREATOR", "HAS_CREATOR",
                 "KNOWS", "KNOWS", "KNOWS"],
    })


def test_optional_match_pandas_oracle() -> None:
    g = graphistry.nodes(_nodes_pd(), "id").edges(_edges_pd(), "src", "dst")
    res = g.gfql(IS7_SHAPED, params={"messageId": 1}, engine="pandas")
    rows = res._nodes.reset_index(drop=True)
    assert list(rows["commentId"]) == [3, 2]
    # creator (12) KNOWS alice (10) but not bob (11): discriminating flags
    assert list(rows["replyAuthorKnowsOriginalMessageAuthor"]) == [False, True]


IS7_SHAPED_NO_CASE = """
MATCH (m:Message {id: $messageId })<-[:REPLY_OF]-(c:Comment)-[:HAS_CREATOR]->(p:Person)
    OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:Person)-[r:KNOWS]-(p)
    RETURN c.id AS commentId,
        c.creationDate AS commentCreationDate,
        p.id AS replyAuthorId
    ORDER BY commentCreationDate DESC, replyAuthorId
"""


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_optional_match_polars_native_end_to_end() -> None:
    """Connected OPTIONAL MATCH + simple RETURN runs natively on polars, oracle-exact."""
    nodes = pl.from_pandas(_nodes_pd())
    edges = pl.from_pandas(_edges_pd())
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    res = g.gfql(IS7_SHAPED_NO_CASE, params={"messageId": 1}, engine="polars")
    rows = res._nodes.to_pandas() if hasattr(res._nodes, "to_pandas") else res._nodes
    rows = rows.reset_index(drop=True)
    assert list(rows["commentId"]) == [3, 2]
    assert list(rows["replyAuthorId"]) == [11, 10]


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_optional_match_polars_no_pandasism_crash() -> None:
    """Full IS7 (CASE r WHEN null projection) runs natively on polars, oracle-exact.

    The simple-CASE null-literal equality (`__cypher_case_eq__(x, null)`) lowers to
    `is_null()` on polars (pandas-parity null-mask semantics). If a future edit
    re-declines it, the honest NIE branch keeps this from crashing dishonestly."""
    nodes = pl.from_pandas(_nodes_pd())
    edges = pl.from_pandas(_edges_pd())
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    try:
        res = g.gfql(IS7_SHAPED, params={"messageId": 1}, engine="polars")
    except NotImplementedError:
        return  # honest parity-or-error decline is acceptable; AttributeError is not
    rows = res._nodes.to_pandas() if hasattr(res._nodes, "to_pandas") else res._nodes
    rows = rows.reset_index(drop=True)
    assert list(rows["commentId"]) == [3, 2]
    assert list(rows["replyAuthorKnowsOriginalMessageAuthor"]) == [False, True]


# ---------------------------------------------------------------------------
# Amplification round 001 (plans/gfql-optional-match-polars-amplification):
# oracle-gated parity matrix for the polars-native connected OPTIONAL MATCH
# lowering (_optional_arm_membership_chain pruning, polars arm-join twin,
# engine-polymorphic seed extraction).
#
# Two assertion modes:
# - _assert_parity: shape runs natively on polars today; polars must return
#   exactly the pandas-oracle rows (regression to NIE or wrong rows fails).
# - _assert_parity_or_nie: shape currently declines honestly on polars
#   (parity-or-error contract); a decline must be NotImplementedError, and if
#   polars ever starts running the shape it must match the oracle. Silent
#   wrong results or a pandas-ism crash always fail.
#
# Fixtures are deliberately asymmetric (distinct score/weight per node/edge)
# so mis-joins produce wrong VALUES, not accidentally-identical rows.
# ---------------------------------------------------------------------------


def _amp_nodes_pd() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 10, 11, 20, 21],
        "label__Message": [True, True, False, False, False, False],
        "label__Person": [False, False, True, True, False, False],
        "label__Tag": [False, False, False, False, True, True],
        "score": [11, 22, 43, 54, 75, 86],  # unique per node: discriminating
    })


def _amp_edges_pd() -> pd.DataFrame:
    # msg1 -HAS_CREATOR-> alice(10) -LIKES-> tag20
    # msg2 -HAS_CREATOR-> bob(11)            (bob likes nothing: unmatched seed)
    # msg1 -HAS_TAG-> tag20, msg2 -HAS_TAG-> tag21
    return pd.DataFrame({
        "src": [1, 2, 10, 1, 2],
        "dst": [10, 11, 20, 20, 21],
        "type": ["HAS_CREATOR", "HAS_CREATOR", "LIKES", "HAS_TAG", "HAS_TAG"],
        "weight": [1.5, 2.5, 3.5, 4.5, 5.5],  # unique per edge: discriminating
    })


def _run_engine(nodes: pd.DataFrame, edges: pd.DataFrame, query: str, engine: str) -> pd.DataFrame:
    if engine == "polars":
        g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "src", "dst")
    else:
        g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    res = g.gfql(query, engine=engine)
    out = res._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted row set with dtype normalization (int/bool/float -> float64, NaN==null)."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("float64")
        elif df[c].isna().all():
            # all-null column: unify to float64 NaN regardless of source dtype
            df[c] = df[c].astype("float64")
        else:
            # unify null representation (polars->pandas gives None, pandas gives NaN)
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    cols = sorted(df.columns)
    return df[cols].sort_values(cols, na_position="last").reset_index(drop=True)


def _assert_parity(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    query: str,
    expected: "Optional[pd.DataFrame]" = None,
) -> None:
    oracle = _run_engine(nodes, edges, query, "pandas")
    if expected is not None:
        pd.testing.assert_frame_equal(
            _normalize(oracle[expected.columns.tolist()]), _normalize(expected),
            check_dtype=False,
        )
    got = _run_engine(nodes, edges, query, "polars")
    assert sorted(got.columns) == sorted(oracle.columns)
    pd.testing.assert_frame_equal(_normalize(got), _normalize(oracle), check_dtype=False)


def _assert_parity_or_nie(nodes: pd.DataFrame, edges: pd.DataFrame, query: str) -> None:
    oracle = _run_engine(nodes, edges, query, "pandas")
    try:
        got = _run_engine(nodes, edges, query, "polars")
    except NotImplementedError:
        return  # honest parity-or-error decline; any other exception fails the test
    assert sorted(got.columns) == sorted(oracle.columns)
    pd.testing.assert_frame_equal(_normalize(got), _normalize(oracle), check_dtype=False)


Q_ARM_LIKES = """
MATCH (m:Message)-[:HAS_CREATOR]->(p:Person)
OPTIONAL MATCH (p)-[r:LIKES]->(t:Tag)
RETURN m.id AS mid, p.id AS pid, t.id AS tid, t.score AS tscore
ORDER BY mid
"""

pytestmark_polars = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")


@pytestmark_polars
def test_amp_partial_match_mixed_seeds() -> None:
    """Mix of matched (alice->tag20) and unmatched (bob) seeds: null-extension
    row must carry nulls, matched row the RIGHT tag's value (score 75)."""
    expected = pd.DataFrame({
        "mid": [1, 2], "pid": [10, 11],
        "tid": [20.0, None], "tscore": [75.0, None],
    })
    _assert_parity(_amp_nodes_pd(), _amp_edges_pd(), Q_ARM_LIKES, expected)


@pytestmark_polars
def test_amp_zero_match_arm_case_flag() -> None:
    """Arm matches nothing anywhere (no NOPE edges): all seeds get flag False
    via the CASE r WHEN null -> is_null lowering."""
    q = """
    MATCH (m:Message)-[:HAS_CREATOR]->(p:Person)
    OPTIONAL MATCH (p)-[r:NOPE]->(t:Tag)
    RETURN m.id AS mid, p.id AS pid,
        CASE r WHEN null THEN false ELSE true END AS hasR
    ORDER BY mid
    """
    expected = pd.DataFrame({"mid": [1, 2], "pid": [10, 11], "hasR": [False, False]})
    _assert_parity(_amp_nodes_pd(), _amp_edges_pd(), q, expected)


@pytestmark_polars
def test_amp_zero_match_arm_property_projection() -> None:
    """Zero-match arm + projecting t.id: pandas emits all-null column; polars
    currently declines (NIE on 'select': no t.* columns to project)."""
    q = """
    MATCH (m:Message)-[:HAS_CREATOR]->(p:Person)
    OPTIONAL MATCH (p)-[r:NOPE]->(t:Tag)
    RETURN m.id AS mid, p.id AS pid, t.id AS tid
    ORDER BY mid
    """
    _assert_parity_or_nie(_amp_nodes_pd(), _amp_edges_pd(), q)


@pytestmark_polars
def test_amp_multiple_optional_arms() -> None:
    """Two OPTIONAL MATCH arms; polars currently declines the second arm
    (NIE on 'rows'); must never silently mis-join."""
    q = """
    MATCH (m:Message)-[:HAS_CREATOR]->(p:Person)
    OPTIONAL MATCH (p)-[r:LIKES]->(t:Tag)
    OPTIONAL MATCH (m)-[h:HAS_TAG]->(t2:Tag)
    RETURN m.id AS mid, p.id AS pid, t.id AS tid, t2.id AS t2id
    ORDER BY mid
    """
    _assert_parity_or_nie(_amp_nodes_pd(), _amp_edges_pd(), q)


@pytestmark_polars
def test_amp_arm_where_keeps_rows() -> None:
    """WHERE on the arm alias that keeps rows (score>=70 keeps tag20). WHERE
    arms skip seeding/pruning entirely (run unseeded) — exercises that branch."""
    q = """
    MATCH (m:Message)-[:HAS_CREATOR]->(p:Person)
    OPTIONAL MATCH (p)-[r:LIKES]->(t:Tag)
    WHERE t.score >= 70
    RETURN m.id AS mid, p.id AS pid, t.id AS tid, t.score AS tscore
    ORDER BY mid
    """
    expected = pd.DataFrame({
        "mid": [1, 2], "pid": [10, 11],
        "tid": [20.0, None], "tscore": [75.0, None],
    })
    _assert_parity(_amp_nodes_pd(), _amp_edges_pd(), q, expected)


@pytestmark_polars
def test_amp_arm_where_filters_all_rows() -> None:
    """WHERE filtering the arm to zero rows: pandas null-extends every seed;
    polars currently declines (NIE) — must not fabricate matched rows."""
    q = """
    MATCH (m:Message)-[:HAS_CREATOR]->(p:Person)
    OPTIONAL MATCH (p)-[r:LIKES]->(t:Tag)
    WHERE t.score > 9000
    RETURN m.id AS mid, p.id AS pid, t.id AS tid
    ORDER BY mid
    """
    _assert_parity_or_nie(_amp_nodes_pd(), _amp_edges_pd(), q)


@pytestmark_polars
def test_amp_duplicate_seed_ids() -> None:
    """Both messages share one creator (alice): the shared-alias join key is
    duplicated across base rows; each row must still get alice's arm match."""
    edges = _amp_edges_pd()
    edges.loc[1, "dst"] = 10  # msg2 also HAS_CREATOR -> alice
    expected = pd.DataFrame({
        "mid": [1, 2], "pid": [10, 10],
        "tid": [20.0, 20.0], "tscore": [75.0, 75.0],
    })
    _assert_parity(_amp_nodes_pd(), edges, Q_ARM_LIKES, expected)


@pytestmark_polars
def test_amp_empty_seed_set() -> None:
    """Main MATCH matches nothing (id 999): empty result, no crash in seed
    extraction/pruning over an empty joined frame."""
    q = """
    MATCH (m:Message {id: 999})-[:HAS_CREATOR]->(p:Person)
    OPTIONAL MATCH (p)-[r:LIKES]->(t:Tag)
    RETURN m.id AS mid, t.id AS tid
    """
    oracle = _run_engine(_amp_nodes_pd(), _amp_edges_pd(), q, "pandas")
    got = _run_engine(_amp_nodes_pd(), _amp_edges_pd(), q, "polars")
    assert len(oracle) == 0
    assert len(got) == 0


@pytestmark_polars
def test_amp_arm_fanout_multiplicity() -> None:
    """One seed matches two arm rows (alice likes tag20 AND tag21): left join
    must fan out to 3 rows total with per-tag values, not dedupe or cross."""
    edges = pd.concat([_amp_edges_pd(), pd.DataFrame({
        "src": [10], "dst": [21], "type": ["LIKES"], "weight": [9.5],
    })], ignore_index=True)
    q = """
    MATCH (m:Message)-[:HAS_CREATOR]->(p:Person)
    OPTIONAL MATCH (p)-[r:LIKES]->(t:Tag)
    RETURN m.id AS mid, p.id AS pid, t.id AS tid, t.score AS tscore
    ORDER BY mid, tid
    """
    expected = pd.DataFrame({
        "mid": [1, 1, 2], "pid": [10, 10, 11],
        "tid": [20.0, 21.0, None], "tscore": [75.0, 86.0, None],
    })
    _assert_parity(_amp_nodes_pd(), edges, q, expected)


@pytestmark_polars
def test_amp_null_carrying_arm_value_column() -> None:
    """Matched arm row whose projected value is itself null (tag20.score=null):
    null-from-match must survive the join identically to null-from-no-match."""
    nodes = _amp_nodes_pd()
    nodes["score"] = nodes["score"].astype("float64")
    nodes.loc[nodes["id"] == 20, "score"] = None
    expected = pd.DataFrame({
        "mid": [1, 2], "pid": [10, 11],
        "tid": [20.0, None], "tscore": [None, None],
    })
    _assert_parity(nodes, _amp_edges_pd(), Q_ARM_LIKES, expected)


@pytestmark_polars
def test_amp_string_ids() -> None:
    """String node ids: membership-pruning filter and joins on Utf8 keys."""
    nodes = _amp_nodes_pd()
    nodes["id"] = "n" + nodes["id"].astype(str)
    edges = _amp_edges_pd()
    edges["src"] = "n" + edges["src"].astype(str)
    edges["dst"] = "n" + edges["dst"].astype(str)
    expected = pd.DataFrame({
        "mid": ["n1", "n2"], "pid": ["n10", "n11"],
        "tid": ["n20", None], "tscore": [75.0, None],
    })
    _assert_parity(nodes, edges, Q_ARM_LIKES, expected)


@pytestmark_polars
def test_amp_float_join_key_dtype_divergence() -> None:
    """Float edge endpoints vs int node ids, and all-float ids: dtype-divergent
    join keys must give parity or an honest decline (currently NIE on polars),
    never a silently-empty or mis-typed join."""
    edges_f = _amp_edges_pd()
    edges_f["src"] = edges_f["src"].astype("float64")
    edges_f["dst"] = edges_f["dst"].astype("float64")
    _assert_parity_or_nie(_amp_nodes_pd(), edges_f, Q_ARM_LIKES)

    nodes_f = _amp_nodes_pd()
    nodes_f["id"] = nodes_f["id"].astype("float64")
    _assert_parity_or_nie(nodes_f, edges_f, Q_ARM_LIKES)


@pytestmark_polars
def test_amp_pruning_fallback_id_constrained_first_op() -> None:
    """Arm's first node op already carries an id constraint ({id: 10}):
    _optional_arm_membership_chain must decline (no double-constraint on the id
    key) and the unseeded fallback must still produce oracle-exact rows —
    alice's row matched, bob's row null-extended."""
    q = """
    MATCH (m:Message)-[:HAS_CREATOR]->(p:Person)
    OPTIONAL MATCH (p {id: 10})-[r:LIKES]->(t:Tag)
    RETURN m.id AS mid, p.id AS pid, t.id AS tid
    ORDER BY mid
    """
    expected = pd.DataFrame({
        "mid": [1, 2], "pid": [10, 11], "tid": [20.0, None],
    })
    _assert_parity(_amp_nodes_pd(), _amp_edges_pd(), q, expected)


@pytestmark_polars
def test_amp_pruning_large_arm_space() -> None:
    """Arm alias space >> seed space (120 extra persons each liking a decoy
    tag with a distinct score; only 2 seeds): the first-alias id-membership
    pruning is logically engaged and must not change results — decoy persons'
    arm rows must never leak into the join."""
    n_extra = 120
    extra_ids = list(range(1000, 1000 + n_extra))
    decoy_tag = 5000
    nodes = pd.concat([_amp_nodes_pd(), pd.DataFrame({
        "id": extra_ids + [decoy_tag],
        "label__Message": False,
        "label__Person": [True] * n_extra + [False],
        "label__Tag": [False] * n_extra + [True],
        "score": list(range(2000, 2000 + n_extra)) + [9999],
    })], ignore_index=True)
    edges = pd.concat([_amp_edges_pd(), pd.DataFrame({
        "src": extra_ids,
        "dst": decoy_tag,
        "type": "LIKES",
        "weight": [float(i) for i in range(n_extra)],
    })], ignore_index=True)
    expected = pd.DataFrame({
        "mid": [1, 2], "pid": [10, 11],
        "tid": [20.0, None], "tscore": [75.0, None],
    })
    _assert_parity(nodes, edges, Q_ARM_LIKES, expected)
