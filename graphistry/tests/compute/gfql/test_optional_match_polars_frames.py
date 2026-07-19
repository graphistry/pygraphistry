"""Regression: connected OPTIONAL MATCH seed extraction must be engine-polymorphic.

_optional_arm_start_nodes (gfql_unified.py) applied pandas-only frame ops
(.dropna()/.drop_duplicates()/.rename(columns=)/boolean-mask __getitem__) to the joined
binding rows, so an IS7-shaped Cypher query (MATCH ... OPTIONAL MATCH ...) on
engine='polars' crashed with AttributeError before reaching the row pipeline
(LDBC SNB interactive-short-7 via the pyg-bench harness). The polars row pipeline may
still honestly decline the query (NotImplementedError, parity-or-error by design), but
it must never crash with a pandas-ism.
"""
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
    return pd.DataFrame({
        "src": [2, 3, 2, 3, 1, 10, 11],
        "dst": [1, 1, 10, 11, 12, 11, 10],
        "type": ["REPLY_OF", "REPLY_OF", "HAS_CREATOR", "HAS_CREATOR", "HAS_CREATOR",
                 "KNOWS", "KNOWS"],
    })


def test_optional_match_pandas_oracle() -> None:
    g = graphistry.nodes(_nodes_pd(), "id").edges(_edges_pd(), "src", "dst")
    res = g.gfql(IS7_SHAPED, params={"messageId": 1}, engine="pandas")
    rows = res._nodes.reset_index(drop=True)
    assert list(rows["commentId"]) == [3, 2]
    # message creator (12) KNOWS nobody -> flag false on both rows
    assert list(rows["replyAuthorKnowsOriginalMessageAuthor"]) == [False, False]


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
    """Full IS7 (CASE projection) must run or honestly decline — never a pandas-ism crash."""
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
    assert list(rows["replyAuthorKnowsOriginalMessageAuthor"]) == [False, False]
