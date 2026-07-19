"""Regression: LDBC IC4-shaped Cypher (comma-MATCH + WITH DISTINCT + CASE + whole-entity
sum-aggregation) runs natively on polars, parity-exact with the pandas oracle.

Three polars lowerings this pins (each previously an honest NIE):
1. HAS_<Label> destination disambiguation in ``binding_rows_polars`` (pandas'
   ``_gfql_disambiguate_has_edge_destination_nodes`` candidate-domain rule: narrow an
   UNLABELED next-node op after a HAS_<Label> edge to that label ONLY when candidate node
   ids collide across labels).
2. ``alias.__gfql_node_id__`` (whole-entity identity key, #1650) resolves to the bare
   ``alias`` id column (the polars bindings table doesn't carry pandas' join-residue
   columns; the bare alias column IS the identity key).
3. ``group_by(key_prefixes=...)`` whole-entity key expansion (every ``<prefix>*`` column
   joins the key set — functionally dependent on the identity key, so group sizes are
   unchanged).
"""
import pandas as pd
import pytest

import graphistry

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


IC4_SHAPED = """
MATCH (person:Person {id: $personId })-[:KNOWS]-(friend:Person),
      (friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag)
WITH DISTINCT tag, post
WITH tag,
     CASE
       WHEN $startDate <= post.creationDate < $endDate THEN 1
       ELSE 0
     END AS valid,
     CASE
       WHEN post.creationDate < $startDate THEN 1
       ELSE 0
     END AS inValid
WITH tag, sum(valid) AS postCount, sum(inValid) AS inValidPostCount
WHERE postCount>0 AND inValidPostCount=0
RETURN tag.name AS tagName, postCount
ORDER BY postCount DESC, tagName ASC
LIMIT 10
"""


def _graph_frames():
    # person 1 KNOWS 2,3; posts 100(by 2, cD 150 in-window), 101(by 3, cD 50 BEFORE window
    # -> its tag t2 is invalidated), 102(by 2, cD 250 in-window). Tags: t1 on 100+102
    # (postCount 2), t2 on 101+102 (invalidated by 101). Tag node id 200 COLLIDES with
    # nothing here; node 300 is a Forum sharing no ids (label narrowing exercised via the
    # unlabeled `tag` op after HAS_TAG).
    nodes = pd.DataFrame({
        "id": [1, 2, 3, 100, 101, 102, 200, 201, 300],
        "label__Person": [True, True, True, False, False, False, False, False, False],
        "label__Post": [False, False, False, True, True, True, False, False, False],
        "label__Tag": [False, False, False, False, False, False, True, True, False],
        "label__Forum": [False, False, False, False, False, False, False, False, True],
        "creationDate": [None, None, None, 150, 50, 250, None, None, None],
        "name": [None, None, None, None, None, None, "t1", "t2", None],
    })
    edges = pd.DataFrame({
        "src": [1, 1, 100, 101, 102, 100, 102, 101, 102],
        "dst": [2, 3, 2, 3, 2, 200, 200, 201, 201],
        "type": ["KNOWS", "KNOWS", "HAS_CREATOR", "HAS_CREATOR", "HAS_CREATOR",
                 "HAS_TAG", "HAS_TAG", "HAS_TAG", "HAS_TAG"],
    })
    return nodes, edges


PARAMS = {"personId": 1, "startDate": 100, "endDate": 300}
EXPECTED = [("t1", 2)]  # t2 invalidated by pre-window post 101


def test_ic4_shaped_pandas_oracle() -> None:
    nodes, edges = _graph_frames()
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    res = g.gfql(IC4_SHAPED, params=PARAMS, engine="pandas")
    rows = res._nodes.reset_index(drop=True)
    assert list(zip(rows["tagName"], rows["postCount"])) == EXPECTED


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_ic4_shaped_polars_native_parity() -> None:
    nodes, edges = _graph_frames()
    g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "src", "dst")
    res = g.gfql(IC4_SHAPED, params=PARAMS, engine="polars")
    rows = res._nodes.to_pandas() if hasattr(res._nodes, "to_pandas") else res._nodes
    rows = rows.reset_index(drop=True)
    assert list(zip(rows["tagName"], rows["postCount"])) == EXPECTED
