"""Regression: LDBC IC4-shaped Cypher (comma-MATCH + WITH DISTINCT + CASE + whole-entity
sum-aggregation) runs natively on polars, parity-exact with the pandas oracle.

Three polars lowerings this pins (each previously an honest NIE):
1. HAS_<Label> destination gate in ``binding_rows_polars``: on UNIQUE-node-id graphs the
   pattern runs native (pandas' ``_gfql_disambiguate_has_edge_destination_nodes`` would
   not narrow there either — parity-exact); on DUPLICATE-id graphs polars declines to the
   honest NIE (the chain-combine result has already deduplicated nodes, so the label row
   pandas narrows to may be gone — a native answer would be row-order-dependent).
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


TAG_SCAN = """
MATCH (post:Post {id: $postId })-[:HAS_TAG]->(tag)
RETURN tag.name AS tagName
ORDER BY tagName
"""


def _disambiguation_frames():
    # Node id 500 is duplicated across Tag/Forum but UNREACHABLE from post 600's hop;
    # reached ids {201, 300} are unique -> pandas does NOT narrow, so the unlabeled
    # `tag` op binds the Forum node 300 too. Node id 400 IS duplicated among reached
    # (Tag row + Forum row) from post 601 -> pandas narrows to the Tag row only.
    nodes = pd.DataFrame({
        "id": [600, 601, 201, 300, 400, 400, 500, 500],
        "label__Post": [True, True, False, False, False, False, False, False],
        "label__Tag": [False, False, True, False, True, False, True, False],
        "label__Forum": [False, False, False, True, False, True, False, True],
        "name": [None, None, "t1", "f-node", "t4", "f4", "t5", "f5"],
    })
    edges = pd.DataFrame({
        "src": [600, 600, 601],
        "dst": [201, 300, 400],
        "type": ["HAS_TAG", "HAS_TAG", "HAS_TAG"],
    })
    return nodes, edges


@pytest.mark.parametrize("engine", ["pandas"] + (["polars"] if HAS_POLARS else []))
def test_has_label_narrowing_skipped_when_reached_ids_unique(engine: str) -> None:
    """Global id collisions must NOT trigger narrowing when the REACHED ids are unique
    (pandas probes duplicates on candidate_source ∩ wavefront, not the whole table).
    NOTE: this single-MATCH shape routes through the plain chain executor, NOT
    ``binding_rows_polars`` — it pins that path's parity; the binding-rows gate is
    pinned by the ``test_binding_rows_dup_id_*`` tests below."""
    nodes, edges = _disambiguation_frames()
    if engine == "polars":
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    res = g.gfql(TAG_SCAN, params={"postId": 600}, engine=engine)
    rows = res._nodes.to_pandas() if hasattr(res._nodes, "to_pandas") else res._nodes
    assert sorted(rows["tagName"]) == ["f-node", "t1"]


@pytest.mark.parametrize("engine", ["pandas"] + (["polars"] if HAS_POLARS else []))
def test_has_label_narrowing_applies_on_reached_collision(engine: str) -> None:
    """A reached id colliding across labels narrows the unlabeled op to the HAS_<Label> label.
    NOTE: single-MATCH shape — plain chain executor path, not ``binding_rows_polars``."""
    nodes, edges = _disambiguation_frames()
    if engine == "polars":
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    res = g.gfql(TAG_SCAN, params={"postId": 601}, engine=engine)
    rows = res._nodes.to_pandas() if hasattr(res._nodes, "to_pandas") else res._nodes
    assert sorted(rows["tagName"]) == ["t4"]


# Comma-MATCH multi-alias shape: lowers to rows(binding_ops=...) and (on polars) routes
# through binding_rows_polars — the path carrying the HAS_<Label> duplicate-id gate.
COLLIDE_BINDINGS = """
MATCH (person:Person {id: $personId })-[:KNOWS]-(friend:Person),
      (friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag)
WITH DISTINCT tag, post
RETURN tag.name AS tagName, post.creationDate AS cd
ORDER BY tagName, cd
"""


def _dup_id_frames(tag_first: bool):
    # Node id 200 has TWO rows: a Tag row ('t1') and a Forum row ('FX'); `tag_first`
    # controls their order so a first-occurrence-dependent answer flips between runs.
    # pandas narrows to the Tag row in BOTH orders; a deduplicated native polars run
    # would return whichever row happened to come first — hence the honest NIE.
    rows = [
        (1, True, False, False, False, None, None),
        (2, True, False, False, False, None, None),
        (100, False, True, False, False, 150, None),
    ]
    tag_row = (200, False, False, True, False, None, "t1")
    forum_row = (200, False, False, False, True, None, "FX")
    rows += [tag_row, forum_row] if tag_first else [forum_row, tag_row]
    nodes = pd.DataFrame(
        rows,
        columns=["id", "label__Person", "label__Post", "label__Tag", "label__Forum",
                 "creationDate", "name"],
    )
    edges = pd.DataFrame({
        "src": [1, 100, 100],
        "dst": [2, 2, 200],
        "type": ["KNOWS", "HAS_CREATOR", "HAS_TAG"],
    })
    return nodes, edges


@pytest.mark.parametrize("tag_first", [True, False])
def test_binding_rows_dup_id_pandas_narrows_order_independent(tag_first: bool) -> None:
    """Oracle: pandas narrows to the HAS_TAG label row regardless of node row order."""
    nodes, edges = _dup_id_frames(tag_first)
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    res = g.gfql(COLLIDE_BINDINGS, params={"personId": 1}, engine="pandas")
    assert sorted(zip(res._nodes["tagName"], res._nodes["cd"])) == [("t1", 150.0)]


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
@pytest.mark.parametrize("tag_first", [True, False])
def test_binding_rows_dup_id_polars_declines_nie(tag_first: bool) -> None:
    """Duplicate node ids + HAS_<Label> gate shape → honest NIE on polars, never a
    silently row-order-dependent native answer (parity-or-error contract)."""
    nodes, edges = _dup_id_frames(tag_first)
    g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "src", "dst")
    with pytest.raises(NotImplementedError):
        g.gfql(COLLIDE_BINDINGS, params={"personId": 1}, engine="polars")


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_binding_rows_unique_ids_native_parity() -> None:
    """Same gate shape with UNIQUE node ids stays native and parity-exact (the Forum
    row gets its own id, so pandas would not narrow either)."""
    nodes, edges = _dup_id_frames(tag_first=True)
    nodes = nodes.copy()
    nodes.loc[nodes["label__Forum"].fillna(False).astype(bool), "id"] = 300
    gp = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    expected = sorted(zip(
        gp.gfql(COLLIDE_BINDINGS, params={"personId": 1}, engine="pandas")._nodes["tagName"],
        gp.gfql(COLLIDE_BINDINGS, params={"personId": 1}, engine="pandas")._nodes["cd"],
    ))
    gl = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "src", "dst")
    res = gl.gfql(COLLIDE_BINDINGS, params={"personId": 1}, engine="polars")
    out = res._nodes.to_pandas() if hasattr(res._nodes, "to_pandas") else res._nodes
    assert sorted(zip(out["tagName"], out["cd"])) == expected == [("t1", 150.0)]
