"""Differential parity: native polars ``WITH ... MATCH ...`` re-entry == pandas (oracle) (#1273).

A ``WITH`` clause that projects a whole-row node alias (optionally ``DISTINCT`` / filtered /
ordered+bounded) feeds a SUBSEQUENT ``MATCH`` that re-traverses from those carried nodes. Before
this change the polars engine declined the WITH->MATCH boundary ("could not recover carried node
identities") because the native projector never emitted the whole-entity id side-channel and the
bounded-reentry executor + seeded binding pipeline were pandas-only.

Pandas is the oracle: every supported query returns an identical result table, polars-typed (no
pandas bridge). Shapes still out of subset (scalar WITH columns carried alongside the whole-row
alias) must raise NotImplementedError, never silently diverge. Companion of the seeded
``rows(binding_ops)`` work in test_engine_polars_binding_rows.
"""
import random

import pandas as pd
import pytest

import graphistry

pl = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

NODES = pd.DataFrame({
    "id": [0, 1, 2, 3, 4, 5],
    "kind": ["person", "person", "post", "post", "post", "comment"],
    "val": [10, 20, 30, 40, 50, 60],
})
EDGES = pd.DataFrame({
    "s": [0, 0, 1, 1, 2, 3, 0, 4],
    "d": [1, 2, 2, 3, 4, 4, 3, 5],
    "type": ["KNOWS", "HAS_CREATOR", "HAS_CREATOR", "KNOWS", "LINK", "LINK", "KNOWS", "HAS_CREATOR"],
})
BASE = graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")


def _to_pandas(df):
    if df is not None and "polars" in type(df).__module__:
        return df.to_pandas()
    return df


def _round_floats(df):
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.astype("float64").round(6)
    return out


def _norm(df):
    df = _round_floats(df).where(lambda d: d.notna(), "∅")
    return df.astype(str).sort_values(list(df.columns)).reset_index(drop=True)


def _assert_parity(g, query, *, order_sensitive=False):
    rpd = g.gfql(query, engine="pandas")._nodes
    rpl = g.gfql(query, engine="polars")._nodes
    assert "polars" in type(rpl).__module__, f"expected polars frame for {query!r}"
    a = _to_pandas(rpd).reset_index(drop=True)
    b = _to_pandas(rpl).reset_index(drop=True)
    assert list(a.columns) == list(b.columns), (
        f"columns differ for {query!r}: {list(a.columns)} vs {list(b.columns)}"
    )
    assert len(a) == len(b), f"row count differs for {query!r}: {len(a)} vs {len(b)}"
    if len(a) == 0:
        return
    an, bn = _norm(a), _norm(b)
    if order_sensitive:
        an = _round_floats(a).where(lambda d: d.notna(), "∅").astype(str)
        bn = _round_floats(b).where(lambda d: d.notna(), "∅").astype(str)
    pd.testing.assert_frame_equal(an, bn, check_dtype=False)


# ---------------------------------------------------------------------------
# Curated parity corpus (whole-row WITH -> MATCH re-entry, native on both engines)
# ---------------------------------------------------------------------------

CORPUS = [
    # DISTINCT whole-row carry -> forward re-entry, various RETURN shapes
    "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend "
    "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN count(post) AS c",
    "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend "
    "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN count(*) AS c",
    "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend "
    "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN post.id AS pid",
    "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend "
    "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN post.id AS pid, post.kind AS pk",
    "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend "
    "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN post",
    "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend "
    "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN DISTINCT post.kind AS pk",
    # non-DISTINCT whole-row carry (prefix may yield duplicate friend rows)
    "MATCH (p {id:0})-[:KNOWS]-(friend) WITH friend "
    "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN count(post) AS c",
    # filtered WITH prefix
    "MATCH (p {id:0})-[:KNOWS]-(friend) WHERE friend.val > 15 WITH DISTINCT friend "
    "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN count(post) AS c",
    "MATCH (p {id:0})-[:KNOWS]-(friend) WHERE friend.kind = 'person' WITH DISTINCT friend "
    "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN post.id AS pid",
    # ORDER BY + bounded LIMIT prefix (preserves WITH row order into re-entry)
    "MATCH (p {id:0})-[:KNOWS]-(friend) WITH friend ORDER BY friend.val LIMIT 2 "
    "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN post.id AS pid",
    # directed re-entry (forward / reverse) and undirected prefix
    "MATCH (p {id:0})<-[:KNOWS]-(friend) WITH DISTINCT friend "
    "MATCH (friend)-[:HAS_CREATOR]->(post) RETURN count(post) AS c",
    "MATCH (p {id:0})-[:KNOWS]->(friend) WITH DISTINCT friend "
    "MATCH (friend)<-[:HAS_CREATOR]-(post) RETURN count(post) AS c",
    # suffix endpoint property filter
    "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend "
    "MATCH (friend)-[:HAS_CREATOR]-(post {kind:'post'}) RETURN count(post) AS c",
]


@pytest.mark.parametrize("query", CORPUS)
def test_with_match_reentry_parity(query):
    _assert_parity(BASE, query)


def test_reentry_count_pin():
    """Exact-value pin for the canonical IC6/IC11-shaped re-entry (regression guard)."""
    q = (
        "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend "
        "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN count(post) AS c"
    )
    rpl = _to_pandas(BASE.gfql(q, engine="polars")._nodes)
    rpd = _to_pandas(BASE.gfql(q, engine="pandas")._nodes)
    assert rpl["c"].tolist() == rpd["c"].tolist()
    assert rpl["c"].tolist() == [1]


def test_projector_emits_entity_projection_meta_polars():
    """The native polars projector must emit the whole-entity id side-channel the bounded
    reentry executor reads to recover carried node identities (the root-cause gap). A terminal
    ``RETURN friend`` exercises the same whole-entity projection branch."""
    prefix = "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend RETURN friend"
    res = BASE.gfql(prefix, engine="polars")
    meta = getattr(res, "_cypher_entity_projection_meta", None)
    assert isinstance(meta, dict) and "friend" in meta, meta
    entry = meta["friend"]
    assert entry["table"] == "nodes"
    assert entry["id_column"] == "id"
    # ids Series is polars-typed and row-aligned with the (post-DISTINCT) prefix rows.
    ids = entry["ids"]
    assert "polars" in type(ids).__module__
    assert ids.to_list() == [1, 3]


def test_projector_text_path_emits_meta_polars():
    """The entity-TEXT projection branch (structured=False, single int/str/bool node) must also
    emit the id side-channel — mirrors the pandas projector, which records meta for whole-entity
    columns regardless of structured/text rendering."""
    from graphistry.compute.gfql.cypher.lowering import ResultProjectionColumn, ResultProjectionPlan
    from graphistry.compute.gfql.lazy.engine.polars.projection import apply_result_projection_polars

    rows = pl.DataFrame({"id": [7, 9], "val": [1, 2], "n": [True, True]})
    g = graphistry.nodes(rows, "id")
    plan = ResultProjectionPlan(
        alias="n",
        table="nodes",
        columns=(ResultProjectionColumn(output_name="n", kind="whole_row", source_name="n"),),
        exclude_columns=(),
    )
    out = apply_result_projection_polars(g, plan, structured=False)
    meta = getattr(out, "_cypher_entity_projection_meta", None)
    assert isinstance(meta, dict) and "n" in meta
    assert meta["n"]["id_column"] == "id"
    assert meta["n"]["ids"].to_list() == [7, 9]


def test_binding_rows_seed_pandas_and_missing_id_col():
    """Seed-frame normalization branches in binding_rows_polars: a pandas seed is converted to
    polars; a seed lacking the node-id column declines (None)."""
    from graphistry.compute.ast import n as _n, e_forward as _ef, serialize_binding_ops as _ser
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import binding_rows_polars

    bo = _ser([_n(name="a"), _ef(), _n(name="b")])
    g = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")

    # pandas seed frame -> df_to_engine conversion path
    seeded_pd = g.bind()
    setattr(seeded_pd, "_gfql_start_nodes", pd.DataFrame({"id": [0]}))
    out = binding_rows_polars(seeded_pd, bo)
    assert out is not None
    assert set(out._nodes["a"].to_list()) == {0}

    # seed frame missing the node-id column -> decline
    seeded_bad = g.bind()
    setattr(seeded_bad, "_gfql_start_nodes", pl.DataFrame({"other": [0]}))
    assert binding_rows_polars(seeded_bad, bo) is None

    # seeded node-cartesian (disconnected trailing aliases) -> decline: the cartesian builder
    # doesn't thread the seed, so running it would silently ignore the WITH constraint.
    from graphistry.compute.ast import n as _n2, serialize_binding_ops as _ser2
    cart = _ser2([_n2(name="a"), _n2(name="b")])
    seeded_cart = g.bind()
    setattr(seeded_cart, "_gfql_start_nodes", pl.DataFrame({"id": [0]}))
    assert binding_rows_polars(seeded_cart, cart) is None


def test_with_match_reentry_binding_table_parity(monkeypatch):
    """End-to-end parity for the case that routes the WITH re-entry seed through the multi-alias
    BINDING-ROWS path (`binding_rows_polars`), not the single-alias hop path: a trailing MATCH that
    binds two fresh aliases and RETURNs both forces a binding table, seeded from the carried WITH ids.
    The direct-call tests above pin the seed branches in isolation; this proves the whole cypher
    stack (lower -> reentry -> seeded binding_rows -> project) matches pandas on real output columns.
    Guards that the seeded binding path is actually exercised, so it can't silently stop covering it."""
    from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as _rp

    seeded_calls = {"n": 0}
    _orig = _rp.binding_rows_polars

    def _counting(g, binding_ops, *a, **k):
        if getattr(g, "_gfql_start_nodes", None) is not None:
            seeded_calls["n"] += 1
        return _orig(g, binding_ops, *a, **k)

    monkeypatch.setattr(_rp, "binding_rows_polars", _counting)

    # (friend)-[:HAS_CREATOR]-(a)-[:LINK]-(b): two fresh aliases + multi-alias RETURN -> binding table,
    # with the first alias (friend) constrained to the carried WITH ids.
    q = (
        "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend "
        "MATCH (friend)-[:HAS_CREATOR]-(a)-[:LINK]-(b) RETURN a.id AS aid, b.id AS bid"
    )
    _assert_parity(BASE, q)
    assert seeded_calls["n"] > 0, "query did not route through the seeded binding_rows path"


def test_scalar_carry_declines_cleanly_polars():
    """WITH that carries a scalar column ALONGSIDE the whole-row alias into the trailing MATCH is
    pandas-only so far; polars must DECLINE (NotImplementedError), never crash or silently diverge.
    pandas still answers it (oracle), so this documents an honest capability gap."""
    q = (
        "MATCH (p {id:0})-[:KNOWS]-(friend) WITH DISTINCT friend, friend.val AS fv "
        "MATCH (friend)-[:HAS_CREATOR]-(post) RETURN post.id AS pid, fv"
    )
    # pandas oracle succeeds
    _to_pandas(BASE.gfql(q, engine="pandas")._nodes)
    # polars declines honestly
    with pytest.raises(NotImplementedError):
        BASE.gfql(q, engine="polars")


# ---------------------------------------------------------------------------
# Seeded differential fuzz (bounded, deterministic) vs the pandas oracle
# ---------------------------------------------------------------------------

_RELS = ["KNOWS", "HAS_CREATOR", "LIKES", "LINK"]
_KINDS = ["person", "post", "comment", "forum"]


def _fuzz_graph(seed, n):
    rng = random.Random(seed)
    nodes = pd.DataFrame({
        "id": list(range(n)),
        "kind": [rng.choice(_KINDS) for _ in range(n)],
        "val": [rng.randint(0, 20) for _ in range(n)],
    })
    m = rng.randint(n, n * 3)
    edges = pd.DataFrame({
        "s": [rng.randint(0, n - 1) for _ in range(m)],
        "d": [rng.randint(0, n - 1) for _ in range(m)],
        "type": [rng.choice(_RELS) for _ in range(m)],
    })
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _fuzz_query(rng, n):
    x = rng.randint(0, n - 1)
    r1, r2 = rng.choice(_RELS), rng.choice(_RELS)

    def arrow(d, rel):
        return {"->": f"-[:{rel}]->", "<-": f"<-[:{rel}]-"}.get(d, f"-[:{rel}]-")

    d1, d2 = rng.choice(["-", "->", "<-"]), rng.choice(["-", "->", "<-"])
    distinct = rng.choice(["DISTINCT ", ""])
    where = rng.choice(["", f" WHERE a.val > {rng.randint(0, 20)}", f" WHERE a.kind = '{rng.choice(_KINDS)}'"])
    tail = rng.choice(["", "", f" ORDER BY a.val LIMIT {rng.randint(1, 4)}"])
    prefix = f"MATCH (p {{id:{x}}}){arrow(d1, r1)}(a){where} WITH {distinct}a{tail}"
    b_filt = rng.choice(["", "", f" {{kind:'{rng.choice(_KINDS)}'}}"])
    suffix_ret = rng.choice([
        "RETURN b.id AS bid",
        "RETURN b.id AS bid, b.kind AS bk",
        "RETURN b",
        "RETURN count(b) AS c",
        "RETURN count(*) AS c",
        "RETURN DISTINCT b.kind AS bk",
    ])
    return f"{prefix} MATCH (a){arrow(d2, r2)}(b{b_filt}) {suffix_ret}"


def test_with_match_reentry_differential_fuzz():
    """Bounded seeded fuzz: for every supported WITH->MATCH shape, polars must match the pandas
    oracle exactly; unsupported shapes must decline (NotImplementedError). No silent-wrong."""
    rng = random.Random(20260718)
    agree = decline = 0
    checked = 0
    for _ in range(120):
        g = _fuzz_graph(rng.randint(0, 10_000), rng.randint(4, 12))
        q = _fuzz_query(rng, len(g._nodes))
        try:
            rpd = _to_pandas(g.gfql(q, engine="pandas")._nodes)
        except Exception:
            # pandas oracle itself declined at parse/lower — polars must not succeed here.
            with pytest.raises(Exception):
                g.gfql(q, engine="polars")
            continue
        try:
            rpl = _to_pandas(g.gfql(q, engine="polars")._nodes)
        except NotImplementedError:
            decline += 1
            continue
        checked += 1
        assert list(rpd.columns) == list(rpl.columns), f"cols differ for {q!r}"
        if len(rpd) or len(rpl):
            pd.testing.assert_frame_equal(_norm(rpd), _norm(rpl), check_dtype=False)
        agree += 1
    # Guard: the sweep must actually exercise the native re-entry path, not just declines.
    assert checked > 20, f"fuzz did not exercise enough native shapes (checked={checked})"
