"""Crash-family pins: #1882 (+#1913 finding-4) and #1879 on pandas and polars.

Every case reproduces a verified-live crash at master ``e6625ed28`` with a hand-computed
oracle, plus the both-sides pins that lock the behavior that already worked:

* #1882 -- PUBLIC ``filter_nodes_by_dict`` / ``filter_edges_by_dict`` / ``filter_by_dict``
  on polars frames raised ``AttributeError: 'DataFrame' object has no attribute 'assign'``
  (``filter_by_dict.py`` built the mask with pandas idioms after the resolver routed
  POLARS). Fixed by dispatching to the existing native ``filter_by_dict_polars``; a
  polars-in graph must come back polars (no silent engine swap).
* #1913 finding-4 -- same family: ``prune_self_edges`` on polars edges hit polars'
  column-selecting boolean ``__getitem__`` (raw ``ValueError``; silently selects COLUMNS
  when the row count equals the column count). Now polars-native, with pandas'
  ``NaN != x -> keep`` null-endpoint semantics.
* #1879 pandas half -- Cypher/chain on a nodes-only graph (edges NEVER bound) died with a
  bare ``TypeError: 'NoneType' object is not subscriptable`` (``ast.py`` slicing
  ``g._edges[:0]``), with or without a policy attached (the policy toggle was inert
  because BOTH paths crashed). Now a node-only pattern is served (it needs no edges) and
  an edge pattern declines with a typed GFQLSchemaError.
* #1879 polars half: polars declined every query on a graph without edges with a NotImplementedError.
  It now serves patterns with only nodes like pandas and declines edge patterns with the same
  typed GFQLSchemaError.
* #1879 OPTIONAL MATCH: an optional edge arm on a graph without edges raised a bare
  TypeError on pandas and E304 on polars. An optional arm matches nothing there, so every
  engine now returns the base rows with nulls.
* empty-but-BOUND edges kept working throughout -- pinned on both engines so the
  nodes-only fix can never regress it.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n
from graphistry.compute.exceptions import GFQLSchemaError
from graphistry.compute.filter_by_dict import filter_by_dict

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]


def _frame(engine, df):
    return pl.from_pandas(df) if engine == "polars" else df


def _pd(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _attr_graph(engine):
    """0(a) -x-> 1(b) -y-> 2(a)"""
    ndf = pd.DataFrame({"id": [0, 1, 2], "kind": ["a", "b", "a"]})
    edf = pd.DataFrame({"s": [0, 1], "d": [1, 2], "rel": ["x", "y"]})
    return graphistry.nodes(_frame(engine, ndf), "id").edges(_frame(engine, edf), "s", "d")


def _nodes_only(engine):
    ndf = pd.DataFrame({"id": [0, 1, 2], "label__Person": [True, True, False], "v": [10, 20, 30]})
    return graphistry.nodes(_frame(engine, ndf), "id")


# --------------------------------------------------------------------- #1882 helpers
class TestFilterByDictPolars:
    """#1882: the three public surfaces serve polars frames, polars-in => polars-out."""

    @polars_only
    def test_filter_nodes_by_dict_polars(self):
        out = _attr_graph("polars").filter_nodes_by_dict({"kind": "a"})
        assert isinstance(out._nodes, pl.DataFrame)  # no silent engine swap
        assert sorted(_pd(out._nodes)["id"].tolist()) == [0, 2]

    @polars_only
    def test_filter_edges_by_dict_polars(self):
        out = _attr_graph("polars").filter_edges_by_dict({"rel": "x"})
        assert isinstance(out._edges, pl.DataFrame)
        assert _pd(out._edges)[["s", "d"]].values.tolist() == [[0, 1]]

    @polars_only
    def test_filter_by_dict_bare_polars(self):
        df = pl.DataFrame({"id": [0, 1, 2], "kind": ["a", "b", "a"]})
        out = filter_by_dict(df, {"kind": "a"})
        assert isinstance(out, pl.DataFrame)
        assert sorted(out["id"].to_list()) == [0, 2]

    @polars_only
    @pytest.mark.parametrize("filter_dict", [
        {"kind": "a"},
        {"kind": ["a", "b", None]},  # membership 3VL: a null cell is never a member
        {"id": 2},
    ])
    def test_polars_matches_pandas_oracle(self, filter_dict):
        ndf = pd.DataFrame({"id": [0, 1, 2, 3], "kind": ["a", "b", None, "a"]})
        expect = filter_by_dict(ndf, filter_dict)["id"].tolist()
        got = filter_by_dict(pl.from_pandas(ndf), filter_dict)["id"].to_list()
        assert got == expect

    def test_filter_nodes_by_dict_pandas_unchanged(self):
        out = _attr_graph("pandas").filter_nodes_by_dict({"kind": "a"})
        assert isinstance(out._nodes, pd.DataFrame)
        assert sorted(out._nodes["id"].tolist()) == [0, 2]


class TestPruneSelfEdges:
    """#1913 finding-4: prune_self_edges is engine-dispatched, identical row semantics."""

    #: (0,0) self -> drop; (1,2) keep; (2,2) self -> drop; (None,None) keep (pandas NaN != NaN)
    _EDF = pd.DataFrame({"s": [0.0, 1.0, 2.0, None], "d": [0.0, 2.0, 2.0, None]})

    @pytest.mark.parametrize("engine", ENGINES)
    def test_prune_self_edges(self, engine):
        g = graphistry.nodes(_frame(engine, pd.DataFrame({"id": [0, 1, 2]})), "id") \
            .edges(_frame(engine, self._EDF), "s", "d")
        out = g.prune_self_edges()
        got = _pd(out._edges)
        assert got[["s", "d"]].values.tolist()[0] == [1.0, 2.0]
        assert len(got) == 2 and got[["s", "d"]].isna().all(axis=1).iloc[1]
        if engine == "polars":
            assert isinstance(out._edges, pl.DataFrame)  # no silent engine swap


# --------------------------------------------------------------------- #1879 pandas half
NODES_ONLY_QUERIES = [
    ("plain", "MATCH (a) RETURN a.id AS id ORDER BY id", [0, 1, 2]),
    ("label", "MATCH (a:Person) RETURN a.id AS id ORDER BY id", [0, 1]),
    ("where", "MATCH (a) WHERE a.v > 15 RETURN a.id AS id ORDER BY id", [1, 2]),
    ("agg", "MATCH (a) RETURN count(*) AS c", None),
]


class TestNodesOnlyCypherPandas:
    """#1879: pandas serves node-only patterns on a graph whose edges were NEVER bound."""

    @pytest.mark.parametrize("_name,query,expect", NODES_ONLY_QUERIES)
    def test_nodes_only_serves(self, _name, query, expect):
        res = _nodes_only("pandas").gfql(query, engine="pandas")
        if expect is None:
            assert res._nodes["c"].tolist() == [3]
        else:
            assert res._nodes["id"].tolist() == expect

    @pytest.mark.parametrize("_name,query,expect", NODES_ONLY_QUERIES)
    def test_policy_does_not_toggle_answers(self, _name, query, expect):
        """#1879: attaching a policy must not change the answer (it used to flip paths)."""
        calls = []
        res = _nodes_only("pandas").gfql(
            query, engine="pandas", policy={"preload": lambda ctx: calls.append(ctx["hook"])})
        assert calls == ["preload"]  # the hook actually fired
        baseline = _nodes_only("pandas").gfql(query, engine="pandas")
        pd.testing.assert_frame_equal(
            res._nodes.reset_index(drop=True), baseline._nodes.reset_index(drop=True))

    def test_chain_node_only_serves_and_result_stays_nodes_only(self):
        out = _nodes_only("pandas").chain([n({"id": 0})])
        assert out._nodes["id"].tolist() == [0]
        assert out._edges is None  # result mirrors the nodes-only input

    def test_chain_with_policy_serves_and_result_stays_nodes_only(self):
        """A policy forces the non-fast chain path -- the path that crashed at ast.py:265 --
        and the served result must still drop the internally synthesized empty edges."""
        out = _nodes_only("pandas").gfql(
            [n({"id": 0})], engine="pandas", policy={"preload": lambda ctx: None})
        assert out._nodes["id"].tolist() == [0]
        assert out._edges is None  # result mirrors the nodes-only input

    def test_edge_pattern_declines_typed_cypher(self):
        with pytest.raises(GFQLSchemaError, match="no edges"):
            _nodes_only("pandas").gfql("MATCH (a)-[r]->(b) RETURN a.id AS id", engine="pandas")

    def test_edge_op_declines_typed_chain(self):
        with pytest.raises(GFQLSchemaError, match="no edges"):
            _nodes_only("pandas").chain([n(), e_forward(), n()])


class TestNodesOnlyCypherPolars:
    """#1879 polars half: polars serves patterns with only nodes like pandas on a graph without edges."""

    @polars_only
    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    @pytest.mark.parametrize("_name,query,expect", NODES_ONLY_QUERIES)
    def test_nodes_only_matches_pandas(self, _name, query, expect, lazy):
        g = _nodes_only("polars")
        if lazy:
            g = g.nodes(g._nodes.lazy(), "id")
        res = g.gfql(query, engine="polars")
        assert isinstance(res._nodes, pl.DataFrame)
        assert res._edges is None
        baseline = _nodes_only("pandas").gfql(query, engine="pandas")
        pd.testing.assert_frame_equal(
            res._nodes.to_pandas(), baseline._nodes.reset_index(drop=True), check_dtype=False)

    @polars_only
    @pytest.mark.parametrize("_name,query,expect", NODES_ONLY_QUERIES)
    def test_policy_does_not_toggle_answers(self, _name, query, expect):
        calls = []
        res = _nodes_only("polars").gfql(
            query, engine="polars", policy={"preload": lambda ctx: calls.append(ctx["hook"])})
        assert calls == ["preload"]
        baseline = _nodes_only("polars").gfql(query, engine="polars")
        assert res._nodes.equals(baseline._nodes)

    @polars_only
    def test_chain_node_only_serves_and_result_stays_nodes_only(self):
        out = _nodes_only("polars").gfql([n({"id": 0})], engine="polars")
        assert out._nodes["id"].to_list() == [0]
        assert out._edges is None

    @polars_only
    def test_edge_pattern_declines_typed_cypher(self):
        with pytest.raises(GFQLSchemaError, match="no edges"):
            _nodes_only("polars").gfql("MATCH (a)-[r]->(b) RETURN a.id AS id", engine="polars")

    @polars_only
    def test_edge_op_declines_typed_chain(self):
        with pytest.raises(GFQLSchemaError, match="no edges"):
            _nodes_only("polars").gfql([n(), e_forward(), n()], engine="polars")


OPTIONAL_ON_NODES_ONLY = [
    ("ids", "MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n.id AS id ORDER BY id",
     {"id": [0, 1, 2]}),
    ("nulls", "MATCH (n) WHERE n.id = 1 OPTIONAL MATCH (n)<-[r]-(m) RETURN n.id AS id, m.id AS mid",
     {"id": [1], "mid": [None]}),
    ("count", "MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n.id AS id, count(m) AS c ORDER BY id",
     {"id": [0, 1, 2], "c": [0, 0, 0]}),
]


class TestNodesOnlyOptionalMatch:
    """#1879: an OPTIONAL edge arm on a graph without edges matches nothing on every engine."""

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("with_policy", [False, True], ids=["no_policy", "policy"])
    @pytest.mark.parametrize("_name,query,expect", OPTIONAL_ON_NODES_ONLY)
    def test_optional_arm_returns_base_rows(self, engine, with_policy, _name, query, expect):
        kwargs = {"policy": {"preload": lambda ctx: None}} if with_policy else {}
        res = _nodes_only(engine).gfql(query, engine=engine, **kwargs)
        got = _pd(res._nodes)
        assert got.astype(object).where(got.notna(), None).to_dict("list") == expect

    @polars_only
    def test_optional_arm_on_lazy_polars(self):
        g = _nodes_only("polars")
        g = g.nodes(g._nodes.lazy(), "id")
        res = g.gfql(OPTIONAL_ON_NODES_ONLY[0][1], engine="polars")
        assert res._nodes["id"].to_list() == [0, 1, 2]

    @pytest.mark.parametrize("engine", ENGINES)
    def test_required_edge_pattern_still_declines_typed(self, engine):
        with pytest.raises(GFQLSchemaError, match="no edges"):
            _nodes_only(engine).gfql("MATCH (n)-[r]->(m) RETURN n.id AS id", engine=engine)


class TestEmptyBoundEdgesStillServe:
    """Both-sides pin: empty-but-BOUND edges worked before the fix and must keep working."""

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("_name,query,expect", NODES_ONLY_QUERIES)
    def test_empty_bound_edges(self, engine, _name, query, expect):
        g = _nodes_only(engine).edges(
            _frame(engine, pd.DataFrame({"s": pd.Series([], dtype="int64"),
                                         "d": pd.Series([], dtype="int64")})), "s", "d")
        res = g.gfql(query, engine=engine)
        got = _pd(res._nodes)
        if expect is None:
            assert got["c"].tolist() == [3]
        else:
            assert got["id"].tolist() == expect
