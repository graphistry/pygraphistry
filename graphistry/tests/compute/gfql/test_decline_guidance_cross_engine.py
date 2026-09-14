"""Polars decline guidance must name every engine that actually serves the query.

The polars engine declines some shapes with an honest NotImplementedError. Those shapes run
on the pandas-idiom path, which serves pandas AND cuDF — so the decline message must offer
BOTH engines, or a GPU user is sent to CPU for no reason (the #1928 report). Each pin here
was verified empirically: polars raises a typed NIE naming the unsupported feature, while
pandas and cuDF both serve the query with the hand-computed values asserted below.

Counter-pins at the bottom cover shapes cuDF does NOT serve (mixed-type IsIn, List-vs-scalar):
their messages stay pandas-only on purpose, and a test asserts they never advertise cuDF.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import pandas as pd
import pytest

import graphistry
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTLet, call
from graphistry.compute.ast import n as ast_n
from graphistry.compute.predicates.is_in import is_in

BOTH_ENGINES = "use engine='pandas' or engine='cudf'"
PANDAS_ONLY = "use engine='pandas'"

NODES = pd.DataFrame({
    "id": ["a", "b", "c", "d"],
    "v": [1, 5, 3, 7],
    "kind": ["person", "person", "company", "person"],
})
EDGES = pd.DataFrame({"s": ["a", "b", "a", "d"], "d": ["b", "c", "c", "a"]})

#: Hand-computed from NODES/EDGES: edges where source v < destination v.
SAME_PATH_LT_ROWS = [("a", "b"), ("a", "c")]
#: Hand-computed: (person v, out-neighbor id) over all out-edges of person nodes.
SCALAR_CARRY_ROWS = [(1, "b"), (1, "c"), (5, "c"), (7, "a")]
#: Hand-computed: nodes reachable from 'a' in exactly 2..3 forward hops (a->b->c only).
MIN_HOPS_TARGETS = ["c"]
#: Hand-computed: each node's v twice, sorted.
UNWIND_TWICE_SORTED = [1, 1, 3, 3, 5, 5, 7, 7]
#: Hand-computed: node ids ordered by v descending.
ORDER_BY_V_DESC_IDS = ["d", "b", "c", "a"]


def _graph(engine: str) -> Plottable:
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return graphistry.nodes(cudf.from_pandas(NODES), "id").edges(cudf.from_pandas(EDGES), "s", "d")
    return graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")


def _rows(g: Plottable, cols: List[str]) -> List[Tuple[Any, ...]]:
    df = g._nodes
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    return sorted(pdf[cols].itertuples(index=False, name=None))


SAME_PATH_Q = "MATCH (x)-[e]->(y) WHERE x.v < y.v RETURN x.id AS xid, y.id AS yid"
SCALAR_CARRY_Q = (
    "MATCH (p {kind:'person'}) WITH p, p.v AS pv MATCH (p)-[]->(q) RETURN pv, q.id AS qid"
)
MIN_HOPS_Q = "MATCH (r {id:'a'})-[*2..3]->(t) RETURN t.id AS tid"
UNWIND_Q = "MATCH (x) UNWIND [x.v, x.v] AS z RETURN z"


class TestPolarsDeclineNamesBothServingEngines:
    """Each polars NIE below is a shape pandas AND cuDF serve: the message must say so."""

    @pytest.mark.parametrize("query,feature", [
        (SAME_PATH_Q, "cross-entity (same-path) WHERE"),
        (SCALAR_CARRY_Q, "scalar WITH columns into the trailing MATCH"),
        (MIN_HOPS_Q, "not yet hop-gated"),
        (UNWIND_Q, "cypher row op"),
    ])
    def test_polars_decline_offers_cudf(self, query: str, feature: str) -> None:
        with pytest.raises(NotImplementedError) as nie:
            _graph("polars").gfql(query, engine="polars")
        msg = str(nie.value)
        assert feature in msg, msg
        assert BOTH_ENGINES in msg, msg

    @pytest.mark.parametrize("engine", ["pandas", "cudf"])
    def test_same_path_where_served(self, engine: str) -> None:
        got = _rows(_graph(engine).gfql(SAME_PATH_Q, engine=engine), ["xid", "yid"])
        assert got == SAME_PATH_LT_ROWS

    @pytest.mark.parametrize("engine", ["pandas", "cudf"])
    def test_scalar_with_carry_served(self, engine: str) -> None:
        got = _rows(_graph(engine).gfql(SCALAR_CARRY_Q, engine=engine), ["pv", "qid"])
        assert [(int(pv), qid) for pv, qid in got] == SCALAR_CARRY_ROWS

    @pytest.mark.parametrize("engine", ["pandas", "cudf"])
    def test_min_hops_alias_served(self, engine: str) -> None:
        got = _rows(_graph(engine).gfql(MIN_HOPS_Q, engine=engine), ["tid"])
        assert [t for (t,) in got] == MIN_HOPS_TARGETS

    @pytest.mark.parametrize("engine", ["pandas", "cudf"])
    def test_unwind_row_op_served(self, engine: str) -> None:
        got = _rows(_graph(engine).gfql(UNWIND_Q, engine=engine), ["z"])
        assert [int(z) for (z,) in got] == UNWIND_TWICE_SORTED


class TestRowPipelineDagDecline:
    """call('order_by') in a let() DAG: polars row pipeline declines naming both engines."""

    DAG = ASTLet({"k": call("order_by", {"keys": [["v", "desc"]]})})

    def test_polars_decline_offers_cudf(self) -> None:
        with pytest.raises(NotImplementedError) as nie:
            _graph("polars").gfql(self.DAG, output="k", engine="polars")
        msg = str(nie.value)
        assert "polars row pipeline does not yet support op 'order_by'" in msg, msg
        assert BOTH_ENGINES in msg, msg

    @pytest.mark.parametrize("engine", ["pandas", "cudf"])
    def test_served_in_order(self, engine: str) -> None:
        got = _graph(engine).gfql(self.DAG, output="k", engine=engine)._nodes
        pdf = got.to_pandas() if hasattr(got, "to_pandas") else got
        assert list(pdf["id"]) == ORDER_BY_V_DESC_IDS


FLOAT_ENTITY_NODES = pd.DataFrame({
    "id": ["s", "m", "c"], "lbl": ["Single", "M", "C"], "f": [1.5, 2.5, 3.5],
})
FLOAT_ENTITY_EDGES = pd.DataFrame({"s": ["s", "m"], "d": ["m", "c"]})
FLOAT_ENTITY_Q = (
    "MATCH (a {lbl:'Single'}), (x {lbl:'C'}) OPTIONAL MATCH (a)-[*]->(x) RETURN x"
)
#: Hand-computed legacy entity text for node c (id excluded): its lbl and float f.
FLOAT_ENTITY_TEXT = "({lbl: 'C', f: 3.5})"


class TestFloatWholeEntityProjectionDecline:
    """Entity-text RETURN over a float column: polars declines naming both engines."""

    @staticmethod
    def _fgraph(engine: str) -> Plottable:
        if engine == "polars":
            pl = pytest.importorskip("polars")
            return graphistry.nodes(pl.from_pandas(FLOAT_ENTITY_NODES), "id").edges(
                pl.from_pandas(FLOAT_ENTITY_EDGES), "s", "d")
        if engine == "cudf":
            cudf = pytest.importorskip("cudf")
            return graphistry.nodes(cudf.from_pandas(FLOAT_ENTITY_NODES), "id").edges(
                cudf.from_pandas(FLOAT_ENTITY_EDGES), "s", "d")
        return graphistry.nodes(FLOAT_ENTITY_NODES, "id").edges(FLOAT_ENTITY_EDGES, "s", "d")

    def test_polars_decline_offers_cudf(self) -> None:
        with pytest.raises(NotImplementedError) as nie:
            self._fgraph("polars").gfql(FLOAT_ENTITY_Q, engine="polars")
        msg = str(nie.value)
        assert "cypher result projection" in msg, msg
        assert BOTH_ENGINES in msg, msg

    @pytest.mark.parametrize("engine", ["pandas", "cudf"])
    def test_served(self, engine: str) -> None:
        got = self._fgraph(engine).gfql(FLOAT_ENTITY_Q, engine=engine)._nodes
        pdf = got.to_pandas() if hasattr(got, "to_pandas") else got
        assert list(pdf["x"]) == [FLOAT_ENTITY_TEXT]


NUM_VS_STR_Q = "MATCH (x) WHERE x.v > 'foo' RETURN x.id AS i"


class TestNumericVsStringPredicateDecline:
    """Cypher numeric-vs-string compare: polars declines; pandas/cuDF answer 0 rows."""

    def test_polars_decline_offers_cudf(self) -> None:
        with pytest.raises(NotImplementedError) as nie:
            _graph("polars").gfql(NUM_VS_STR_Q, engine="polars")
        msg = str(nie.value)
        assert "numeric-vs-string" in msg, msg
        assert BOTH_ENGINES in msg, msg

    @pytest.mark.parametrize("engine", ["pandas", "cudf"])
    def test_served_empty(self, engine: str) -> None:
        got = _graph(engine).gfql(NUM_VS_STR_Q, engine=engine)._nodes
        assert len(got) == 0


class TestPandasOnlyDeclinesStayPandasOnly:
    """Counter-pins: shapes cuDF does NOT serve keep a pandas-only suggestion."""

    MIXED_ISIN = [ast_n({"v": is_in([1, "a"])})]

    def test_polars_mixed_isin_decline_stays_pandas_only(self) -> None:
        with pytest.raises(NotImplementedError) as nie:
            _graph("polars").gfql(self.MIXED_ISIN, engine="polars")
        msg = str(nie.value)
        assert "IsIn predicate" in msg, msg
        assert PANDAS_ONLY in msg, msg
        assert "engine='cudf'" not in msg, msg

    def test_pandas_serves_mixed_isin(self) -> None:
        got = _graph("pandas").gfql(self.MIXED_ISIN, engine="pandas")._nodes
        assert sorted(got["id"]) == ["a"]

    def test_cudf_errors_on_mixed_isin(self) -> None:
        # cuDF cannot build the mixed-type membership column, so the message must not offer it.
        pytest.importorskip("cudf")
        with pytest.raises(Exception):
            _graph("cudf").gfql(self.MIXED_ISIN, engine="cudf")
