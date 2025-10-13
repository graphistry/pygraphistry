# -*- coding: utf-8 -*-
import os, pandas as pd, pytest, unittest

from graphistry.compute import ComputeMixin
from graphistry.plotter import PlotterBase
from graphistry.tests.common import NoAuthTestCase
from graphistry.umap_utils import UMAPMixin
from graphistry.feature_utils import FeatureMixin


class CG(ComputeMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ComputeMixin.__init__(self, *args, **kwargs)


class CGFull(UMAPMixin, FeatureMixin, ComputeMixin, PlotterBase, object):
    def __init__(self, *args, **kwargs):
        print("CGFull init")
        super(CGFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        FeatureMixin.__init__(self, *args, **kwargs)
        UMAPMixin.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)


class TestComputeMixin(NoAuthTestCase):
    def test_materialize(self):
        cg = CGFull()
        g = cg.edges(
            pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "a", "d"]}), "s", "d"
        )
        g = g.materialize_nodes()
        assert g._nodes.to_dict(orient="records") == [
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ]
        assert g._node == "id"

    def test_materialize_reuse(self):
        cg = CGFull()
        g = cg.edges(
            pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "a", "d"]}), "s", "d"
        )
        g = g.nodes(pd.DataFrame({"id": ["a", "b", "c", "d"], "v": [2, 4, 6, 8]}), "id")
        g = g.materialize_nodes()
        assert g._nodes.to_dict(orient="records") == [
            {"id": "a", "v": 2},
            {"id": "b", "v": 4},
            {"id": "c", "v": 6},
            {"id": "d", "v": 8},
        ]
        assert g._node == "id"


    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1",
    )
    def test_materialize_nodes_cudf(self):

        import cudf

        cg = CGFull()
        g = cg.edges(
            cudf.from_pandas(pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "a", "d"]})),
            "s", "d"
        )
        g = g.materialize_nodes()
        assert g._nodes.to_pandas().to_dict(orient="records") == [
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ]
        assert g._node == "id"


    def test_materialize_empty_edges(self):
        """Test materialize_nodes() with empty edges DataFrame.

        This is an edge case where materialize_nodes() returns early without
        setting _node binding, which is why validation checks in hop() are necessary.
        """
        cg = CGFull()
        # Create graph with empty edges
        g = cg.edges(pd.DataFrame({"s": [], "d": []}), "s", "d")

        # materialize_nodes() should return early for empty edges
        g2 = g.materialize_nodes()

        # The critical assertion: _node should be None because materialize_nodes()
        # returns early at ComputeMixin.py line 196 without calling .nodes()
        assert g2._node is None, "Empty edges should leave _node as None"
        assert g2._nodes is None or len(g2._nodes) == 0

    def test_degrees_in(self):
        cg = CGFull()
        g = cg.edges(
            pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "a", "d"]}), "s", "d"
        )
        g2 = g.get_indegrees()
        assert g2._nodes.to_dict(orient="records") == [
            {"id": "a", "degree_in": 1},
            {"id": "b", "degree_in": 1},
            {"id": "c", "degree_in": 0},
            {"id": "d", "degree_in": 1},
        ]
        assert g2._node == "id"

    def test_degrees_out(self):
        cg = CGFull()
        g = cg.edges(
            pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "a", "d"]}), "s", "d"
        )
        g2 = g.get_outdegrees()
        assert g2._nodes.to_dict(orient="records") == [
            {"id": "b", "degree_out": 1},
            {"id": "a", "degree_out": 1},
            {"id": "d", "degree_out": 0},
            {"id": "c", "degree_out": 1},
        ]
        assert g2._node == "id"

    def test_degrees(self):
        cg = CGFull()
        g = cg.edges(
            pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "a", "d"]}), "s", "d"
        )
        g2 = g.get_degrees()
        assert g2._nodes.to_dict(orient="records") == [
            {"id": "a", "degree_in": 1, "degree_out": 1, "degree": 2},
            {"id": "b", "degree_in": 1, "degree_out": 1, "degree": 2},
            {"id": "c", "degree_in": 0, "degree_out": 1, "degree": 1},
            {"id": "d", "degree_in": 1, "degree_out": 0, "degree": 1},
        ]
        assert g2._node == "id"

    def test_get_topological_levels_mt(self):
        cg = CGFull()
        g = cg.edges(
            pd.DataFrame({"s": [], "d": []}), "s", "d"
        ).get_topological_levels()
        assert g._edges is None or len(g._edges) == 0

    def test_get_topological_levels_1(self):
        cg = CGFull()
        g = cg.edges(
            pd.DataFrame({"s": ["a"], "d": ["b"]}), "s", "d"
        ).get_topological_levels()
        assert g._nodes.to_dict(orient="records") == [
            {"id": "a", "level": 0},
            {"id": "b", "level": 1},
        ]

    def test_get_topological_levels_1_aliasing(self):
        cg = CGFull()
        g = (
            cg.edges(pd.DataFrame({"s": ["a"], "d": ["b"]}), "s", "d")
            .nodes(pd.DataFrame({"n": ["a", "b"], "degree": ["x", "y"]}), "n")
            .get_topological_levels()
        )
        assert g._nodes.to_dict(orient="records") == [
            {"n": "a", "level": 0, "degree": "x"},
            {"n": "b", "level": 1, "degree": "y"},
        ]

    def test_get_topological_levels_cycle_exn(self):
        cg = CGFull()
        with pytest.raises(ValueError):
            cg.edges(
                pd.DataFrame({"s": ["a", "b"], "d": ["b", "a"]}), "s", "d"
            ).get_topological_levels(allow_cycles=False)

    def test_get_topological_levels_cycle_override(self):
        cg = CGFull()
        g = cg.edges(
            pd.DataFrame({"s": ["a", "b"], "d": ["b", "a"]}), "s", "d"
        ).get_topological_levels(allow_cycles=True)
        assert g._nodes.to_dict(orient="records") == [
            {"id": "a", "level": 0},
            {"id": "b", "level": 1},
        ]

    def test_drop_nodes(self):
        cg = CGFull()
        g = cg.edges(
            pd.DataFrame({"x": ["m", "m", "n", "m"], "y": ["a", "b", "c", "d"]}),
            "x",
            "y",
        )
        g2 = g.drop_nodes(["m"])
        assert g2._edges.to_dict(orient="records") == [{"x": "n", "y": "c"}]

    def test_keep_nodes(self):
        cg = CGFull()
        g = cg.edges(
            pd.DataFrame({"x": ["m", "m", "m", "n", "m"], "y": ["m", "a", "b", "c", "d"]}),
            "x",
            "y",
        )
        assert g.keep_nodes(["z"])._nodes.to_dict(orient="records") == []
        assert (
            g.keep_nodes(["m", "a"])._nodes
            .sort_values(by='id')
            .to_dict(orient="records") == [{"id": "a"}, {"id": "m"}]
        )
        assert (
            g.keep_nodes(["m", "a", "b"])._edges
            .sort_values(by='y')
            .to_dict(orient="records") == [
                {"x": "m", "y": "a"}, {"x": "m", "y": "b"}, {"x": "m", "y": "m"}
            ]
        )
        assert (
            g.keep_nodes(pd.Series(["m", "a", "b", "c"]))._nodes
            .sort_values(by='id')
            .to_dict(orient="records") == [
                {"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "m"}
            ]
        )
        #TODO test dict

    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1",
    )
    def test_keep_nodes_cudf(self):
        import cudf
        cg = CGFull()
        g = cg.edges(
            cudf.DataFrame({"x": ["m", "m", "m", "n", "m"], "y": ["m", "a", "b", "c", "d"]}),
            "x",
            "y",
        )
        assert g.keep_nodes(["z"])._nodes.to_pandas().to_dict(orient="records") == []
        assert (
            g.keep_nodes(["m", "a"])._nodes
            .sort_values(by='id')
            .to_pandas().to_dict(orient="records") == [{"id": "a"}, {"id": "m"}]
        )
        assert (
            g.keep_nodes(["m", "a", "b"])._edges
            .sort_values(by='y')
            .to_pandas().to_dict(orient="records") == [
                {"x": "m", "y": "a"}, {"x": "m", "y": "b"}, {"x": "m", "y": "m"}
            ]
        )
        assert (
            g.keep_nodes(cudf.Series(["m", "a", "b", "c"]))._nodes
            .sort_values(by='id')
            .to_pandas().to_dict(orient="records") == [
                {"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "m"}
            ]
        )
        #TODO test dict


if __name__ == "__main__":
    unittest.main()
