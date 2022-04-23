# -*- coding: utf-8 -*-
from functools import lru_cache

import pandas as pd
import pytest
import unittest
from unittest import TestCase

# from common import NoAuthTestCase

from graphistry.compute import ComputeMixin
from graphistry.plotter import PlotterBase


class CG(ComputeMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ComputeMixin.__init__(self, *args, **kwargs)


class CGFull(ComputeMixin, PlotterBase, object):
    def __init__(self, *args, **kwargs):
        print("CGFull init")
        super(CGFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)


def get_collapse_graph(as_string=False):
    edges = pd.DataFrame(
        {
            "src": [0, -1, 1, 2, 3, 3, 4, 3, 0, 6, 2, 8, -1, 1, 6],
            "dst": [-1, 1, 2, 3, 4, 7, 5, 5, 6, 5, 8, 3, 9, 9, 10],
        }
    )
    nodes = pd.DataFrame(
        {
            "node": [0, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10],
            "level": [0, 1, 1, 1, 2, 2, 1, 1, 0, 1, 2, 1],
        }
    )
    if as_string:
        g = (
            CGFull()
            .edges(edges.astype(str), "src", "dst")
            .nodes(nodes.astype(str), "node")
        )
    else:
        g = CGFull().edges(edges, "src", "dst").nodes(nodes, "node")
    return g


parent_result_nodes_should_stay_same = {
    "node": {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "-1",
        9: "8",
        10: "9",
        11: "10",
    },
    "level": {
        0: "0",
        1: "1",
        2: "1",
        3: "1",
        4: "2",
        5: "2",
        6: "1",
        7: "1",
        8: "0",
        9: "1",
        10: "2",
        11: "1",
    },
}

parent_result_edges_should_stay_same = {
    "src": {
        0: "0",
        1: "-1",
        2: "1",
        3: "2",
        4: "3",
        5: "3",
        6: "4",
        7: "3",
        8: "0",
        9: "6",
        10: "2",
        11: "8",
        12: "-1",
        13: "1",
        14: "6",
    },
    "dst": {
        0: "-1",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "7",
        6: "5",
        7: "5",
        8: "6",
        9: "5",
        10: "8",
        11: "3",
        12: "9",
        13: "9",
        14: "10",
    },
}

collapse_result_nodes_overwrite = {
    "node": {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "-1",
        9: "8",
        10: "9",
        11: "10",
    },
    "level": {
        0: "0",
        1: "1",
        2: "1",
        3: "1",
        4: "2",
        5: "2",
        6: "1",
        7: "1",
        8: "0",
        9: "1",
        10: "2",
        11: "1",
    },
    "node_collapse": {
        0: "None",
        1: "~1~ ~2~ ~3~ ~7~ ~8~",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "None",
        5: "None",
        6: "~10~ ~6~",
        7: "~1~ ~2~ ~3~ ~7~ ~8~",
        8: "None",
        9: "~1~ ~2~ ~3~ ~7~ ~8~",
        10: "None",
        11: "~10~ ~6~",
    },
    "node_final": {
        0: "0",
        1: "~1~ ~2~ ~3~ ~7~ ~8~",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "4",
        5: "5",
        6: "~10~ ~6~",
        7: "~1~ ~2~ ~3~ ~7~ ~8~",
        8: "-1",
        9: "~1~ ~2~ ~3~ ~7~ ~8~",
        10: "9",
        11: "~10~ ~6~",
    },
}


collapse_result_edges_overwrite = {
    "src": {
        0: "0",
        1: "-1",
        2: "1",
        3: "2",
        4: "3",
        5: "3",
        6: "4",
        7: "3",
        8: "0",
        9: "6",
        10: "2",
        11: "8",
        12: "-1",
        13: "1",
        14: "6",
    },
    "dst": {
        0: "-1",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "7",
        6: "5",
        7: "5",
        8: "6",
        9: "5",
        10: "8",
        11: "3",
        12: "9",
        13: "9",
        14: "10",
    },
    "src_collapse": {
        0: "None",
        1: "None",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "~1~ ~2~ ~3~ ~7~ ~8~",
        5: "~1~ ~2~ ~3~ ~7~ ~8~",
        6: "None",
        7: "~1~ ~2~ ~3~ ~7~ ~8~",
        8: "None",
        9: "~10~ ~6~",
        10: "~1~ ~2~ ~3~ ~7~ ~8~",
        11: "~1~ ~2~ ~3~ ~7~ ~8~",
        12: "None",
        13: "~1~ ~2~ ~3~ ~7~ ~8~",
        14: "~10~ ~6~",
    },
    "dst_collapse": {
        0: "None",
        1: "~1~ ~2~ ~3~ ~7~ ~8~",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "None",
        5: "~1~ ~2~ ~3~ ~7~ ~8~",
        6: "None",
        7: "None",
        8: "~10~ ~6~",
        9: "None",
        10: "~1~ ~2~ ~3~ ~7~ ~8~",
        11: "~1~ ~2~ ~3~ ~7~ ~8~",
        12: "None",
        13: "None",
        14: "~10~ ~6~",
    },
    "src_final": {
        0: "0",
        1: "-1",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "~1~ ~2~ ~3~ ~7~ ~8~",
        5: "~1~ ~2~ ~3~ ~7~ ~8~",
        6: "4",
        7: "~1~ ~2~ ~3~ ~7~ ~8~",
        8: "0",
        9: "~10~ ~6~",
        10: "~1~ ~2~ ~3~ ~7~ ~8~",
        11: "~1~ ~2~ ~3~ ~7~ ~8~",
        12: "-1",
        13: "~1~ ~2~ ~3~ ~7~ ~8~",
        14: "~10~ ~6~",
    },
    "dst_final": {
        0: "-1",
        1: "~1~ ~2~ ~3~ ~7~ ~8~",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "4",
        5: "~1~ ~2~ ~3~ ~7~ ~8~",
        6: "5",
        7: "5",
        8: "~10~ ~6~",
        9: "5",
        10: "~1~ ~2~ ~3~ ~7~ ~8~",
        11: "~1~ ~2~ ~3~ ~7~ ~8~",
        12: "9",
        13: "9",
        14: "~10~ ~6~",
    },
}

collapse_all_nodes = {
    "node": {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "-1",
        9: "8",
        10: "9",
        11: "10",
    },
    "level": {
        0: "0",
        1: "1",
        2: "1",
        3: "1",
        4: "2",
        5: "2",
        6: "1",
        7: "1",
        8: "0",
        9: "1",
        10: "2",
        11: "1",
    },
    "node_collapse": {
        0: "~-1~ ~0~",
        1: "~1~ ~2~ ~3~ ~7~ ~8~",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "~4~ ~5~",
        5: "~4~ ~5~",
        6: "~10~ ~6~",
        7: "~1~ ~2~ ~3~ ~7~ ~8~",
        8: "~-1~ ~0~",
        9: "~1~ ~2~ ~3~ ~7~ ~8~",
        10: "None",
        11: "~10~ ~6~",
    },
    "node_final": {
        0: "~-1~ ~0~",
        1: "~1~ ~2~ ~3~ ~7~ ~8~",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "~4~ ~5~",
        5: "~4~ ~5~",
        6: "~10~ ~6~",
        7: "~1~ ~2~ ~3~ ~7~ ~8~",
        8: "~-1~ ~0~",
        9: "~1~ ~2~ ~3~ ~7~ ~8~",
        10: "9",
        11: "~10~ ~6~",
    },
}

collapse_all_edges = {
    "src": {
        0: "0",
        1: "-1",
        2: "1",
        3: "2",
        4: "3",
        5: "3",
        6: "4",
        7: "3",
        8: "0",
        9: "6",
        10: "2",
        11: "8",
        12: "-1",
        13: "1",
        14: "6",
    },
    "dst": {
        0: "-1",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "7",
        6: "5",
        7: "5",
        8: "6",
        9: "5",
        10: "8",
        11: "3",
        12: "9",
        13: "9",
        14: "10",
    },
    "src_collapse": {
        0: "~-1~ ~0~",
        1: "~-1~ ~0~",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "~1~ ~2~ ~3~ ~7~ ~8~",
        5: "~1~ ~2~ ~3~ ~7~ ~8~",
        6: "~4~ ~5~",
        7: "~1~ ~2~ ~3~ ~7~ ~8~",
        8: "~-1~ ~0~",
        9: "~10~ ~6~",
        10: "~1~ ~2~ ~3~ ~7~ ~8~",
        11: "~1~ ~2~ ~3~ ~7~ ~8~",
        12: "~-1~ ~0~",
        13: "~1~ ~2~ ~3~ ~7~ ~8~",
        14: "~10~ ~6~",
    },
    "dst_collapse": {
        0: "~-1~ ~0~",
        1: "~1~ ~2~ ~3~ ~7~ ~8~",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "~4~ ~5~",
        5: "~1~ ~2~ ~3~ ~7~ ~8~",
        6: "~4~ ~5~",
        7: "~4~ ~5~",
        8: "~10~ ~6~",
        9: "~4~ ~5~",
        10: "~1~ ~2~ ~3~ ~7~ ~8~",
        11: "~1~ ~2~ ~3~ ~7~ ~8~",
        12: "None",
        13: "None",
        14: "~10~ ~6~",
    },
    "src_final": {
        0: "~-1~ ~0~",
        1: "~-1~ ~0~",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "~1~ ~2~ ~3~ ~7~ ~8~",
        5: "~1~ ~2~ ~3~ ~7~ ~8~",
        6: "~4~ ~5~",
        7: "~1~ ~2~ ~3~ ~7~ ~8~",
        8: "~-1~ ~0~",
        9: "~10~ ~6~",
        10: "~1~ ~2~ ~3~ ~7~ ~8~",
        11: "~1~ ~2~ ~3~ ~7~ ~8~",
        12: "~-1~ ~0~",
        13: "~1~ ~2~ ~3~ ~7~ ~8~",
        14: "~10~ ~6~",
    },
    "dst_final": {
        0: "~-1~ ~0~",
        1: "~1~ ~2~ ~3~ ~7~ ~8~",
        2: "~1~ ~2~ ~3~ ~7~ ~8~",
        3: "~1~ ~2~ ~3~ ~7~ ~8~",
        4: "~4~ ~5~",
        5: "~1~ ~2~ ~3~ ~7~ ~8~",
        6: "~4~ ~5~",
        7: "~4~ ~5~",
        8: "~10~ ~6~",
        9: "~4~ ~5~",
        10: "~1~ ~2~ ~3~ ~7~ ~8~",
        11: "~1~ ~2~ ~3~ ~7~ ~8~",
        12: "9",
        13: "9",
        14: "~10~ ~6~",
    },
}


@lru_cache(maxsize=1)
def hops_graph():
    nodes_df = pd.DataFrame(
        [
            {"node": "a"},
            {"node": "b"},
            {"node": "c"},
            {"node": "d"},
            {"node": "e"},
            {"node": "f"},
            {"node": "g"},
            {"node": "h"},
            {"node": "i"},
            {"node": "j"},
            {"node": "k"},
            {"node": "l"},
            {"node": "m"},
            {"node": "n"},
            {"node": "o"},
            {"node": "p"},
        ]
    ).assign(type="n")

    edges_df = pd.DataFrame(
        [
            {"s": "e", "d": "l"},
            {"s": "l", "d": "b"},
            {"s": "k", "d": "a"},
            {"s": "e", "d": "g"},
            {"s": "g", "d": "a"},
            {"s": "d", "d": "f"},
            {"s": "d", "d": "c"},
            {"s": "d", "d": "j"},
            {"s": "d", "d": "i"},
            {"s": "d", "d": "h"},
            {"s": "j", "d": "p"},
            {"s": "i", "d": "n"},
            {"s": "h", "d": "m"},
            {"s": "j", "d": "o"},
            {"s": "o", "d": "b"},
            {"s": "m", "d": "a"},
            {"s": "n", "d": "a"},
            {"s": "p", "d": "b"},
        ]
    ).assign(type="e")

    return CGFull().nodes(nodes_df, "node").edges(edges_df, "s", "d")


class TestComputeMixin(TestCase):
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

    def test_hop_0(self):

        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: []}), 0)
        assert g2._nodes.shape == (0, 2)
        assert g2._edges.shape == (0, 3)

    def test_hop_0b(self):

        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ["d"]}), 0)
        assert g2._nodes.shape == (0, 2)
        assert g2._edges.shape == (0, 3)

    def test_hop_1_1_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ["d"]}), 1)
        assert g2._nodes.shape == (6, 2)
        assert g2._nodes[g2._node].sort_values().to_list() == sorted(  # noqa: W504
            ["f", "j", "d", "i", "c", "h"]
        )
        assert g2._edges.shape == (5, 3)

    def test_hop_2_1_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ["k", "d"]}), 1)
        assert g2._nodes.shape == (8, 2)
        assert g2._edges.shape == (6, 3)

    def test_hop_2_2_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ["k", "d"]}), 2)
        assert g2._nodes.shape == (12, 2)
        assert g2._edges.shape == (10, 3)

    def test_hop_2_all_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ["k", "d"]}), to_fixed_point=True)
        assert g2._nodes.shape == (13, 2)
        assert g2._edges.shape == (14, 3)

    def test_hop_1_2_undirected(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ["j"]}), 2, direction="undirected")
        assert g2._nodes.shape == (9, 2)
        assert g2._edges.shape == (9, 3)

    def test_hop_1_all_reverse(self):
        g = hops_graph()
        g2 = g.hop(
            pd.DataFrame({g._node: ["b"]}), direction="reverse", to_fixed_point=True
        )
        assert g2._nodes.shape == (7, 2)
        assert g2._edges.shape == (7, 3)

    def _test_graph_collapse(self, g, g2):
        assert len(g._nodes) == len(g2._nodes) # now we don't drop any since we write to FINAL columns instead. so nodes table will be same
        assert len(g._edges) >= len(g2._edges)
        assert g2._nodes.astype(str).to_dict() == collapse_result_nodes_overwrite, "collapsed nodes dataframe.astype(str) should match collapsed nodes df"
        assert g._nodes.astype(str).to_dict() == parent_result_nodes_should_stay_same, "original node dataframe.astype(str) should match parent ndf"
        assert g2._edges.astype(str).to_dict() == collapse_result_edges_overwrite, "collapsed edges dataframe.astype(str) should match collapsed edges df"
        assert g._edges.astype(str).to_dict() == parent_result_edges_should_stay_same, "original edge dataframe.astype(str) should match parent edf"

    def _test_graph_chain_collapse(self, g, g2):
        assert len(g._nodes) == len(g2._nodes)  # now we don't drop any since we write to FINAL columns instead. so nodes table will be same
        assert len(g._edges) >= len(g2._edges)
        assert g2._nodes.astype(str).to_dict() == collapse_all_nodes, "chained collapsed nodes dataframe.astype(str) should match collapsed nodes df"
        assert g2._edges.astype(str).to_dict() == collapse_all_edges, "chained collapsed edges dataframe.astype(str) should match collapsed edges df"

    def test_collapse_over_string_values(self):
        g = get_collapse_graph(as_string=True)
        g2 = g.collapse(
            node="0", attribute="1", column="level", unwrap=False, verbose=True
        )
        self._test_graph_collapse(g, g2)

    def test_chained_collapse(self):
        g = get_collapse_graph(as_string=True)  # order matters here
        g2 = g.collapse(node="0", attribute="1", column="level", unwrap=False)
        g3 = g2.collapse(node="0", attribute="2", column="level", unwrap=False)
        g4 = g3.collapse(node="0", attribute="0", column="level", unwrap=False)
        self._test_graph_chain_collapse(g, g4)


if __name__ == "__main__":
    unittest.main()
