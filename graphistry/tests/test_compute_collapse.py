import pandas as pd
from common import NoAuthTestCase

from graphistry.tests.test_compute import CGFull


# ##############################################################################


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


# ##############################################################################


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


class TestCollapse(NoAuthTestCase):

    def _test_graph_collapse(self, g, g2):
        assert len(g._nodes) == len(g2._nodes)  # now we don't drop any since we write to FINAL columns instead. so nodes table will be same
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
