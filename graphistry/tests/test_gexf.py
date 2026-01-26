import os

import pytest

import graphistry
from common import NoAuthTestCase


DATA_DIR = os.path.join("graphistry", "tests", "data", "gexf")


class TestGEXF(NoAuthTestCase):
    def test_gexf_11draft_basic(self):
        path = os.path.join(DATA_DIR, "sample-1.1draft-basic.gexf")
        g = graphistry.gexf(path)

        self.assertEqual(g._node, "node_id")
        self.assertEqual(g._source, "source")
        self.assertEqual(g._destination, "target")
        self.assertEqual(len(g._nodes), 3)
        self.assertEqual(len(g._edges), 2)
        self.assertEqual(g._point_title, "label")
        self.assertEqual(g._edge_title, "label")
        self.assertEqual(g._description, "Sample 1.1draft")

        node_n2 = g._nodes[g._nodes["node_id"] == "n2"].iloc[0]
        assert node_n2["role"] == "member"

        node_n0 = g._nodes[g._nodes["node_id"] == "n0"].iloc[0]
        assert node_n0["score"] == pytest.approx(1.5)

        edge_e1 = g._edges[g._edges["edge_id"] == "e1"].iloc[0]
        assert edge_e1["weight_attr"] == pytest.approx(0.75)

    def test_gexf_12draft_viz(self):
        path = os.path.join(DATA_DIR, "sample-1.2draft-viz.gexf")
        g = graphistry.gexf(path)

        self.assertEqual(g._point_color, "viz_color")
        self.assertEqual(g._point_size, "viz_size")
        self.assertEqual(g._point_x, "viz_x")
        self.assertEqual(g._point_y, "viz_y")
        self.assertEqual(g._point_opacity, "viz_opacity")
        self.assertEqual(g._edge_color, "viz_color")
        self.assertEqual(g._edge_size, "viz_thickness")
        self.assertEqual(g._edge_opacity, "viz_opacity")
        self.assertEqual(g._edge_weight, "weight")
        self.assertEqual(g._url_params.get("play"), 0)

        node_n10 = g._nodes[g._nodes["node_id"] == "n10"].iloc[0]
        assert node_n10["viz_color"] == "#EFAD42"
        assert node_n10["viz_opacity"] == pytest.approx(0.5)
        assert node_n10["viz_x"] == pytest.approx(10.0)
        assert node_n10["viz_y"] == pytest.approx(20.5)
        assert node_n10["viz_size"] == pytest.approx(2.5)

        edge_e10 = g._edges[g._edges["edge_id"] == "e10"].iloc[0]
        assert edge_e10["viz_color"] == "#9DD54E"
        assert edge_e10["viz_opacity"] == pytest.approx(0.8)
        assert edge_e10["viz_thickness"] == pytest.approx(3.5)
        assert edge_e10["weight"] == pytest.approx(2.0)

    def test_gexf_13_viz_hex(self):
        path = os.path.join(DATA_DIR, "sample-1.3-viz.gexf")
        g = graphistry.gexf(path)

        node_u0 = g._nodes[g._nodes["node_id"] == "u0"].iloc[0]
        assert node_u0["viz_color"] == "#FF7700"
        assert node_u0["viz_opacity"] == pytest.approx(0.5)

        edge_e20 = g._edges[g._edges["edge_id"] == "e20"].iloc[0]
        assert edge_e20["viz_color"] == "#112233"
        assert edge_e20["viz_opacity"] == pytest.approx(0.6)

    def test_gexf_missing_node_id(self):
        path = os.path.join(DATA_DIR, "invalid-missing-node-id.gexf")
        with pytest.raises(ValueError, match="node missing id"):
            graphistry.gexf(path)

    def test_gexf_missing_edge_node(self):
        path = os.path.join(DATA_DIR, "invalid-edge-missing-node.gexf")
        with pytest.raises(ValueError, match="missing node ids"):
            graphistry.gexf(path)
