import importlib.util
import io
import os
import tempfile
from numbers import Integral, Real
from unittest.mock import patch

import pandas as pd
import pytest

import graphistry
from common import NoAuthTestCase
from graphistry.io.metadata import (
    serialize_edge_bindings,
    serialize_node_bindings,
    serialize_plottable_metadata,
)
from graphistry.plugins.gexf import reader as gexf_reader


DATA_DIR = os.path.join("graphistry", "tests", "data", "gexf")


def _sorted_frame(df: pd.DataFrame, cols, sort_cols):
    return df[cols].sort_values(sort_cols).reset_index(drop=True)


def _build_roundtrip_plotter():
    nodes = pd.DataFrame(
        {
            "node_id": ["a", "b"],
            "label": ["Alpha", "Beta"],
            "viz_color": ["#112233", "#445566"],
            "viz_size": [2.5, 3.5],
            "viz_x": [1.0, 2.0],
            "viz_y": [3.0, 4.0],
            "viz_opacity": [0.2, 0.8],
            "group": ["g1", "g2"],
        }
    )
    edges = pd.DataFrame(
        {
            "source": ["a"],
            "target": ["b"],
            "label": ["connects"],
            "viz_color": ["#AABBCC"],
            "viz_thickness": [1.5],
            "viz_opacity": [0.6],
            "weight": [2.5],
            "relation": ["r1"],
        }
    )
    return (
        graphistry.edges(edges, "source", "target")
        .nodes(nodes, "node_id")
        .bind(
            point_title="label",
            point_color="viz_color",
            point_size="viz_size",
            point_x="viz_x",
            point_y="viz_y",
            point_opacity="viz_opacity",
            edge_title="label",
            edge_color="viz_color",
            edge_size="viz_thickness",
            edge_opacity="viz_opacity",
            edge_weight="weight",
        )
    )


def _assert_roundtrip_payload(g2):
    assert set(g2._nodes["group"]) == {"g1", "g2"}
    assert g2._edges["relation"].iloc[0] == "r1"
    assert g2._nodes["viz_color"].iloc[0] == "#112233"
    assert g2._edges["viz_color"].iloc[0] == "#AABBCC"
    assert g2._edges["viz_thickness"].iloc[0] == pytest.approx(1.5)
    assert g2._edges["weight"].iloc[0] == pytest.approx(2.5)


def _read_gexf_bytes(filename: str) -> bytes:
    with open(os.path.join(DATA_DIR, filename), "rb") as f:
        return f.read()


class TestGEXF(NoAuthTestCase):
    def test_gexf_url_fetch_uses_default_timeout(self):
        payload = _read_gexf_bytes("sample-1.1draft-basic.gexf")

        class _FakeHTTPResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return payload

        with patch.object(gexf_reader, "urlopen") as mocked_urlopen:
            mocked_urlopen.return_value = _FakeHTTPResponse()
            g = graphistry.gexf("https://example.com/sample.gexf", parse_engine="stdlib")

        mocked_urlopen.assert_called_once_with("https://example.com/sample.gexf", timeout=10.0)
        self.assertEqual(len(g._nodes), 3)

    def test_gexf_11draft_basic(self):
        path = os.path.join(DATA_DIR, "sample-1.1draft-basic.gexf")
        g = graphistry.gexf(path)

        node_bindings = serialize_node_bindings(g)
        edge_bindings = serialize_edge_bindings(g)
        metadata = serialize_plottable_metadata(g)

        self.assertEqual(node_bindings.get("node"), "node_id")
        self.assertEqual(edge_bindings.get("source"), "source")
        self.assertEqual(edge_bindings.get("destination"), "target")
        self.assertEqual(len(g._nodes), 3)
        self.assertEqual(len(g._edges), 2)
        self.assertEqual(node_bindings.get("node_title"), "label")
        self.assertEqual(edge_bindings.get("edge_title"), "label")
        self.assertEqual(metadata.get("metadata", {}).get("description"), "Sample 1.1draft")
        self.assertNotIn("node_color", node_bindings)
        self.assertNotIn("edge_color", edge_bindings)
        self.assertNotIn("node_size", node_bindings)
        self.assertNotIn("edge_size", edge_bindings)
        self.assertNotIn("node_x", node_bindings)
        self.assertNotIn("node_y", node_bindings)
        self.assertIsNone(g._url_params.get("play"))

        node_n2 = g._nodes[g._nodes["node_id"] == "n2"].iloc[0]
        assert node_n2["role"] == "member"

        node_n0 = g._nodes[g._nodes["node_id"] == "n0"].iloc[0]
        assert node_n0["score"] == pytest.approx(1.5)

        edge_e1 = g._edges[g._edges["edge_id"] == "e1"].iloc[0]
        assert edge_e1["weight_attr"] == pytest.approx(0.75)

    def test_gexf_namespace_alias(self):
        path = os.path.join(DATA_DIR, "sample-1.1draft-basic.gexf")
        with open(path, "rb") as f:
            data = f.read()
        data = data.replace(
            b"http://www.gephi.org/gexf/1.1draft",
            b"http://gexf.net/1.1draft",
        )
        g = graphistry.gexf(data)
        self.assertEqual(len(g._nodes), 3)

    def test_gexf_12draft_viz(self):
        path = os.path.join(DATA_DIR, "sample-1.2draft-viz.gexf")
        g = graphistry.gexf(path)

        node_bindings = serialize_node_bindings(g)
        edge_bindings = serialize_edge_bindings(g)

        self.assertEqual(node_bindings.get("node_color"), "viz_color")
        self.assertEqual(node_bindings.get("node_size"), "viz_size")
        self.assertEqual(node_bindings.get("node_x"), "viz_x")
        self.assertEqual(node_bindings.get("node_y"), "viz_y")
        self.assertEqual(node_bindings.get("node_opacity"), "viz_opacity")
        self.assertEqual(edge_bindings.get("edge_color"), "viz_color")
        self.assertEqual(edge_bindings.get("edge_size"), "viz_thickness")
        self.assertEqual(edge_bindings.get("edge_opacity"), "viz_opacity")
        self.assertEqual(edge_bindings.get("edge_weight"), "weight")
        self.assertEqual(g._url_params.get("play"), 0)
        self.assertEqual(node_bindings.get("node_icon"), "viz_shape_icon")

        node_n10 = g._nodes[g._nodes["node_id"] == "n10"].iloc[0]
        assert node_n10["viz_color"] == "#EFAD42"
        assert node_n10["viz_opacity"] == pytest.approx(0.5)
        assert node_n10["viz_x"] == pytest.approx(10.0)
        assert node_n10["viz_y"] == pytest.approx(20.5)
        assert node_n10["viz_size"] == pytest.approx(2.5)
        assert node_n10["viz_shape_icon"] == "circle"

        edge_e10 = g._edges[g._edges["edge_id"] == "e10"].iloc[0]
        assert edge_e10["viz_color"] == "#9DD54E"
        assert edge_e10["viz_opacity"] == pytest.approx(0.8)
        assert edge_e10["viz_thickness"] == pytest.approx(3.5)
        assert edge_e10["weight"] == pytest.approx(2.0)

        node_n11 = g._nodes[g._nodes["node_id"] == "n11"].iloc[0]
        assert node_n11["viz_shape_icon"] == "square"

    def test_gexf_13_viz_hex(self):
        path = os.path.join(DATA_DIR, "sample-1.3-viz.gexf")
        g = graphistry.gexf(path)

        node_u0 = g._nodes[g._nodes["node_id"] == "u0"].iloc[0]
        assert node_u0["viz_color"] == "#FF7700"
        assert node_u0["viz_opacity"] == pytest.approx(0.5)

        edge_e20 = g._edges[g._edges["edge_id"] == "e20"].iloc[0]
        assert edge_e20["viz_color"] == "#112233"
        assert edge_e20["viz_opacity"] == pytest.approx(0.6)

    def test_gexf_in_memory_bytes_and_stream(self):
        payload = _read_gexf_bytes("sample-1.2draft-viz.gexf")
        g_bytes = graphistry.gexf(payload, parse_engine="stdlib")
        g_stream = graphistry.gexf(io.BytesIO(payload), parse_engine="stdlib")

        node_cols = ["node_id", "label", "viz_color", "viz_size", "viz_x", "viz_y", "viz_opacity", "viz_shape_icon"]
        edge_cols = ["source", "target", "label", "viz_color", "viz_thickness", "viz_opacity", "weight"]

        pd.testing.assert_frame_equal(
            _sorted_frame(g_bytes._nodes, node_cols, "node_id"),
            _sorted_frame(g_stream._nodes, node_cols, "node_id"),
        )
        pd.testing.assert_frame_equal(
            _sorted_frame(g_bytes._edges, edge_cols, ["source", "target"]),
            _sorted_frame(g_stream._edges, edge_cols, ["source", "target"]),
        )

    def test_gexf_node_viz_only(self):
        path = os.path.join(DATA_DIR, "sample-1.2draft-node-viz-only.gexf")
        g = graphistry.gexf(path)

        node_bindings = serialize_node_bindings(g)
        edge_bindings = serialize_edge_bindings(g)

        self.assertEqual(node_bindings.get("node_color"), "viz_color")
        self.assertEqual(node_bindings.get("node_size"), "viz_size")
        self.assertEqual(node_bindings.get("node_x"), "viz_x")
        self.assertEqual(node_bindings.get("node_y"), "viz_y")
        self.assertEqual(node_bindings.get("node_opacity"), "viz_opacity")
        self.assertNotIn("edge_color", edge_bindings)
        self.assertNotIn("edge_size", edge_bindings)
        self.assertNotIn("edge_opacity", edge_bindings)
        self.assertEqual(g._url_params.get("play"), 0)

        node_n0 = g._nodes[g._nodes["node_id"] == "n0"].iloc[0]
        assert node_n0["viz_color"] == "#FF0000"
        assert node_n0["viz_opacity"] == pytest.approx(0.7)
        assert node_n0["viz_size"] == pytest.approx(3.0)

    def test_gexf_edge_viz_only(self):
        path = os.path.join(DATA_DIR, "sample-1.2draft-edge-viz-only.gexf")
        g = graphistry.gexf(path)

        node_bindings = serialize_node_bindings(g)
        edge_bindings = serialize_edge_bindings(g)

        self.assertEqual(edge_bindings.get("edge_color"), "viz_color")
        self.assertEqual(edge_bindings.get("edge_size"), "viz_thickness")
        self.assertEqual(edge_bindings.get("edge_opacity"), "viz_opacity")
        self.assertNotIn("node_color", node_bindings)
        self.assertNotIn("node_size", node_bindings)
        self.assertNotIn("node_opacity", node_bindings)
        self.assertIsNone(g._url_params.get("play"))

        edge_e0 = g._edges[g._edges["edge_id"] == "e0"].iloc[0]
        assert edge_e0["viz_color"] == "#0080FF"
        assert edge_e0["viz_opacity"] == pytest.approx(0.6)
        assert edge_e0["viz_thickness"] == pytest.approx(2.5)

    def test_gexf_viz_binding_options(self):
        path = os.path.join(DATA_DIR, "sample-1.2draft-viz.gexf")
        g = graphistry.gexf(path, bind_node_viz=["position"], bind_edge_viz=[])

        node_bindings = serialize_node_bindings(g)
        edge_bindings = serialize_edge_bindings(g)

        self.assertNotIn("node_color", node_bindings)
        self.assertNotIn("node_size", node_bindings)
        self.assertNotIn("node_opacity", node_bindings)
        self.assertNotIn("node_icon", node_bindings)
        self.assertEqual(node_bindings.get("node_x"), "viz_x")
        self.assertEqual(node_bindings.get("node_y"), "viz_y")
        self.assertNotIn("edge_color", edge_bindings)
        self.assertNotIn("edge_size", edge_bindings)
        self.assertNotIn("edge_opacity", edge_bindings)
        self.assertEqual(g._url_params.get("play"), 0)

    def test_gexf_viz_binding_invalid(self):
        path = os.path.join(DATA_DIR, "sample-1.2draft-viz.gexf")
        with pytest.raises(ValueError, match="Unsupported node viz bindings"):
            graphistry.gexf(path, bind_node_viz=["color", "not-a-viz-field"])
        with pytest.raises(ValueError, match="Unsupported edge viz bindings"):
            graphistry.gexf(path, bind_edge_viz=["opacity", "not-a-viz-field"])

    def test_gexf_parse_engine_stdlib(self):
        path = os.path.join(DATA_DIR, "sample-1.1draft-basic.gexf")
        g = graphistry.gexf(path, parse_engine="stdlib")
        node_bindings = serialize_node_bindings(g)
        self.assertEqual(node_bindings.get("node"), "node_id")

    def test_gexf_parse_engine_defused(self):
        if importlib.util.find_spec("defusedxml") is None:
            pytest.skip("defusedxml not installed")
        path = os.path.join(DATA_DIR, "sample-1.1draft-basic.gexf")
        g = graphistry.gexf(path, parse_engine="defused")
        node_bindings = serialize_node_bindings(g)
        self.assertEqual(node_bindings.get("node"), "node_id")

    def test_gexf_parse_engine_invalid(self):
        path = os.path.join(DATA_DIR, "sample-1.1draft-basic.gexf")
        with pytest.raises(ValueError, match="Unsupported parse_engine"):
            graphistry.gexf(path, parse_engine="bogus")

    def test_gexf_type_handling_roundtrip(self):
        path = os.path.join(DATA_DIR, "sample-1.2draft-types.gexf")
        g = graphistry.gexf(path)

        node_n0 = g._nodes[g._nodes["node_id"] == "n0"].iloc[0]
        node_n1 = g._nodes[g._nodes["node_id"] == "n1"].iloc[0]
        edge_e0 = g._edges.iloc[0]

        assert isinstance(node_n0["count"], Integral) and not isinstance(node_n0["count"], bool)
        assert pd.api.types.is_bool_dtype(g._nodes["active"])
        assert isinstance(node_n0["ratio"], Real) and not isinstance(node_n0["ratio"], bool)
        assert node_n0["count"] == 3
        assert bool(node_n0["active"]) is False
        assert node_n0["ratio"] == pytest.approx(1.5)

        assert bool(node_n1["active"]) is True
        assert node_n1["count"] == 7
        assert node_n1["ratio"] == pytest.approx(2.0)

        assert edge_e0["weight_int"] == 4
        assert bool(edge_e0["flagged"]) is True

        xml_str = g.to_gexf(version="1.2draft")
        g2 = graphistry.gexf(xml_str.encode("utf-8"))

        node_cols = ["node_id", "count", "active", "ratio"]
        edge_cols = ["source", "target", "weight_int", "flagged"]
        pd.testing.assert_frame_equal(
            _sorted_frame(g._nodes, node_cols, "node_id"),
            _sorted_frame(g2._nodes, node_cols, "node_id"),
        )
        pd.testing.assert_frame_equal(
            _sorted_frame(g._edges, edge_cols, ["source", "target"]),
            _sorted_frame(g2._edges, edge_cols, ["source", "target"]),
        )

    def test_gexf_viz_roundtrip_from_file(self):
        path = os.path.join(DATA_DIR, "sample-1.2draft-viz.gexf")
        g = graphistry.gexf(path)

        xml_str = g.to_gexf(version="1.2draft")
        g2 = graphistry.gexf(xml_str.encode("utf-8"))

        node_cols = [
            "node_id",
            "label",
            "viz_color",
            "viz_size",
            "viz_x",
            "viz_y",
            "viz_opacity",
            "viz_shape",
        ]
        edge_cols = [
            "source",
            "target",
            "label",
            "viz_color",
            "viz_thickness",
            "viz_opacity",
            "weight",
            "relation",
        ]

        pd.testing.assert_frame_equal(
            _sorted_frame(g._nodes, node_cols, "node_id"),
            _sorted_frame(g2._nodes, node_cols, "node_id"),
        )
        pd.testing.assert_frame_equal(
            _sorted_frame(g._edges, edge_cols, ["source", "target"]),
            _sorted_frame(g2._edges, edge_cols, ["source", "target"]),
        )

    def test_gexf_missing_node_id(self):
        path = os.path.join(DATA_DIR, "invalid-missing-node-id.gexf")
        with pytest.raises(ValueError, match="node missing id"):
            graphistry.gexf(path)

    def test_gexf_missing_edge_node(self):
        path = os.path.join(DATA_DIR, "invalid-edge-missing-node.gexf")
        with pytest.raises(ValueError, match="missing node ids"):
            graphistry.gexf(path)

    def test_to_gexf_roundtrip(self):
        g = _build_roundtrip_plotter()
        xml_str = g.to_gexf(version="1.2draft")
        _assert_roundtrip_payload(graphistry.gexf(xml_str.encode("utf-8")))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.gexf")
            g.to_gexf(out_path, version="1.2draft")
            assert os.path.exists(out_path)
            _assert_roundtrip_payload(graphistry.gexf(out_path, parse_engine="stdlib"))

    def test_to_gexf_roundtrip_all_supported_versions(self):
        g = _build_roundtrip_plotter()
        for version in ("1.1draft", "1.2draft", "1.3"):
            with self.subTest(version=version):
                xml_str = g.to_gexf(version=version)
                _assert_roundtrip_payload(graphistry.gexf(xml_str.encode("utf-8"), parse_engine="stdlib"))
