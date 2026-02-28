from __future__ import annotations

from typing import IO, Dict, Iterable, List, Optional, Set, Tuple, Union
import importlib
import types
import os
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import pandas as pd

from graphistry.Plottable import Plottable
from graphistry.plugins_types.gexf_types import GexfEdgeViz, GexfNodeViz, GexfParseEngine

DEFUSED_ET: Optional[types.ModuleType]
try:
    _DEFUSED_ET = importlib.import_module("defusedxml.ElementTree")
except Exception:
    DEFUSED_ET = None
else:
    DEFUSED_ET = _DEFUSED_ET

GEXF_NAMESPACES = {
    "http://www.gephi.org/gexf/1.1draft",
    "http://www.gexf.net/1.1draft",
    "http://gexf.net/1.1draft",
    "http://www.gexf.net/1.2draft",
    "http://gexf.net/1.2draft",
    "http://gexf.net/1.3",
}
VIZ_NAMESPACES = {"http://www.gephi.org/gexf/1.1draft/viz", "http://www.gexf.net/1.1draft/viz", "http://www.gexf.net/1.2draft/viz", "http://gexf.net/1.3/viz", "http://www.gexf.net/1.3/viz"}
GEXF_NODE_SHAPE_ICON_MAP = {"disc": "circle", "square": "square", "triangle": "caret-up", "diamond": "diamond", "image": "picture-o"}
GEXF_NODE_VIZ_ALLOWED: Set[str] = {"color", "size", "opacity", "position", "icon"}
GEXF_EDGE_VIZ_ALLOWED: Set[str] = {"color", "size", "opacity"}
DEFAULT_GEXF_URL_TIMEOUT_SECONDS = 10.0

GexfSource = Union[str, bytes, bytearray, IO[bytes], IO[str]]
ScalarValue = Optional[Union[int, float, bool, str]]
AttrDef = Tuple[str, str, Optional[str]]

def _read_source_bytes(source: GexfSource) -> bytes:
    if hasattr(source, "read"):
        data = source.read()
        return bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            with urlopen(source, timeout=DEFAULT_GEXF_URL_TIMEOUT_SECONDS) as resp:
                return resp.read()
        if not os.path.exists(source):
            raise ValueError(f"GEXF file not found: {source}")
        with open(source, "rb") as f:
            return f.read()
    raise ValueError("Unsupported GEXF source type")
def _coerce_value(raw: Optional[str], attr_type: str) -> ScalarValue:
    if raw is None:
        return None
    value = raw.strip()
    if attr_type in {"integer", "long", "int"}:
        try:
            return int(value)
        except ValueError:
            return None
    if attr_type in {"float", "double", "decimal"}:
        try:
            return float(value)
        except ValueError:
            return None
    if attr_type in {"boolean", "bool"}:
        return value.lower() in {"true", "1", "yes"}
    return value
def _rgb_to_hex(r: Optional[str], g: Optional[str], b: Optional[str]) -> Optional[str]:
    if r is None or g is None or b is None:
        return None
    try:
        return "#{:02X}{:02X}{:02X}".format(int(float(r)), int(float(g)), int(float(b)))
    except ValueError:
        return None
def _collect_attribute_defs(graph_elem: ET.Element, class_name: str) -> Dict[str, AttrDef]:
    defs: Dict[str, AttrDef] = {}
    for attrs_elem in graph_elem.findall("{*}attributes"):
        if attrs_elem.attrib.get("class") != class_name:
            continue
        for attr_elem in attrs_elem.findall("{*}attribute"):
            attr_id = attr_elem.attrib.get("id")
            if attr_id is None:
                continue
            title = attr_elem.attrib.get("title") or f"attr_{attr_id}"
            attr_type = attr_elem.attrib.get("type", "string")
            default_elem = attr_elem.find("{*}default")
            default = default_elem.text.strip() if default_elem is not None and default_elem.text else None
            defs[attr_id] = (title, attr_type, default)
    return defs
def _apply_attvalues(parent: ET.Element, attr_defs: Dict[str, AttrDef], row: Dict[str, ScalarValue]) -> None:
    for _attr_id, (title, attr_type, default) in attr_defs.items():
        if default is not None and title not in row:
            row[title] = _coerce_value(default, attr_type)
    attvalues_elem = parent.find("{*}attvalues")
    if attvalues_elem is None:
        return
    for attvalue in attvalues_elem.findall("{*}attvalue"):
        attr_id = attvalue.attrib.get("for")
        if attr_id is None:
            continue
        raw_value = attvalue.attrib.get("value")
        attr_def = attr_defs.get(attr_id)
        if attr_def is None:
            row[f"attr_{attr_id}"] = raw_value
        else:
            title, attr_type, _default = attr_def
            row[title] = _coerce_value(raw_value, attr_type)
def _parse_viz(parent: ET.Element, row: Dict[str, ScalarValue], element_kind: str) -> None:
    for child in parent:
        tag = child.tag
        if not tag.startswith("{"):
            continue
        ns, local = tag[1:].split("}", 1)
        if ns not in VIZ_NAMESPACES:
            continue
        attrib = child.attrib
        if local == "color":
            hex_val = attrib.get("hex") or _rgb_to_hex(attrib.get("r"), attrib.get("g"), attrib.get("b"))
            if hex_val is not None:
                row["viz_color"] = hex_val
            alpha_val = attrib.get("alpha") or attrib.get("a")
            if alpha_val is not None:
                row["viz_opacity"] = _coerce_value(alpha_val, "float")
        elif local == "position":
            for axis in ("x", "y", "z"):
                value = attrib.get(axis)
                if value is not None:
                    row[f"viz_{axis}"] = _coerce_value(value, "float")
        elif local in {"size", "thickness"}:
            value = attrib.get("value")
            if value is not None:
                row["viz_size" if local == "size" else "viz_thickness"] = _coerce_value(value, "float")
        elif local == "shape":
            value = attrib.get("value")
            if value is None:
                continue
            row["viz_shape"] = value
            if element_kind != "node":
                continue
            icon_value = GEXF_NODE_SHAPE_ICON_MAP.get(value)
            if value == "image":
                uri = attrib.get("uri")
                if uri:
                    row["viz_shape_uri"] = uri
                    icon_value = uri
            if icon_value is not None:
                row["viz_shape_icon"] = icon_value
def _normalize_viz_fields(value: Optional[Iterable[str]], allowed: Set[str], label: str) -> Set[str]:
    if value is None:
        return set(allowed)
    try:
        fields = {value} if isinstance(value, str) else set(value)
    except TypeError as exc:
        raise ValueError(f"{label} viz bindings must be an iterable of strings") from exc
    if not all(isinstance(v, str) for v in fields):
        raise ValueError(f"{label} viz bindings must be strings")
    unknown = fields.difference(allowed)
    if unknown:
        raise ValueError(f"Unsupported {label} viz bindings: {', '.join(sorted(unknown))}")
    return fields
def gexf_to_dfs(
    source: GexfSource,
    node_col: str = "node_id",
    source_col: str = "source",
    destination_col: str = "target",
    edge_id_col: str = "edge_id",
    parse_engine: GexfParseEngine = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Optional[str]]]:
    data = _read_source_bytes(source)
    parser: types.ModuleType
    if parse_engine == "stdlib":
        parser = ET
    elif parse_engine == "defused":
        if DEFUSED_ET is None:
            raise ValueError("defusedxml is not installed; install it or use parse_engine='stdlib'")
        parser = DEFUSED_ET
    elif parse_engine == "auto":
        parser = DEFUSED_ET if DEFUSED_ET is not None else ET
    else:
        raise ValueError(f"Unsupported parse_engine: {parse_engine}")
    try:
        root = parser.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid GEXF XML: {exc}") from exc
    tag = root.tag
    if tag.startswith("{"):
        ns, root_name = tag[1:].split("}", 1)
    else:
        ns, root_name = "", tag
    if root_name != "gexf":
        raise ValueError("Invalid GEXF: root tag is not <gexf>")
    if ns not in GEXF_NAMESPACES:
        raise ValueError(f"Unsupported GEXF namespace: {ns}")
    graph_elem = root.find("{*}graph")
    if graph_elem is None:
        raise ValueError("Invalid GEXF: missing <graph> element")
    meta: Dict[str, Optional[str]] = {"namespace": ns, "version": root.attrib.get("version"), "defaultedgetype": graph_elem.attrib.get("defaultedgetype")}
    meta_elem = root.find("{*}meta")
    if meta_elem is not None:
        for key in ("creator", "description"):
            text = meta_elem.findtext(f"{{*}}{key}")
            if text:
                meta[key] = text.strip()
    node_attr_defs = _collect_attribute_defs(graph_elem, "node")
    edge_attr_defs = _collect_attribute_defs(graph_elem, "edge")
    nodes_elem = graph_elem.find("{*}nodes")
    if nodes_elem is None:
        raise ValueError("Invalid GEXF: missing <nodes> element")
    node_rows: List[Dict[str, ScalarValue]] = []
    for node in nodes_elem.findall("{*}node"):
        node_id = node.attrib.get("id")
        if node_id is None:
            raise ValueError("Invalid GEXF: node missing id attribute")
        row: Dict[str, ScalarValue] = {node_col: str(node_id)}
        label = node.attrib.get("label")
        if label is not None:
            row["label"] = label
        _apply_attvalues(node, node_attr_defs, row)
        _parse_viz(node, row, "node")
        node_rows.append(row)
    nodes_df = pd.DataFrame(node_rows)
    if nodes_df.empty:
        raise ValueError("Invalid GEXF: no nodes found")
    if nodes_df[node_col].duplicated().any():
        raise ValueError("Invalid GEXF: duplicate node ids")
    node_ids = set(nodes_df[node_col].astype(str))
    edges_elem = graph_elem.find("{*}edges")
    edge_rows: List[Dict[str, ScalarValue]] = []
    if edges_elem is not None:
        for edge in edges_elem.findall("{*}edge"):
            src = edge.attrib.get("source")
            dst = edge.attrib.get("target")
            if src is None or dst is None:
                raise ValueError("Invalid GEXF: edge missing source/target")
            row = {source_col: str(src), destination_col: str(dst)}
            edge_id = edge.attrib.get("id")
            if edge_id is not None:
                row[edge_id_col] = str(edge_id)
            label = edge.attrib.get("label")
            if label is not None:
                row["label"] = label
            weight = edge.attrib.get("weight")
            if weight is not None:
                row["weight"] = _coerce_value(weight, "float")
            _apply_attvalues(edge, edge_attr_defs, row)
            _parse_viz(edge, row, "edge")
            edge_rows.append(row)

    edges_df = pd.DataFrame(columns=[source_col, destination_col]) if edges_elem is None else pd.DataFrame(edge_rows)

    if len(edges_df) > 0:
        missing = set(edges_df[source_col].astype(str)) | set(edges_df[destination_col].astype(str))
        missing = missing.difference(node_ids)
        if missing:
            missing_list = ", ".join(sorted(list(missing))[:5])
            raise ValueError(f"Invalid GEXF: edges reference missing node ids ({missing_list})")

    return edges_df, nodes_df, meta
def from_gexf(
    self: Plottable,
    source: GexfSource,
    name: Optional[str] = None,
    description: Optional[str] = None,
    bind_node_viz: Optional[Iterable[GexfNodeViz]] = None,
    bind_edge_viz: Optional[Iterable[GexfEdgeViz]] = None,
    parse_engine: GexfParseEngine = "auto",
) -> Plottable:
    """Load a GEXF file/URL/stream into a PyGraphistry plotter."""
    edges_df, nodes_df, meta = gexf_to_dfs(source, parse_engine=parse_engine)
    g = self.edges(edges_df, "source", "target").nodes(nodes_df, "node_id")
    bindings: Dict[str, str] = {"node": "node_id", "source": "source", "destination": "target"}

    node_viz = _normalize_viz_fields(bind_node_viz, GEXF_NODE_VIZ_ALLOWED, "node")
    edge_viz = _normalize_viz_fields(bind_edge_viz, GEXF_EDGE_VIZ_ALLOWED, "edge")

    node_specs = [(None, "point_title", "label"), ("color", "point_color", "viz_color"), ("size", "point_size", "viz_size"), ("opacity", "point_opacity", "viz_opacity"), ("position", "point_x", "viz_x"), ("position", "point_y", "viz_y"), ("icon", "point_icon", "viz_shape_icon")]
    for viz_key, binding_key, col in node_specs:
        if col in nodes_df.columns and (viz_key is None or viz_key in node_viz):
            bindings[binding_key] = col

    edge_specs = [(None, "edge_title", "label"), ("color", "edge_color", "viz_color"), ("size", "edge_size", "viz_thickness"), ("opacity", "edge_opacity", "viz_opacity")]
    for viz_key, binding_key, col in edge_specs:
        if col in edges_df.columns and (viz_key is None or viz_key in edge_viz):
            bindings[binding_key] = col
    if "weight" in edges_df.columns:
        bindings["edge_weight"] = "weight"

    g = g.bind(**bindings)

    if "position" in node_viz and "viz_x" in nodes_df.columns and "viz_y" in nodes_df.columns:
        g = g.settings(url_params={"play": 0})

    if name is not None:
        g = g.name(name)
    if description is not None:
        g = g.description(description)
    else:
        meta_description = meta.get("description")
        if meta_description is not None:
            g = g.description(meta_description)

    return g
