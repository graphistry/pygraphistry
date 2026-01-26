from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import pandas as pd

from graphistry.Plottable import Plottable


GEXF_NAMESPACES = {
    "http://www.gephi.org/gexf/1.1draft",
    "http://www.gexf.net/1.1draft",
    "http://www.gexf.net/1.2draft",
    "http://gexf.net/1.3",
}

VIZ_NAMESPACES = {
    "http://www.gephi.org/gexf/1.1draft/viz",
    "http://www.gexf.net/1.1draft/viz",
    "http://www.gexf.net/1.2draft/viz",
    "http://gexf.net/1.3/viz",
    "http://www.gexf.net/1.3/viz",
}


def _namespace(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return ""


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_child(elem: ET.Element, local_name: str) -> Optional[ET.Element]:
    for child in list(elem):
        if _local_name(child.tag) == local_name:
            return child
    return None


def _iter_children(elem: ET.Element, local_name: str) -> Iterable[ET.Element]:
    for child in list(elem):
        if _local_name(child.tag) == local_name:
            yield child


def _read_source_bytes(source: Any) -> bytes:
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        return str(data).encode("utf-8")
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, str):
        if source.startswith("http://") or source.startswith("https://"):
            with urlopen(source) as resp:
                return resp.read()
        if not os.path.exists(source):
            raise ValueError(f"GEXF file not found: {source}")
        with open(source, "rb") as f:
            return f.read()
    raise ValueError("Unsupported GEXF source type")


def _coerce_value(raw: Optional[str], attr_type: str) -> Any:
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


def _rgb_to_hex(r: str, g: str, b: str) -> Optional[str]:
    try:
        return "#{:02X}{:02X}{:02X}".format(int(float(r)), int(float(g)), int(float(b)))
    except ValueError:
        return None


AttrDef = Tuple[str, str, Optional[str]]


def _collect_attribute_defs(graph_elem: ET.Element, class_name: str) -> Dict[str, AttrDef]:
    defs: Dict[str, AttrDef] = {}
    for attrs_elem in _iter_children(graph_elem, "attributes"):
        if attrs_elem.attrib.get("class") != class_name:
            continue
        for attr_elem in _iter_children(attrs_elem, "attribute"):
            attr_id = attr_elem.attrib.get("id")
            if attr_id is None:
                continue
            title = attr_elem.attrib.get("title") or f"attr_{attr_id}"
            attr_type = attr_elem.attrib.get("type", "string")
            default = None
            default_elem = _find_child(attr_elem, "default")
            if default_elem is not None and default_elem.text is not None:
                default = default_elem.text.strip()
            defs[attr_id] = (title, attr_type, default)
    return defs


def _apply_attr_defaults(attr_defs: Dict[str, AttrDef], row: Dict[str, Any]) -> None:
    for _, (title, attr_type, default) in attr_defs.items():
        if default is None:
            continue
        if title not in row:
            row[title] = _coerce_value(default, attr_type)


def _parse_attvalues(parent: ET.Element, attr_defs: Dict[str, AttrDef], row: Dict[str, Any]) -> None:
    attvalues_elem = _find_child(parent, "attvalues")
    if attvalues_elem is None:
        return
    for attvalue in _iter_children(attvalues_elem, "attvalue"):
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


def _parse_viz(parent: ET.Element, row: Dict[str, Any]) -> None:
    for child in list(parent):
        ns = _namespace(child.tag)
        if ns not in VIZ_NAMESPACES:
            continue
        local = _local_name(child.tag)
        if local == "color":
            hex_val = child.attrib.get("hex")
            alpha_val = child.attrib.get("alpha") or child.attrib.get("a")
            if hex_val is None:
                r = child.attrib.get("r")
                g = child.attrib.get("g")
                b = child.attrib.get("b")
                if r is not None and g is not None and b is not None:
                    hex_val = _rgb_to_hex(r, g, b)
            if hex_val is not None:
                row["viz_color"] = hex_val
            if alpha_val is not None:
                row["viz_opacity"] = _coerce_value(alpha_val, "float")
        elif local == "position":
            x = child.attrib.get("x")
            y = child.attrib.get("y")
            z = child.attrib.get("z")
            if x is not None:
                row["viz_x"] = _coerce_value(x, "float")
            if y is not None:
                row["viz_y"] = _coerce_value(y, "float")
            if z is not None:
                row["viz_z"] = _coerce_value(z, "float")
        elif local == "size":
            value = child.attrib.get("value")
            if value is not None:
                row["viz_size"] = _coerce_value(value, "float")
        elif local == "thickness":
            value = child.attrib.get("value")
            if value is not None:
                row["viz_thickness"] = _coerce_value(value, "float")
        elif local == "shape":
            value = child.attrib.get("value")
            if value is not None:
                row["viz_shape"] = value


def gexf_to_dfs(
    source: Any,
    node_col: str = "node_id",
    source_col: str = "source",
    destination_col: str = "target",
    edge_id_col: str = "edge_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    data = _read_source_bytes(source)
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid GEXF XML: {exc}") from exc

    if _local_name(root.tag) != "gexf":
        raise ValueError("Invalid GEXF: root tag is not <gexf>")
    ns = _namespace(root.tag)
    if ns not in GEXF_NAMESPACES:
        raise ValueError(f"Unsupported GEXF namespace: {ns}")

    graph_elem = _find_child(root, "graph")
    if graph_elem is None:
        raise ValueError("Invalid GEXF: missing <graph> element")

    meta = {
        "namespace": ns,
        "version": root.attrib.get("version"),
        "defaultedgetype": graph_elem.attrib.get("defaultedgetype"),
    }
    meta_elem = _find_child(root, "meta")
    if meta_elem is not None:
        creator_elem = _find_child(meta_elem, "creator")
        description_elem = _find_child(meta_elem, "description")
        if creator_elem is not None and creator_elem.text is not None:
            meta["creator"] = creator_elem.text.strip()
        if description_elem is not None and description_elem.text is not None:
            meta["description"] = description_elem.text.strip()

    node_attr_defs = _collect_attribute_defs(graph_elem, "node")
    edge_attr_defs = _collect_attribute_defs(graph_elem, "edge")

    nodes_elem = _find_child(graph_elem, "nodes")
    if nodes_elem is None:
        raise ValueError("Invalid GEXF: missing <nodes> element")

    node_rows: List[Dict[str, Any]] = []
    node_ids: List[str] = []
    for node in _iter_children(nodes_elem, "node"):
        node_id = node.attrib.get("id")
        if node_id is None:
            raise ValueError("Invalid GEXF: node missing id attribute")
        node_id = str(node_id)
        node_ids.append(node_id)
        row: Dict[str, Any] = {node_col: node_id}
        label = node.attrib.get("label")
        if label is not None:
            row["label"] = label
        _apply_attr_defaults(node_attr_defs, row)
        _parse_attvalues(node, node_attr_defs, row)
        _parse_viz(node, row)
        node_rows.append(row)

    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Invalid GEXF: duplicate node ids")

    edges_elem = _find_child(graph_elem, "edges")
    edge_rows: List[Dict[str, Any]] = []
    if edges_elem is not None:
        for edge in _iter_children(edges_elem, "edge"):
            edge_id = edge.attrib.get("id")
            src = edge.attrib.get("source")
            dst = edge.attrib.get("target")
            if src is None or dst is None:
                raise ValueError("Invalid GEXF: edge missing source/target")
            row = {
                source_col: str(src),
                destination_col: str(dst),
            }
            if edge_id is not None:
                row[edge_id_col] = str(edge_id)
            label = edge.attrib.get("label")
            if label is not None:
                row["label"] = label
            weight = edge.attrib.get("weight")
            if weight is not None:
                row["weight"] = _coerce_value(weight, "float")
            _apply_attr_defaults(edge_attr_defs, row)
            _parse_attvalues(edge, edge_attr_defs, row)
            _parse_viz(edge, row)
            edge_rows.append(row)

    nodes_df = pd.DataFrame(node_rows)
    if len(nodes_df) == 0:
        raise ValueError("Invalid GEXF: no nodes found")

    if edges_elem is None:
        edges_df = pd.DataFrame(columns=[source_col, destination_col])
    else:
        edges_df = pd.DataFrame(edge_rows)

    node_id_set = set(nodes_df[node_col].astype(str))
    if len(edges_df) > 0:
        missing = set(edges_df[source_col].astype(str)) | set(edges_df[destination_col].astype(str))
        missing = missing.difference(node_id_set)
        if missing:
            missing_list = ", ".join(sorted(list(missing))[:5])
            raise ValueError(f"Invalid GEXF: edges reference missing node ids ({missing_list})")

    return edges_df, nodes_df, meta


def from_gexf(
    self: Plottable,
    source: Any,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Plottable:
    """
    Convert a GEXF file/URL/stream into a PyGraphistry graph.

    :param source: Path, URL, bytes, or file-like object containing GEXF XML
    :param name: Optional Graphistry dataset name override
    :param description: Optional Graphistry dataset description override
    :return: Graphistry plotter with nodes/edges/bindings populated
    """
    edges_df, nodes_df, meta = gexf_to_dfs(source)
    g = self.edges(edges_df, "source", "target").nodes(nodes_df, "node_id")
    bindings: Dict[str, Any] = {"node": "node_id", "source": "source", "destination": "target"}

    if "label" in nodes_df.columns:
        bindings["point_title"] = "label"
    if "viz_color" in nodes_df.columns:
        bindings["point_color"] = "viz_color"
    if "viz_size" in nodes_df.columns:
        bindings["point_size"] = "viz_size"
    if "viz_opacity" in nodes_df.columns:
        bindings["point_opacity"] = "viz_opacity"
    if "viz_x" in nodes_df.columns:
        bindings["point_x"] = "viz_x"
    if "viz_y" in nodes_df.columns:
        bindings["point_y"] = "viz_y"

    if "label" in edges_df.columns:
        bindings["edge_title"] = "label"
    if "viz_color" in edges_df.columns:
        bindings["edge_color"] = "viz_color"
    if "viz_thickness" in edges_df.columns:
        bindings["edge_size"] = "viz_thickness"
    if "viz_opacity" in edges_df.columns:
        bindings["edge_opacity"] = "viz_opacity"
    if "weight" in edges_df.columns:
        bindings["edge_weight"] = "weight"

    g = g.bind(**bindings)

    if "viz_x" in nodes_df.columns and "viz_y" in nodes_df.columns:
        g = g.settings(url_params={"play": 0})

    if name is not None:
        g = g.name(name)
    if description is not None:
        g = g.description(description)
    elif meta.get("description"):
        g = g.description(meta["description"])

    return g
