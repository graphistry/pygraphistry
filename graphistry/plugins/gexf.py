from __future__ import annotations

from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import os
from urllib.request import urlopen
import xml.etree.ElementTree as ET

try:
    from defusedxml import ElementTree as DEFUSED_ET
except Exception:
    DEFUSED_ET = None

import pandas as pd

from graphistry.Plottable import Plottable
from graphistry.plugins_types.gexf_types import GexfEdgeViz, GexfNodeViz, GexfParseEngine
GEXF_NAMESPACES = {
    "http://www.gephi.org/gexf/1.1draft", "http://www.gexf.net/1.1draft",
    "http://www.gexf.net/1.2draft", "http://gexf.net/1.3",
}
VIZ_NAMESPACES = {
    "http://www.gephi.org/gexf/1.1draft/viz", "http://www.gexf.net/1.1draft/viz",
    "http://www.gexf.net/1.2draft/viz", "http://gexf.net/1.3/viz", "http://www.gexf.net/1.3/viz",
}
GEXF_NODE_SHAPE_ICON_MAP = {
    "disc": "circle", "square": "square", "triangle": "caret-up",
    "diamond": "diamond", "image": "picture-o",
}
EXPORT_VERSION_CONFIG = {
    "1.1draft": ("http://www.gexf.net/1.1draft", "http://www.gexf.net/1.1draft/viz",
                 "http://www.gexf.net/1.1draft/gexf.xsd", "1.1"),
    "1.2draft": ("http://www.gexf.net/1.2draft", "http://www.gexf.net/1.2draft/viz",
                 "http://www.gexf.net/1.2draft/gexf.xsd", "1.2"),
    "1.3": ("http://gexf.net/1.3", "http://gexf.net/1.3/viz", "http://gexf.net/1.3/gexf.xsd", "1.3"),
}
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
GEXF_NODE_VIZ_ALLOWED: Set[str] = {"color", "size", "opacity", "position", "icon"}
GEXF_EDGE_VIZ_ALLOWED: Set[str] = {"color", "size", "opacity"}
GexfSource = Union[str, bytes, bytearray, IO[bytes], IO[str]]

def _split_tag(tag: str) -> Tuple[str, str]:
    if tag.startswith("{") and "}" in tag:
        ns, local = tag[1:].split("}", 1)
        return ns, local
    return "", tag

def _find_child(elem: ET.Element, local_name: str) -> Optional[ET.Element]:
    return next((child for child in elem if _split_tag(child.tag)[1] == local_name), None)

def _iter_children(elem: ET.Element, local_name: str) -> Iterable[ET.Element]:
    for child in elem:
        if _split_tag(child.tag)[1] == local_name:
            yield child

def _read_source_bytes(source: GexfSource) -> bytes:
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

ScalarValue = Optional[Union[int, float, bool, str]]

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

def _rgb_to_hex(r: str, g: str, b: str) -> Optional[str]:
    try:
        return "#{:02X}{:02X}{:02X}".format(int(float(r)), int(float(g)), int(float(b)))
    except ValueError:
        return None

def _infer_attr_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "string"

def _is_na(value: Any) -> bool:
    if value is None:
        return True
    if not pd.api.types.is_scalar(value):
        return False
    return bool(pd.isna(value))

def _format_attr_value(value: ScalarValue, attr_type: str) -> Optional[str]:
    if _is_na(value):
        return None
    if attr_type == "boolean":
        return "true" if bool(value) else "false"
    return str(value)

def _parse_hex_color(value: Any, label: str) -> Optional[Tuple[int, int, int, Optional[float]]]:
    if _is_na(value):
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid {label} color value {value!r}; expected hex string like #RRGGBB")
    v = value.strip()
    if not v.startswith("#"):
        raise ValueError(f"Invalid {label} color value {value!r}; expected hex string like #RRGGBB")
    hex_body = v[1:]
    if len(hex_body) not in {6, 8}:
        raise ValueError(f"Invalid {label} color value {value!r}; expected #RRGGBB or #RRGGBBAA")
    try:
        r = int(hex_body[0:2], 16)
        g = int(hex_body[2:4], 16)
        b = int(hex_body[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"Invalid {label} color value {value!r}; expected hex string like #RRGGBB") from exc
    alpha = None
    if len(hex_body) == 8:
        alpha = int(hex_body[6:8], 16) / 255.0
    return r, g, b, alpha

def _coerce_float(value: Any, label: str) -> Optional[float]:
    if _is_na(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} value {value!r}; expected numeric") from exc

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

def _apply_attvalues(parent: ET.Element, attr_defs: Dict[str, AttrDef], row: Dict[str, ScalarValue]) -> None:
    for _, (title, attr_type, default) in attr_defs.items():
        if default is not None and title not in row:
            row[title] = _coerce_value(default, attr_type)
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

def _parse_viz(parent: ET.Element, row: Dict[str, ScalarValue], element_kind: str) -> None:
    for child in list(parent):
        ns, local = _split_tag(child.tag)
        if ns not in VIZ_NAMESPACES:
            continue
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
            for axis in ("x", "y", "z"):
                value = child.attrib.get(axis)
                if value is not None:
                    row[f"viz_{axis}"] = _coerce_value(value, "float")
        elif local in {"size", "thickness"}:
            value = child.attrib.get("value")
            if value is not None:
                row["viz_size" if local == "size" else "viz_thickness"] = _coerce_value(value, "float")
        elif local == "shape":
            value = child.attrib.get("value")
            if value is not None:
                row["viz_shape"] = value
                if element_kind == "node":
                    icon_value = GEXF_NODE_SHAPE_ICON_MAP.get(value)
                    if value == "image":
                        uri = child.attrib.get("uri")
                        if uri:
                            row["viz_shape_uri"] = uri
                            icon_value = uri
                    if icon_value is not None:
                        row["viz_shape_icon"] = icon_value

def _normalize_viz_fields(
    value: Optional[Iterable[str]],
    allowed: Set[str],
    label: str,
) -> Set[str]:
    if value is None:
        return set(allowed)
    if isinstance(value, str):
        fields = {value}
    else:
        try:
            fields = set(value)
        except TypeError as exc:
            raise ValueError(f"{label} viz bindings must be an iterable of strings") from exc
    if not all(isinstance(v, str) for v in fields):
        raise ValueError(f"{label} viz bindings must be strings")
    unknown = fields.difference(allowed)
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported {label} viz bindings: {unknown_list}")
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

    ns, root_name = _split_tag(root.tag)
    if root_name != "gexf":
        raise ValueError("Invalid GEXF: root tag is not <gexf>")
    if ns not in GEXF_NAMESPACES:
        raise ValueError(f"Unsupported GEXF namespace: {ns}")

    graph_elem = _find_child(root, "graph")
    if graph_elem is None:
        raise ValueError("Invalid GEXF: missing <graph> element")

    meta: Dict[str, Optional[str]] = {
        "namespace": ns,
        "version": root.attrib.get("version"),
        "defaultedgetype": graph_elem.attrib.get("defaultedgetype"),
    }
    meta_elem = _find_child(root, "meta")
    if meta_elem is not None:
        for key in ("creator", "description"):
            elem = _find_child(meta_elem, key)
            if elem is not None and elem.text is not None:
                meta[key] = elem.text.strip()

    node_attr_defs = _collect_attribute_defs(graph_elem, "node")
    edge_attr_defs = _collect_attribute_defs(graph_elem, "edge")

    nodes_elem = _find_child(graph_elem, "nodes")
    if nodes_elem is None:
        raise ValueError("Invalid GEXF: missing <nodes> element")

    node_rows: List[Dict[str, ScalarValue]] = []
    node_ids: Set[str] = set()
    for node in _iter_children(nodes_elem, "node"):
        node_id = node.attrib.get("id")
        if node_id is None:
            raise ValueError("Invalid GEXF: node missing id attribute")
        node_id = str(node_id)
        if node_id in node_ids:
            raise ValueError("Invalid GEXF: duplicate node ids")
        node_ids.add(node_id)
        row: Dict[str, ScalarValue] = {node_col: node_id}
        label = node.attrib.get("label")
        if label is not None:
            row["label"] = label
        _apply_attvalues(node, node_attr_defs, row)
        _parse_viz(node, row, "node")
        node_rows.append(row)

    edges_elem = _find_child(graph_elem, "edges")
    edge_rows: List[Dict[str, ScalarValue]] = []
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
            _apply_attvalues(edge, edge_attr_defs, row)
            _parse_viz(edge, row, "edge")
            edge_rows.append(row)

    nodes_df = pd.DataFrame(node_rows)
    if len(nodes_df) == 0:
        raise ValueError("Invalid GEXF: no nodes found")

    if edges_elem is None:
        edges_df = pd.DataFrame(columns=[source_col, destination_col])
    else:
        edges_df = pd.DataFrame(edge_rows)

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
    """
    Convert a GEXF file/URL/stream into a PyGraphistry graph.

    :param source: Path, URL, bytes, or file-like object containing GEXF XML
    :param name: Optional Graphistry dataset name override
    :param description: Optional Graphistry dataset description override
    :param bind_node_viz: Optional list of node viz fields to bind (honor). None means bind all available node viz fields.
        Empty list means bind none. Choices: "color", "size", "opacity", "position", "icon".
    :param bind_edge_viz: Optional list of edge viz fields to bind (honor). None means bind all available edge viz fields.
        Empty list means bind none. Choices: "color", "size", "opacity".
    :param parse_engine: XML parser to use: "auto" (prefer defusedxml if installed), "defused", or "stdlib".
    :return: Graphistry plotter with nodes/edges/bindings populated

    If ``defusedxml`` is installed, it is used automatically for safer XML parsing of untrusted inputs.
    """
    edges_df, nodes_df, meta = gexf_to_dfs(source, parse_engine=parse_engine)
    g = self.edges(edges_df, "source", "target").nodes(nodes_df, "node_id")
    bindings: Dict[str, str] = {"node": "node_id", "source": "source", "destination": "target"}

    node_viz = _normalize_viz_fields(bind_node_viz, GEXF_NODE_VIZ_ALLOWED, "node")
    edge_viz = _normalize_viz_fields(bind_edge_viz, GEXF_EDGE_VIZ_ALLOWED, "edge")

    node_specs = [
        (None, "point_title", "label"),
        ("color", "point_color", "viz_color"),
        ("size", "point_size", "viz_size"),
        ("opacity", "point_opacity", "viz_opacity"),
        ("position", "point_x", "viz_x"),
        ("position", "point_y", "viz_y"),
        ("icon", "point_icon", "viz_shape_icon"),
    ]
    for viz_key, binding_key, col in node_specs:
        if col in nodes_df.columns and (viz_key is None or viz_key in node_viz):
            bindings[binding_key] = col

    edge_specs = [
        (None, "edge_title", "label"),
        ("color", "edge_color", "viz_color"),
        ("size", "edge_size", "viz_thickness"),
        ("opacity", "edge_opacity", "viz_opacity"),
    ]
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
    elif meta.get("description"):
        g = g.description(meta["description"])

    return g

def _df_to_pandas(df: Any, label: str) -> pd.DataFrame:
    if df is None:
        raise ValueError(f"Missing {label} dataframe")
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    raise ValueError(f"Unsupported {label} dataframe type: {type(df)}")

def _resolve_attr_columns(
    df: pd.DataFrame,
    include: Optional[List[str]],
    exclude: Iterable[str],
    label: str,
) -> List[str]:
    exclude_set = set(exclude)
    if include is None:
        return [c for c in df.columns if c not in exclude_set]
    missing = [c for c in include if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} attribute columns: {missing}")
    conflicts = [c for c in include if c in exclude_set]
    if conflicts:
        raise ValueError(f"Reserved {label} attribute columns: {conflicts}")
    return list(include)

def _select_col(
    df: pd.DataFrame,
    bound: Optional[str],
    fallback: str,
    label: str,
) -> Optional[str]:
    if bound is not None:
        if bound not in df.columns:
            raise ValueError(f"Missing {label} column '{bound}'")
        return bound
    if fallback in df.columns:
        return fallback
    return None

AttrExportDef = Dict[str, Tuple[str, str]]

def _write_attr_defs(
    graph_elem: ET.Element,
    gexf_ns: str,
    class_name: str,
    df: pd.DataFrame,
    attr_cols: List[str],
) -> AttrExportDef:
    if not attr_cols:
        return {}
    attrs_elem = ET.SubElement(graph_elem, f"{{{gexf_ns}}}attributes", attrib={"class": class_name})
    attr_defs = {col: (str(i), _infer_attr_type(df[col])) for i, col in enumerate(attr_cols)}
    for col, (attr_id, attr_type) in attr_defs.items():
        ET.SubElement(
            attrs_elem,
            f"{{{gexf_ns}}}attribute",
            attrib={"id": attr_id, "title": col, "type": attr_type},
        )
    return attr_defs

def _write_attvalues(
    elem: ET.Element,
    gexf_ns: str,
    row: pd.Series,
    attr_defs: AttrExportDef,
) -> None:
    if not attr_defs:
        return
    attvalues = []
    for col, (attr_id, attr_type) in attr_defs.items():
        value = _format_attr_value(row.get(col), attr_type)
        if value is not None:
            attvalues.append((attr_id, value))
    if not attvalues:
        return
    attvalues_elem = ET.SubElement(elem, f"{{{gexf_ns}}}attvalues")
    for attr_id, value in attvalues:
        ET.SubElement(
            attvalues_elem,
            f"{{{gexf_ns}}}attvalue",
            attrib={"for": attr_id, "value": value},
        )

def _write_viz(
    elem: ET.Element,
    viz_ns: str,
    row: pd.Series,
    *,
    color_col: Optional[str],
    opacity_col: Optional[str],
    size_col: Optional[str],
    size_label: str,
    size_tag: str,
    shape_col: Optional[str],
    position_cols: Optional[Tuple[Optional[str], Optional[str], Optional[str]]],
    label: str,
) -> None:
    color_value = row.get(color_col) if color_col is not None else None
    opacity_value = row.get(opacity_col) if opacity_col is not None else None
    if color_value is not None or opacity_value is not None:
        color_tuple = _parse_hex_color(color_value, label) if color_value is not None else None
        if color_tuple is not None:
            opacity = _coerce_float(opacity_value, f"{label} opacity") if opacity_value is not None else None
            r, g_val, b_val, alpha = color_tuple
            color_attrib = {"r": str(r), "g": str(g_val), "b": str(b_val)}
            alpha_val = opacity if opacity is not None else alpha
            if alpha_val is not None:
                color_attrib["a"] = str(alpha_val)
            ET.SubElement(elem, f"{{{viz_ns}}}color", attrib=color_attrib)

    size_value = row.get(size_col) if size_col is not None else None
    if size_value is not None:
        size = _coerce_float(size_value, size_label)
        if size is not None:
            ET.SubElement(elem, f"{{{viz_ns}}}{size_tag}", attrib={"value": str(size)})

    if position_cols is not None:
        x_col, y_col, z_col = position_cols
        if x_col is not None and y_col is not None:
            x = _coerce_float(row.get(x_col), "node x")
            y = _coerce_float(row.get(y_col), "node y")
            z = _coerce_float(row.get(z_col), "node z") if z_col is not None else None
            if x is not None and y is not None:
                pos_attrib = {"x": str(x), "y": str(y)}
                if z is not None:
                    pos_attrib["z"] = str(z)
                ET.SubElement(elem, f"{{{viz_ns}}}position", attrib=pos_attrib)

    shape_value = row.get(shape_col) if shape_col is not None else None
    if shape_col is not None and not _is_na(shape_value):
        ET.SubElement(elem, f"{{{viz_ns}}}shape", attrib={"value": str(shape_value)})

def to_gexf(
    self: Plottable,
    path: Optional[str] = None,
    *,
    version: str = "1.2draft",
    directed: bool = True,
    include_viz: bool = True,
    include_meta: bool = True,
    meta_creator: Optional[str] = "graphistry",
    meta_description: Optional[str] = None,
    meta_lastmodifieddate: Optional[str] = None,
    node_attributes: Optional[List[str]] = None,
    edge_attributes: Optional[List[str]] = None,
) -> str:
    """
    Export the current graph to a GEXF string (optionally writing to disk).

    :param path: Optional output path to write the GEXF XML
    :param version: GEXF version key ("1.1draft", "1.2draft", or "1.3")
    :param directed: Whether edges should be marked as directed
    :param include_viz: Whether to export viz attributes (color/size/position/opacity/thickness)
    :param include_meta: Whether to include a <meta> section
    :param meta_creator: Optional meta.creator value
    :param meta_description: Optional meta.description value (defaults to g._description)
    :param meta_lastmodifieddate: Optional meta lastmodifieddate attribute
    :param node_attributes: Optional list of node attribute columns to include
    :param edge_attributes: Optional list of edge attribute columns to include
    :return: GEXF XML string
    """
    try:
        gexf_ns, viz_ns, schema, gexf_version = EXPORT_VERSION_CONFIG[version]
    except KeyError as exc:
        raise ValueError(f"Unsupported GEXF export version: {version}") from exc

    g = self.materialize_nodes()
    if g._edges is None:
        raise ValueError("Missing edges")
    if g._source is None or g._destination is None:
        raise ValueError("Missing source/destination bindings")

    nodes_df = _df_to_pandas(g._nodes, "nodes")
    edges_df = _df_to_pandas(g._edges, "edges")

    node_col = g._node if g._node is not None else "id"
    if node_col not in nodes_df.columns:
        raise ValueError(f"Missing node id column '{node_col}'")
    if g._source not in edges_df.columns or g._destination not in edges_df.columns:
        raise ValueError("Missing edge source/destination columns")

    if nodes_df[node_col].isna().any():
        raise ValueError("GEXF export requires non-null node ids")
    if nodes_df[node_col].duplicated().any():
        raise ValueError("GEXF export requires unique node ids")
    if edges_df[g._source].isna().any() or edges_df[g._destination].isna().any():
        raise ValueError("GEXF export requires non-null edge endpoints")

    node_label_col = g._point_title or g._point_label
    if node_label_col is not None and node_label_col not in nodes_df.columns:
        raise ValueError(f"Missing node label column '{node_label_col}'")
    edge_label_col = g._edge_title or g._edge_label
    if edge_label_col is not None and edge_label_col not in edges_df.columns:
        raise ValueError(f"Missing edge label column '{edge_label_col}'")

    edge_id_col = g._edge if g._edge is not None else None
    if edge_id_col is not None and edge_id_col not in edges_df.columns:
        raise ValueError(f"Missing edge id column '{edge_id_col}'")

    edge_weight_col = None
    if g._edge_weight is not None:
        if g._edge_weight not in edges_df.columns:
            raise ValueError(f"Missing edge weight column '{g._edge_weight}'")
        edge_weight_col = g._edge_weight
    elif "weight" in edges_df.columns:
        edge_weight_col = "weight"

    point_color_col = point_size_col = point_opacity_col = point_x_col = point_y_col = None
    point_z_col = point_shape_col = edge_color_col = edge_size_col = edge_opacity_col = edge_shape_col = None

    if include_viz:
        point_color_col, point_size_col, point_opacity_col = (
            _select_col(nodes_df, g._point_color, "viz_color", "point_color"),
            _select_col(nodes_df, g._point_size, "viz_size", "point_size"),
            _select_col(nodes_df, g._point_opacity, "viz_opacity", "point_opacity"),
        )
        point_x_col, point_y_col = (
            _select_col(nodes_df, g._point_x, "viz_x", "point_x"),
            _select_col(nodes_df, g._point_y, "viz_y", "point_y"),
        )
        point_z_col = "viz_z" if "viz_z" in nodes_df.columns else None
        point_shape_col = "viz_shape" if "viz_shape" in nodes_df.columns else None
        edge_color_col, edge_size_col, edge_opacity_col = (
            _select_col(edges_df, g._edge_color, "viz_color", "edge_color"),
            _select_col(edges_df, g._edge_size, "viz_thickness", "edge_size"),
            _select_col(edges_df, g._edge_opacity, "viz_opacity", "edge_opacity"),
        )
        edge_shape_col = "viz_shape" if "viz_shape" in edges_df.columns else None

        if (point_x_col is None) != (point_y_col is None):
            raise ValueError("GEXF export requires both point_x and point_y columns when exporting positions")

    node_viz_cols = [
        point_color_col, point_size_col, point_opacity_col, point_x_col, point_y_col, point_z_col, point_shape_col,
    ]
    edge_viz_cols = [edge_color_col, edge_size_col, edge_opacity_col, edge_shape_col]

    node_exclude = {node_col}
    if node_label_col is not None:
        node_exclude.add(node_label_col)
    node_exclude.update(col for col in node_viz_cols if col is not None)

    edge_exclude = {g._source, g._destination}
    if edge_label_col is not None:
        edge_exclude.add(edge_label_col)
    if edge_id_col is not None:
        edge_exclude.add(edge_id_col)
    if edge_weight_col is not None:
        edge_exclude.add(edge_weight_col)
    edge_exclude.update(col for col in edge_viz_cols if col is not None)

    node_attr_cols = _resolve_attr_columns(nodes_df, node_attributes, node_exclude, "node")
    edge_attr_cols = _resolve_attr_columns(edges_df, edge_attributes, edge_exclude, "edge")

    ET.register_namespace("", gexf_ns)
    ET.register_namespace("xsi", XSI_NS)
    if include_viz and any(node_viz_cols + edge_viz_cols):
        ET.register_namespace("viz", viz_ns)

    root = ET.Element(
        f"{{{gexf_ns}}}gexf",
        attrib={
            "version": gexf_version,
            f"{{{XSI_NS}}}schemaLocation": f"{gexf_ns} {schema}",
        },
    )

    if include_meta:
        meta_description = meta_description if meta_description is not None else g._description
        meta_attrib: Dict[str, str] = {}
        if meta_lastmodifieddate is not None:
            meta_attrib["lastmodifieddate"] = meta_lastmodifieddate
        meta_elem = ET.SubElement(root, f"{{{gexf_ns}}}meta", attrib=meta_attrib)
        if meta_creator:
            creator_elem = ET.SubElement(meta_elem, f"{{{gexf_ns}}}creator")
            creator_elem.text = str(meta_creator)
        if meta_description:
            desc_elem = ET.SubElement(meta_elem, f"{{{gexf_ns}}}description")
            desc_elem.text = str(meta_description)

    graph_elem = ET.SubElement(
        root,
        f"{{{gexf_ns}}}graph",
        attrib={
            "mode": "static",
            "defaultedgetype": "directed" if directed else "undirected",
        },
    )

    node_attr_defs = _write_attr_defs(graph_elem, gexf_ns, "node", nodes_df, node_attr_cols)
    edge_attr_defs = _write_attr_defs(graph_elem, gexf_ns, "edge", edges_df, edge_attr_cols)

    nodes_elem = ET.SubElement(graph_elem, f"{{{gexf_ns}}}nodes")
    for _, row in nodes_df.iterrows():
        node_id = str(row[node_col])
        node_attrib = {"id": node_id}
        if node_label_col is not None:
            label_val = row.get(node_label_col)
            if not _is_na(label_val):
                node_attrib["label"] = str(label_val)
        node_elem = ET.SubElement(nodes_elem, f"{{{gexf_ns}}}node", attrib=node_attrib)

        if include_viz:
            _write_viz(
                node_elem,
                viz_ns,
                row,
                color_col=point_color_col,
                opacity_col=point_opacity_col,
                size_col=point_size_col,
                size_label="node size",
                size_tag="size",
                shape_col=point_shape_col,
                position_cols=(point_x_col, point_y_col, point_z_col),
                label="node",
            )

        _write_attvalues(node_elem, gexf_ns, row, node_attr_defs)

    edges_elem = ET.SubElement(graph_elem, f"{{{gexf_ns}}}edges")
    for idx, row in edges_df.iterrows():
        edge_id = row.get(edge_id_col) if edge_id_col is not None else idx
        edge_attrib = {
            "id": str(edge_id),
            "source": str(row[g._source]),
            "target": str(row[g._destination]),
        }
        if edge_label_col is not None:
            label_val = row.get(edge_label_col)
            if not _is_na(label_val):
                edge_attrib["label"] = str(label_val)
        if edge_weight_col is not None:
            weight_val = _coerce_float(row.get(edge_weight_col), "edge weight")
            if weight_val is not None:
                edge_attrib["weight"] = str(weight_val)
        edge_elem = ET.SubElement(edges_elem, f"{{{gexf_ns}}}edge", attrib=edge_attrib)

        if include_viz:
            _write_viz(
                edge_elem,
                viz_ns,
                row,
                color_col=edge_color_col,
                opacity_col=edge_opacity_col,
                size_col=edge_size_col,
                size_label="edge thickness",
                size_tag="thickness",
                shape_col=edge_shape_col,
                position_cols=None,
                label="edge",
            )

        _write_attvalues(edge_elem, gexf_ns, row, edge_attr_defs)

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    xml_str = xml_bytes.decode("utf-8")
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml_str)
    return xml_str
