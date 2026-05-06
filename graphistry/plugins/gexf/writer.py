from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import pandas as pd

from graphistry.Plottable import Plottable

EXPORT_VERSION_CONFIG = {
    "1.1draft": ("http://www.gexf.net/1.1draft", "http://www.gexf.net/1.1draft/viz", "http://www.gexf.net/1.1draft/gexf.xsd", "1.1"),
    "1.2draft": ("http://www.gexf.net/1.2draft", "http://www.gexf.net/1.2draft/viz", "http://www.gexf.net/1.2draft/gexf.xsd", "1.2"),
    "1.3": ("http://gexf.net/1.3", "http://gexf.net/1.3/viz", "http://gexf.net/1.3/gexf.xsd", "1.3"),
}
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

ScalarValue = Optional[Union[int, float, bool, str]]
AttrExportDef = Dict[str, Tuple[str, str]]
def _is_na(value: Any) -> bool:
    return value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value)))
def _infer_attr_type(series: pd.Series) -> str:
    return "boolean" if pd.api.types.is_bool_dtype(series) else "integer" if pd.api.types.is_integer_dtype(series) else "float" if pd.api.types.is_float_dtype(series) else "string"
def _format_attr_value(value: object, attr_type: str) -> Optional[str]:
    if _is_na(value):
        return None
    if attr_type == "boolean":
        return "true" if bool(value) else "false"
    return str(value)
def _parse_hex_color(value: Any, label: str) -> Optional[Tuple[int, int, int, Optional[float]]]:
    if _is_na(value):
        return None
    if not isinstance(value, str) or not value.strip().startswith("#"):
        raise ValueError(f"Invalid {label} color value {value!r}; expected hex string like #RRGGBB")
    hex_body = value.strip()[1:]
    if len(hex_body) not in {6, 8}:
        raise ValueError(f"Invalid {label} color value {value!r}; expected #RRGGBB or #RRGGBBAA")
    try:
        r, g, b = (int(hex_body[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"Invalid {label} color value {value!r}; expected hex string like #RRGGBB") from exc
    alpha = int(hex_body[6:8], 16) / 255.0 if len(hex_body) == 8 else None
    return r, g, b, alpha
def _coerce_float(value: Any, label: str) -> Optional[float]:
    if _is_na(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} value {value!r}; expected numeric") from exc
def _df_to_pandas(df: Any, label: str) -> pd.DataFrame:
    if df is None:
        raise ValueError(f"Missing {label} dataframe")
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    raise ValueError(f"Unsupported {label} dataframe type: {type(df)}")
def _require_cols(df: pd.DataFrame, cols: Iterable[Optional[str]], label: str) -> None:
    missing = [c for c in cols if c is not None and c not in df.columns]
    if not missing:
        return
    if len(missing) == 1:
        raise ValueError(f"Missing {label} column '{missing[0]}'")
    raise ValueError(f"Missing {label} columns: {missing}")
def _resolve_attr_columns(df: pd.DataFrame, include: Optional[List[str]], exclude: Iterable[str], label: str) -> List[str]:
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
def _select_col(df: pd.DataFrame, bound: Optional[str], fallback: str, label: str) -> Optional[str]:
    if bound is not None:
        if bound not in df.columns:
            raise ValueError(f"Missing {label} column '{bound}'")
        return bound
    return fallback if fallback in df.columns else None
def _write_attr_defs(graph_elem: ET.Element, gexf_ns: str, class_name: str, df: pd.DataFrame, attr_cols: List[str]) -> AttrExportDef:
    if not attr_cols:
        return {}
    attrs_elem = ET.SubElement(graph_elem, f"{{{gexf_ns}}}attributes", attrib={"class": class_name})
    attr_defs = {col: (str(i), _infer_attr_type(df[col])) for i, col in enumerate(attr_cols)}
    for col, (attr_id, attr_type) in attr_defs.items():
        ET.SubElement(attrs_elem, f"{{{gexf_ns}}}attribute", attrib={"id": attr_id, "title": col, "type": attr_type})
    return attr_defs
def _write_attvalues(elem: ET.Element, gexf_ns: str, row: pd.Series, attr_defs: AttrExportDef) -> None:
    if not attr_defs:
        return
    attvalues = [(attr_id, value) for col, (attr_id, attr_type) in attr_defs.items() if (value := _format_attr_value(row.get(col), attr_type)) is not None]
    if not attvalues:
        return
    attvalues_elem = ET.SubElement(elem, f"{{{gexf_ns}}}attvalues")
    for attr_id, value in attvalues:
        ET.SubElement(attvalues_elem, f"{{{gexf_ns}}}attvalue", attrib={"for": attr_id, "value": value})
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
    """Export the current graph to a GEXF string (optionally writing to disk)."""
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
    node_col = g._node or "id"
    _require_cols(nodes_df, [node_col], "node id")
    _require_cols(edges_df, [g._source, g._destination], "edge endpoint")
    if nodes_df[node_col].isna().any():
        raise ValueError("GEXF export requires non-null node ids")
    if nodes_df[node_col].duplicated().any():
        raise ValueError("GEXF export requires unique node ids")
    if edges_df[g._source].isna().any() or edges_df[g._destination].isna().any():
        raise ValueError("GEXF export requires non-null edge endpoints")
    node_label_col = g._point_title or g._point_label
    if node_label_col is not None:
        _require_cols(nodes_df, [node_label_col], "node label")
    edge_label_col = g._edge_title or g._edge_label
    if edge_label_col is not None:
        _require_cols(edges_df, [edge_label_col], "edge label")
    edge_id_col = g._edge if g._edge is not None else None
    if edge_id_col is not None:
        _require_cols(edges_df, [edge_id_col], "edge id")
    edge_weight_col = None
    if g._edge_weight is not None:
        _require_cols(edges_df, [g._edge_weight], "edge weight")
        edge_weight_col = g._edge_weight
    elif "weight" in edges_df.columns:
        edge_weight_col = "weight"
    point_color_col = point_size_col = point_opacity_col = None
    point_x_col = point_y_col = point_z_col = point_shape_col = None
    edge_color_col = edge_size_col = edge_opacity_col = edge_shape_col = None

    if include_viz:
        point_color_col = _select_col(nodes_df, g._point_color, "viz_color", "point_color")
        point_size_col = _select_col(nodes_df, g._point_size, "viz_size", "point_size")
        point_opacity_col = _select_col(nodes_df, g._point_opacity, "viz_opacity", "point_opacity")
        point_x_col = _select_col(nodes_df, g._point_x, "viz_x", "point_x")
        point_y_col = _select_col(nodes_df, g._point_y, "viz_y", "point_y")
        point_z_col = "viz_z" if "viz_z" in nodes_df.columns else None
        point_shape_col = "viz_shape" if "viz_shape" in nodes_df.columns else None
        edge_color_col = _select_col(edges_df, g._edge_color, "viz_color", "edge_color")
        edge_size_col = _select_col(edges_df, g._edge_size, "viz_thickness", "edge_size")
        edge_opacity_col = _select_col(edges_df, g._edge_opacity, "viz_opacity", "edge_opacity")
        edge_shape_col = "viz_shape" if "viz_shape" in edges_df.columns else None
        if (point_x_col is None) != (point_y_col is None):
            raise ValueError("GEXF export requires both point_x and point_y columns when exporting positions")

    node_viz_cols = [c for c in (point_color_col, point_size_col, point_opacity_col, point_x_col, point_y_col, point_z_col, point_shape_col) if c is not None]
    edge_viz_cols = [c for c in (edge_color_col, edge_size_col, edge_opacity_col, edge_shape_col) if c is not None]

    node_exclude = {c for c in (node_col, node_label_col, *node_viz_cols) if c}
    edge_exclude = {c for c in (g._source, g._destination, edge_label_col, edge_id_col, edge_weight_col, *edge_viz_cols) if c}

    node_attr_cols = _resolve_attr_columns(nodes_df, node_attributes, node_exclude, "node")
    edge_attr_cols = _resolve_attr_columns(edges_df, edge_attributes, edge_exclude, "edge")

    ET.register_namespace("", gexf_ns)
    ET.register_namespace("xsi", XSI_NS)
    if include_viz and (node_viz_cols or edge_viz_cols):
        ET.register_namespace("viz", viz_ns)

    root = ET.Element(f"{{{gexf_ns}}}gexf", attrib={"version": gexf_version, f"{{{XSI_NS}}}schemaLocation": f"{gexf_ns} {schema}"})

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

    graph_elem = ET.SubElement(root, f"{{{gexf_ns}}}graph", attrib={"mode": "static", "defaultedgetype": "directed" if directed else "undirected"})

    node_attr_defs = _write_attr_defs(graph_elem, gexf_ns, "node", nodes_df, node_attr_cols)
    edge_attr_defs = _write_attr_defs(graph_elem, gexf_ns, "edge", edges_df, edge_attr_cols)

    nodes_elem = ET.SubElement(graph_elem, f"{{{gexf_ns}}}nodes")
    for _, row in nodes_df.iterrows():
        node_attrib = {"id": str(row[node_col])}
        if node_label_col is not None:
            label_val = row.get(node_label_col)
            if not _is_na(label_val):
                node_attrib["label"] = str(label_val)
        node_elem = ET.SubElement(nodes_elem, f"{{{gexf_ns}}}node", attrib=node_attrib)
        if include_viz and node_viz_cols:
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
        edge_attrib = {"id": str(edge_id), "source": str(row[g._source]), "target": str(row[g._destination])}
        if edge_label_col is not None:
            label_val = row.get(edge_label_col)
            if not _is_na(label_val):
                edge_attrib["label"] = str(label_val)
        if edge_weight_col is not None:
            weight_val = _coerce_float(row.get(edge_weight_col), "edge weight")
            if weight_val is not None:
                edge_attrib["weight"] = str(weight_val)
        edge_elem = ET.SubElement(edges_elem, f"{{{gexf_ns}}}edge", attrib=edge_attrib)
        if include_viz and edge_viz_cols:
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

    xml_str = ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml_str)
    return xml_str
