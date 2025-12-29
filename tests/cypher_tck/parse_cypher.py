import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from tests.cypher_tck.models import GraphFixture


@dataclass
class ParseContext:
    nodes_by_id: Dict[str, Dict[str, Any]]
    var_to_id: Dict[str, str]
    node_counter: int
    rel_counter: int


_CREATE_SPLIT_RE = re.compile(r"\bCREATE\b", flags=re.IGNORECASE)


def _split_top_level(text: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0
    in_quote = False
    for ch in text:
        if ch == "'":
            in_quote = not in_quote
        if not in_quote:
            if ch == '(':
                depth_paren += 1
            elif ch == ')':
                depth_paren = max(depth_paren - 1, 0)
            elif ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace = max(depth_brace - 1, 0)
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket = max(depth_bracket - 1, 0)
        if ch == ',' and not in_quote and depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
            part = ''.join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = ''.join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> Tuple[str, str]:
    depth = 0
    start = None
    in_quote = False
    for idx, ch in enumerate(text):
        if ch == "'":
            in_quote = not in_quote
        if in_quote:
            continue
        if ch == open_ch:
            if depth == 0:
                start = idx
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : idx + 1], text[idx + 1 :]
    raise ValueError(f"Unbalanced {open_ch}{close_ch} in: {text}")


def _parse_properties(prop_text: str) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    inner = prop_text.strip()[1:-1].strip()
    if not inner:
        return props
    items = _split_top_level(inner)
    for item in items:
        if ':' not in item:
            continue
        key, raw = item.split(':', 1)
        key = key.strip()
        raw = raw.strip()
        if raw.startswith("'") and raw.endswith("'"):
            value: Any = raw[1:-1]
        elif re.fullmatch(r"-?\d+", raw):
            value = int(raw)
        elif re.fullmatch(r"-?\d+\.\d+", raw):
            value = float(raw)
        elif raw.lower() == "null":
            value = None
        elif raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            value = raw
        props[key] = value
    return props


def _parse_node(node_text: str, ctx: ParseContext) -> str:
    inner = node_text.strip()[1:-1].strip()
    props: Dict[str, Any] = {}
    if '{' in inner:
        before, prop_part = inner.split('{', 1)
        inner = before.strip()
        props = _parse_properties('{' + prop_part)
    var: str | None = None
    labels: List[str] = []
    if inner:
        if inner.startswith(':'):
            var_part = ''
            label_part = inner
        else:
            var_part, *rest = inner.split(':')
            label_part = ':' + ':'.join(rest) if rest else ''
        var_part = var_part.strip()
        if var_part:
            var = var_part
        if label_part:
            labels = [lab for lab in label_part.split(':') if lab]
    if var and var in ctx.var_to_id:
        node_id = ctx.var_to_id[var]
    else:
        node_id = var or f"anon_{ctx.node_counter}"
        ctx.node_counter += 1
        if var:
            ctx.var_to_id[var] = node_id
    if node_id in ctx.nodes_by_id:
        node = ctx.nodes_by_id[node_id]
        existing_labels = list(node.get("labels", []))
        for lab in labels:
            if lab not in existing_labels:
                existing_labels.append(lab)
        node["labels"] = existing_labels
        for key, value in props.items():
            node.setdefault(key, value)
    else:
        node = {"id": node_id, "labels": labels, **props}
        ctx.nodes_by_id[node_id] = node
    return node_id


def _parse_rel_segment(
    rel_segment: str, ctx: ParseContext
) -> Tuple[str, str | None, Dict[str, Any]]:
    rel_inner = rel_segment.strip()[1:-1].strip()
    rel_props: Dict[str, Any] = {}
    if '{' in rel_inner:
        before, prop_part = rel_inner.split('{', 1)
        rel_inner = before.strip()
        rel_props = _parse_properties('{' + prop_part)
    rel_var = None
    rel_type = None
    if rel_inner:
        rel_parts = rel_inner.split(':')
        rel_var = rel_parts[0].strip() or None
        if len(rel_parts) > 1:
            rel_type = rel_parts[1].strip() or None
    edge_id = rel_var or f"rel_{ctx.rel_counter}"
    ctx.rel_counter += 1
    return edge_id, rel_type, rel_props

def _edge_from_segment(
    left_id: str,
    right_id: str,
    rel_segment: str,
    left_dir: str | None,
    right_dir: str | None,
    ctx: ParseContext,
) -> Dict[str, Any]:
    edge_id, rel_type, rel_props = _parse_rel_segment(rel_segment, ctx)
    if left_dir == '<-' and right_dir == '-':
        src, dst = right_id, left_id
    elif left_dir == '-' and right_dir == '->':
        src, dst = left_id, right_id
    elif left_dir == '<-' and right_dir == '->':
        src, dst = right_id, left_id
    else:
        src, dst = left_id, right_id
    edge = {
        "edge_id": edge_id,
        "src": src,
        "dst": dst,
        "type": rel_type,
        "undirected": left_dir == '-' and right_dir == '-',
    }
    for key, value in rel_props.items():
        if key in edge:
            edge[f"prop__{key}"] = value
        else:
            edge[key] = value
    return edge


def _parse_chain(pattern: str, ctx: ParseContext) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    node_text, rest = _extract_balanced(pattern.strip(), '(', ')')
    left_id = _parse_node(node_text, ctx)
    text = rest.strip()
    while text:
        left_dir = None
        if text.startswith('<-'):
            left_dir = '<-'
            text = text[2:]
        elif text.startswith('-'):
            left_dir = '-'
            text = text[1:]
        else:
            break

        rel_segment, rest = _extract_balanced(text.strip(), '[', ']')
        text = rest.strip()

        right_dir = None
        if text.startswith('->'):
            right_dir = '->'
            text = text[2:]
        elif text.startswith('-'):
            right_dir = '-'
            text = text[1:]

        node_text, rest = _extract_balanced(text.strip(), '(', ')')
        right_id = _parse_node(node_text, ctx)
        text = rest.strip()

        edges.append(_edge_from_segment(left_id, right_id, rel_segment, left_dir, right_dir, ctx))
        left_id = right_id
    return edges


def _extract_create_clauses(script: str) -> List[str]:
    normalized = " ".join(line.strip() for line in script.strip().splitlines())
    parts = _CREATE_SPLIT_RE.split(normalized)
    clauses: List[str] = []
    for part in parts[1:]:
        clause = part.strip()
        if clause:
            clauses.append(clause)
    return clauses


def graph_fixture_from_create(script: str) -> GraphFixture:
    ctx = ParseContext(nodes_by_id={}, var_to_id={}, node_counter=1, rel_counter=1)
    edges: List[Dict[str, Any]] = []
    for clause in _extract_create_clauses(script):
        for pattern in _split_top_level(clause):
            if '[' in pattern and ']' in pattern:
                edges.extend(_parse_chain(pattern, ctx))
            else:
                _parse_node(pattern, ctx)
    return GraphFixture(
        nodes=list(ctx.nodes_by_id.values()),
        edges=edges,
        edge_columns=("src", "dst", "edge_id", "type", "undirected"),
    )


def merge_fixtures(fixtures: Iterable[GraphFixture]) -> GraphFixture:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    for fixture in fixtures:
        nodes.extend(fixture.nodes)
        edges.extend(fixture.edges)
    return GraphFixture(nodes=nodes, edges=edges)
