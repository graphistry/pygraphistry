"""Text-level helpers for GFQL row-expression parsing.

These helpers are intentionally side-effect free so expression tokenization and
literal handling can evolve independently from row-table execution semantics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


_GFQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def gfql_strip_outer_parens(expr: str) -> str:
    txt = expr.strip()
    while txt.startswith("(") and txt.endswith(")"):
        depth = 0
        in_single = False
        in_double = False
        balanced = True
        for idx, ch in enumerate(txt):
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if in_single or in_double:
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx < len(txt) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        txt = txt[1:-1].strip()
    return txt


def gfql_split_top_level_keyword(expr: str, keyword: str) -> Optional[Tuple[str, str]]:
    txt = expr
    upper = txt.upper()
    needle = keyword.upper()
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_single = False
    in_double = False
    idx = 0
    while idx < len(txt):
        ch = txt[idx]
        if ch == "'" and not in_double:
            in_single = not in_single
            idx += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            idx += 1
            continue
        if in_single or in_double:
            idx += 1
            continue
        if ch == "(":
            depth_paren += 1
            idx += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            idx += 1
            continue
        if ch == "[":
            depth_bracket += 1
            idx += 1
            continue
        if ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            idx += 1
            continue
        if ch == "{":
            depth_brace += 1
            idx += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            idx += 1
            continue
        if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0 and upper.startswith(needle, idx):
            left_ok = idx == 0 or not (upper[idx - 1].isalnum() or upper[idx - 1] == "_")
            right_idx = idx + len(needle)
            right_ok = right_idx >= len(upper) or not (upper[right_idx].isalnum() or upper[right_idx] == "_")
            if left_ok and right_ok:
                left = txt[:idx].strip()
                right = txt[right_idx:].strip()
                if left and right:
                    return left, right
        idx += 1
    return None


def gfql_split_top_level_operator(
    expr: str, operators: Sequence[str]
) -> Optional[Tuple[str, str, str]]:
    txt = expr
    upper = txt.upper()
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    case_depth = 0
    in_single = False
    in_double = False
    idx = 0
    while idx < len(txt):
        ch = txt[idx]
        if ch == "'" and not in_double:
            in_single = not in_single
            idx += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            idx += 1
            continue
        if in_single or in_double:
            idx += 1
            continue
        if ch == "(":
            depth_paren += 1
            idx += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            idx += 1
            continue
        if ch == "[":
            depth_bracket += 1
            idx += 1
            continue
        if ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            idx += 1
            continue
        if ch == "{":
            depth_brace += 1
            idx += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            idx += 1
            continue
        if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            if upper.startswith("CASE", idx):
                left_ok = idx == 0 or not (upper[idx - 1].isalnum() or upper[idx - 1] == "_")
                right_idx = idx + 4
                right_ok = right_idx >= len(upper) or not (upper[right_idx].isalnum() or upper[right_idx] == "_")
                if left_ok and right_ok:
                    case_depth += 1
                    idx = right_idx
                    continue
            if case_depth > 0 and upper.startswith("END", idx):
                left_ok = idx == 0 or not (upper[idx - 1].isalnum() or upper[idx - 1] == "_")
                right_idx = idx + 3
                right_ok = right_idx >= len(upper) or not (upper[right_idx].isalnum() or upper[right_idx] == "_")
                if left_ok and right_ok:
                    case_depth = max(0, case_depth - 1)
                    idx = right_idx
                    continue
        if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0 and case_depth == 0:
            for op in operators:
                if not txt.startswith(op, idx):
                    continue
                if op in {"+", "-"}:
                    prev_idx = idx - 1
                    while prev_idx >= 0 and txt[prev_idx].isspace():
                        prev_idx -= 1
                    if prev_idx < 0 or txt[prev_idx] in {"(", "[", "{", ",", "+", "-", "*", "/", "%", "=", "<", ">"}:
                        continue
                left = txt[:idx].strip()
                right = txt[idx + len(op) :].strip()
                if left and right:
                    return left, op, right
        idx += 1
    return None


def gfql_split_top_level_commas(expr: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_single = False
    in_double = False
    escaped = False
    for ch in expr:
        if in_single or in_double:
            current.append(ch)
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if in_single and ch == "'":
                in_single = False
            elif in_double and ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            current.append(ch)
            continue
        if ch == '"':
            in_double = True
            current.append(ch)
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        if ch == "," and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def gfql_parse_quantifier_expr(expr: str) -> Optional[Tuple[str, str, str, str]]:
    txt = expr.strip()
    head = re.match(r"(?is)^(any|all|none|single)\s*\(", txt)
    if head is None:
        return None
    fn = head.group(1).lower()
    open_idx = txt.find("(", head.start())
    if open_idx < 0:
        return None
    depth = 0
    in_single = False
    in_double = False
    escaped = False
    close_idx = -1
    for idx in range(open_idx, len(txt)):
        ch = txt[idx]
        if in_single or in_double:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if in_single and ch == "'":
                in_single = False
            elif in_double and ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                close_idx = idx
                break
            if depth < 0:
                return None
    if close_idx < 0:
        return None
    if txt[close_idx + 1 :].strip() != "":
        return None
    body = txt[open_idx + 1 : close_idx].strip()
    in_split = gfql_split_top_level_keyword(body, "IN")
    if in_split is None:
        return None
    var = in_split[0].strip()
    if _GFQL_IDENT_RE.fullmatch(var) is None:
        return None
    where_split = gfql_split_top_level_keyword(in_split[1], "WHERE")
    if where_split is None:
        return None
    list_expr = where_split[0].strip()
    predicate_expr = where_split[1].strip()
    if list_expr == "" or predicate_expr == "":
        return None
    return fn, var, list_expr, predicate_expr


def gfql_parse_function_call(expr: str, fn_name: str) -> Optional[str]:
    txt = expr.strip()
    head = re.match(rf"(?is)^{re.escape(fn_name)}\s*\(", txt)
    if head is None:
        return None
    open_idx = txt.find("(", head.start())
    if open_idx < 0:
        return None
    depth = 0
    in_single = False
    in_double = False
    escaped = False
    close_idx = -1
    for idx in range(open_idx, len(txt)):
        ch = txt[idx]
        if in_single or in_double:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if in_single and ch == "'":
                in_single = False
            elif in_double and ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                close_idx = idx
                break
            if depth < 0:
                return None
    if close_idx < 0:
        return None
    if txt[close_idx + 1 :].strip() != "":
        return None
    return txt[open_idx + 1 : close_idx].strip()


def gfql_parse_list_comprehension_expr(
    expr: str,
) -> Optional[Tuple[str, str, Optional[str], str]]:
    txt = expr.strip()
    if not (txt.startswith("[") and txt.endswith("]")):
        return None
    body = txt[1:-1].strip()
    if body == "":
        return None
    in_split = gfql_split_top_level_keyword(body, "IN")
    if in_split is None:
        return None
    var = in_split[0].strip()
    if _GFQL_IDENT_RE.fullmatch(var) is None:
        return None
    rhs = in_split[1].strip()
    pipe_split = gfql_split_top_level_operator(rhs, ["|"])
    if pipe_split is not None:
        lhs = pipe_split[0].strip()
        proj_expr = pipe_split[2].strip()
    else:
        lhs = rhs
        proj_expr = var
    if lhs == "" or proj_expr == "":
        return None
    where_split = gfql_split_top_level_keyword(lhs, "WHERE")
    if where_split is not None:
        list_expr = where_split[0].strip()
        predicate_expr: Optional[str] = where_split[1].strip()
    else:
        list_expr = lhs.strip()
        predicate_expr = None
    if list_expr == "":
        return None
    if predicate_expr is not None and predicate_expr == "":
        return None
    return var, list_expr, predicate_expr, proj_expr


def gfql_find_matching_case_end(expr: str) -> int:
    txt = expr.strip()
    n = len(txt)
    if n < 8:
        return -1

    def _is_word_boundary(pos: int) -> bool:
        return pos < 0 or pos >= n or not (txt[pos].isalnum() or txt[pos] == "_")

    in_single = False
    in_double = False
    escaped = False
    depth = 0
    i = 0
    while i < n:
        ch = txt[i]
        if in_single or in_double:
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == "\\":
                escaped = True
                i += 1
                continue
            if in_single and ch == "'":
                in_single = False
            elif in_double and ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue
        if i + 4 <= n and txt[i : i + 4].upper() == "CASE":
            if _is_word_boundary(i - 1) and _is_word_boundary(i + 4):
                depth += 1
                i += 4
                continue
        if i + 3 <= n and txt[i : i + 3].upper() == "END":
            if _is_word_boundary(i - 1) and _is_word_boundary(i + 3):
                if depth <= 0:
                    return -1
                depth -= 1
                if depth == 0:
                    return i + 2
                i += 3
                continue
        i += 1
    return -1


def gfql_parse_case_when_expr(expr: str) -> Optional[Tuple[str, str, str]]:
    txt = expr.strip()
    if not txt.upper().startswith("CASE "):
        return None
    end_idx = gfql_find_matching_case_end(txt)
    if end_idx < 0 or end_idx != (len(txt) - 1):
        return None
    body = txt[4 : end_idx - 2].strip()
    if body.upper().startswith("WHEN "):
        body = body[5:].strip()
    then_split = gfql_split_top_level_keyword(body, "THEN")
    if then_split is None:
        return None
    else_split = gfql_split_top_level_keyword(then_split[1], "ELSE")
    if else_split is None:
        return None
    cond_expr = then_split[0].strip()
    true_expr = else_split[0].strip()
    false_expr = else_split[1].strip()
    if cond_expr == "" or true_expr == "" or false_expr == "":
        return None
    return cond_expr, true_expr, false_expr


def gfql_replace_identifier(expr: str, identifier: str, replacement: str) -> str:
    return re.sub(
        rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
        replacement,
        expr,
    )


def gfql_find_top_level_char(text: str, target: str) -> int:
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(text):
        if in_single or in_double:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if in_single and ch == "'":
                in_single = False
            elif in_double and ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "(":
            depth_paren += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            continue
        if ch == "{":
            depth_brace += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            continue
        if ch == "[":
            depth_bracket += 1
            continue
        if ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            continue
        if depth_paren != 0 or depth_brace != 0 or depth_bracket != 0:
            continue
        if ch == target:
            return idx
    return -1


def gfql_parse_subscript_expr(expr: str) -> Optional[Tuple[str, str]]:
    txt = expr.strip()
    if len(txt) < 3 or not txt.endswith("]"):
        return None
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0
    in_single = False
    in_double = False
    escaped = False
    open_idx = -1
    for idx, ch in enumerate(txt):
        if in_single or in_double:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if in_single and ch == "'":
                in_single = False
            elif in_double and ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "(":
            depth_paren += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            continue
        if ch == "{":
            depth_brace += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            continue
        if ch == "[":
            if depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
                open_idx = idx
            depth_bracket += 1
            continue
        if ch == "]":
            depth_bracket -= 1
            if depth_bracket < 0:
                return None
            continue
    if open_idx <= 0:
        return None
    base = txt[:open_idx].strip()
    key = txt[open_idx + 1 : -1].strip()
    if base == "" or key == "":
        return None
    return base, key


def gfql_parse_quoted_string_literal(token: str) -> str:
    if len(token) < 2 or token[0] != token[-1] or token[0] not in {"'", '"'}:
        raise ValueError("invalid quoted string literal")
    quote = token[0]
    out: List[str] = []
    i = 1
    end = len(token) - 1
    while i < end:
        ch = token[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= end:
            raise ValueError("invalid quoted string literal escape")
        nxt = token[i + 1]
        if nxt == quote or nxt == "\\":
            out.append(nxt)
        elif nxt == "n":
            out.append("\n")
        elif nxt == "r":
            out.append("\r")
        elif nxt == "t":
            out.append("\t")
        elif nxt == "b":
            out.append("\b")
        elif nxt == "f":
            out.append("\f")
        else:
            out.append(nxt)
        i += 2
    return "".join(out)


def gfql_parse_cypher_literal(text: str) -> Any:
    token = text.strip()
    if token == "":
        raise ValueError("empty literal")
    lower = token.lower()
    if lower in {"null", "none"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        try:
            return gfql_parse_quoted_string_literal(token)
        except Exception:
            return token[1:-1]
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?", token):
        return float(token)
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if inner == "":
            return []
        return [gfql_parse_cypher_literal(part) for part in gfql_split_top_level_commas(inner)]
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        if inner == "":
            return {}
        out: Dict[str, Any] = {}
        for item in gfql_split_top_level_commas(inner):
            colon_idx = gfql_find_top_level_char(item, ":")
            if colon_idx <= 0:
                raise ValueError(f"invalid map entry: {item}")
            key_token = item[:colon_idx].strip()
            value_token = item[colon_idx + 1 :].strip()
            if key_token == "":
                raise ValueError(f"empty map key in: {item}")
            if len(key_token) >= 2 and key_token[0] == key_token[-1] and key_token[0] in {"'", '"'}:
                parsed_key = gfql_parse_cypher_literal(key_token)
                if not isinstance(parsed_key, str):
                    raise ValueError(f"map key must be string: {item}")
                key = parsed_key
            elif _GFQL_IDENT_RE.fullmatch(key_token) is not None:
                key = key_token
            else:
                raise ValueError(f"invalid map key: {key_token}")
            out[key] = gfql_parse_cypher_literal(value_token)
        return out
    raise ValueError(f"unsupported literal token: {token}")


def gfql_parse_cypher_structured_literal(text: str) -> Tuple[bool, Any]:
    try:
        parsed = gfql_parse_cypher_literal(text)
    except Exception:
        return False, None
    return True, parsed


def gfql_parse_structured_literal(text: str) -> Tuple[bool, Any]:
    try:
        parsed = gfql_parse_cypher_literal(text)
    except Exception:
        return False, None
    return True, parsed


def gfql_normalize_json_like_literal(value: Any) -> Tuple[bool, Any]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True, value
    if isinstance(value, list):
        out: List[Any] = []
        for item in value:
            ok, normalized = gfql_normalize_json_like_literal(item)
            if not ok:
                return False, None
            out.append(normalized)
        return True, out
    if isinstance(value, dict):
        out_dict: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return False, None
            ok, normalized = gfql_normalize_json_like_literal(item)
            if not ok:
                return False, None
            out_dict[key] = normalized
        return True, out_dict
    return False, None


def gfql_parse_literal_token(token: str) -> Tuple[bool, Any]:
    txt = token.strip()
    if txt == "":
        return False, None
    low = txt.lower()
    if low == "null":
        return True, None
    if low == "true":
        return True, True
    if low == "false":
        return True, False
    if len(txt) >= 2 and txt[0] == txt[-1] and txt[0] in {"'", '"'}:
        return True, txt[1:-1]
    if re.fullmatch(r"-?\d+", txt):
        return True, int(txt)
    if re.fullmatch(r"-?\d+\.\d+", txt):
        return True, float(txt)
    if txt.startswith("[") and txt.endswith("]"):
        ok, parsed = gfql_parse_structured_literal(txt)
        if not ok:
            ok, parsed = gfql_parse_cypher_structured_literal(txt)
        if not ok:
            return False, None
        ok_norm, normalized = gfql_normalize_json_like_literal(parsed)
        if ok_norm and isinstance(normalized, list):
            return True, normalized
        return False, None
    if txt.startswith("{") and txt.endswith("}"):
        ok, parsed = gfql_parse_structured_literal(txt)
        if not ok:
            ok, parsed = gfql_parse_cypher_structured_literal(txt)
        if not ok:
            return False, None
        ok_norm, normalized = gfql_normalize_json_like_literal(parsed)
        if ok_norm and isinstance(normalized, dict):
            return True, normalized
        return False, None
    return False, None
