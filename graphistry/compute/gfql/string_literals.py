from __future__ import annotations

import re


_ESCAPE_RE = re.compile(r"\\(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)")
_PARSE_ESCAPES = {
    "\\": "\\\\",
    "'": "'",
    '"': '"',
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "b": "\b",
    "f": "\f",
}
_RENDER_ESCAPES = str.maketrans(
    {
        "\\": r"\u005C",
        "'": r"\u0027",
        "\n": r"\u000A",
        "\r": r"\u000D",
        "\t": r"\u0009",
        "\b": r"\u0008",
        "\f": r"\u000C",
    }
)


def parse_cypher_string_token(token: str) -> str:
    if len(token) < 2 or token[0] != token[-1] or token[0] not in {"'", '"'}:
        raise ValueError("Invalid string literal")

    body = token[1:-1]
    trailing_slashes = len(body) - len(body.rstrip("\\"))
    if trailing_slashes % 2 == 1:
        raise ValueError("Invalid string literal")

    def replace_escape(match: re.Match[str]) -> str:
        escape = match.group(1)
        if escape[0] in {"u", "U"}:
            return chr(int(escape[1:], 16))
        if escape in _PARSE_ESCAPES:
            return _PARSE_ESCAPES[escape]
        raise ValueError("Invalid string literal")

    return _ESCAPE_RE.sub(replace_escape, body)


def render_cypher_string_literal(value: str) -> str:
    return "'" + value.translate(_RENDER_ESCAPES) + "'"
