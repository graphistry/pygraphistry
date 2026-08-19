"""Shared client-side error surfacing for remote GFQL / remote Python calls.

Every failure a user can hit through the public remote APIs is raised as a typed
GFQL error carrying the HTTP status and the server's own message; no ``requests``
or ``zipfile`` exception reaches the caller.
"""
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import Literal

import requests

from graphistry.Plottable import Plottable
from graphistry.compute.exceptions import ErrorCode, GFQLRemoteError, GFQLSchemaError


ZipMemberKind = Literal['nodes', 'edges']

_BODY_CHARS = 500


def _body_text(response: requests.Response) -> str:
    try:
        return response.text[:_BODY_CHARS]
    except Exception:
        return f"<{len(response.content)} undecodable bytes>"


def parse_json_body(response: requests.Response) -> Optional[Any]:  # hygiene-ok: explicit-any -- an arbitrary server JSON document
    """Decode a response body as JSON, or ``None`` when it is not JSON at all.

    :param response: The HTTP response to decode.
    :return: The decoded document, or ``None`` when the body does not decode.
    """
    if not response.headers.get('content-type', '').startswith('application/json'):
        return None
    try:
        return response.json()
    # requests' JSONDecodeError subclasses ValueError, so a ValueError arm cannot be the fallback.
    except Exception:
        return None


def server_error_message(response: requests.Response) -> Optional[str]:
    """Extract the server's own error text from a JSON body, when it carries one.

    :param response: The HTTP response to inspect.
    :return: The server's message, or ``None`` when the body is not a JSON error document.
    """
    body = parse_json_body(response)
    if isinstance(body, dict) and 'error' in body:
        return str(body['error'])
    return None


def json_body_is_error(body: Any) -> bool:  # hygiene-ok: explicit-any -- an arbitrary server JSON document
    """Whether a decoded 200-response body is an error document rather than a result."""
    return isinstance(body, dict) and 'error' in body


def raise_for_remote_error(response: requests.Response, api_name: str) -> None:
    """Raise a typed error for a non-2xx response, preferring the server's message.

    :param response: The failed HTTP response.
    :param api_name: Public entry point named in the error message.
    :raises GFQLRemoteError: Always, when ``response`` is not ok.
    """
    if response.ok:
        return
    server_msg = server_error_message(response)
    detail = server_msg if server_msg is not None else _body_text(response)
    raise GFQLRemoteError(
        ErrorCode.E401,
        f"{api_name} failed (HTTP {response.status_code}): {detail}",
        status_code=response.status_code,
        server_message=server_msg,
    )


def error_document_error(
    response: requests.Response,
    api_name: str,
    expected: str,
) -> GFQLRemoteError:
    """Build (do not raise) a typed error for a 200 response that is not the expected payload.

    :param response: The HTTP response whose body was not usable.
    :param api_name: Public entry point named in the error message.
    :param expected: What the client expected to decode, e.g. ``"a zip archive"``.
    :return: The error to raise at the call site.
    """
    server_msg = server_error_message(response)
    status = f"HTTP {response.status_code}"
    if server_msg is not None:
        message = f"{api_name} failed ({status}): {server_msg}"
    else:
        message = f"{api_name} failed ({status}): expected {expected}, got: {_body_text(response)}"
    return GFQLRemoteError(
        ErrorCode.E402,
        message,
        status_code=response.status_code,
        server_message=server_msg,
    )


def decode_json_body(response: requests.Response, api_name: str) -> Any:  # hygiene-ok: explicit-any -- an arbitrary server JSON document
    """Decode a success response as JSON, raising typed instead of leaking a decoder error.

    :param response: The HTTP response to decode.
    :param api_name: Public entry point named in the error message.
    :return: The decoded document, whatever its shape.
    :raises GFQLRemoteError: When the body does not decode as JSON.
    """
    try:
        return response.json()
    # requests' JSONDecodeError subclasses ValueError, so a ValueError arm cannot be the fallback.
    except Exception as e:
        raise error_document_error(response, api_name, "a JSON result") from e


def decode_json_result(response: requests.Response, api_name: str) -> Any:  # hygiene-ok: explicit-any -- an arbitrary server JSON document
    """Decode a success response as a graph/table result, refusing error documents.

    :param response: The HTTP response to decode.
    :param api_name: Public entry point named in the error message.
    :return: The decoded result document.
    :raises GFQLRemoteError: When the body does not decode, or is an error document.
    """
    body = decode_json_body(response, api_name)
    if json_body_is_error(body):
        raise error_document_error(response, api_name, "a JSON result")
    return body


def select_zip_member(names: Sequence[str], kind: ZipMemberKind, api_name: str) -> str:
    """Pick the zip member holding the ``kind`` table by name, never by substring.

    :param names: Member names in the archive.
    :param kind: ``"nodes"`` or ``"edges"``.
    :param api_name: Public entry point named in the error message.
    :return: The selected member name.
    :raises GFQLRemoteError: When no member matches, or the match is ambiguous.
    """
    exact = [nm for nm in names if PurePosixPath(nm).stem == kind]
    if len(exact) == 1:
        return exact[0]
    if not exact:
        # A server may name members differently; accept only an unambiguous looser match.
        loose = [nm for nm in names if kind in PurePosixPath(nm).name]
        if len(loose) == 1:
            return loose[0]
        if not loose:
            raise GFQLRemoteError(
                ErrorCode.E402,
                f"{api_name} failed: server zip response has no '{kind}' member",
                member=kind,
                members=list(names),
            )
        exact = loose
    raise GFQLRemoteError(
        ErrorCode.E402,
        f"{api_name} failed: server zip response has {len(exact)} candidate '{kind}' members, cannot pick one",
        member=kind,
        members=list(names),
        candidates=exact,
    )


def require_json_result_keys(
    body: Any,  # hygiene-ok: explicit-any -- an arbitrary server JSON document
    keys: Sequence[str],
    response: requests.Response,
    api_name: str,
) -> Dict[str, Any]:  # hygiene-ok: explicit-any -- an arbitrary server JSON document
    """Require a decoded JSON result to be an object carrying ``keys``.

    :param body: The decoded response body.
    :param keys: Keys the result must provide.
    :param response: The originating response, used for the error message.
    :param api_name: Public entry point named in the error message.
    :return: The validated body.
    :raises GFQLRemoteError: When the body is an error document or misses a key.
    """
    if json_body_is_error(body) or not isinstance(body, dict):
        raise error_document_error(response, api_name, f"a JSON object with {list(keys)}")
    missing = [k for k in keys if k not in body]
    if missing:
        raise GFQLRemoteError(
            ErrorCode.E402,
            f"{api_name} failed: server JSON response is missing {missing}",
            status_code=response.status_code,
            missing=missing,
            keys=sorted(str(k) for k in body.keys()),
        )
    return body


def check_subset_result_bindings(
    g: 'Plottable',
    node_col_subset: Optional[List[str]],
    edge_col_subset: Optional[List[str]],
    api_name: str,
) -> None:
    """Reject a requested column subset that dropped a column the result graph is bound to.

    Runs on the final result, so server-supplied metadata bindings are the ones checked.

    :param g: The Plottable about to be returned to the caller.
    :param node_col_subset: The caller's requested node columns, or ``None``.
    :param edge_col_subset: The caller's requested edge columns, or ``None``.
    :param api_name: Public entry point named in the error message.
    :raises GFQLSchemaError: When a bound column is absent from a returned frame.
    """
    checks: List[Any] = []  # hygiene-ok: explicit-any -- heterogeneous (frame, binding name, kwarg name, table) tuples
    if node_col_subset is not None:
        checks.append((g._nodes, g._node, 'node_col_subset', 'nodes'))
    if edge_col_subset is not None:
        checks.append((g._edges, g._source, 'edge_col_subset', 'edges'))
        checks.append((g._edges, g._destination, 'edge_col_subset', 'edges'))
    for df, col, subset_arg, table in checks:
        if df is None or col is None:
            continue
        columns = list(getattr(df, 'columns', []))
        if not columns:
            continue
        if col not in columns:
            raise GFQLSchemaError(
                ErrorCode.E301,
                f"{api_name} returned {table} without the bound '{col}' column, "
                f"so the result graph would be unusable",
                field=col,
                value=columns,
                suggestion=f"Include '{col}' in {subset_arg}, or rebind the result with g.{table}(df, ...)",
            )
