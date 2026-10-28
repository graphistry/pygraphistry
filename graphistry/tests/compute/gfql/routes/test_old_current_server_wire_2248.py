"""Cross-version GFQL route contract (#2248).

The fixtures are source-mapped hermetic HTTP routes:

- old server ``054e79c87d74a87dbd68c40d4d43a9a6e23f61c3``:
  ``server/routes/gfql.py`` requires ``gfql_operations``.
- current server ``72d2664c2e61859cb813cb94c6479e3bc0824291``:
  ``server/routes/gfql.py`` accepts ``gfql_query`` or ``gfql_operations``.
- both handlers delegate execution through
  ``server/client/graph/processing_handler.py::process_gfql_query`` to ``g.gfql``.

The public client sends real HTTP requests to local ThreadingHTTPServer routes.
No credentials, datasets, deployment, or Hub endpoint is involved.
"""
from __future__ import annotations

import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import ASTEdge, ASTLet, ASTNode
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import (
    GFQLRemoteError,
    GFQLTypeError,
    GFQLValidationError,
)


OLD_SERVER = "054e79c87d74a87dbd68c40d4d43a9a6e23f61c3"
CURRENT_SERVER = "72d2664c2e61859cb813cb94c6479e3bc0824291"
DATASET_ID = "0d3d9a5a-880f-45f1-8486-b7983f6f9228"


NODES = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "kind": ["A", "B", "B"],
    }
)
EDGES = pd.DataFrame(
    {
        "src": [1, 2],
        "dst": [2, 3],
    }
)


def public_let() -> ASTLet:
    path = Chain(
        [
            ASTNode(filter_dict={"kind": "A"}, name="a"),
            ASTEdge(name="r"),
            ASTNode(filter_dict={"kind": "B"}, name="b"),
        ]
    )
    return ASTLet({"path": path, "result": path})


def cypher_let() -> str:
    return (
        "GRAPH path = GRAPH { MATCH (a {kind:'A'})-[r]->(b) } "
        "USE path MATCH (x) RETURN x"
    )


def flat_chain() -> List[ASTNode]:
    return [ASTNode(filter_dict={"kind": "A"})]


def _records(frame: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if frame is None:
        return []
    return frame.to_dict(orient="records")


def _json_response(payload: Mapping[str, Any]) -> Tuple[int, bytes, str]:
    return 200, json.dumps(payload).encode("utf-8"), "application/json"


def _parquet_all_response(result: graphistry.Plottable) -> Tuple[int, bytes, str]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, frame in (("nodes", result._nodes), ("edges", result._edges)):
            frame_buffer = io.BytesIO()
            (frame if frame is not None else pd.DataFrame()).to_parquet(frame_buffer, index=False)
            archive.writestr(f"{name}.parquet", frame_buffer.getvalue())
    return 200, buffer.getvalue(), "application/zip"


def _error_response(status: int, message: str) -> Tuple[int, bytes, str]:
    return status, json.dumps({"success": False, "error_code": status, "message": message}).encode("utf-8"), "application/json"


def _execute(
    body: Mapping[str, Any],
    graph: graphistry.Plottable,
    *,
    mode: str,
) -> graphistry.Plottable:
    if mode == "current":
        operations = body.get("gfql_query") or body.get("gfql_operations")
    else:
        operations = body["gfql_operations"]
    output = body.get("gfql_output")
    return graph.gfql(operations, output=output, engine="pandas")


class RouteServer:
    def __init__(self, mode: str):
        self.mode = mode
        self.requests: List[Dict[str, Any]] = []
        self.port = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _handler(self, *args: Any, **kwargs: Any) -> BaseHTTPRequestHandler:
        return _RequestHandler(self, *args, **kwargs)


class _RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, route_server: RouteServer, *args: Any, **kwargs: Any):
        self.route_server = route_server
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        assert self.command == "POST"
        expected_path = f"/api/v2/etl/datasets/{DATASET_ID}/gfql/all"
        assert self.path == expected_path

        raw_length = self.headers.get("Content-Length", "0")
        raw = self.rfile.read(int(raw_length))
        body = json.loads(raw.decode("utf-8")) if raw else {}
        self.route_server.requests.append({"path": self.path, "body": body})

        if self.route_server.mode == "old":
            route = _old_route(body)
        else:
            route = _current_route(body)
        status, payload, content_type = route
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _old_route(body: Mapping[str, Any]) -> Tuple[int, bytes, str]:
    # Source map: old gfql.py lines 303-323 only consult gfql_operations.
    if "gfql_operations" not in body:
        return _error_response(422, 'Missing required "gfql_operations" field in payload.')
    result = _execute(body, _local_graph(), mode="old")
    return _json_response({"nodes": _records(result._nodes), "edges": _records(result._edges)})


def _current_route(body: Mapping[str, Any]) -> Tuple[int, bytes, str]:
    # Source map: current gfql.py lines 303-322 consult gfql_query first.
    if body.get("gfql_operations") is None and body.get("gfql_query") is None:
        return _error_response(
            422,
            'Missing required "gfql_operations" or "gfql_query" field in payload.',
        )
    result = _execute(body, _local_graph(), mode="current")
    if body.get("format") == "parquet":
        return _parquet_all_response(result)
    return _json_response({"nodes": _records(result._nodes), "edges": _records(result._edges)})


def _local_graph() -> graphistry.Plottable:
    return graphistry.edges(EDGES, "src", "dst").nodes(NODES, "id")


@pytest.fixture(params=["old", "current"])
def route_server(request: pytest.FixtureRequest) -> Any:
    server = RouteServer(request.param)
    server.start()
    try:
        yield server
    finally:
        server.close()


def _bound_graph(server: RouteServer) -> graphistry.Plottable:
    graph = graphistry.edges(EDGES, "src", "dst").nodes(NODES, "id")
    graph._dataset_id = DATASET_ID
    graph.session.protocol = "http"
    graph.session.hostname = f"127.0.0.1:{server.port}"
    return graph


def _remote(
    graph: graphistry.Plottable,
    query: Any,
    *,
    output: Optional[str] = None,
    data_format: str = "json",
) -> graphistry.Plottable:
    return graph.gfql_remote(
        query,
        api_token="wire-test-token",
        output=output,
        output_type="all",
        format=data_format,
        engine="pandas",
    )


@pytest.mark.parametrize("data_format", ["json", "parquet"])
def test_current_server_accepts_all_query_shapes(route_server: RouteServer, data_format: str) -> None:
    if route_server.mode != "current":
        pytest.skip("current-server half of paired fixture")

    graph = _bound_graph(route_server)

    let_result = _remote(graph, public_let(), output="path", data_format=data_format)
    assert sorted(let_result._nodes["id"].tolist()) == [1, 2]
    assert list(zip(let_result._edges["src"], let_result._edges["dst"])) == [(1, 2)]

    serialized_result = _remote(
        graph,
        public_let().to_json(),
        output="path",
        data_format=data_format,
    )
    assert sorted(serialized_result._nodes["id"].tolist()) == [1, 2]
    assert list(zip(serialized_result._edges["src"], serialized_result._edges["dst"])) == [(1, 2)]

    cypher_result = _remote(graph, cypher_let(), data_format=data_format)
    assert sorted(cypher_result._nodes["id"].tolist()) == [1, 2]
    assert len(cypher_result._edges) == 0

    flat_result = _remote(graph, flat_chain(), data_format=data_format)
    assert flat_result._nodes["id"].tolist() == [1]
    assert len(flat_result._edges) == 0

    assert len(route_server.requests) == 4
    request_bodies = [request["body"] for request in route_server.requests]
    assert request_bodies[0]["gfql_query"]["type"] == "Let"
    assert "gfql_operations" not in request_bodies[0]
    assert request_bodies[1]["gfql_query"]["type"] == "Let"
    assert request_bodies[2]["gfql_query"]["type"] == "Let"
    assert request_bodies[3]["gfql_query"]["type"] == "Chain"
    assert request_bodies[3]["gfql_operations"]


@pytest.mark.parametrize("query,output", [
    pytest.param(public_let, "path", id="public-client-let"),
    pytest.param(lambda: public_let().to_json(), "path", id="serialized-let"),
    pytest.param(cypher_let, None, id="compiled-cypher-let"),
])
def test_old_server_rejects_let(route_server: RouteServer, query: Any, output: Optional[str]) -> None:
    if route_server.mode != "old":
        pytest.skip("old-server half of paired fixture")

    with pytest.raises(GFQLRemoteError) as error:
        _remote(_bound_graph(route_server), query(), output=output)

    assert "gfql_operations" in str(error.value)
    assert error.value.context["status_code"] == 422
    assert "HTTP 422" in str(error.value)
    assert not isinstance(error.value, ConnectionError)
    assert len(route_server.requests) == 1
    assert "gfql_operations" not in route_server.requests[0]["body"]


def test_old_server_accepts_nonempty_flat_chain(route_server: RouteServer) -> None:
    if route_server.mode != "old":
        pytest.skip("old-server half of paired fixture")

    result = _remote(_bound_graph(route_server), flat_chain())
    assert result._nodes["id"].tolist() == [1]
    assert len(result._edges) == 0
    assert route_server.requests[0]["body"]["gfql_operations"]


def test_public_client_rejects_invalid_query_type_without_request(route_server: RouteServer) -> None:
    graph = _bound_graph(route_server)

    with pytest.raises(GFQLValidationError) as missing:
        _remote(graph, {})

    assert missing.value.code == "missing-required-field"
    assert missing.value.context["field"] == "type"

    with pytest.raises(GFQLTypeError) as invalid:
        _remote(graph, {"type": "bogus"})

    assert invalid.value.code == "invalid-query-type"
    assert invalid.value.context["field"] == "type"
    assert invalid.value.context["value"] == "bogus"
    assert not route_server.requests
