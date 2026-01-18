import json
from unittest import mock

import pandas as pd

import graphistry
from graphistry.compute.ast import n, e_forward


TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


def _mock_response(json_data=None, status=200):
    resp = mock.Mock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.json = mock.Mock(return_value=json_data or {})
    resp.headers = {"content-type": "application/json"}
    resp.text = json.dumps(json_data or {})
    resp.raise_for_status = mock.Mock()
    return resp


def _make_graph():
    edges = pd.DataFrame({"src": [1, 2], "dst": [2, 3]})
    nodes = pd.DataFrame({"id": [1, 2, 3]})
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    g.session.api_token = "tok"
    g.session.certificate_validation = True
    g.session.privacy = None
    g._privacy = None
    g._pygraphistry.refresh = mock.Mock()
    return g


def _inject_trace(headers):
    return {**headers, "traceparent": TRACEPARENT}


def _post_response_for_plot(url: str):
    if "/api/v2/upload/datasets/" in url and "/edges/arrow" in url:
        return _mock_response({"success": True})
    if "/api/v2/upload/datasets/" in url and "/nodes/arrow" in url:
        return _mock_response({"success": True})
    if url.rstrip("/").endswith("/api/v2/upload/datasets"):
        return _mock_response({"success": True, "data": {"dataset_id": "ds1"}})
    if url.rstrip("/").endswith("/api/v2/files"):
        return _mock_response({"file_id": "file1"})
    if "/api/v2/upload/files/" in url:
        return _mock_response({"is_valid": True, "is_uploaded": True})
    if "/api/v2/share/link/" in url:
        return _mock_response({"success": True})
    raise AssertionError(f"Unexpected POST url: {url}")


@mock.patch("graphistry.arrow_uploader.inject_trace_headers")
@mock.patch("requests.post")
def test_plot_injects_traceparent(mock_post, mock_inject):
    mock_inject.side_effect = _inject_trace
    headers_seen = []

    def _fake_post(url, **kwargs):
        headers_seen.append(kwargs.get("headers", {}))
        return _post_response_for_plot(url)

    mock_post.side_effect = _fake_post

    g = _make_graph()
    g.plot(render="g", as_files=False, validate=False, warn=False, memoize=False)

    assert headers_seen
    assert all(h.get("traceparent") == TRACEPARENT for h in headers_seen)


@mock.patch("graphistry.ArrowFileUploader.inject_trace_headers")
@mock.patch("graphistry.arrow_uploader.inject_trace_headers")
@mock.patch("requests.post")
def test_upload_injects_traceparent(mock_post, mock_inject, mock_inject_files):
    mock_inject.side_effect = _inject_trace
    mock_inject_files.side_effect = _inject_trace
    headers_seen = []

    def _fake_post(url, **kwargs):
        headers_seen.append(kwargs.get("headers", {}))
        return _post_response_for_plot(url)

    mock_post.side_effect = _fake_post

    g = _make_graph()
    g.upload(validate=False, warn=False, memoize=False, erase_files_on_fail=False)

    assert headers_seen
    assert all(h.get("traceparent") == TRACEPARENT for h in headers_seen)


@mock.patch("graphistry.compute.chain_remote.inject_trace_headers")
@mock.patch("graphistry.compute.chain_remote.requests.post")
def test_gfql_remote_injects_traceparent(mock_post, mock_inject):
    mock_inject.side_effect = _inject_trace

    response = _mock_response({"nodes": [], "edges": []}, status=200)
    mock_post.return_value = response

    g = _make_graph()
    g._dataset_id = "dataset_remote"
    g.gfql_remote(
        [n(), e_forward(), n()],
        api_token="tok",
        dataset_id="dataset_remote",
        output_type="all",
        format="json",
    )

    headers = mock_post.call_args[1]["headers"]
    assert headers["traceparent"] == TRACEPARENT
