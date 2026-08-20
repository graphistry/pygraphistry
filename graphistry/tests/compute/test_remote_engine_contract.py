"""Contract tests for explicit engines on remote compute calls."""

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from graphistry.compute.ast import ASTNode
from graphistry.compute.chain import Chain
from graphistry.compute.chain_remote import chain_remote_generic
from graphistry.compute.exceptions import ErrorCode, GFQLRemoteError
from graphistry.compute.python_remote import python_remote_generic


TASK = "def task(g):\n    return g\n"
QUERY = Chain([ASTNode(filter_dict={"type": "Person"})])


class Posted(Exception):
    """Stop a test after the request reaches the mocked transport."""


def mock_plottable(dataset_id: str | None = None) -> MagicMock:
    """Build the minimum graph state used by both remote entry points."""
    graph = MagicMock()
    graph._dataset_id = dataset_id
    graph._edges = pd.DataFrame({"s": [0], "d": [1]})
    graph._nodes = pd.DataFrame({"id": [0, 1]})
    graph._privacy = None
    graph._url_params = {}
    graph.session.api_token = "refreshed-token"
    graph.session.certificate_validation = True
    graph.base_url_server.return_value = "https://test.graphistry.com"

    def upload(*args: Any, **kwargs: Any) -> MagicMock:
        graph._dataset_id = "uploaded-dataset"
        return graph

    graph.upload.side_effect = upload
    return graph


def call_remote(api_name: str, graph: MagicMock, engine: str, *, with_creds: bool) -> Any:
    """Call one remote entry point with matching mock credentials."""
    credentials = {"api_token": "token", "dataset_id": "dataset"} if with_creds else {}
    if api_name == "gfql_remote":
        return chain_remote_generic(
            graph,
            QUERY,
            engine=engine,  # type: ignore[arg-type]
            format="json",
            validate=False,
            **credentials,
        )
    return python_remote_generic(
        graph,
        TASK,
        engine=engine,  # type: ignore[arg-type]
        format="json",
        validate=False,
        **credentials,
    )


@pytest.mark.parametrize(
    ("api_name", "post_target"),
    [
        ("gfql_remote", "graphistry.compute.chain_remote.requests.post"),
        ("python_remote", "graphistry.compute.python_remote.requests.post"),
    ],
)
@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_explicit_supported_engine_is_sent_unchanged(
    api_name: str, post_target: str, engine: str
) -> None:
    graph = mock_plottable("dataset")
    with patch(post_target, side_effect=Posted) as post:
        with pytest.raises(Posted):
            call_remote(api_name, graph, engine, with_creds=True)
    assert post.call_args.kwargs["json"]["engine"] == engine


@pytest.mark.parametrize(
    ("api_name", "post_target"),
    [
        ("gfql_remote", "graphistry.compute.chain_remote.requests.post"),
        ("python_remote", "graphistry.compute.python_remote.requests.post"),
    ],
)
@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_explicit_unsupported_engine_declines_before_side_effects(
    api_name: str, post_target: str, engine: str
) -> None:
    graph = mock_plottable()
    with patch(post_target) as post:
        with pytest.raises(GFQLRemoteError) as excinfo:
            call_remote(api_name, graph, engine, with_creds=False)

    assert excinfo.value.code == ErrorCode.E405
    assert excinfo.value.context["field"] == "engine"
    assert excinfo.value.context["value"] == engine
    assert "pandas" in str(excinfo.value)
    assert "cudf" in str(excinfo.value)
    assert "Dask engines" not in str(excinfo.value)
    graph._pygraphistry.refresh.assert_not_called()
    graph.upload.assert_not_called()
    post.assert_not_called()
