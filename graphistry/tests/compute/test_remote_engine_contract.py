"""Contract tests for explicit engines on remote compute calls."""

import typing
from typing import Optional
from unittest.mock import MagicMock, patch
from typing_extensions import Literal

import pandas as pd
import pytest

from graphistry.Engine import EngineAbstractType
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTNode
from graphistry.compute.chain import Chain
from graphistry.compute.chain_remote import chain_remote_generic
from graphistry.compute.exceptions import ErrorCode, GFQLRemoteError
from graphistry.compute.python_remote import python_remote_generic
from graphistry.compute.remote_df_io import RemoteAPIName


TASK = "def task(g):\n    return g\n"
QUERY = Chain([ASTNode(filter_dict={"type": "Person"})])
_PostTarget = Literal[
    "graphistry.compute.chain_remote.requests.post",
    "graphistry.compute.python_remote.requests.post",
]



class Posted(Exception):
    """Stop a test after the request reaches the mocked transport."""


def mock_plottable(dataset_id: Optional[str] = None) -> MagicMock:
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

    def upload(*, validate: bool) -> MagicMock:
        graph._dataset_id = "uploaded-dataset"
        return graph

    graph.upload.side_effect = upload
    return graph


def call_remote(
    api_name: RemoteAPIName,
    graph: Plottable,
    engine: EngineAbstractType,
    *,
    with_creds: bool,
) -> typing.NoReturn:
    """Call one remote entry point with matching mock credentials."""
    api_token = "token" if with_creds else None
    dataset_id = "dataset" if with_creds else None
    if api_name == "gfql_remote":
        chain_remote_generic(
            graph,
            QUERY,
            api_token=api_token,
            dataset_id=dataset_id,
            engine=engine,
            format="json",
            validate=False,
        )
    else:
        python_remote_generic(
            graph,
            TASK,
            api_token=api_token,
            dataset_id=dataset_id,
            engine=engine,
            format="json",
            output_type="json",
            validate=False,
        )
    raise AssertionError("remote call returned before transport")


@pytest.mark.parametrize(
    ("api_name", "post_target"),
    [
        ("gfql_remote", "graphistry.compute.chain_remote.requests.post"),
        ("python_remote", "graphistry.compute.python_remote.requests.post"),
    ],
)
@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_explicit_supported_engine_is_sent_unchanged(
    api_name: RemoteAPIName, post_target: _PostTarget, engine: EngineAbstractType
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
    api_name: RemoteAPIName, post_target: _PostTarget, engine: EngineAbstractType
) -> None:
    graph = mock_plottable()
    with patch(post_target) as post:
        with pytest.raises(GFQLRemoteError) as excinfo:
            call_remote(api_name, graph, engine, with_creds=False)

    assert excinfo.value.code == ErrorCode.E405
    assert excinfo.value.context["field"] == "engine"
    assert excinfo.value.context["value"] == engine
    graph._pygraphistry.refresh.assert_not_called()
    graph.upload.assert_not_called()
    post.assert_not_called()
