"""Remote result decoding must preserve server-side values or decline loudly."""
from io import BytesIO
from unittest.mock import MagicMock, patch
import os
import zipfile

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n


skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)


# Server-side truth: leading-zero ids, pandas NA-vocabulary strings, int64 beyond float53
NODES = pd.DataFrame({
    'id': ['007', '08', 'NA'],
    'name': ['', 'null', 'x'],
    'big': [4611686018427387904, 4611686018427387905, 3],
})
EDGES = pd.DataFrame({'s': ['007'], 'd': ['08']})

FAITHFUL_ARGS = {
    'dtype': {'id': str, 'name': str, 's': str, 'd': str},
    'keep_default_na': False,
    'na_values': [],
}


def build_zip(fmt: str) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        if fmt == 'csv':
            z.writestr('nodes.csv', NODES.to_csv(index=False))
            z.writestr('edges.csv', EDGES.to_csv(index=False))
        else:
            nb = BytesIO()
            NODES.to_parquet(nb, index=False)
            eb = BytesIO()
            EDGES.to_parquet(eb, index=False)
            z.writestr('nodes.parquet', nb.getvalue())
            z.writestr('edges.parquet', eb.getvalue())
    return buf.getvalue()


def build_table(fmt: str) -> bytes:
    if fmt == 'csv':
        return NODES.to_csv(index=False).encode('utf-8')
    buf = BytesIO()
    NODES.to_parquet(buf, index=False)
    return buf.getvalue()


def mock_response(content: bytes) -> MagicMock:
    resp = MagicMock()
    resp.ok = True
    resp.content = content
    resp.headers = {}
    resp.raise_for_status.return_value = None
    return resp


def bound_graph():
    g = graphistry.edges(EDGES, 's', 'd').nodes(NODES, 'id')
    g._dataset_id = 'ds_test'
    return g


def norm(df: pd.DataFrame):
    return [
        {k: (None if isinstance(v, float) and v != v else v) for k, v in rec.items()}
        for rec in df.to_dict('records')
    ]


class TestGfqlRemoteCsvFidelity:

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_declines_when_no_import_args(self, mock_post):
        mock_post.return_value = mock_response(build_zip('csv'))
        with pytest.raises(ValueError) as excinfo:
            bound_graph().gfql_remote([n()], format='csv', api_token='t')
        msg = str(excinfo.value)
        assert 'df_import_args' in msg
        assert 'parquet' in msg
        assert not mock_post.called

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_declines_for_nodes_output_type(self, mock_post):
        mock_post.return_value = mock_response(build_table('csv'))
        with pytest.raises(ValueError):
            bound_graph().gfql_remote([n()], output_type='nodes', format='csv', api_token='t')
        assert not mock_post.called

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_shape_csv_declines_when_no_import_args(self, mock_post):
        mock_post.return_value = mock_response(build_table('csv'))
        with pytest.raises(ValueError):
            bound_graph().gfql_remote_shape([n()], format='csv', api_token='t')
        assert not mock_post.called

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_rejects_non_dict_import_args(self, mock_post):
        mock_post.return_value = mock_response(build_zip('csv'))
        with pytest.raises(ValueError):
            bound_graph().gfql_remote(
                [n()], format='csv', api_token='t',
                df_import_args='dtype=str',  # type: ignore[arg-type]
            )
        assert not mock_post.called

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_opt_in_preserves_string_ids(self, mock_post):
        mock_post.return_value = mock_response(build_zip('csv'))
        out = bound_graph().gfql_remote(
            [n()], format='csv', api_token='t', df_import_args=FAITHFUL_ARGS
        )
        assert not pd.api.types.is_numeric_dtype(out._nodes['id'])
        assert list(out._nodes['id']) == ['007', '08', 'NA']
        assert list(out._nodes['name']) == ['', 'null', 'x']
        assert list(out._nodes['big']) == [4611686018427387904, 4611686018427387905, 3]
        assert out._nodes['name'].isna().sum() == 0

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_opt_in_keeps_node_edge_join_coherent(self, mock_post):
        mock_post.return_value = mock_response(build_zip('csv'))
        out = bound_graph().gfql_remote(
            [n()], format='csv', api_token='t', df_import_args=FAITHFUL_ARGS
        )
        assert str(out._nodes['id'].dtype) == str(out._edges['s'].dtype)
        assert set(out._edges['s']).issubset(set(out._nodes['id']))
        assert set(out._edges['d']).issubset(set(out._nodes['id']))
        joined = out._edges.merge(out._nodes, left_on='s', right_on='id', how='inner')
        assert len(joined) == len(out._edges)

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_opt_in_matches_parquet_values(self, mock_post):
        mock_post.return_value = mock_response(build_zip('parquet'))
        ref = bound_graph().gfql_remote([n()], format='parquet', api_token='t')

        mock_post.return_value = mock_response(build_zip('csv'))
        out = bound_graph().gfql_remote(
            [n()], format='csv', api_token='t', df_import_args=FAITHFUL_ARGS
        )
        assert norm(out._nodes) == norm(ref._nodes)
        assert norm(out._edges) == norm(ref._edges)

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_parquet_needs_no_import_args(self, mock_post):
        mock_post.return_value = mock_response(build_zip('parquet'))
        out = bound_graph().gfql_remote([n()], format='parquet', api_token='t')
        assert list(out._nodes['id']) == ['007', '08', 'NA']

    @skip_gpu
    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_declines_on_cudf_graph(self, mock_post):
        import cudf
        mock_post.return_value = mock_response(build_zip('csv'))
        g = graphistry.edges(cudf.from_pandas(EDGES), 's', 'd').nodes(cudf.from_pandas(NODES), 'id')
        g._dataset_id = 'ds_test'
        with pytest.raises(ValueError):
            g.gfql_remote([n()], format='csv', api_token='t')
        assert not mock_post.called

    @skip_gpu
    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_opt_in_preserves_string_ids_on_cudf_graph(self, mock_post):
        import cudf
        mock_post.return_value = mock_response(build_zip('csv'))
        g = graphistry.edges(cudf.from_pandas(EDGES), 's', 'd').nodes(cudf.from_pandas(NODES), 'id')
        g._dataset_id = 'ds_test'
        out = g.gfql_remote(
            [n()], format='csv', api_token='t',
            df_import_args={'dtype': {'id': 'str', 'name': 'str', 's': 'str', 'd': 'str'},
                            'keep_default_na': False, 'na_values': []},
        )
        assert list(out._nodes['id'].to_pandas()) == ['007', '08', 'NA']
        assert str(out._nodes['id'].dtype) == str(out._edges['s'].dtype)


class TestPythonRemoteCsvFidelity:

    @patch('graphistry.compute.python_remote.requests.post')
    def test_python_remote_table_csv_declines_when_no_import_args(self, mock_post):
        mock_post.return_value = mock_response(build_table('csv'))
        code = "def task(g):\n    return g._nodes\n"
        with pytest.raises(ValueError) as excinfo:
            bound_graph().python_remote_table(code, format='csv', api_token='t')
        assert 'df_import_args' in str(excinfo.value)
        assert not mock_post.called

    @patch('graphistry.compute.python_remote.requests.post')
    def test_python_remote_g_csv_declines_when_no_import_args(self, mock_post):
        mock_post.return_value = mock_response(build_zip('csv'))
        code = "def task(g):\n    return g\n"
        with pytest.raises(ValueError):
            bound_graph().python_remote_g(code, format='csv', api_token='t')
        assert not mock_post.called

    @patch('graphistry.compute.python_remote.requests.post')
    def test_python_remote_table_csv_opt_in_preserves_string_ids(self, mock_post):
        mock_post.return_value = mock_response(build_table('csv'))
        code = "def task(g):\n    return g._nodes\n"
        out = bound_graph().python_remote_table(
            code, format='csv', api_token='t', df_import_args=FAITHFUL_ARGS
        )
        assert list(out['id']) == ['007', '08', 'NA']
        assert list(out['name']) == ['', 'null', 'x']
        assert list(out['big']) == [4611686018427387904, 4611686018427387905, 3]

    @patch('graphistry.compute.python_remote.requests.post')
    def test_python_remote_table_parquet_needs_no_import_args(self, mock_post):
        mock_post.return_value = mock_response(build_table('parquet'))
        code = "def task(g):\n    return g._nodes\n"
        out = bound_graph().python_remote_table(code, format='parquet', api_token='t')
        assert list(out['id']) == ['007', '08', 'NA']


def test_csv_decline_is_a_typed_gfql_error() -> None:
    from graphistry.compute.exceptions import (
        ErrorCode, GFQLRemoteError, GFQLValidationError
    )
    from graphistry.compute.remote_df_io import require_csv_opt_in

    with pytest.raises(GFQLRemoteError) as excinfo:
        require_csv_opt_in(None, "gfql_remote")
    assert excinfo.value.code == ErrorCode.E403

    # Catchable the documented GFQL way ...
    with pytest.raises(GFQLValidationError):
        require_csv_opt_in(None, "gfql_remote")

    # ... and still as ValueError, so pre-existing callers keep working.
    with pytest.raises(ValueError):
        require_csv_opt_in(None, "gfql_remote")


def test_csv_decline_on_non_dict_import_args_is_typed() -> None:
    from graphistry.compute.exceptions import ErrorCode, GFQLRemoteError
    from graphistry.compute.remote_df_io import require_csv_opt_in

    with pytest.raises(GFQLRemoteError) as excinfo:
        require_csv_opt_in("nope", "gfql_remote")  # type: ignore[arg-type]
    assert excinfo.value.code == ErrorCode.E403
