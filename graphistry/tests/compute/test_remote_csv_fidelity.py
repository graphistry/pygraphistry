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


def lossy_warning(rec) -> str:
    hits = [str(w.message) for w in rec if 'df_import_args' in str(w.message)]
    assert len(hits) == 1, [str(w.message) for w in rec]
    return hits[0]


def norm(df: pd.DataFrame):
    return [
        {k: (None if isinstance(v, float) and v != v else v) for k, v in rec.items()}
        for rec in df.to_dict('records')
    ]


class TestGfqlRemoteCsvFidelity:

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_warns_and_serves_when_no_import_args(self, mock_post):
        mock_post.return_value = mock_response(build_zip('csv'))
        with pytest.warns(UserWarning) as rec:
            out = bound_graph().gfql_remote([n()], format='csv', api_token='t')
        msg = str(rec[0].message)
        assert 'df_import_args' in msg
        assert 'parquet' in msg
        assert mock_post.called
        assert out._nodes is not None

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_warns_and_serves_for_nodes_output_type(self, mock_post):
        mock_post.return_value = mock_response(build_table('csv'))
        with pytest.warns(UserWarning):
            out = bound_graph().gfql_remote([n()], output_type='nodes', format='csv', api_token='t')
        assert mock_post.called
        assert out._nodes is not None

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_shape_csv_warns_and_serves(self, mock_post):
        mock_post.return_value = mock_response(build_table('csv'))
        with pytest.warns(UserWarning):
            out = bound_graph().gfql_remote_shape([n()], format='csv', api_token='t')
        assert mock_post.called
        assert out is not None

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_warns_and_serves_when_import_args_govern_nothing(self, mock_post):
        mock_post.return_value = mock_response(build_zip('csv'))
        with pytest.warns(UserWarning) as rec:
            out = bound_graph().gfql_remote(
                [n()], format='csv', api_token='t', df_import_args={'sep': ','}
            )
        assert 'dtype inference' in lossy_warning(rec)
        assert mock_post.called
        assert pd.api.types.is_numeric_dtype(out._nodes['id'])
        assert out._nodes['name'].isna().sum() == 2

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_gfql_remote_csv_warns_and_serves_on_empty_import_args(self, mock_post):
        mock_post.return_value = mock_response(build_zip('csv'))
        with pytest.warns(UserWarning) as rec:
            out = bound_graph().gfql_remote(
                [n()], format='csv', api_token='t', df_import_args={}
            )
        assert 'NA substitution' in lossy_warning(rec)
        assert mock_post.called
        assert list(out._nodes['id'])[:2] == [7.0, 8.0]

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
    def test_gfql_remote_csv_warns_and_serves_on_cudf_graph(self, mock_post):
        import cudf
        mock_post.return_value = mock_response(build_zip('csv'))
        g = graphistry.edges(cudf.from_pandas(EDGES), 's', 'd').nodes(cudf.from_pandas(NODES), 'id')
        g._dataset_id = 'ds_test'
        with pytest.warns(UserWarning) as rec:
            out = g.gfql_remote([n()], format='csv', api_token='t')
        msg = lossy_warning(rec)
        assert 'parquet' in msg
        assert mock_post.called
        assert out._nodes is not None
        assert len(out._nodes) == len(NODES)

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
    def test_python_remote_table_csv_warns_and_serves(self, mock_post):
        mock_post.return_value = mock_response(build_table('csv'))
        code = "def task(g):\n    return g._nodes\n"
        with pytest.warns(UserWarning) as rec:
            out = bound_graph().python_remote_table(code, format='csv', api_token='t')
        assert 'df_import_args' in str(rec[0].message)
        assert mock_post.called
        assert out is not None

    @patch('graphistry.compute.python_remote.requests.post')
    def test_python_remote_g_csv_warns_and_serves(self, mock_post):
        mock_post.return_value = mock_response(build_zip('csv'))
        code = "def task(g):\n    return g\n"
        with pytest.warns(UserWarning):
            out = bound_graph().python_remote_g(code, format='csv', api_token='t')
        assert mock_post.called
        assert out._nodes is not None

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


def test_missing_import_args_warns_and_yields_inferring_reader() -> None:
    from graphistry.compute.remote_df_io import resolve_csv_import_args

    with pytest.warns(UserWarning) as rec:
        args = resolve_csv_import_args(None, "gfql_remote")
    assert args == {}
    assert 'parquet' in str(rec[0].message)


def test_import_args_governing_both_axes_do_not_warn() -> None:
    import warnings as _w
    from graphistry.compute.remote_df_io import resolve_csv_import_args

    with _w.catch_warnings():
        _w.simplefilter("error")
        assert resolve_csv_import_args(FAITHFUL_ARGS, "gfql_remote") == FAITHFUL_ARGS


def test_converters_govern_both_axes_and_do_not_warn() -> None:
    import warnings as _w
    from graphistry.compute.remote_df_io import resolve_csv_import_args

    args = {'converters': {'id': str}}
    with _w.catch_warnings():
        _w.simplefilter("error")
        assert resolve_csv_import_args(args, "gfql_remote") == args


@pytest.mark.parametrize('args', [{}, {'sep': ','}, {'nrows': 10, 'engine': 'c'}])
def test_import_args_governing_neither_axis_warn_about_both(args) -> None:
    from graphistry.compute.remote_df_io import resolve_csv_import_args

    with pytest.warns(UserWarning) as rec:
        assert resolve_csv_import_args(dict(args), "gfql_remote") == args
    msg = lossy_warning(rec)
    assert 'dtype inference' in msg
    assert 'NA substitution' in msg
    assert 'parquet' in msg


def test_dtype_only_import_args_still_warn_about_na_substitution() -> None:
    from graphistry.compute.remote_df_io import resolve_csv_import_args

    with pytest.warns(UserWarning) as rec:
        resolve_csv_import_args({'dtype': str}, "gfql_remote")
    msg = lossy_warning(rec)
    assert 'NA substitution' in msg
    assert 'dtype inference' not in msg


def test_na_only_import_args_still_warn_about_dtype_inference() -> None:
    from graphistry.compute.remote_df_io import resolve_csv_import_args

    with pytest.warns(UserWarning) as rec:
        resolve_csv_import_args({'keep_default_na': False, 'na_values': []}, "gfql_remote")
    msg = lossy_warning(rec)
    assert 'dtype inference' in msg
    assert 'NA substitution' not in msg


def test_each_warned_axis_names_a_real_rewrite_pandas_performs() -> None:
    from io import StringIO

    csv = pd.DataFrame({'id': ['007', '08'], 'name': ['NA', 'x']}).to_csv(index=False)

    dtype_only = pd.read_csv(StringIO(csv), dtype=str)
    assert list(dtype_only['id']) == ['007', '08']
    assert dtype_only['name'].isna().sum() == 1

    na_only = pd.read_csv(StringIO(csv), keep_default_na=False, na_values=[])
    assert list(na_only['name']) == ['NA', 'x']
    assert list(na_only['id']) == [7, 8]

    both = pd.read_csv(StringIO(csv), dtype=str, keep_default_na=False, na_values=[])
    assert list(both['id']) == ['007', '08']
    assert list(both['name']) == ['NA', 'x']


def test_non_dict_import_args_is_a_typed_gfql_error() -> None:
    from graphistry.compute.exceptions import (
        ErrorCode, GFQLRemoteError, GFQLValidationError
    )
    from graphistry.compute.remote_df_io import resolve_csv_import_args

    with pytest.raises(GFQLRemoteError) as excinfo:
        resolve_csv_import_args("nope", "gfql_remote")  # type: ignore[arg-type]
    assert excinfo.value.code == ErrorCode.E403

    # Catchable the documented GFQL way, and still as ValueError.
    with pytest.raises(GFQLValidationError):
        resolve_csv_import_args("nope", "gfql_remote")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        resolve_csv_import_args("nope", "gfql_remote")  # type: ignore[arg-type]


def test_polars_frames_decline_before_the_request() -> None:
    pl = pytest.importorskip("polars")
    from unittest.mock import MagicMock, patch as _patch
    from graphistry.compute.exceptions import ErrorCode, GFQLRemoteError

    g = graphistry.nodes(pl.DataFrame({"id": [0, 1]}), "id").edges(
        pl.DataFrame({"s": [0], "d": [1]}), "s", "d")
    g._dataset_id = "ds"
    resp = MagicMock()
    resp.status_code = 200
    resp.content = b"id\n1\n"
    resp.headers = {"content-type": "text/csv"}

    with _patch("graphistry.compute.chain_remote.requests.post", return_value=resp) as mp:
        with pytest.raises(GFQLRemoteError) as excinfo:
            g.gfql_remote([n()], format="parquet", api_token="t")
        assert excinfo.value.code == ErrorCode.E404
        assert "polars" in str(excinfo.value).lower()
        assert not mp.called


def test_supported_frame_library_resolves_pandas_and_none() -> None:
    from graphistry.compute.remote_df_io import require_supported_frame_library

    assert require_supported_frame_library(None, None, "gfql_remote") == "pandas"
    assert require_supported_frame_library(
        pd.DataFrame({"id": [0]}), pd.DataFrame({"s": [0]}), "gfql_remote") == "pandas"
