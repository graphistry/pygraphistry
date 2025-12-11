# -*- coding: utf-8 -*-

import copy, datetime as dt, graphistry, os, pandas as pd, pyarrow as pa, pytest

from mock import patch
from graphistry.tests.common import NoAuthTestCase
from graphistry.constants import NODE


maybe_cudf = None
if "TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1":
    import cudf

    maybe_cudf = cudf


triangleEdges = pd.DataFrame({"src": ["a", "b", "c"], "dst": ["b", "c", "a"]})
triangleNodes = pd.DataFrame(
    {"id": ["a", "b", "c"], "a1": [1, 2, 3], "a2": ["red", "blue", "green"]}
)
triangleNodesRich = pd.DataFrame(
    {
        "id": ["a", "b", "c"],
        "a1": [1, 2, 3],
        "a2": ["red", "blue", "green"],
        "a3": [True, False, False],
        "a4": [0.5, 1.5, 1000.3],
        "a5": [
            dt.datetime.fromtimestamp(x) for x in [1440643875, 1440644191, 1440645638]
        ],
        "a6": [u"æski ēˈmōjē", u"😋", "s"],
    }
)

squareEvil = pd.DataFrame(
    {
        "src": [0, 1, 2, 3],
        "dst": [1, 2, 3, 0],
        "colors": [1, 1, 2, 2],
        "list_int": [[1], [2, 3], [4], []],
        "list_str": [["x"], ["1", "2"], ["y"], []],
        "list_bool": [[True], [True, False], [False], []],
        "list_date_str": [
            ["2018-01-01 00:00:00"],
            ["2018-01-02 00:00:00", "2018-01-03 00:00:00"],
            ["2018-01-05 00:00:00"],
            [],
        ],
        "list_date": [
            [pd.Timestamp("2018-01-05")],
            [pd.Timestamp("2018-01-05"), pd.Timestamp("2018-01-05")],
            [],
            [],
        ],
        "list_mixed": [[1], ["1", "2"], [False, None], []],
        "bool": [True, False, True, True],
        "char": ["a", "b", "c", "d"],
        "str": ["a", "b", "c", "d"],
        "ustr": [u"a", u"b", u"c", u"d"],
        "emoji": ["😋", "😋😋", "😋", "😋"],
        "int": [0, 1, 2, 3],
        "num": [0.5, 1.5, 2.5, 3.5],
        "date_str": [
            "2018-01-01 00:00:00",
            "2018-01-02 00:00:00",
            "2018-01-03 00:00:00",
            "2018-01-05 00:00:00",
        ],
        # API 1 BUG: Try with https://github.com/graphistry/pygraphistry/pull/126
        "date": [
            dt.datetime(2018, 1, 1),
            dt.datetime(2018, 1, 1),
            dt.datetime(2018, 1, 1),
            dt.datetime(2018, 1, 1),
        ],
        "time": [
            pd.Timestamp("2018-01-05"),
            pd.Timestamp("2018-01-05"),
            pd.Timestamp("2018-01-05"),
            pd.Timestamp("2018-01-05"),
        ],
        # API 2 BUG: Need timedelta in https://github.com/graphistry/pygraphistry/blob/master/graphistry/vgraph.py#L108
        "delta": [
            pd.Timedelta("1 day"),
            pd.Timedelta("1 day"),
            pd.Timedelta("1 day"),
            pd.Timedelta("1 day"),
        ],
    }
)
for c in squareEvil.columns:
    try:
        squareEvil[c + "_cat"] = squareEvil[c].astype("category")
    except:
        # lists aren't categorical
        # print('could not make categorical', c)
        1


class Fake_Response(object):
    def raise_for_status(self):
        pass

    def json(self):
        # Include all fields needed by different endpoints:
        # - 'token': for refresh endpoint
        # - 'data': for create_dataset endpoint (api=3)
        # - 'dataset', 'viztoken': for legacy compatibility
        return {
            "success": True,
            "token": "faketoken",
            "dataset": "fakedatasetname",
            "viztoken": "faketoken",
            "data": {"dataset_id": "fakedatasetname", "viztoken": "faketoken"}
        }

    status_code = 200
    text = '{"success": true}'  # Required for error handling in arrow_uploader

def assertFrameEqual(df1, df2, **kwds):
    """Assert that two dataframes are equal, ignoring ordering of columns"""

    from pandas.testing import assert_frame_equal

    return assert_frame_equal(
        df1.sort_index(axis=1), df2.sort_index(axis=1), check_names=True, **kwds
    )


class TestPlotterConversions(NoAuthTestCase):
    @pytest.mark.xfail(raises=ModuleNotFoundError)
    def test_igraph2pandas(self):
        import igraph

        ig = igraph.Graph.Tree(4, 2)
        ig.vs["vattrib"] = 0
        ig.es["eattrib"] = 1

        with pytest.warns(DeprecationWarning):
            (e, n) = graphistry.bind(source="src", destination="dst").igraph2pandas(ig)

        edges = pd.DataFrame(
            {
                "dst": {0: 1, 1: 2, 2: 3},
                "src": {0: 0, 1: 0, 2: 1},
                "eattrib": {0: 1, 1: 1, 2: 1},
            }
        )
        nodes = pd.DataFrame(
            {
                NODE: {0: 0, 1: 1, 2: 2, 3: 3},
                "vattrib": {0: 0, 1: 0, 2: 0, 3: 0},
            }
        )

        assertFrameEqual(e, edges)
        assertFrameEqual(n, nodes)

    @pytest.mark.xfail(raises=ModuleNotFoundError)
    def test_pandas2igraph(self):
        plotter = graphistry.bind(source="src", destination="dst", node="id")
        with pytest.warns(DeprecationWarning):
            ig = plotter.pandas2igraph(triangleEdges)
        with pytest.warns(DeprecationWarning):
            (e, n) = plotter.igraph2pandas(ig)
        assertFrameEqual(e, triangleEdges[["src", "dst"]])
        assertFrameEqual(n, triangleNodes[["id"]])

    @pytest.mark.xfail(raises=ModuleNotFoundError)
    @pytest.mark.xfail(raises=ImportError)
    def test_networkx2igraph(self):
        import networkx as nx

        ng = nx.complete_graph(3)
        vs = [int(x) for x in nx.__version__.split(".")]
        x = vs[0]
        if x == 1:
            nx.set_node_attributes(ng, "vattrib", 0)
            nx.set_edge_attributes(ng, "eattrib", 1)
        else:
            nx.set_node_attributes(ng, 0, "vattrib")
            nx.set_edge_attributes(ng, 1, "eattrib")
        (e, n) = graphistry.bind(source="src", destination="dst").networkx2pandas(ng)

        edges = pd.DataFrame(
            {
                "dst": {0: 1, 1: 2, 2: 2},
                "src": {0: 0, 1: 0, 2: 1},
                "eattrib": {0: 1, 1: 1, 2: 1},
            }
        )
        nodes = pd.DataFrame(
            {NODE: {0: 0, 1: 1, 2: 2}, "vattrib": {0: 0, 1: 0, 2: 0}}
        )

        assertFrameEqual(e, edges)
        assertFrameEqual(n, nodes)

    def test_cypher_unconfigured(self):
        with pytest.raises(ValueError):
            graphistry.bind().cypher("MATCH (a)-[b]-(c) RETURN a,b,c")


class TestPlotterNameBindings(NoAuthTestCase):
    def test_bind_name(self):
        plotter = graphistry.bind().name("n")
        assert plotter._name == "n"

    def test_bind_description(self):
        plotter = graphistry.bind().description("d")
        assert plotter._description == "d"


class TestPlotterPandasConversions(NoAuthTestCase):
    def test_table_to_pandas_from_none(self):
        plotter = graphistry.bind()
        assert plotter._table_to_pandas(None) is None

    def test_table_to_pandas_from_pandas(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({"x": []})
        assert isinstance(plotter._table_to_pandas(df), pd.DataFrame)

    def test_table_to_pandas_from_arrow(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({"x": []})
        arr = pa.Table.from_pandas(df)
        assert isinstance(plotter._table_to_pandas(arr), pd.DataFrame)

    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1",
    )
    def test_table_to_pandas_from_cudf(self):
        import cudf

        plotter = graphistry.bind()
        df = pd.DataFrame({"x": [1, 2, 3]})
        gdf = cudf.from_pandas(df)
        out = plotter._table_to_pandas(gdf)
        assert isinstance(out, pd.DataFrame)
        assertFrameEqual(out, df)

    @pytest.mark.skipif(
        not ("TEST_DASK" in os.environ and os.environ["TEST_DASK"] == "1"),
        reason="dask tests need TEST_DASK=1",
    )
    def test_table_to_pandas_from_dask(self):
        import dask, dask.dataframe
        from dask.distributed import Client

        with Client(processes=True):
            plotter = graphistry.bind()
            df = pd.DataFrame({"x": [1, 2, 3]})
            ddf = dask.dataframe.from_pandas(df, npartitions=2)
            out = plotter._table_to_pandas(ddf)
            assert isinstance(out, pd.DataFrame)
            assertFrameEqual(out, df)

    @pytest.mark.skipif(
        not ("TEST_DASK_CUDF" in os.environ and os.environ["TEST_DASK_CUDF"] == "1"),
        reason="dask_cudf tests need TEST_DASK_CUDF=1",
    )
    def test_table_to_pandas_from_dask_cudf(self):
        import cudf, dask_cudf

        plotter = graphistry.bind()
        df = pd.DataFrame({"x": [1, 2, 3]})
        gdf = cudf.from_pandas(df)
        dgdf = dask_cudf.from_cudf(gdf, npartitions=2)
        out = plotter._table_to_pandas(dgdf)
        assert isinstance(out, pd.DataFrame)
        assertFrameEqual(out, df)


class TestPlotterLabelInference(NoAuthTestCase):
    def test_infer_labels_predefined_title(self):
        g = graphistry.nodes(pd.DataFrame({"x": [1, 2], "y": [1, 2]})).bind(
            point_title="y"
        )
        g2 = g.infer_labels()
        assert g2._point_title == "y"

    def test_infer_labels_predefined_label(self):
        g = graphistry.nodes(pd.DataFrame({"x": [1, 2], "y": [1, 2]})).bind(
            point_label="y"
        )
        g2 = g.infer_labels()
        assert g2._point_label == "y"

    def test_infer_labels_infer_name_exact(self):
        g = graphistry.nodes(pd.DataFrame({"x": [1, 2], "name": [1, 2]}))
        g2 = g.infer_labels()
        assert g2._point_title == "name"

    def test_infer_labels_infer_name_substr(self):
        g = graphistry.nodes(pd.DataFrame({"x": [1, 2], "thename": [1, 2]}))
        g2 = g.infer_labels()
        assert g2._point_title == "thename"

    def test_infer_labels_infer_name_id_fallback(self):
        g = graphistry.nodes(pd.DataFrame({"x": [1, 2], "y": [1, 2]}), "y")
        g2 = g.infer_labels()
        assert g2._point_title == "y"

    def test_infer_labels_exn_unknown(self):
        g = graphistry.nodes(pd.DataFrame({"x": [1, 2], "y": [1, 2]}))
        with pytest.raises(ValueError):
            g.infer_labels()


class TestPlotterArrowConversions(NoAuthTestCase):
    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.pygraphistry.PyGraphistry.store_token_creds_in_memory(True)
        graphistry.pygraphistry.PyGraphistry.relogin = lambda: True
        graphistry.register(api=3)

    def test_table_to_arrow_from_none(self):
        plotter = graphistry.bind()
        assert plotter._table_to_arrow(None) is None

    def test_table_to_arrow_from_pandas(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({"x": []})
        assert isinstance(plotter._table_to_arrow(df), pa.Table)

    def test_table_to_arrow_from_arrow(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({"x": []})
        arr = pa.Table.from_pandas(df)
        assert isinstance(plotter._table_to_arrow(arr), pa.Table)

    def test_api3_plot_from_pandas(self):
        g = graphistry.edges(pd.DataFrame({"s": [0], "d": [0]})).bind(
            source="s", destination="d"
        )
        ds = g.plot(skip_upload=True)
        assert isinstance(ds.edges, pa.Table)

    @pytest.mark.xfail(raises=ModuleNotFoundError)
    def test_api3_plot_from_igraph(self):
        g = graphistry.bind(source="src", destination="dst", node="id")
        with pytest.warns(DeprecationWarning):
            ig = g.pandas2igraph(triangleEdges)
        with pytest.warns(DeprecationWarning):
            (e, n) = g.igraph2pandas(ig)
        g = g.edges(e).nodes(n)
        ds = g.plot(skip_upload=True)
        assert isinstance(ds.edges, pa.Table)
        assert isinstance(ds.nodes, pa.Table)

    def test_api3_plot_from_arrow(self):
        g = graphistry.edges(
            pa.Table.from_pandas(pd.DataFrame({"s": [0], "d": [0]}))
        ).bind(source="s", destination="d")
        ds = g.plot(skip_upload=True)
        assert isinstance(ds.edges, pa.Table)

    def test_api3_pdf_to_arrow_memoization(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({"x": [1]})
        arr1 = plotter._table_to_arrow(df)
        arr2 = plotter._table_to_arrow(df)
        assert isinstance(arr1, pa.Table)
        assert arr1 is arr2

        arr3 = plotter._table_to_arrow(pd.DataFrame({"x": [1]}))
        assert arr1 is arr3

    def test_api3_pdf_to_renamed_arrow_memoization(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({'x': [1]})
        arr1 = plotter._table_to_arrow(df)
        arr2 = plotter._table_to_arrow(df)
        assert isinstance(arr1, pa.Table)
        assert arr1 is arr2

        df2 = df.rename(columns={'x': 'y'})
        arr3 = plotter._table_to_arrow(df2)
        assert not (arr1 is arr3)

    def test_api3_pdf_to_arrow_memoization_forgets(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({"x": [0]})
        arr1 = plotter._table_to_arrow(df)
        for i in range(1, 110):
            plotter._table_to_arrow(pd.DataFrame({"x": [i]}))

        assert not (arr1 is plotter._table_to_arrow(df))

    # ==========================================================================
    # Auto-coerce mixed-type columns tests (Issue #867)
    # ==========================================================================

    def test_table_to_arrow_mixed_bytes_float_string(self):
        """Test that mixed bytes/float/string columns are auto-coerced to string."""
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'src': [1, 2, 3],
            'dst': [2, 3, 1],
            'amount': [b'bytes_value', 1.5, 'string_value']  # Mixed: bytes, float, string
        })
        # Should not raise, should succeed with coercion
        with pytest.warns(RuntimeWarning, match='Coerced mixed-type columns to string'):
            arr = plotter._table_to_arrow(df)
        assert isinstance(arr, pa.Table)
        # amount column should be coerced to string type
        assert pa.types.is_string(arr.schema.field('amount').type) or pa.types.is_large_string(arr.schema.field('amount').type)

    def test_table_to_arrow_mixed_datetime_int_string(self):
        """Test that mixed datetime/int/string columns are auto-coerced to string."""
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'timestamp': [dt.datetime(2020, 10, 20), 1, 'string_value']  # Mixed: datetime, int, string
        })
        with pytest.warns(RuntimeWarning, match='Coerced mixed-type columns to string'):
            arr = plotter._table_to_arrow(df)
        assert isinstance(arr, pa.Table)
        assert pa.types.is_string(arr.schema.field('timestamp').type) or pa.types.is_large_string(arr.schema.field('timestamp').type)

    def test_table_to_arrow_mixed_list_scalar(self):
        """Test that mixed list/scalar columns are auto-coerced to string (ArrowInvalid)."""
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'roles': [['admin', 'user'], 'not_a_list', None]  # Mixed: list, string, None
        })
        with pytest.warns(RuntimeWarning, match='Coerced mixed-type columns to string'):
            arr = plotter._table_to_arrow(df)
        assert isinstance(arr, pa.Table)
        assert pa.types.is_string(arr.schema.field('roles').type) or pa.types.is_large_string(arr.schema.field('roles').type)

    def test_table_to_arrow_mixed_dict_scalar(self):
        """Test that mixed dict/scalar columns are auto-coerced to string (ArrowInvalid)."""
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'attrs': [{'a': 1}, 'not_a_dict', None]  # Mixed: dict, string, None
        })
        with pytest.warns(RuntimeWarning, match='Coerced mixed-type columns to string'):
            arr = plotter._table_to_arrow(df)
        assert isinstance(arr, pa.Table)
        assert pa.types.is_string(arr.schema.field('attrs').type) or pa.types.is_large_string(arr.schema.field('attrs').type)

    def test_table_to_arrow_clean_data_no_warning(self):
        """Test that clean data does not emit warnings."""
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'src': [1, 2, 3],
            'dst': [2, 3, 1],
            'weight': [1.0, 2.0, 3.0]
        })
        # Should not warn for clean data
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            arr = plotter._table_to_arrow(df)
        assert isinstance(arr, pa.Table)

    def test_table_to_arrow_multiple_bad_columns(self):
        """Test that multiple bad columns are all coerced and reported."""
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'good': [1, 2, 3],
            'bad1': [b'bytes', 1.5, 'str'],  # Mixed types
            'bad2': [['list'], 'scalar', None],  # Mixed list/scalar
        })
        with pytest.warns(RuntimeWarning, match='bad1.*bad2|bad2.*bad1'):
            arr = plotter._table_to_arrow(df)
        assert isinstance(arr, pa.Table)

    def test_table_to_arrow_memoization_with_coercion(self):
        """Test that memoization works correctly with coerced data."""
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'x': [1, 2],
            'mixed': [b'bytes', 1.5]  # Will be coerced
        })
        # First call - coerces and caches
        with pytest.warns(RuntimeWarning, match='Coerced mixed-type columns'):
            arr1 = plotter._table_to_arrow(df)

        # Second call with same data - should hit cache, no warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            arr2 = plotter._table_to_arrow(df)

        # Should be same cached object
        assert arr1 is arr2

    def test_to_arrow_public_method(self):
        """Test public to_arrow() helper method."""
        df = pd.DataFrame({'src': [1, 2, 3], 'dst': [2, 3, 1]})
        g = graphistry.edges(df, 'src', 'dst')

        # Convert explicit dataframe
        arr = g.to_arrow(df)
        assert isinstance(arr, pa.Table)

        # Convert bound edges (default)
        arr2 = g.to_arrow()
        assert isinstance(arr2, pa.Table)

    def test_to_arrow_with_mixed_types(self):
        """Test that to_arrow() also handles mixed types."""
        df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 1],
            'mixed': [b'bytes', 1.5]
        })
        g = graphistry.edges(df, 'src', 'dst')

        with pytest.warns(RuntimeWarning, match='Coerced mixed-type columns'):
            arr = g.to_arrow(df)
        assert isinstance(arr, pa.Table)

    # ==========================================================================
    # Validate mode tests (Issue #867 - strict vs autofix)
    # ==========================================================================

    def test_validate_strict_fails_on_mixed_types(self):
        """validate='strict' should raise ArrowConversionError on mixed-type columns."""
        from graphistry.exceptions import ArrowConversionError
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'src': [1, 2, 3],
            'dst': [2, 3, 1],
            'amount': [b'strict_bytes', 1.5, 'strict_string']  # Mixed types
        })
        with pytest.raises(ArrowConversionError, match='Arrow conversion failed'):
            plotter._table_to_arrow(df, memoize=False, validate_mode='strict')

    def test_validate_strict_fast_fails_on_mixed_types(self):
        """validate='strict-fast' should also fail on type issues."""
        from graphistry.exceptions import ArrowConversionError
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'src': [1, 2, 3],
            'dst': [2, 3, 1],
            'amount': [b'strictfast_bytes', 1.5, 'strictfast_string']  # Mixed types
        })
        with pytest.raises(ArrowConversionError, match='Arrow conversion failed'):
            plotter._table_to_arrow(df, memoize=False, validate_mode='strict-fast')

    def test_validate_autofix_coerces_mixed_types(self):
        """validate='autofix' should coerce and warn (default behavior)."""
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'src': [5, 6, 7],
            'dst': [6, 7, 5],
            'amount': [b'autofix_bytes', 5.5, 'autofix_str']
        })
        with pytest.warns(RuntimeWarning, match='Coerced mixed-type columns'):
            arr = plotter._table_to_arrow(df, memoize=False, validate_mode='autofix')
        assert isinstance(arr, pa.Table)

    def test_validate_default_is_autofix(self):
        """Default validate mode should be autofix (coerce and warn)."""
        plotter = graphistry.bind()
        # Use unique values to avoid memoization cache hits from other tests
        df = pd.DataFrame({
            'src': [100, 200, 300],
            'dst': [200, 300, 100],
            'amount': [b'unique_bytes', 99.99, 'unique_string']
        })
        # Default validate_mode='autofix' should coerce and warn
        with pytest.warns(RuntimeWarning, match='Coerced mixed-type columns'):
            arr = plotter._table_to_arrow(df, memoize=False)  # default validate_mode='autofix'
        assert isinstance(arr, pa.Table)

    def test_validate_true_maps_to_strict(self):
        """validate=True should behave like 'strict' (for backward compat)."""
        from graphistry.exceptions import ArrowConversionError
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'src': [10, 20, 30],
            'dst': [20, 30, 10],
            'amount': [b'test_true_bytes', 2.5, 'test_true_str']
        })
        # True maps to 'strict' - test by using 'strict' directly
        with pytest.raises(ArrowConversionError):
            plotter._table_to_arrow(df, memoize=False, validate_mode='strict')

    def test_validate_false_maps_to_autofix_silent(self):
        """validate=False should behave like autofix without warnings (at plot level)."""
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'src': [11, 21, 31],
            'dst': [21, 31, 11],
            'amount': [b'test_false_bytes', 3.5, 'test_false_str']
        })
        # False maps to 'autofix' - test by using 'autofix' directly
        # (warn suppression is handled at plot() level via warnings.catch_warnings)
        with pytest.warns(RuntimeWarning, match='Coerced mixed-type columns'):
            arr = plotter._table_to_arrow(df, memoize=False, validate_mode='autofix')
        assert isinstance(arr, pa.Table)

    def test_validate_strict_allows_clean_data(self):
        """validate='strict' should allow clean data through without error."""
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'src': [1000, 2000, 3000],
            'dst': [2000, 3000, 1000],
            'weight': [1.1, 2.2, 3.3]  # Clean data
        })
        # Should not raise
        arr = plotter._table_to_arrow(df, memoize=False, validate_mode='strict')
        assert isinstance(arr, pa.Table)

    def test_validate_strict_error_message_is_helpful(self):
        """ArrowConversionError should include helpful information."""
        from graphistry.exceptions import ArrowConversionError
        plotter = graphistry.bind()
        df = pd.DataFrame({
            'src': [12, 22, 32],
            'dst': [22, 32, 12],
            'amount': [b'helpful_bytes', 4.5, 'helpful_str']
        })
        with pytest.raises(ArrowConversionError) as exc_info:
            plotter._table_to_arrow(df, memoize=False, validate_mode='strict')
        error_msg = str(exc_info.value)
        # Should mention autofix as alternative
        assert 'autofix' in error_msg.lower() or 'auto' in error_msg.lower()
        # Should mention the problematic column(s)
        assert 'amount' in error_msg or 'columns' in error_msg

    # ==========================================================================
    # Phase 8: Comprehensive validation scenario tests (Issue #867)
    # Tests the full matrix of (validate x warn x data_state x encoding_state)
    # ==========================================================================

    # --- Arrow validation through plot() entrypoint ---

    def test_plot_strict_clean_data_passes(self):
        """Scenario 1: strict + clean data should pass without errors."""
        g = graphistry.edges(pd.DataFrame({
            'src': [1, 2, 3],
            'dst': [2, 3, 1],
            'weight': [1.0, 2.0, 3.0]  # Clean data
        }), 'src', 'dst')
        # Should not raise
        result = g.plot(skip_upload=True, validate='strict')
        assert result is not None

    def test_plot_strict_mixed_data_raises(self):
        """Scenario 3: strict + mixed data should raise ArrowConversionError."""
        from graphistry.exceptions import ArrowConversionError
        g = graphistry.edges(pd.DataFrame({
            'src': [1, 2, 3],
            'dst': [2, 3, 1],
            'mixed': [b'bytes', 1.5, 'string']  # Mixed types
        }), 'src', 'dst')
        with pytest.raises(ArrowConversionError):
            g.plot(skip_upload=True, validate='strict')

    def test_plot_autofix_warn_true_clean_no_warning(self):
        """Scenario 10: autofix + warn=True + clean data should pass without warning."""
        import warnings
        g = graphistry.edges(pd.DataFrame({
            'src': [100, 200, 300],
            'dst': [200, 300, 100],
            'weight': [1.0, 2.0, 3.0]  # Clean data - unique values to avoid cache
        }), 'src', 'dst')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = g.plot(skip_upload=True, validate='autofix', warn=True)
            # Filter for our specific warnings
            coerce_warnings = [x for x in w if 'Coerced' in str(x.message)]
            assert len(coerce_warnings) == 0, f"Expected no coercion warnings, got: {coerce_warnings}"
        assert result is not None

    def test_plot_autofix_warn_true_mixed_warns_and_coerces(self):
        """Scenario 12: autofix + warn=True + mixed data should warn and coerce."""
        g = graphistry.edges(pd.DataFrame({
            'src': [101, 201, 301],
            'dst': [201, 301, 101],
            'mixed': [b'warn_bytes', 1.5, 'warn_string']  # Mixed types
        }), 'src', 'dst')
        with pytest.warns(RuntimeWarning, match='Coerced mixed-type columns'):
            result = g.plot(skip_upload=True, validate='autofix', warn=True)
        assert result is not None

    def test_plot_autofix_warn_false_mixed_coerces_silently(self):
        """Scenario 16: autofix + warn=False + mixed data should coerce silently."""
        import warnings
        g = graphistry.edges(pd.DataFrame({
            'src': [102, 202, 302],
            'dst': [202, 302, 102],
            'mixed': [b'silent_bytes', 2.5, 'silent_string']  # Mixed types
        }), 'src', 'dst')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = g.plot(skip_upload=True, validate='autofix', warn=False)
            # Filter for coercion warnings
            coerce_warnings = [x for x in w if 'Coerced' in str(x.message)]
            assert len(coerce_warnings) == 0, f"Expected no warnings with warn=False, got: {coerce_warnings}"
        assert result is not None

    def test_plot_validate_false_coerces_silently(self):
        """Scenario 19: validate=False should coerce silently (maps to autofix+warn=False)."""
        import warnings
        g = graphistry.edges(pd.DataFrame({
            'src': [103, 203, 303],
            'dst': [203, 303, 103],
            'mixed': [b'false_bytes', 3.5, 'false_string']  # Mixed types
        }), 'src', 'dst')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = g.plot(skip_upload=True, validate=False)
            coerce_warnings = [x for x in w if 'Coerced' in str(x.message)]
            assert len(coerce_warnings) == 0, f"Expected no warnings with validate=False, got: {coerce_warnings}"
        assert result is not None

    def test_plot_validate_true_raises_on_mixed(self):
        """Scenario 18: validate=True should raise (maps to strict)."""
        from graphistry.exceptions import ArrowConversionError
        g = graphistry.edges(pd.DataFrame({
            'src': [104, 204, 304],
            'dst': [204, 304, 104],
            'mixed': [b'true_bytes', 4.5, 'true_string']  # Mixed types
        }), 'src', 'dst')
        with pytest.raises(ArrowConversionError):
            g.plot(skip_upload=True, validate=True)

    # --- Encoding validation tests (Phase 8.B) ---
    # Note: These test encoding validation at the validate_encodings level.
    # Full integration with upload() would require server mocking.

    def test_encoding_validation_strict_invalid_raises(self):
        """Scenario 2: strict + invalid encoding should raise ValueError."""
        from graphistry.validate.validate_encodings import validate_encodings

        # Encoding references a column that doesn't exist
        node_enc = {
            "bindings": {"node": "n"},
            "complex": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "nonexistent_column",  # Invalid!
                        "variation": "categorical",
                        "mapping": {"categorical": {"fixed": {"a": "red"}, "other": "blue"}}
                    }
                }
            }
        }
        edge_enc = {"bindings": {"source": "s", "destination": "d"}}

        # With node_attributes specified, validation should fail
        with pytest.raises(ValueError):
            validate_encodings(node_enc, edge_enc, node_attributes=['n', 'real_column'])

    def test_encoding_validation_valid_passes(self):
        """Encoding validation with valid attributes should pass."""
        from graphistry.validate.validate_encodings import validate_encodings

        node_enc = {
            "bindings": {"node": "n"},
            "complex": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "type",  # Valid - in attributes list
                        "variation": "categorical",
                        "mapping": {"categorical": {"fixed": {"a": "red"}, "other": "blue"}}
                    }
                }
            }
        }
        edge_enc = {"bindings": {"source": "s", "destination": "d"}}

        # Should not raise
        result = validate_encodings(node_enc, edge_enc, node_attributes=['n', 'type'])
        assert result is not None

    # --- Encoding validation warn mode tests (Phase 8.D) ---

    def test_encoding_validation_warn_mode_emits_warning(self):
        """Scenario 11: autofix + warn=True + invalid encoding should warn, not raise."""
        from graphistry.arrow_uploader import ArrowUploader
        import warnings

        # Create a minimal uploader with nodes that have columns - this enables attribute validation
        uploader = ArrowUploader.__new__(ArrowUploader)
        uploader.nodes = pa.table({'n': [1, 2], 'real_column': ['a', 'b']})  # Only has 'n' and 'real_column'
        uploader.edges = pa.table({'s': [1, 2], 'd': [2, 1]})

        invalid_json = {
            "node_encodings": {
                "bindings": {"node": "n"},
                "complex": {
                    "default": {
                        "pointColorEncoding": {
                            "graphType": "point",
                            "encodingType": "color",
                            "attribute": "nonexistent",  # Invalid - not in node columns!
                            "variation": "categorical",
                            "mapping": {"categorical": {"fixed": {"a": "red"}, "other": "blue"}}
                        }
                    }
                }
            },
            "edge_encodings": {"bindings": {"source": "s", "destination": "d"}}
        }

        # With strict=False, warn=True: should emit warning, not raise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This will fail at the network call, but encoding validation should warn first
            try:
                uploader.create_dataset(invalid_json, validate=True, strict=False, warn=True)
            except Exception:
                pass  # Expected to fail at network call
            # Check for encoding validation warning
            encoding_warnings = [x for x in w if 'Encoding validation warning' in str(x.message)]
            assert len(encoding_warnings) > 0, f"Expected encoding warning, got: {[str(x.message) for x in w]}"

    def test_encoding_validation_strict_mode_raises(self):
        """Scenario 2/6: strict mode + invalid encoding should raise."""
        from graphistry.arrow_uploader import ArrowUploader

        uploader = ArrowUploader.__new__(ArrowUploader)
        uploader.nodes = pa.table({'n': [1, 2], 'real_column': ['a', 'b']})  # Only has 'n' and 'real_column'
        uploader.edges = pa.table({'s': [1, 2], 'd': [2, 1]})

        invalid_json = {
            "node_encodings": {
                "bindings": {"node": "n"},
                "complex": {
                    "default": {
                        "pointColorEncoding": {
                            "graphType": "point",
                            "encodingType": "color",
                            "attribute": "nonexistent",  # Invalid - not in node columns!
                            "variation": "categorical",
                            "mapping": {"categorical": {"fixed": {"a": "red"}, "other": "blue"}}
                        }
                    }
                }
            },
            "edge_encodings": {"bindings": {"source": "s", "destination": "d"}}
        }

        # With strict=True: should raise ValueError
        with pytest.raises(ValueError):
            uploader.create_dataset(invalid_json, validate=True, strict=True, warn=True)

    def test_encoding_validation_silent_mode(self):
        """Scenario 15/17: warn=False should be silent on invalid encoding."""
        from graphistry.arrow_uploader import ArrowUploader
        import warnings

        uploader = ArrowUploader.__new__(ArrowUploader)
        uploader.nodes = pa.table({'n': [1, 2], 'real_column': ['a', 'b']})  # Only has 'n' and 'real_column'
        uploader.edges = pa.table({'s': [1, 2], 'd': [2, 1]})

        invalid_json = {
            "node_encodings": {
                "bindings": {"node": "n"},
                "complex": {
                    "default": {
                        "pointColorEncoding": {
                            "graphType": "point",
                            "encodingType": "color",
                            "attribute": "nonexistent",  # Invalid - not in node columns!
                            "variation": "categorical",
                            "mapping": {"categorical": {"fixed": {"a": "red"}, "other": "blue"}}
                        }
                    }
                }
            },
            "edge_encodings": {"bindings": {"source": "s", "destination": "d"}}
        }

        # With warn=False: should NOT emit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                uploader.create_dataset(invalid_json, validate=True, strict=False, warn=False)
            except Exception:
                pass  # Expected to fail at network call
            encoding_warnings = [x for x in w if 'Encoding validation warning' in str(x.message)]
            assert len(encoding_warnings) == 0, f"Expected no encoding warnings with warn=False, got: {encoding_warnings}"

    # --- Combined scenario tests (Phase 8.C) ---

    def test_strict_mixed_data_fails_before_encoding_check(self):
        """Scenario 4: With mixed data, Arrow conversion fails before encoding validation."""
        from graphistry.exceptions import ArrowConversionError

        # Both mixed data AND invalid encoding - Arrow should fail first
        g = graphistry.edges(pd.DataFrame({
            'src': [1, 2, 3],
            'dst': [2, 3, 1],
            'mixed': [b'bytes', 1.5, 'string']  # Mixed types cause Arrow failure
        }), 'src', 'dst')

        # Add an invalid encoding
        g = g.encode_point_color('nonexistent_column', categorical_mapping={'a': 'red'})

        # Should raise ArrowConversionError, not encoding ValueError
        with pytest.raises(ArrowConversionError):
            g.plot(skip_upload=True, validate='strict')

    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1",
    )
    def test_api3_cudf_to_arrow_memoization(self):
        maybe_cudf = None
        try:
            import cudf

            maybe_cudf = cudf
        except ImportError:
            1
        if maybe_cudf is None:
            return

        plotter = graphistry.bind()
        df = maybe_cudf.DataFrame({"x": [1]})
        arr1 = plotter._table_to_arrow(df)
        arr2 = plotter._table_to_arrow(df)
        assert isinstance(arr1, pa.Table)
        assert arr1 is arr2

        arr3 = plotter._table_to_arrow(maybe_cudf.DataFrame({"x": [1]}))
        assert arr1 is arr3

    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1",
    )
    def test_api3_cudf_to_arrow_memoization_forgets(self):
        maybe_cudf = None
        try:
            import cudf

            maybe_cudf = cudf
        except ImportError:
            1
        if maybe_cudf is None:
            return

        plotter = graphistry.bind()
        df = maybe_cudf.DataFrame({"x": [0]})
        arr1 = plotter._table_to_arrow(df)
        for i in range(1, 110):
            plotter._table_to_arrow(maybe_cudf.DataFrame({"x": [i]}))

        assert not (arr1 is plotter._table_to_arrow(df))

    @pytest.mark.skipif(
        not ("TEST_DASK" in os.environ and os.environ["TEST_DASK"] == "1"),
        reason="dask tests need TEST_DASK=1",
    )
    def test_api3_dask_to_arrow_memoization(self):
        import dask, dask.dataframe
        from dask.distributed import Client

        with Client(processes=True):
            plotter = graphistry.bind()
            df = pd.DataFrame({"x": [1, 2]})
            ddf = dask.dataframe.from_pandas(df, npartitions=2)
            arr1 = plotter._table_to_arrow(ddf)
            arr2 = plotter._table_to_arrow(ddf)
            assert isinstance(arr1, pa.Table)
            assert arr1 is arr2

            arr3 = plotter._table_to_arrow(
                dask.dataframe.from_pandas(pd.DataFrame({"x": [1, 2]}), npartitions=2)
            )
            assert arr1 is arr3

    @pytest.mark.skipif(
        not ("TEST_DASK" in os.environ and os.environ["TEST_DASK"] == "1"),
        reason="dask tests need TEST_DASK=1",
    )
    def test_api3_dask_to_arrow_memoization_forgets(self):
        import dask, dask.dataframe
        from dask.distributed import Client

        with Client(processes=True):
            plotter = graphistry.bind()
            df = pd.DataFrame({"x": [0]})
            ddf = dask.dataframe.from_pandas(df, npartitions=2)
            arr1 = plotter._table_to_arrow(ddf)
            for i in range(1, 110):
                ddf_i = dask.dataframe.from_pandas(
                    pd.DataFrame({"x": [0, i]}), npartitions=2
                )
                plotter._table_to_arrow(ddf_i)
            assert not (arr1 is plotter._table_to_arrow(ddf))

    @pytest.mark.skipif(
        not ("TEST_DASK_CUDF" in os.environ and os.environ["TEST_DASK_CUDF"] == "1"),
        reason="dask_cudf tests need TEST_DASK_CUDF=1",
    )
    def test_api3_dask_cudf_to_arrow_memoization(self):
        import cudf, dask_cudf

        plotter = graphistry.bind()
        gdf = cudf.DataFrame({"x": [1, 2]})
        dgdf = dask_cudf.from_cudf(gdf, npartitions=2)
        arr1 = plotter._table_to_arrow(dgdf)
        arr2 = plotter._table_to_arrow(dgdf)
        assert isinstance(arr1, pa.Table)
        assert arr1 is arr2

        arr3 = plotter._table_to_arrow(
            dask_cudf.from_cudf(cudf.DataFrame({"x": [1, 2]}), npartitions=2)
        )
        assert arr1 is arr3

    @pytest.mark.skipif(
        not ("TEST_DASK_CUDF" in os.environ and os.environ["TEST_DASK_CUDF"] == "1"),
        reason="dask_cudf tests need TEST_DASK_CUDF=1",
    )
    def test_api3_dask_cudf_to_arrow_memoization_forgets(self):
        import cudf, dask_cudf

        plotter = graphistry.bind()
        gdf = cudf.DataFrame({"x": [1, 0]})
        dgdf = dask_cudf.from_cudf(gdf, npartitions=2)
        arr1 = plotter._table_to_arrow(dgdf)
        for i in range(1, 110):
            dgdf_i = dask_cudf.from_cudf(cudf.DataFrame({"x": [0, i]}), npartitions=2)
            plotter._table_to_arrow(dgdf_i)
        assert not (arr1 is plotter._table_to_arrow(dgdf))


class TestPlotterStylesArrow(NoAuthTestCase):
    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.pygraphistry.PyGraphistry.store_token_creds_in_memory(True)
        graphistry.pygraphistry.PyGraphistry.relogin = lambda: True
        graphistry.register(api=3)

    def test_init(self):
        g = graphistry.bind()
        assert g._style is None

    def test_style_good(self):
        g = graphistry.bind()

        bg = {"color": "red"}
        fg = {"blendMode": 1}
        logo = {"url": "zzz"}
        page = {"title": "zzz"}

        assert g.style()._style == {}

        g.style(fg={"blendMode": "screen"})
        assert g.style()._style == {}

        assert g.style(bg=copy.deepcopy(bg))._style == {"bg": bg}
        assert g.style(bg={"color": "blue"}).style(bg=copy.deepcopy(bg))._style == {
            "bg": bg
        }
        assert g.style(bg={"image": {"url": "http://asdf.com/b.png"}}).style(
            bg=copy.deepcopy(bg)
        )._style == {"bg": bg}
        assert (
            g.style(
                bg=copy.deepcopy(bg),
                fg=copy.deepcopy(fg),
                logo=copy.deepcopy(logo),
                page=copy.deepcopy(page),
            )._style == {"bg": bg, "fg": fg, "logo": logo, "page": page}
        )
        assert g.style(
            bg=copy.deepcopy(bg),
            fg=copy.deepcopy(fg),
            logo=copy.deepcopy(logo),
            page=copy.deepcopy(page),
        ).style(bg={"color": "green"})._style == {
            "bg": {"color": "green"},
            "fg": fg,
            "logo": logo,
            "page": page,
        }

        g2 = graphistry.edges(pd.DataFrame({"s": [0], "d": [0]})).bind(
            source="s", destination="d"
        )
        ds = g2.style(
            bg=copy.deepcopy(bg),
            fg=copy.deepcopy(fg),
            page=copy.deepcopy(page),
            logo=copy.deepcopy(logo),
        ).plot(skip_upload=True)
        assert ds.metadata["bg"] == bg
        assert ds.metadata["fg"] == fg
        assert ds.metadata["logo"] == logo
        assert ds.metadata["page"] == page

    def test_addStyle_good(self):
        g = graphistry.bind()

        bg = {"color": "red"}
        fg = {"blendMode": 1}
        logo = {"url": "zzz"}
        page = {"title": "zzz"}

        assert g.addStyle()._style == {}

        g.addStyle(fg={"blendMode": "screen"})
        assert g.addStyle()._style == {}

        assert g.addStyle(bg=copy.deepcopy(bg))._style == {"bg": bg}
        assert g.addStyle(bg={"color": "blue"}).addStyle(
            bg=copy.deepcopy(bg)
        )._style == {"bg": bg}
        assert g.addStyle(bg={"image": {"url": "http://asdf.com/b.png"}}).addStyle(
            bg=copy.deepcopy(bg)
        )._style == {"bg": {**bg, "image": {"url": "http://asdf.com/b.png"}}}
        assert (
            g.addStyle(
                bg=copy.deepcopy(bg),
                fg=copy.deepcopy(fg),
                logo=copy.deepcopy(logo),
                page=copy.deepcopy(page),
            )._style == {"bg": bg, "fg": fg, "logo": logo, "page": page}
        )
        assert g.addStyle(
            bg=copy.deepcopy(bg),
            fg=copy.deepcopy(fg),
            logo=copy.deepcopy(logo),
            page=copy.deepcopy(page),
        ).addStyle(bg={"color": "green"})._style == {
            "bg": {"color": "green"},
            "fg": fg,
            "logo": logo,
            "page": page,
        }

        g2 = graphistry.edges(pd.DataFrame({"s": [0], "d": [0]})).bind(
            source="s", destination="d"
        )
        ds = g2.addStyle(
            bg=copy.deepcopy(bg),
            fg=copy.deepcopy(fg),
            page=copy.deepcopy(page),
            logo=copy.deepcopy(logo),
        ).plot(skip_upload=True)
        assert ds.metadata["bg"] == bg
        assert ds.metadata["fg"] == fg
        assert ds.metadata["logo"] == logo
        assert ds.metadata["page"] == page


class TestPlotterEncodings(NoAuthTestCase):

    COMPLEX_EMPTY = {
        "node_encodings": {"current": {}, "default": {}},
        "edge_encodings": {"current": {}, "default": {}},
    }

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.pygraphistry.PyGraphistry.store_token_creds_in_memory(True)
        graphistry.pygraphistry.PyGraphistry.relogin = lambda: True
        graphistry.register(api=3)

    def test_init_mt(self):
        assert (
            graphistry.bind()._complex_encodings == TestPlotterEncodings.COMPLEX_EMPTY
        )

    def test_point_color(self):
        assert graphistry.bind().encode_point_color("z")._point_color == "z"
        assert graphistry.bind().encode_point_color(
            "z", ["red", "blue"], as_continuous=True
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "z",
                        "variation": "continuous",
                        "colors": ["red", "blue"],
                    }
                },
                "current": {},
            },
        }
        assert graphistry.bind().encode_point_color(
            "z", ["red", "blue"], as_categorical=True
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "z",
                        "variation": "categorical",
                        "colors": ["red", "blue"],
                    }
                },
                "current": {},
            },
        }
        assert graphistry.bind().encode_point_color(
            "z", categorical_mapping={"truck": "red"}
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "z",
                        "variation": "categorical",
                        "mapping": {"categorical": {"fixed": {"truck": "red"}}},
                    }
                },
                "current": {},
            },
        }
        assert graphistry.bind().encode_point_color(
            "z", categorical_mapping={"truck": "red"}, default_mapping="blue"
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "z",
                        "variation": "categorical",
                        "mapping": {
                            "categorical": {"fixed": {"truck": "red"}, "other": "blue"}
                        },
                    }
                },
                "current": {},
            },
        }

    def test_point_size(self):
        assert graphistry.bind().encode_point_size("z")._point_size == "z"

    def test_point_icon(self):
        assert graphistry.bind().encode_point_icon("z")._point_icon == "z"

        assert graphistry.bind().encode_point_icon(
            "z", categorical_mapping={}, as_text=True, blend_mode="color-dodge"
        )._complex_encodings["node_encodings"] == {
            "current": {},
            "default": {
                "pointIconEncoding": {
                    "graphType": "point",
                    "encodingType": "icon",
                    "attribute": "z",
                    "variation": "categorical",
                    "mapping": {"categorical": {"fixed": {}}},
                    "asText": True,
                    "blendMode": "color-dodge",
                }
            },
        }

        assert graphistry.bind().encode_point_icon(
            "z",
            continuous_binning=[],
            comparator="<=",
            as_text=True,
            blend_mode="color-dodge",
        )._complex_encodings["node_encodings"] == {
            "current": {},
            "default": {
                "pointIconEncoding": {
                    "graphType": "point",
                    "encodingType": "icon",
                    "attribute": "z",
                    "variation": "continuous",
                    "mapping": {"continuous": {"bins": [], "comparator": "<="}},
                    "asText": True,
                    "blendMode": "color-dodge",
                }
            },
        }

    def test_edge_icon(self):
        assert graphistry.bind().encode_edge_icon("z")._edge_icon == "z"

    def test_edge_color(self):
        assert graphistry.bind().encode_edge_color("z")._edge_color == "z"

    def test_badge(self):

        assert graphistry.bind().encode_point_badge(
            "z", position="Top"
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {
                "default": {
                    "pointBadgeTopEncoding": {
                        "graphType": "point",
                        "encodingType": "badgeTop",
                        "attribute": "z",
                        "variation": "categorical",
                    }
                },
                "current": {},
            },
        }

        assert graphistry.bind().encode_edge_badge(
            "z", position="Top"
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "edge_encodings": {
                "default": {
                    "edgeBadgeTopEncoding": {
                        "graphType": "edge",
                        "encodingType": "badgeTop",
                        "attribute": "z",
                        "variation": "categorical",
                    }
                },
                "current": {},
            },
        }

        assert graphistry.bind().encode_point_badge(
            "z",
            position="Top",
            continuous_binning=[[None, "a"]],
            default_mapping="zz",
            comparator="<=",
            color="red",
            bg={"color": "green"},
            fg={"style": {"opacity": 0.5}},
            as_text=True,
            blend_mode="color-dodge",
            style={"opacity": 0.5},
            border={"width": 10, "color": "green", "stroke": "dotted"},
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {
                "default": {
                    "pointBadgeTopEncoding": {
                        "graphType": "point",
                        "encodingType": "badgeTop",
                        "attribute": "z",
                        "variation": "continuous",
                        "mapping": {
                            "continuous": {
                                "bins": [[None, "a"]],
                                "comparator": "<=",
                                "other": "zz",
                            }
                        },
                        "color": "red",
                        "bg": {"color": "green"},
                        "fg": {"style": {"opacity": 0.5}},
                        "asText": True,
                        "blendMode": "color-dodge",
                        "style": {"opacity": 0.5},
                        "border": {"width": 10, "color": "green", "stroke": "dotted"},
                    }
                },
                "current": {},
            },
        }

        assert graphistry.bind().encode_edge_badge(
            "z",
            position="Right",
            categorical_mapping={"a": "b"},
            for_default=False,
            for_current=True,
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "edge_encodings": {
                "default": {},
                "current": {
                    "edgeBadgeRightEncoding": {
                        "graphType": "edge",
                        "encodingType": "badgeRight",
                        "attribute": "z",
                        "variation": "categorical",
                        "mapping": {"categorical": {"fixed": {"a": "b"}}},
                    }
                },
            },
        }

    def test_set_mode(self):
        assert graphistry.bind().encode_point_color(
            "z", categorical_mapping={"a": "b"}
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "z",
                        "variation": "categorical",
                        "mapping": {"categorical": {"fixed": {"a": "b"}}},
                    }
                },
                "current": {},
            },
        }

        assert graphistry.bind().encode_point_color(
            "z", categorical_mapping={"a": "b"}, for_default=False, for_current=False
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {"default": {}, "current": {}},
        }

        assert graphistry.bind().encode_point_color(
            "z", categorical_mapping={"a": "b"}, for_default=True, for_current=False
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "z",
                        "variation": "categorical",
                        "mapping": {"categorical": {"fixed": {"a": "b"}}},
                    }
                },
                "current": {},
            },
        }

        assert graphistry.bind().encode_point_color(
            "z", categorical_mapping={"a": "b"}, for_default=False, for_current=True
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {
                "default": {},
                "current": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "z",
                        "variation": "categorical",
                        "mapping": {"categorical": {"fixed": {"a": "b"}}},
                    }
                },
            },
        }

        assert graphistry.bind().encode_point_color(
            "z", categorical_mapping={"a": "b"}, for_default=True, for_current=True
        )._complex_encodings == {
            **TestPlotterEncodings.COMPLEX_EMPTY,
            "node_encodings": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "z",
                        "variation": "categorical",
                        "mapping": {"categorical": {"fixed": {"a": "b"}}},
                    }
                },
                "current": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "z",
                        "variation": "categorical",
                        "mapping": {"categorical": {"fixed": {"a": "b"}}},
                    }
                },
            },
        }

    def test_composition(self):
        # chaining + overriding
        out = (
            graphistry.bind()
            .encode_point_size("z", categorical_mapping={"m": 2})
            .encode_point_color("z", categorical_mapping={"a": "b"}, for_current=True)
            .encode_point_color("z", categorical_mapping={"a": "b2"})
            .encode_edge_color("z", categorical_mapping={"x": "y"}, for_current=True)
            ._complex_encodings
        )
        assert out["edge_encodings"]["default"] == {
            "edgeColorEncoding": {
                "graphType": "edge",
                "encodingType": "color",
                "attribute": "z",
                "variation": "categorical",
                "mapping": {"categorical": {"fixed": {"x": "y"}}},
            }
        }
        assert out["edge_encodings"]["current"] == {
            "edgeColorEncoding": {
                "graphType": "edge",
                "encodingType": "color",
                "attribute": "z",
                "variation": "categorical",
                "mapping": {"categorical": {"fixed": {"x": "y"}}},
            }
        }
        assert out["node_encodings"]["default"] == {
            "pointSizeEncoding": {
                "graphType": "point",
                "encodingType": "size",
                "attribute": "z",
                "variation": "categorical",
                "mapping": {"categorical": {"fixed": {"m": 2}}},
            },
            "pointColorEncoding": {
                "graphType": "point",
                "encodingType": "color",
                "attribute": "z",
                "variation": "categorical",
                "mapping": {"categorical": {"fixed": {"a": "b2"}}},
            },
        }
        assert out["node_encodings"]["current"] == {
            "pointColorEncoding": {
                "graphType": "point",
                "encodingType": "color",
                "attribute": "z",
                "variation": "categorical",
                "mapping": {"categorical": {"fixed": {"a": "b"}}},
            }
        }


class TestPlotterMixins(NoAuthTestCase):
    @classmethod
    def setUpClass(cls):
        1

    def test_has_degree(self):
        g = graphistry.bind().edges(
            pd.DataFrame({"s": ["a", "b"], "d": ["b", "a"]}), "s", "d"
        )
        g2 = g.get_degrees()
        assert g2._nodes.to_dict(orient="records") == [
            {"id": "a", "degree_in": 1, "degree_out": 1, "degree": 2},
            {"id": "b", "degree_in": 1, "degree_out": 1, "degree": 2},
        ]
