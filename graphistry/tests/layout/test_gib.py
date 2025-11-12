import logging, os, pandas as pd, pytest, warnings
from graphistry.compute import ComputeMixin
from graphistry.layouts import LayoutsMixin
from graphistry.plotter import PlotterBase
from graphistry.tests.common import NoAuthTestCase
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


test_cudf = "TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"

class LG(LayoutsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        LayoutsMixin.__init__(self, *args, **kwargs)


class LGFull(LayoutsMixin, ComputeMixin, PlotterBase):
    def __init__(self, *args, **kwargs):
        print('LGFull init')
        super(LGFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)
        LayoutsMixin.__init__(self, *args, **kwargs)


class Test_gib(NoAuthTestCase):

    def test_gib_pd(self):
        try:
            import igraph
        except:
            return
        
        lg = LGFull()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            g = (
                lg
                .edges(
                    pd.DataFrame({
                        's': ['a', 'b', 'c', 'd', 'm', 'd1'],
                        'd': ['b', 'c', 'd', 'd', 'm', 'd2']
                    }),
                    's', 'd')
                .group_in_a_box_layout()
            )
        assert isinstance(g._nodes, pd.DataFrame)
        assert isinstance(g._edges, pd.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()

    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1")
    def test_gib_cudf(self):
        import cudf

        lg = LGFull()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            g = (
                lg
                .edges(
                    cudf.DataFrame({
                        's': ['a', 'b', 'c', 'd', 'm', 'd1'],
                        'd': ['b', 'c', 'd', 'd', 'm', 'd2']
                    }),
                    's', 'd')
                .group_in_a_box_layout()
            )
        print('g ::', type(g))
        print('g._nodes ::', type(g._nodes))
        print('g._edges ::', type(g._edges))
        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._edges, cudf.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()

    def test_circle_layout_with_partition_pd(self):
        """Test circle_layout with partition_by parameter (pandas) - tests the fixed code path"""
        lg = LGFull()

        # Create nodes with partition assignments
        nodes = pd.DataFrame({
            'id': [0, 1, 2, 3, 4],
            'partition': [0, 0, 1, 1, 1],
            'x': [0.0, 1.0, 2.0, 3.0, 4.0],
            'y': [0.0, 1.0, 2.0, 3.0, 4.0]
        })

        edges = pd.DataFrame({'src': [0], 'dst': [1]})

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            g = lg.nodes(nodes, 'id').edges(edges, 'src', 'dst')

            # Compute bounding boxes per partition
            groupby_partition = g._nodes.groupby('partition')
            min_x = groupby_partition['x'].min().reset_index()
            max_x = groupby_partition['x'].max().reset_index()
            min_y = groupby_partition['y'].min().reset_index()
            max_y = groupby_partition['y'].max().reset_index()

            bounding_boxes = pd.DataFrame({
                'partition_key': min_x['partition'],
                'cx': (min_x['x'] + max_x['x']) * 0.5,
                'cy': (min_y['y'] + max_y['y']) * 0.5,
                'w': max_x['x'] - min_x['x'],
                'h': max_y['y'] - min_y['y']
            })

            # This calls circle_layout with partition_by, which triggers the fixed code path
            result = g.circle_layout(
                bounding_box=bounding_boxes,
                partition_by='partition',
                engine='pandas'
            )

        assert isinstance(result._nodes, pd.DataFrame)
        assert 'x' in result._nodes
        assert 'y' in result._nodes
        assert not result._nodes.x.isna().any(), "circle_layout produced NaN x coordinates"
        assert not result._nodes.y.isna().any(), "circle_layout produced NaN y coordinates"
        assert len(result._nodes) == 5

    def test_gib_pd_with_partitions(self):
        """Test group_in_a_box_layout with multiple communities - tests full integration path"""
        try:
            import igraph
        except:
            return

        lg = LGFull()

        # Create a graph with distinct communities that will trigger partitioning
        # Community 0: nodes a, b, c
        # Community 1: nodes d, e, f
        edges = pd.DataFrame({
            's': ['a', 'b', 'c', 'a', 'b', 'd', 'e', 'f', 'd', 'e'],
            'd': ['b', 'c', 'a', 'c', 'a', 'e', 'f', 'd', 'f', 'd']
        })

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # group_in_a_box_layout uses community detection and internally calls
            # circle_layout with partition_by, triggering the fixed code path
            g = (
                lg
                .edges(edges, 's', 'd')
                .group_in_a_box_layout()
            )

        assert isinstance(g._nodes, pd.DataFrame)
        assert isinstance(g._edges, pd.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any(), "group_in_a_box_layout produced NaN x coordinates"
        assert not g._nodes.y.isna().any(), "group_in_a_box_layout produced NaN y coordinates"
        # Should have 6 unique nodes
        assert len(g._nodes) == 6

    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1")
    def test_circle_layout_with_partition_cudf(self):
        """Test circle_layout with partition_by parameter (cuDF) - tests the fixed code path for GPU"""
        import cudf

        lg = LGFull()

        # Create nodes with partition assignments
        nodes = cudf.DataFrame({
            'id': [0, 1, 2, 3, 4],
            'partition': [0, 0, 1, 1, 1],
            'x': [0.0, 1.0, 2.0, 3.0, 4.0],
            'y': [0.0, 1.0, 2.0, 3.0, 4.0]
        })

        edges = cudf.DataFrame({'src': [0], 'dst': [1]})

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            g = lg.nodes(nodes, 'id').edges(edges, 'src', 'dst')

            # Compute bounding boxes per partition
            groupby_partition = g._nodes.groupby('partition')
            min_x = groupby_partition['x'].min().reset_index()
            max_x = groupby_partition['x'].max().reset_index()
            min_y = groupby_partition['y'].min().reset_index()
            max_y = groupby_partition['y'].max().reset_index()

            bounding_boxes = cudf.DataFrame({
                'partition_key': min_x['partition'],
                'cx': (min_x['x'] + max_x['x']) * 0.5,
                'cy': (min_y['y'] + max_y['y']) * 0.5,
                'w': max_x['x'] - min_x['x'],
                'h': max_y['y'] - min_y['y']
            })

            # This calls circle_layout with partition_by, which triggers the fixed code path
            # The fix replaced groupby.transform('size') with groupby.size() + map()
            result = g.circle_layout(
                bounding_box=bounding_boxes,
                partition_by='partition',
                engine='cudf'
            )

        assert isinstance(result._nodes, cudf.DataFrame)
        assert 'x' in result._nodes
        assert 'y' in result._nodes
        assert not result._nodes.x.isna().any(), "circle_layout produced NaN x coordinates"
        assert not result._nodes.y.isna().any(), "circle_layout produced NaN y coordinates"
        assert len(result._nodes) == 5

    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1")
    def test_gib_cudf_with_partitions(self):
        """Test group_in_a_box_layout on GPU with multiple communities - tests full integration path"""
        import cudf
        try:
            import igraph
        except:
            pytest.skip("igraph not available")

        lg = LGFull()

        # Create a graph with distinct communities that will trigger partitioning
        # Community 0: nodes a, b, c
        # Community 1: nodes d, e, f
        edges = cudf.DataFrame({
            's': ['a', 'b', 'c', 'a', 'b', 'd', 'e', 'f', 'd', 'e'],
            'd': ['b', 'c', 'a', 'c', 'a', 'e', 'f', 'd', 'f', 'd']
        })

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # group_in_a_box_layout uses community detection and internally calls
            # circle_layout with partition_by, triggering the fixed code path
            g = (
                lg
                .edges(edges, 's', 'd')
                .group_in_a_box_layout()
            )

        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._edges, cudf.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any(), "group_in_a_box_layout produced NaN x coordinates"
        assert not g._nodes.y.isna().any(), "group_in_a_box_layout produced NaN y coordinates"
        # Should have 6 unique nodes
        assert len(g._nodes) == 6
