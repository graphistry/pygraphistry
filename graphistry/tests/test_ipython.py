import graphistry, IPython
from common import NoAuthTestCase
from mock import patch
from graphistry.render.resolve_render_mode import resolve_cascaded
from graphistry.tests.test_plotter import Fake_Response, triangleEdges


@patch("webbrowser.open")
@patch("requests.post", return_value=Fake_Response())
class TestPlotterReturnValue(NoAuthTestCase):

    @patch("graphistry.render.resolve_render_mode.in_ipython")
    def test_no_ipython(self, mock_in_ipython, mock_post, mock_open):
        resolve_cascaded.cache_clear()
        mock_in_ipython.return_value = False
        url = graphistry.bind(source="src", destination="dst").plot(triangleEdges, render="browser")
        self.assertIn("fakedatasetname", url)
        # API v3 generates viztoken client-side (UUID), so we check it's present rather than value
        self.assertIn("viztoken=", url)
        self.assertTrue(mock_open.called)
        self.assertTrue(mock_post.called)

    @patch("graphistry.render.resolve_render_mode.in_ipython")
    def test_ipython(self, mock_in_ipython, mock_post, mock_open):
        resolve_cascaded.cache_clear()
        mock_in_ipython.return_value = True

        # The setUpClass in NoAuthTestCase only run once, so, reset the _is_authenticated to True here
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True

        widget = graphistry.bind(source="src", destination="dst").plot(triangleEdges)
        self.assertIsInstance(widget, IPython.core.display.HTML)

        widget = graphistry.bind(source="src", destination="dst").plot(triangleEdges, render="ipython")
        self.assertIsInstance(widget, IPython.core.display.HTML)
