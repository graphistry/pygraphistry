import importlib
import logging, pandas as pd, pytest
import sys
from mock import patch
from unittest.mock import Mock

import graphistry.render
from graphistry.render.resolve_render_mode import resolve_cascaded, resolve_render_mode
from graphistry.tests.common import NoAuthTestCase
from graphistry.tests.test_compute import CGFull
from graphistry.tests.test_plotter import Fake_Response
logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def abc_g() -> CGFull:
    """
    a->b->c
    """
    return (CGFull()
        .edges(pd.DataFrame({
            's': ['a', 'b',],
            'd': ['b', 'c']}),
            's', 'd'))

def test_resolve_concrete(abc_g):
    resolve_cascaded.cache_clear()

    # concrete
    for mode in ['g', 'url', 'browser', 'ipython', 'databricks']:
        assert resolve_render_mode(abc_g, mode) == mode

def test_resolve_sniffed(abc_g):
    resolve_cascaded.cache_clear()

    # bool
    assert resolve_render_mode(abc_g, True) in ['url', 'ipython', 'databricks', 'browser']
    assert resolve_render_mode(abc_g, False) == 'url'

    # none
    assert resolve_render_mode(abc_g, None) in ['url', 'ipython', 'databricks', 'browser']

def test_resolve_cascade(abc_g):
    resolve_cascaded.cache_clear()

    assert resolve_render_mode(abc_g.settings(render='g'), None) == 'g'
    assert resolve_render_mode(abc_g.settings(render='g'), 'url') == 'url'
    assert resolve_render_mode(abc_g.settings(render='g'), True) in ['url', 'ipython', 'databricks', 'browser']
    assert resolve_render_mode(abc_g.settings(render='g'), False) == 'url'


@patch("graphistry.render.resolve_render_mode.in_ipython")
class TestIPython(NoAuthTestCase):

    def test_no_ipython(self, mock_in_ipython):
        resolve_cascaded.cache_clear()
        mock_in_ipython.return_value = False

        mode_render_true = resolve_render_mode(abc_g, True)
        self.assertNotEqual(mode_render_true, 'ipython')

        mode_render_false = resolve_render_mode(abc_g, False)
        self.assertEqual(mode_render_false, 'url')

        mode_render_ipython = resolve_render_mode(abc_g, 'ipython')
        self.assertEqual(mode_render_ipython, 'ipython')

    def test_ipython(self, mock_in_ipython):
        resolve_cascaded.cache_clear()
        mock_in_ipython.return_value = True

        mode_render_true = resolve_render_mode(abc_g, True)
        self.assertEqual(mode_render_true, 'ipython')

        mode_render_false = resolve_render_mode(abc_g, False)
        self.assertEqual(mode_render_false, 'url')

        mode_render_ipython = resolve_render_mode(abc_g, 'ipython')
        self.assertEqual(mode_render_ipython, 'ipython')


@patch("graphistry.render.resolve_render_mode.in_databricks")
class TestDatabricks(NoAuthTestCase):

    def test_no_databricks(self, mock_in_databricks):
        resolve_cascaded.cache_clear()
        mock_in_databricks.return_value = False

        mode_render_true = resolve_render_mode(abc_g, True)
        self.assertNotEqual(mode_render_true, 'databricks')

        mode_render_false = resolve_render_mode(abc_g, False)
        self.assertEqual(mode_render_false, 'url')

        mode_render_ipython = resolve_render_mode(abc_g, 'databricks')
        self.assertEqual(mode_render_ipython, 'databricks')

    def test_ipython(self, mock_in_databricks):
        resolve_cascaded.cache_clear()
        mock_in_databricks.return_value = True

        mode_render_true = resolve_render_mode(abc_g, True)
        self.assertEqual(mode_render_true, 'databricks')

        mode_render_false = resolve_render_mode(abc_g, False)
        self.assertEqual(mode_render_false, 'url')

        mode_render_ipython = resolve_render_mode(abc_g, 'databricks')
        self.assertEqual(mode_render_ipython, 'databricks')
