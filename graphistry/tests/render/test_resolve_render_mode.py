import logging, os, pandas as pd, pytest, warnings
from graphistry.compute import ComputeMixin
from graphistry.layouts import LayoutsMixin
from graphistry.plotter import PlotterBase
from graphistry.render import resolve_render_mode
from graphistry.tests.common import NoAuthTestCase
from graphistry.tests.test_compute import CGFull
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

    # concrete
    for mode in ['g', 'url', 'browser', 'ipython', 'databricks']:
        assert resolve_render_mode(abc_g, mode) == mode

def test_resolve_sniffed(abc_g):

    # bool
    assert resolve_render_mode(abc_g, True) in ['url', 'ipython', 'databricks', 'browser']
    assert resolve_render_mode(abc_g, False) == 'url'

    # none
    assert resolve_render_mode(abc_g, None) in ['url', 'ipython', 'databricks', 'browser']

def test_resolve_cascade(abc_g):

    assert resolve_render_mode(abc_g.settings(render='g'), None) == 'g'
    assert resolve_render_mode(abc_g.settings(render='g'), 'url') == 'url'
    assert resolve_render_mode(abc_g.settings(render='g'), True) in ['url', 'ipython', 'databricks', 'browser']
    assert resolve_render_mode(abc_g.settings(render='g'), False) == 'url'
