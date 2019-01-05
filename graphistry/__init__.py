from graphistry.plotter import Plotter

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

_plotter = Plotter()


def register(**settings):
    Plotter.update_default_settings(settings)


def data(**data):
    global _plotter
    return _plotter.data(**data)


def bind(**bindings):
    global _plotter
    return _plotter.bind(**bindings)


def settings(**settings):
    global _plotter
    return _plotter.settings(**settings)


def cypher(*kwargs):
    global _plotter
    return _plotter.cypher(*kwargs)
