from graphistry.plotter import Plotter
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

def data(**data):
    return Plotter().data(**data)


def bind(**bindings):
    return Plotter().bind(**bindings)


def settings(**settings):
    return Plotter().settings(**settings)
