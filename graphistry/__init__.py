from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from graphistry.pygraphistry import register, bind, edges, nodes, graph, settings, hypergraph, bolt, cypher
