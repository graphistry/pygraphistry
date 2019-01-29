import pyarrow
import os


def decompose(graph):
    for decompose in [_decompose_igraph, _decompose_networkx, _decompose_neo4j]:
        try:
            decomposition = decompose(graph)
            if decomposition is not None:
                return decomposition
        except ImportError:
            continue

    raise TypeError("Unsupported Graph: %s" % (type(graph)))


def _decompose_igraph(graph):
    from graphistry.ext.igraph_to_arrow import to_arrow
    return to_arrow(graph)


def _decompose_networkx(graph):
    from graphistry.ext.networkx_to_arrow import to_arrow
    return to_arrow(graph)


def _decompose_neo4j(graph):
    from graphistry.ext.neo4j_to_arrow import to_arrow
    return to_arrow(graph)
