import igraph
import pyarrow as arrow
import itertools

from ..constants import BINDINGS


def to_arrow(graph):
    if not isinstance(graph, igraph.Graph):
        return None

    nodes = arrow.Table.from_arrays(
        [column for column in itertools.chain(
            _id_columns(graph.vs, BINDINGS.NODE_ID),
            _attribute_columns(graph.vs)
        )]
    )

    edges = arrow.Table.from_arrays(
        [column for column in itertools.chain(
            _id_columns(graph.es, BINDINGS.EDGE_ID),
            _src_dst_columns(graph.es),
            _attribute_columns(graph.es)
        )]
    )

    return (edges, nodes)


def _attribute_columns(sequence):
    for attribute_name in sequence.attributes():
        yield arrow.column(attribute_name, [
            [item[attribute_name] for item in sequence]
        ])


def _id_columns(sequence, id_column_name):
    yield arrow.column(id_column_name, [
        [id for id, _ in enumerate(sequence)]
    ])


def _src_dst_columns(edgeSequence):
    yield arrow.column(BINDINGS.EDGE_SRC, [
        [edge.tuple[0] for edge in edgeSequence]
    ])

    yield arrow.column(BINDINGS.EDGE_DST, [
        [edge.tuple[1] for edge in edgeSequence]
    ])
