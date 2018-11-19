import igraph
import pyarrow as arrow
import itertools

from graphistry.plotter import NODE_ID, EDGE_ID, EDGE_SRC, EDGE_DST


def to_arrow( # TODO(cwharris): move these consts out of here
    graph,
    node_id_column_name=NODE_ID,
    edge_id_column_name=EDGE_ID,
    edge_src_column_name=EDGE_SRC,
    edge_dst_column_name=EDGE_DST
):
    if not isinstance(graph, igraph.Graph):
        return None

    nodes: arrow.Table = arrow.Table.from_arrays(
        [column for column in itertools.chain(
            _id_columns(graph.vs, node_id_column_name),
            _attribute_columns(graph.vs)
        )]
    )

    edges = arrow.Table.from_arrays(
        [column for column in itertools.chain(
            _id_columns(graph.es, edge_id_column_name),
            _src_dst_columns(graph.es, edge_src_column_name, edge_dst_column_name),
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


def _src_dst_columns(edgeSequence, edge_src_column_name, edge_dst_column_name):
    yield arrow.column(edge_src_column_name, [
        [edge.tuple[0] for edge in edgeSequence]
    ])

    yield arrow.column(edge_dst_column_name, [
        [edge.tuple[1] for edge in edgeSequence]
    ])
