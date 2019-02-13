import pyarrow
import networkx
import itertools

from graphistry.constants import BINDING


def to_arrow(graph, bindings):
    return None if not isinstance(graph, networkx.classes.graph.Graph) else (
        pyarrow.Table.from_arrays([column for column in _edge_columns(graph, bindings)]),
        pyarrow.Table.from_arrays([column for column in _node_columns(graph, bindings)])
    )


def _edge_columns(graph, bindings):
    attribute_names = set(
        key
        for _, _, edgeAttributes in graph.edges(data=True)
        for key in edgeAttributes.keys()
    )

    yield pyarrow.column(bindings.get(BINDING.EDGE_SRC), [
        [srcId for srcId, _ in graph.edges()]
    ])

    yield pyarrow.column(bindings.get(BINDING.EDGE_DST), [
        [dstId for _, dstId in graph.edges()]
    ])

    for attributeName in attribute_names:
        attributeValues = graph.get_node_attributes(attributeName)
        yield pyarrow.column(attributeName, [
            [attributeValues[edge]
                if edge in attributeValues else None for edge in graph.edges()]
        ])


def _node_columns(graph, bindings):
    attribute_names = set(
        key
        for _, nodeAttributes in graph.nodes(data=True)
        for key in nodeAttributes.keys()
    )

    yield pyarrow.column(bindings.get(BINDING.NODE_ID), [
        [nodeId for nodeId in graph.nodes()]
    ])

    for attributeName in attribute_names:
        attributeValues = graph.get_node_attributes(attributeName)
        yield pyarrow.column(attributeName, [
            [attributeValues[node]
                if node in attributeValues else None for node in graph.nodes()]
        ])
