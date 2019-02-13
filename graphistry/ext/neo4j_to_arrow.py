import neo4j
import pyarrow as arrow
import itertools

from graphistry.constants import BINDING, BINDING_DEFAULT


def to_arrow(graph, bindings):
    if not isinstance(graph, neo4j.types.graph.Graph):
        return None

    return (
        _edge_table(graph.relationships, bindings),
        _node_table(graph.nodes, bindings)
    )


def _edge_table(relationships, bindings):
    attribute_names = _attributes_for_entities(relationships)
    return arrow.Table.from_arrays(
        [column for column in itertools.chain(
            _intrinsic_edge_columns(
                relationships=relationships,
                bindings=bindings
            ),
            _columns_for_entity(
                entities=relationships,
                entity_attributes=attribute_names
            )
        )]
    )


def _node_table(nodes, bindings):
    attribute_names = _attributes_for_entities(nodes)
    return arrow.Table.from_arrays(
        [column for column in itertools.chain(
            _intrinsic_node_columns(
                nodes=nodes,
                bindings=bindings
            ),
            _columns_for_entity(
                entities=nodes,
                entity_attributes=attribute_names
            )
        )]
    )


def _attributes_for_entities(entities):
    return set(
        key for entity in entities for key in entity.keys()
    )


def _columns_for_entity(
    entities,
    entity_attributes
):
    for attribute in entity_attributes:
        yield arrow.column(attribute, [
            [entity[attribute] if attribute in entity else None for entity in entities]
        ])


def _intrinsic_edge_columns(relationships, bindings):
    # TODO(cwharris): remove the string conversion once server can haandle non-ascending integers.
    # currently, ids will be remapped as part of pre-plot rectification.
    yield arrow.column(bindings.get(BINDING.EDGE_ID), [
        [str(relationship.id) for relationship in relationships]
    ])

    yield arrow.column(bindings.get(BINDING.EDGE_SRC), [
        [str(relationship.start_node.id) for relationship in relationships]
    ])

    yield arrow.column(bindings.get(BINDING.EDGE_DST), [
        [str(relationship.end_node.id) for relationship in relationships]
    ])

    yield arrow.column(BINDING_DEFAULT.NEO4J_TYPE, [
        [relationship.type for relationship in relationships]
    ])


def _intrinsic_node_columns(nodes, bindings):
    # TODO(cwharris): remove the string conversion once server can haandle non-ascending integers.
    # currently, ids will be remapped as part of pre-plot rectification.
    yield arrow.column(bindings.get(BINDING.NODE_ID), [
        [str(node.id) for node in nodes]
    ])

    yield arrow.column(BINDING_DEFAULT.NEO4J_LABEL, [
        [list(node.labels) for node in nodes]
    ])
