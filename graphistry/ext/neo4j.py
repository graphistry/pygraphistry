import neo4j
import pyarrow as arrow
import itertools

from graphistry.plotter import NODE_ID, EDGE_ID, EDGE_SRC, EDGE_DST

def to_arrow( # TODO(cwharris): move these consts out of here
    graph,
    node_id=NODE_ID,
    edge_id=EDGE_ID,
    edge_src=EDGE_SRC,
    edge_dst=EDGE_DST,
    neo4j_type="__neo4j_type__",
    neo4j_label="__neo4j_label__"
):
    edge_table = _edge_table(
        graph.relationships,
        edge_id,
        edge_src,
        edge_dst,
        neo4j_type
    )

    node_table = _node_table(
        graph.nodes,
        node_id,
        neo4j_label
    )

    return (edge_table, node_table)

def _edge_table(
    relationships,
    edge_id,
    edge_src,
    edge_dst,
    neo4j_type
):
    attribute_names = _attributes_for_entities(relationships)
    return arrow.Table.from_arrays(
        [column for column in itertools.chain(
            _intrinsic_edge_columns(
                relationships=relationships,
                edge_id=edge_id,
                edge_src=edge_src,
                edge_dst=edge_dst,
                neo4j_type=neo4j_type
            ),
            _columns_for_entity(
                entities=relationships,
                entity_attributes=attribute_names
            )
        )]
    )

def _node_table(
    nodes,
    node_id,
    neo4j_label
):
    attribute_names = _attributes_for_entities(nodes)
    return arrow.Table.from_arrays(
        [column for column in itertools.chain(
            _intrinsic_node_columns(
                nodes=nodes,
                node_id=node_id,
                neo4j_label=neo4j_label
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

def _intrinsic_edge_columns(
    relationships,
    edge_id,
    edge_src,
    edge_dst,
    neo4j_type
):
    # TODO(cwharris): remove the string conversion once server can haandle non-ascending integers.
    # currently, ids will be remapped as part of pre-plot rectification.
    yield arrow.column(edge_id, [
        [str(relationship.id) for relationship in relationships]
    ])

    yield arrow.column(edge_src, [
        [str(relationship.start_node.id) for relationship in relationships]
    ])

    yield arrow.column(edge_dst, [
        [str(relationship.end_node.id) for relationship in relationships]
    ])

    yield arrow.column(neo4j_type, [
        [relationship.type for relationship in relationships]
    ])

def _intrinsic_node_columns(
    nodes,
    node_id,
    neo4j_label
):
    # TODO(cwharris): remove the string conversion once server can haandle non-ascending integers.
    # currently, ids will be remapped as part of pre-plot rectification.
    yield arrow.column(NODE_ID, [
        [str(node.id) for node in nodes]
    ])

    yield arrow.column(neo4j_label, [
        [list(node.labels) for node in nodes]
    ])
