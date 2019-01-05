import neo4j
import itertools
import pyarrow as arrow
import graphistry

def get_graph(query, args={}):
    driver = neo4j.GraphDatabase.driver("bolt://neo4j:7687")
    with driver.session() as session:
        return session.run(query, args).graph()

def test_hey():
    graph = get_graph("MATCH (a)-[r:PAYMENT]->(b) RETURN distinct * LIMIT 1")
    plotter = graphistry \
        .data(graph=graph) \
        .bind(
            node_id='__node_id__',
            edge_id='__edge_id__',
            edge_src='__edge_src__',
            edge_dst='__edge_dst__'
        )

    plotter.plot()

    for column in plotter._data['edges']:
        for value in column:
            pass

# def _node_fields(node: neo4j.Node): # TODO(cwharris) make keys configurable
#     yield ("__NODE_ID__", node.id)
#     yield ("__NEO4J_LABEL__", node.labels)
#     for item in node.items():
#         yield item

# def _relationship_fields(relationship: neo4j.Relationship): # TODO(cwharris) make keys configurable
#     yield ("__EDGE_ID__", relationship.id)
#     yield ("__EDGE_SRC__", relationship.start_node.id)
#     yield ("__EDGE_DST__", relationship.end_node.id)
#     yield ("__NEO4J_TYPE__", relationship.type)
#     for item in relationship.items():
#         yield item
