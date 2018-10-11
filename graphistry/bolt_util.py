from .pygraphistry import util

node_id_key = u'_bolt_node_id_key'
start_node_id_key = u'_bolt_start_node_id_key'
end_node_id_key = u'_bolt_end_node_id_key'
relationship_id_key = u'_bolt_relationship_id'


def to_bolt_driver(driver):
    try:
        from neo4j import GraphDatabase, Driver
        if driver is None:
            return None
        if isinstance(driver, Driver):
            return driver
        return GraphDatabase.driver(**driver)
    except ImportError:
        raise BoltSupportModuleNotFound()

def bolt_graph_to_edges_dataframe(graph):
    import pandas as pd
    return pd.DataFrame([
        util.merge_two_dicts(
            { key: value for (key, value) in relationship.items() },
            {
                relationship_id_key:    relationship.id,
                start_node_id_key:          relationship.start_node.id,
                end_node_id_key:     relationship.end_node.id
            }
        )
        for relationship in graph.relationships
    ])


def bolt_graph_to_nodes_dataframe(graph):
    import pandas as pd
    return pd.DataFrame([
        util.merge_two_dicts(
            { key: value for (key, value) in node.items() },
            {
                node_id_key: node.id
            }
        )
        for node in graph.nodes
    ])

class BoltSupportModuleNotFound(Exception):
    def __init__(self):
        super(BoltSupportModuleNotFound, self).__init__(
            "The neo4j module was not found but is required for pygraphistry bolt support. Try running `!pip install pygraphistry[bolt]`."
        )
