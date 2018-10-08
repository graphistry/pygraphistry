from .pygraphistry import util


def _to_bolt_driver(driver):
    if driver is None:
        return None
    try:
        from neo4j import GraphDatabase, Driver
        if isinstance(driver, Driver):
            return driver
        return GraphDatabase.driver(**driver)
    except ImportError:
        return None


def _bolt_graph_to_dataframe(graph):
    import pandas as pd
    return pd.DataFrame([
        util.merge_two_dicts(
            { key: value for (key, value) in relationship.items() },
            {
                u'_bolt_relationship_id': relationship.id,
                u'_bolt_source_id': relationship.start_node.id,
                u'_bolt_destination_id': relationship.end_node.id
            }
        )
        for relationship in graph.relationships
    ])
