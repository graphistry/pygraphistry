from graphistry.plotter import Plotter

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

_plotter = Plotter()


def register(**settings):
    Plotter.update_default_settings(settings)


def data(**data):
    global _plotter
    return _plotter.data(**data)


def bind(**bindings):
    """
    Keyword arguments:
        node_id (str): nodeId
        edge_id (str): edgeId
        edge_src (str): source
        edge_dst (str): destination
        edge_color (str): edgeColor
        edge_label (str): edgeLabel
        edge_title (str): edgeTitle
        edge_weight (str): edgeWeight
        node_color (str): pointColor
        node_label (str): pointLabel
        node_title (str): pointTitle
        node_size (str): pointSize
    """
    global _plotter
    return _plotter.bind(**bindings)


def settings(**settings):
    global _plotter
    return _plotter.settings(**settings)


def cypher(*kwargs):
    global _plotter
    return _plotter.cypher(*kwargs)


def hypergraph(
    raw_events,
    entity_types=None,
    opts={},
    drop_na=True,
    drop_edge_attrs=False,
    verbose=True,
    direct=False
):
    """Transform a dataframe into a hypergraph.
    :param Dataframe raw_events: Dataframe to transform
    :param List entity_types: Optional list of columns (strings) to turn into nodes, None signifies all
    :param Dict opts: See below
    :param bool drop_edge_attrs: Whether to include each row's attributes on its edges, defaults to False (include)
    :param bool verbose: Whether to print size information
    :param bool direct: Omit hypernode and instead strongly connect nodes in an event
    Create a graph out of the dataframe, and return the graph components as dataframes,
    and the renderable result Plotter. It reveals relationships between the rows and between column values.
    This transform is useful for lists of events, samples, relationships, and other structured high-dimensional data.
    The transform creates a node for every row, and turns a row's column entries into node attributes.
    If direct=False (default), every unique value within a column is also turned into a node.
    Edges are added to connect a row's nodes to each of its column nodes, or if direct=True, to one another.
    Nodes are given the attribute 'type' corresponding to the originating column name, or in the case of a row, 'EventID'.
    Consider a list of events. Each row represents a distinct event, and each column some metadata about an event.
    If multiple events have common metadata, they will be transitively connected through those metadata values.
    The layout algorithm will try to cluster the events together.
    Conversely, if an event has unique metadata, the unique metadata will turn into nodes that only have connections to the event node, and the clustering algorithm will cause them to form a ring around the event node.
    Best practice is to set EVENTID to a row's unique ID,
    SKIP to all non-categorical columns (or entity_types to all categorical columns),
    and CATEGORY to group columns with the same kinds of values.
    The optional ``opts={...}`` configuration options are:
    * 'EVENTID': Column name to inspect for a row ID. By default, uses the row index.
    * 'CATEGORIES': Dictionary mapping a category name to inhabiting columns. E.g., {'IP': ['srcAddress', 'dstAddress']}.  If the same IP appears in both columns, this makes the transform generate one node for it, instead of one for each column.
    * 'DELIM': When creating node IDs, defines the separator used between the column name and node value
    * 'SKIP': List of column names to not turn into nodes. For example, dates and numbers are often skipped.
    * 'EDGES': For direct=True, instead of making all edges, pick column pairs. E.g., {'a': ['b', 'd'], 'd': ['d']} creates edges between columns a->b and a->d, and self-edges d->d.
    :returns: {'entities': DF, 'events': DF, 'edges': DF, 'nodes': DF, 'graph': Plotter}
    :rtype: Dictionary
    **Example**
        ::
            import graphistry
            h = graphistry.hypergraph(my_df)
            g = h['graph'].plot()
    """
    global _plotter

    from graphistry.hyper import Hypergraph

    return Hypergraph().hypergraph(
        _plotter,
        raw_events,
        entity_types,
        opts,
        drop_na,
        drop_edge_attrs,
        verbose,
        direct
    )
