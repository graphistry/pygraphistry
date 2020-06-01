from __future__ import absolute_import, print_function
from builtins import object, str
import copy, numpy, pandas, pyarrow as pa, sys, uuid

from .pygraphistry import PyGraphistry
from .pygraphistry import util
from .pygraphistry import bolt_util
from .nodexlistry import NodeXLGraphistry
from .tigeristry import Tigeristry
from .arrow_uploader import ArrowUploader

maybe_cudf = None
try:
    import cudf
    maybe_cudf = cudf
except ImportError:
    1

class Plotter(object):
    """Graph plotting class.

    Created using ``Graphistry.bind()``.

    Chained calls successively add data and visual encodings, and end with a plot call.

    To streamline reuse and replayable notebooks, Plotter manipulations are immutable. Each chained call returns a new instance that derives from the previous one. The old plotter or the new one can then be used to create different graphs.

    The class supports convenience methods for mixing calls across Pandas, NetworkX, and IGraph.
    """


    _defaultNodeId = '__nodeid__'


    def __init__(self):
        # Bindings
        self._edges = None
        self._nodes = None
        self._source = None
        self._destination = None
        self._node = None
        self._edge_title = None
        self._edge_label = None
        self._edge_color = None
        self._edge_size = None
        self._edge_weight = None
        self._edge_icon = None
        self._edge_opacity = None
        self._point_title = None
        self._point_label = None
        self._point_color = None
        self._point_size = None
        self._point_weight = None
        self._point_icon = None
        self._point_opacity = None
        # Settings
        self._height = 500
        self._render = True
        self._url_params = {'info': 'true'}
        # Integrations
        self._bolt_driver = None
        self._tigergraph = None


    def __repr__(self):
        bindings = ['edges', 'nodes', 'source', 'destination', 'node', 
                    'edge_label', 'edge_color', 'edge_size', 'edge_weight', 'edge_title', 'edge_icon', 'edge_opacity',
                    'point_label', 'point_color', 'point_size', 'point_weight', 'point_title', 'point_icon', 'point_opacity']
        settings = ['height', 'url_params']

        rep = {'bindings': dict([(f, getattr(self, '_' + f)) for f in bindings]),
               'settings': dict([(f, getattr(self, '_' + f)) for f in settings])}
        if util.in_ipython():
            from IPython.lib.pretty import pretty
            return pretty(rep)
        else:
            return str(rep)


    def bind(self, source=None, destination=None, node=None,
             edge_title=None, edge_label=None, edge_color=None, edge_weight=None, edge_size=None, edge_opacity=None, edge_icon=None,
             point_title=None, point_label=None, point_color=None, point_weight=None, point_size=None, point_opacity=None, point_icon=None):
        """Relate data attributes to graph structure and visual representation.

        To facilitate reuse and replayable notebooks, the binding call is chainable. Invocation does not effect the old binding: it instead returns a new Plotter instance with the new bindings added to the existing ones. Both the old and new bindings can then be used for different graphs.


        :param source: Attribute containing an edge's source ID
        :type source: String.

        :param destination: Attribute containing an edge's destination ID
        :type destination: String.

        :param node: Attribute containing a node's ID
        :type node: String.

        :param edge_title: Attribute overriding edge's minimized label text. By default, the edge source and destination is used.
        :type edge_title: HtmlString.

        :param edge_label: Attribute overriding edge's expanded label text. By default, scrollable list of attribute/value mappings.
        :type edge_label: HtmlString.

        :param edge_color: Attribute overriding edge's color. `See palette definitions <https://graphistry.github.io/docs/legacy/api/0.9.2/api.html#extendedpalette>`_ for values. Based on Color Brewer.
        :type edge_color: String.

        :param edge_weight: Attribute overriding edge weight. Default is 1. Advanced layout controls will relayout edges based on this value.
        :type edge_weight: String.

        :param point_title: Attribute overriding node's minimized label text. By default, the node ID is used.
        :type point_title: HtmlString.

        :param point_label: Attribute overriding node's expanded label text. By default, scrollable list of attribute/value mappings.
        :type point_label: HtmlString.

        :param point_color: Attribute overriding node's color. `See palette definitions <https://graphistry.github.io/docs/legacy/api/0.9.2/api.html#extendedpalette>`_ for values. Based on Color Brewer.
        :type point_color: Integer.

        :param point_size: Attribute overriding node's size. By default, uses the node degree. The visualization will normalize point sizes and adjust dynamically using semantic zoom.
        :type point_size: HtmlString.

        :returns: Plotter.
        :rtype: Plotter.

        **Example: Minimal**
            ::

                import graphistry
                g = graphistry.bind()
                g = g.bind(source='src', destination='dst')

        **Example: Node colors**
            ::

                import graphistry
                g = graphistry.bind()
                g = g.bind(source='src', destination='dst',
                           node='id', point_color='color')

        **Example: Chaining**
            ::

                import graphistry
                g = graphistry.bind(source='src', destination='dst', node='id')

                g1 = g.bind(point_color='color1', point_size='size1')

                g.bind(point_color='color1b')

                g2a = g1.bind(point_color='color2a')
                g2b = g1.bind(point_color='color2b', point_size='size2b')

                g3a = g2a.bind(point_size='size3a')
                g3b = g2b.bind(point_size='size3b')

        In the above **Chaining** example, all bindings use src/dst/id. Colors and sizes bind to:
            ::

                g: default/default
                g1: color1/size1
                g2a: color2a/size1
                g2b: color2b/size2b
                g3a: color2a/size3a
                g3b: color2b/size3b


        """
        res = copy.copy(self)
        res._source = source or self._source
        res._destination = destination or self._destination
        res._node = node or self._node

        res._edge_title = edge_title or self._edge_title
        res._edge_label = edge_label or self._edge_label
        res._edge_color = edge_color or self._edge_color
        res._edge_size = edge_size or self._edge_size
        res._edge_weight = edge_weight or self._edge_weight
        res._edge_icon = edge_icon or self._edge_icon
        res._edge_opacity = edge_opacity or self._edge_opacity

        res._point_title = point_title or self._point_title
        res._point_label = point_label or self._point_label
        res._point_color = point_color or self._point_color
        res._point_size = point_size or self._point_size
        res._point_weight = point_weight or self._point_weight
        res._point_opacity = point_opacity or self._point_opacity
        res._point_icon = point_icon or self._point_icon

        return res


    def nodes(self, nodes):
        """Specify the set of nodes and associated data.

        Must include any nodes referenced in the edge list.

        :param nodes: Nodes and their attributes.
        :type point_size: Pandas dataframe

        :returns: Plotter.
        :rtype: Plotter.

        **Example**
            ::

                import graphistry

                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                g = graphistry
                    .bind(source='src', destination='dst')
                    .edges(es)

                vs = pandas.DataFrame({'v': [0,1,2], 'lbl': ['a', 'b', 'c']})
                g = g.bind(node='v').nodes(vs)

                g.plot()

        """


        res = copy.copy(self)
        res._nodes = nodes
        return res


    def edges(self, edges):
        """Specify edge list data and associated edge attribute values.

        :param edges: Edges and their attributes.
        :type point_size: Pandas dataframe, NetworkX graph, or IGraph graph.

        :returns: Plotter.
        :rtype: Plotter.

        **Example**
            ::

                import graphistry
                df = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                graphistry
                    .bind(source='src', destination='dst')
                    .edges(df)
                    .plot()

        """

        res = copy.copy(self)
        res._edges = edges
        return res


    def graph(self, ig):
        """Specify the node and edge data.

        :param ig: Graph with node and edge attributes.
        :type ig: NetworkX graph or an IGraph graph.

        :returns: Plotter.
        :rtype: Plotter.
        """

        res = copy.copy(self)
        res._edges = ig
        res._nodes = None
        return res


    def settings(self, height=None, url_params={}, render=None):
        """Specify iframe height and add URL parameter dictionary.

        The library takes care of URI component encoding for the dictionary.

        :param height: Height in pixels.
        :type height: Integer.

        :param url_params: Dictionary of querystring parameters to append to the URL.
        :type url_params: Dictionary

        :param render: Whether to render the visualization using the native notebook environment (default True), or return the visualization URL
        :type render: Boolean

        """

        res = copy.copy(self)
        res._height = height or self._height
        res._url_params = dict(self._url_params, **url_params)
        res._render = self._render if render == None else render
        return res


    def plot(self, graph=None, nodes=None, name=None, render=None, skip_upload=False):
        """Upload data to the Graphistry server and show as an iframe of it.

        name, Uses the currently bound schema structure and visual encodings.
        Optional parameters override the current bindings.

        When used in a notebook environment, will also show an iframe of the visualization.

        :param graph: Edge table or graph.
        :type graph: Pandas dataframe, NetworkX graph, or IGraph graph.

        :param nodes: Nodes table.
        :type nodes: Pandas dataframe.

        :param render: Whether to render the visualization using the native notebook environment (default True), or return the visualization URL
        :type render: Boolean

        :param skip_upload: Return node/edge/bindings that would have been uploaded. By default, upload happens.
        :type skip_upload: Boolean. 

        **Example: Simple**
            ::

                import graphistry
                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                graphistry
                    .bind(source='src', destination='dst')
                    .edges(es)
                    .plot()

        **Example: Shorthand**
            ::

                import graphistry
                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                graphistry
                    .bind(source='src', destination='dst')
                    .plot(es)

        """

        if graph is None:
            if self._edges is None:
                util.error('Graph/edges must be specified.')
            g = self._edges
        else:
            g = graph
        n = self._nodes if nodes is None else nodes
        name = name or util.random_string(10)

        self._check_mandatory_bindings(not isinstance(n, type(None)))

        api_version = PyGraphistry.api_version()
        if api_version == 1:
            dataset = self._plot_dispatch(g, n, name, 'json')
            if skip_upload:
                return dataset
            info = PyGraphistry._etl1(dataset)
        elif api_version == 2:
            dataset = self._plot_dispatch(g, n, name, 'vgraph')
            if skip_upload:
                return dataset
            info = PyGraphistry._etl2(dataset)
        elif api_version == 3:
            dataset = self._plot_dispatch(g, n, name, 'arrow')
            if skip_upload:
                return dataset
            #fresh
            dataset.token = PyGraphistry.api_token()
            dataset.post()
            info = {
                'name': dataset.dataset_id,
                'type': 'arrow',
                'viztoken': str(uuid.uuid4())
            }

        viz_url = PyGraphistry._viz_url(info, self._url_params)
        cfg_client_protocol_hostname = PyGraphistry._config['client_protocol_hostname']
        full_url = ('%s:%s' % (PyGraphistry._config['protocol'], viz_url)) if cfg_client_protocol_hostname is None else viz_url

        if render == False or (render == None and not self._render):
            return full_url
        elif util.in_ipython():
            from IPython.core.display import HTML
            return HTML(util.make_iframe(full_url, self._height))
        else:
            import webbrowser
            webbrowser.open(full_url)
            return full_url


    def pandas2igraph(self, edges, directed=True):
        """Convert a pandas edge dataframe to an IGraph graph.

        Uses current bindings. Defaults to treating edges as directed.

        **Example**
            ::

                import graphistry
                g = graphistry.bind()

                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                g = g.bind(source='src', destination='dst')

                ig = g.pandas2igraph(es)
                ig.vs['community'] = ig.community_infomap().membership
                g.bind(point_color='community').plot(ig)
        """


        import igraph
        self._check_mandatory_bindings(False)
        self._check_bound_attribs(edges, ['source', 'destination'], 'Edge')
        
        self._node = self._node or Plotter._defaultNodeId
        eattribs = edges.columns.values.tolist()
        eattribs.remove(self._source)
        eattribs.remove(self._destination)
        cols = [self._source, self._destination] + eattribs
        etuples = [tuple(x) for x in edges[cols].values]
        return igraph.Graph.TupleList(etuples, directed=directed, edge_attrs=eattribs,
                                      vertex_name_attr=self._node)


    def igraph2pandas(self, ig):
        """Under current bindings, transform an IGraph into a pandas edges dataframe and a nodes dataframe.

        **Example**
            ::

                import graphistry
                g = graphistry.bind()

                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                g = g.bind(source='src', destination='dst').edges(es)

                ig = g.pandas2igraph(es)
                ig.vs['community'] = ig.community_infomap().membership

                (es2, vs2) = g.igraph2pandas(ig)
                g.nodes(vs2).bind(point_color='community').plot()
        """

        def get_edgelist(ig):
            idmap = dict(enumerate(ig.vs[self._node]))
            for e in ig.es:
                t = e.tuple
                yield dict({self._source: idmap[t[0]], self._destination: idmap[t[1]]},
                            **e.attributes())

        self._check_mandatory_bindings(False)
        if self._node is None:
            ig.vs[Plotter._defaultNodeId] = [v.index for v in ig.vs]
            self._node = Plotter._defaultNodeId
        elif self._node not in ig.vs.attributes():
            util.error('Vertex attribute "%s" bound to "node" does not exist.' % self._node)

        edata = get_edgelist(ig)
        ndata = [v.attributes() for v in ig.vs]
        nodes = pandas.DataFrame(ndata, columns=ig.vs.attributes())
        cols = [self._source, self._destination] + ig.es.attributes()
        edges = pandas.DataFrame(edata, columns=cols)
        return (edges, nodes)


    def networkx_checkoverlap(self, g):

        import networkx as nx
        [x, y] = [int(x) for x in nx.__version__.split('.')]

        vattribs = None
        if x == 1:
            vattribs = g.nodes(data=True)[0][1] if g.number_of_nodes() > 0 else []
        else:
            vattribs = g.nodes(data=True) if g.number_of_nodes() > 0 else []
        if not (self._node is None) and self._node in vattribs:
            util.error('Vertex attribute "%s" already exists.' % self._node)

    def networkx2pandas(self, g):

        def get_nodelist(g):
            for n in g.nodes(data=True):
                yield dict({self._node: n[0]}, **n[1])
        def get_edgelist(g):
            for e in g.edges(data=True):
                yield dict({self._source: e[0], self._destination: e[1]}, **e[2])

        self._check_mandatory_bindings(False)
        self.networkx_checkoverlap(g)
        
        self._node = self._node or Plotter._defaultNodeId
        nodes = pandas.DataFrame(get_nodelist(g))
        edges = pandas.DataFrame(get_edgelist(g))
        return (edges, nodes)


    def _check_mandatory_bindings(self, node_required):
        if self._source is None or self._destination is None:
            util.error('Both "source" and "destination" must be bound before plotting.')
        if node_required and self._node is None:
            util.error('Node identifier must be bound when using node dataframe.')


    def _check_bound_attribs(self, df, attribs, typ):
        cols = df.columns.values.tolist()
        for a in attribs:
            b = getattr(self, '_' + a)
            if b not in cols:
                util.error('%s attribute "%s" bound to "%s" does not exist.' % (typ, a, b))


    def _plot_dispatch(self, graph, nodes, name, mode='json'):

        if isinstance(graph, pandas.core.frame.DataFrame) \
            or isinstance(graph, pa.Table) \
            or ( not (maybe_cudf is None) and isinstance(graph, maybe_cudf.DataFrame) ):
            return self._make_dataset(graph, nodes, name, mode)

        try:
            import igraph
            if isinstance(graph, igraph.Graph):
                (e, n) = self.igraph2pandas(graph)
                return self._make_dataset(e, n, name, mode)
        except ImportError:
            pass

        try:
            import networkx
            if isinstance(graph, networkx.classes.graph.Graph) or \
               isinstance(graph, networkx.classes.digraph.DiGraph) or \
               isinstance(graph, networkx.classes.multigraph.MultiGraph) or \
               isinstance(graph, networkx.classes.multidigraph.MultiDiGraph):
                (e, n) = self.networkx2pandas(graph)
                return self._make_dataset(e, n, name, mode)
        except ImportError:
            pass

        util.error('Expected Pandas/Arrow/cuDF dataframe(s) or Igraph/NetworkX graph.')


    # Sanitize node/edge dataframe by
    # - dropping indices
    # - dropping edges with NAs in source or destination
    # - dropping nodes with NAs in nodeid
    # - creating a default node table if none was provided.
    # - inferring numeric types of all columns containing numpy objects
    def _sanitize_dataset(self, edges, nodes, nodeid):
        self._check_bound_attribs(edges, ['source', 'destination'], 'Edge')
        elist = edges.reset_index(drop=True) \
                     .dropna(subset=[self._source, self._destination])

        obj_df = elist.select_dtypes(include=[numpy.object_])
        elist[obj_df.columns] = obj_df.apply(pandas.to_numeric, errors='ignore')

        if nodes is None:
            nodes = pandas.DataFrame()
            nodes[nodeid] = pandas.concat([edges[self._source], edges[self._destination]],
                                           ignore_index=True).drop_duplicates()
        else:
            self._check_bound_attribs(nodes, ['node'], 'Vertex')

        nlist = nodes.reset_index(drop=True) \
                     .dropna(subset=[nodeid]) \
                     .drop_duplicates(subset=[nodeid])

        obj_df = nlist.select_dtypes(include=[numpy.object_])
        nlist[obj_df.columns] = obj_df.apply(pandas.to_numeric, errors='ignore')

        return (elist, nlist)


    def _check_dataset_size(self, elist, nlist):
        edge_count = len(elist.index)
        node_count = len(nlist.index)
        graph_size = edge_count + node_count
        if edge_count > 8e6:
            util.error('Maximum number of edges (8M) exceeded: %d.' % edge_count)
        if node_count > 8e6:
            util.error('Maximum number of nodes (8M) exceeded: %d.' % node_count)
        if graph_size > 1e6:
            util.warn('Large graph: |nodes| + |edges| = %d. Layout/rendering might be slow.' % graph_size)


    # Bind attributes for ETL1 by creating a copy of the designated column renamed
    # with magic names understood by ETL1 (eg. pointColor, etc)
    def _bind_attributes_v1(self, edges, nodes):
        def bind(df, pbname, attrib, default=None):
            bound = getattr(self, attrib)
            if bound:
                if bound in df.columns.tolist():
                    df[pbname] = df[bound]
                else:
                    util.warn('Attribute "%s" bound to %s does not exist.' % (bound, attrib))
            elif default:
                df[pbname] = df[default]

        nodeid = self._node or Plotter._defaultNodeId
        (elist, nlist) = self._sanitize_dataset(edges, nodes, nodeid)
        self._check_dataset_size(elist, nlist)

        bind(elist, 'edgeColor', '_edge_color')
        bind(elist, 'edgeLabel', '_edge_label')
        bind(elist, 'edgeTitle', '_edge_title')
        bind(elist, 'edgeSize', '_edge_size')
        bind(elist, 'edgeWeight', '_edge_weight')
        bind(elist, 'edgeOpacity', '_edge_opacity')
        bind(elist, 'edgeIcon', '_edge_icon')
        bind(nlist, 'pointColor', '_point_color')
        bind(nlist, 'pointLabel', '_point_label')
        bind(nlist, 'pointTitle', '_point_title', nodeid)
        bind(nlist, 'pointSize', '_point_size')
        bind(nlist, 'pointWeight', '_point_weight')
        bind(nlist, 'pointOpacity', '_point_opacity')
        bind(nlist, 'pointIcon', '_point_icon')
        return (elist, nlist)

    # Bind attributes for ETL2 by an encodings map storing the visual semantic of
    # each bound column.
    def _bind_attributes_v2(self, edges, nodes):
        def bind(enc, df, pbname, attrib, default=None):
            bound = getattr(self, attrib)
            if bound:
                if bound in df.columns.tolist():
                    enc[pbname] = {'attributes' : [bound]}
                else:
                    util.warn('Attribute "%s" bound to %s does not exist.' % (bound, attrib))
            elif default:
                enc[pbname] = {'attributes': [default]}

        nodeid = self._node or Plotter._defaultNodeId
        (elist, nlist) = self._sanitize_dataset(edges, nodes, nodeid)
        self._check_dataset_size(elist, nlist)

        edge_encodings = {
            'source': {'attributes' : [self._source]},
            'destination': {'attributes': [self._destination]},
        }
        node_encodings = {
            'nodeId': {'attributes': [nodeid]}
        }
        bind(edge_encodings, elist, 'edgeColor', '_edge_color')
        bind(edge_encodings, elist, 'edgeLabel', '_edge_label')
        bind(edge_encodings, elist, 'edgeTitle', '_edge_title')
        bind(edge_encodings, elist, 'edgeSize', '_edge_size')
        bind(edge_encodings, elist, 'edgeWeight', '_edge_weight')
        bind(edge_encodings, elist, 'edgeOpacity', '_edge_opacity')
        bind(edge_encodings, elist, 'edgeIcon', '_edge_icon')
        bind(node_encodings, nlist, 'pointColor', '_point_color')
        bind(node_encodings, nlist, 'pointLabel', '_point_label')
        bind(node_encodings, nlist, 'pointTitle', '_point_title', nodeid)
        bind(node_encodings, nlist, 'pointSize', '_point_size')
        bind(node_encodings, nlist, 'pointWeight', '_point_weight')
        bind(node_encodings, nlist, 'pointOpacity', '_point_opacity')
        bind(node_encodings, nlist, 'pointIcon', '_point_icon')

        encodings = {
            'nodes': node_encodings,
            'edges': edge_encodings
        }
        return (elist, nlist, encodings)

    def _table_to_pandas(self, table) -> pandas.DataFrame:

        if table is None:
            return table

        if isinstance(table, pandas.DataFrame):
            return table

        if isinstance(table, pa.Table):
            return table.to_pandas()
        
        if not (maybe_cudf is None) and isinstance(table, maybe_cudf.DataFrame):
            return table.to_pandas()
        
        raise Exception('Unknown type %s: Could not convert data to Pandas dataframe' % str(type(table)))

    def _table_to_arrow(self, table) -> pa.Table:

        if table is None:
            return table

        if isinstance(table, pa.Table):
            return table
        
        if isinstance(table, pandas.DataFrame):
            return pa.Table.from_pandas(table, preserve_index=False).replace_schema_metadata({})

        if not (maybe_cudf is None) and isinstance(table, maybe_cudf.DataFrame):
            return table.to_arrow()
        
        raise Exception('Unknown type %s: Could not convert data to Arrow' % str(type(table)))


    def _make_dataset(self, edges, nodes, name, mode):
        try:
            if len(edges) == 0:
                util.warn('Graph has no edges, may have rendering issues')
        except:
            1

        if mode == 'json':
            edges_df = self._table_to_pandas(edges)
            nodes_df = self._table_to_pandas(nodes)
            return self._make_json_dataset(edges_df, nodes_df, name)
        elif mode == 'vgraph':
            edges_df = self._table_to_pandas(edges)
            nodes_df = self._table_to_pandas(nodes)
            return self._make_vgraph_dataset(edges_df, nodes_df, name)
        elif mode == 'arrow':
            edges_arr = self._table_to_arrow(edges)
            nodes_arr = self._table_to_arrow(nodes)
            return self._make_arrow_dataset(edges_arr, nodes_arr, name)
            #token=None, dataset_id=None, url_params = None)
        else:
            raise ValueError('Unknown mode: ' + mode)


    # Main helper for creating ETL1 payload
    def _make_json_dataset(self, edges, nodes, name):
        (elist, nlist) = self._bind_attributes_v1(edges, nodes)
        edict = elist.where((pandas.notnull(elist)), None).to_dict(orient='records')

        bindings = {'idField': self._node or Plotter._defaultNodeId,
                    'destinationField': self._destination, 'sourceField': self._source}
        dataset = {'name': PyGraphistry._config['dataset_prefix'] + name,
                   'bindings': bindings, 'type': 'edgelist', 'graph': edict}

        if nlist is not None:
            ndict = nlist.where((pandas.notnull(nlist)), None).to_dict(orient='records')
            dataset['labels'] = ndict
        return dataset


    # Main helper for creating ETL2 payload
    def _make_vgraph_dataset(self, edges, nodes, name):
        from . import vgraph

        (elist, nlist, encodings) = self._bind_attributes_v2(edges, nodes)
        nodeid = self._node or Plotter._defaultNodeId

        sources = elist[self._source]
        dests = elist[self._destination]
        elist.drop([self._source, self._destination], axis=1, inplace=True)

        # Filter out nodes which have no edges
        lnodes = pandas.concat([sources, dests], ignore_index=True).unique()
        lnodes_df = pandas.DataFrame(lnodes, columns=[nodeid])
        filtered_nlist = pandas.merge(lnodes_df, nlist, on=nodeid, how='left')

        # Create a map from nodeId to a continuous range of integer [0, #nodes-1].
        # The vgraph protobuf format uses the continous integer ranger as internal nodeIds.
        node_map = dict([(v, i) for i, v in enumerate(lnodes.tolist())])

        dataset = vgraph.create(elist, filtered_nlist, sources, dests, nodeid, node_map, name)
        dataset['encodings'] = encodings
        return dataset

    def _make_arrow_dataset(self, edges: pa.Table, nodes: pa.Table, name: str) -> ArrowUploader:
        au = ArrowUploader(
            server_base_path=PyGraphistry.protocol() + '://' + PyGraphistry.server(),
            edges=edges, nodes=nodes, name=name,
            metadata={
                'usertag': PyGraphistry._tag,
                'key': PyGraphistry.api_key(),
                'agent': 'pygraphistry',
                'apiversion' : '3',
                'agentversion': sys.modules['graphistry'].__version__,
            },
            certificate_validation=PyGraphistry.certificate_validation())
        au.edge_encodings = au.g_to_edge_encodings(self)
        au.node_encodings = au.g_to_node_encodings(self)
        return au

    def bolt(self, driver):
        res = copy.copy(self)
        res._bolt_driver = bolt_util.to_bolt_driver(driver)
        return res


    def cypher(self, query, params={}):
        res = copy.copy(self)
        driver = self._bolt_driver or PyGraphistry._config['bolt_driver']
        with driver.session() as session:
            bolt_statement = session.run(query, **params)
            graph = bolt_statement.graph()
            edges = bolt_util.bolt_graph_to_edges_dataframe(graph)
            nodes = bolt_util.bolt_graph_to_nodes_dataframe(graph)
        return res\
            .bind(\
                node=bolt_util.node_id_key,\
                source=bolt_util.start_node_id_key,\
                destination=bolt_util.end_node_id_key
            )\
            .nodes(nodes)\
            .edges(edges)

    def nodexl(self, xls_or_url, source='default', engine=None, verbose=False):
        
        if not (engine is None):
            print('WARNING: Engine currently ignored, please contact if critical')
        
        return NodeXLGraphistry(self, engine).xls(xls_or_url, source, verbose)


    def tigergraph(self,
        protocol = 'http',
        server = 'localhost',
        web_port = 14240,
        api_port = 9000,
        db = None,
        user = 'tigergraph',
        pwd = 'tigergraph',
        verbose = False
    ):
        """Register Tigergraph connection setting defaults
    
        :param protocol: Protocol used to contact the database.
        :type protocol: Optional string.
        :param server: Domain of the database
        :type server: Optional string.
        :param web_port: 
        :type web_port: Optional integer.
        :param api_port: 
        :type api_port: Optional integer.
        :param db: Name of the database
        :type db: Optional string.    
        :param user:
        :type user: Optional string.    
        :param pwd: 
        :type pwd: Optional string.
        :param verbose: Whether to print operations
        :type verbose: Optional bool.         
        :returns: Plotter.
        :rtype: Plotter.


        **Example: Standard**
                ::

                    import graphistry
                    tg = graphistry.tigergraph(protocol='https', server='acme.com', db='my_db', user='alice', pwd='tigergraph2')                    

        """
        res = copy.copy(self)
        res._tigergraph = Tigeristry(self, protocol, server, web_port, api_port, db, user, pwd, verbose)
        return res


    def gsql_endpoint(self, method_name, args = {}, bindings = {}, db = None, dry_run = False):
        """Invoke Tigergraph stored procedure at a user-definend endpoint and return transformed Plottable
    
        :param method_name: Stored procedure name
        :type method_name: String.
        :param args: Named endpoint arguments
        :type args: Optional dictionary.
        :param bindings: Mapping defining names of returned 'edges' and/or 'nodes', defaults to @@nodeList and @@edgeList
        :type bindings: Optional dictionary.
        :param db: Name of the database, defaults to value set in .tigergraph(...)
        :type db: Optional string.
        :param dry_run: Return target URL without running
        :type dry_run: Bool, defaults to False
        :returns: Plotter.
        :rtype: Plotter.

        **Example: Minimal**
                ::

                    import graphistry
                    tg = graphistry.tigergraph(db='my_db')
                    tg.gsql_endpoint('neighbors').plot()

        **Example: Full**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    tg.gsql_endpoint('neighbors', {'k': 2}, {'edges': 'my_edge_list'}, 'my_db').plot()

        **Example: Read data**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    out = tg.gsql_endpoint('neighbors')
                    (nodes_df, edges_df) = (out._nodes, out._edges)

        """
        return self._tigergraph.gsql_endpoint(self, method_name, args, bindings, db, dry_run)

    def gsql(self, query, bindings = {}, dry_run = False):
        """Run Tigergraph query in interpreted mode and return transformed Plottable
    
        :param query: Code to run
        :type query: String.
        :param bindings: Mapping defining names of returned 'edges' and/or 'nodes', defaults to @@nodeList and @@edgeList
        :type bindings: Optional dictionary.
        :param dry_run: Return target URL without running
        :type dry_run: Bool, defaults to False        
        :returns: Plotter.
        :rtype: Plotter.

        **Example: Minimal**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    tg.gsql(\"\"\"
INTERPRET QUERY () FOR GRAPH Storage { 
    
    OrAccum<BOOL> @@stop;
    ListAccum<EDGE> @@edgeList;
    SetAccum<vertex> @@set;
    
    @@set += to_vertex("61921", "Pool");

    Start = @@set;

    while Start.size() > 0 and @@stop == false do

      Start = select t from Start:s-(:e)-:t
      where e.goUpper == TRUE
      accum @@edgeList += e
      having t.type != "Service";
    end;

    print @@edgeList;
  }
                    \"\"\").plot()

       **Example: Full**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    tg.gsql(\"\"\"
INTERPRET QUERY () FOR GRAPH Storage { 
    
    OrAccum<BOOL> @@stop;
    ListAccum<EDGE> @@edgeList;
    SetAccum<vertex> @@set;
    
    @@set += to_vertex("61921", "Pool");

    Start = @@set;

    while Start.size() > 0 and @@stop == false do

      Start = select t from Start:s-(:e)-:t
      where e.goUpper == TRUE
      accum @@edgeList += e
      having t.type != "Service";
    end;

    print @@my_edge_list;
  }
                    \"\"\", {'edges': 'my_edge_list'}).plot()
        """        
        return self._tigergraph.gsql(self, query, bindings, dry_run)







