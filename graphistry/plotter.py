from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
import random
import string
import copy
import types
import pandas

from . import pygraphistry
from . import util


class Plotter(object):
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
        self._edge_weight = None
        self._point_title = None
        self._point_label = None
        self._point_color = None
        self._point_size = None
        # Settings
        self._height = 500
        self._url_params = {'info': 'true'}

    def __repr__(self):
        bnds = ['edges', 'nodes', 'source', 'destination', 'node', 'edge_title',
                'edge_label', 'edge_color', 'edge_weight', 'point_title',
                'point_label', 'point_color', 'point_size']
        stgs = ['height', 'url_params']

        rep = {'bindings': dict([(f, getattr(self, '_' + f)) for f in bnds]),
               'settings': dict([(f, getattr(self, '_' + f)) for f in stgs])}
        if util.in_ipython():
            from IPython.lib.pretty import pretty
            return pretty(rep)
        else:
            return str(rep)

    def bind(self, source=None, destination=None, node=None,
             edge_title=None, edge_label=None, edge_color=None, edge_weight=None,
             point_title=None, point_label=None, point_color=None, point_size=None):
        res = copy.copy(self)
        res._source = source or self._source
        res._destination = destination or self._destination
        res._node = node or self._node

        res._edge_title = edge_title or self._edge_title
        res._edge_label = edge_label or self._edge_label
        res._edge_color = edge_color or self._edge_color
        res._edge_weight = edge_weight or self._edge_weight

        res._point_title = point_title or self._point_title
        res._point_label = point_label or self._point_label
        res._point_color = point_color or self._point_color
        res._point_size = point_size or self._point_size
        return res

    def nodes(self, nodes):
        res = copy.copy(self)
        res._nodes = nodes
        return res

    def edges(self, edges):
        res = copy.copy(self)
        res._edges = edges
        return res

    def graph(self, ig):
        res = copy.copy(self)
        res._edges = ig
        res._nodes = None
        return res

    def settings(self, height=None, url_params={}):
        res = copy.copy(self)
        res._height = height or self._height
        res._url_params = dict(self._url_params, **url_params)
        return res

    def plot(self, graph=None, nodes=None):
        if graph is None:
            if self._edges is None:
                util.error('Graph/edges must be specified.')
            g = self._edges
        else:
            g = graph
        n = self._nodes if nodes is None else nodes

        self._check_mandatory_bindings(not isinstance(n, type(None)))
        dataset = self._plot_dispatch(g, n)
        if dataset is None:
            util.error('Expected Pandas dataframe(s) or Igraph/NetworkX graph.')

        dataset_name = pygraphistry.PyGraphistry._etl(dataset)
        viz_url = pygraphistry.PyGraphistry._viz_url(dataset_name, self._url_params)

        if util.in_ipython() is True:
            from IPython.core.display import HTML
            return HTML(self._iframe(viz_url))
        else:
            print('Url: ', viz_url)
            import webbrowser
            webbrowser.open(viz_url)
            return self

    def pandas2igraph(self, edges, directed=True):
        import igraph
        self._check_mandatory_bindings(False)
        self._check_bound_attribs(edges, ['source', 'destination'], 'Edge')
        if self._node is None:
            util.warn('"node" is unbound, automatically binding it to "%s".' % Plotter._defaultNodeId)

        self._node = self._node or Plotter._defaultNodeId
        eattribs = edges.columns.values.tolist()
        eattribs.remove(self._source)
        eattribs.remove(self._destination)
        cols = [self._source, self._destination] + eattribs
        etuples = [tuple(x) for x in edges[cols].values]
        return igraph.Graph.TupleList(etuples, directed=directed, edge_attrs=eattribs,
                                      vertex_name_attr=self._node)

    def igraph2pandas(self, ig):
        def get_edgelist(ig):
            idmap = dict(enumerate(ig.vs[self._node]))
            for e in ig.es:
                t = e.tuple
                yield dict({self._source: idmap[t[0]], self._destination: idmap[t[1]]},
                            **e.attributes())

        self._check_mandatory_bindings(False)
        if self._node is None:
            util.warn('"node" is unbound, automatically binding it to "%s".' % Plotter._defaultNodeId)
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

    def networkx2pandas(self, g):
        def get_nodelist(g):
            for n in g.nodes(data=True):
                yield dict({self._node: n[0]}, **n[1])
        def get_edgelist(g):
            for e in g.edges(data=True):
                yield dict({self._source: e[0], self._destination: e[1]}, **e[2])

        self._check_mandatory_bindings(False)
        vattribs = g.nodes(data=True)[0][1] if g.number_of_nodes() > 0 else []
        if self._node is None:
            util.warn('"node" is unbound, automatically binding it to "%s".' % Plotter._defaultNodeId)
            self._node = Plotter._defaultNodeId
        elif self._node not in vattribs:
            util.error('Vertex attribute "%s" bound to "node" does not exist.' % self._node)

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

    def _plot_dispatch(self, graph, nodes):
        if isinstance(graph, pandas.core.frame.DataFrame):
            return self._pandas2dataset(graph, nodes)

        try:
            import igraph
            if isinstance(graph, igraph.Graph):
                (e, n) = self.igraph2pandas(graph)
                return self._pandas2dataset(e, n)
        except ImportError:
            pass

        try:
            import networkx
            if isinstance(graph, networkx.classes.graph.Graph) or \
               isinstance(graph, networkx.classes.digraph.DiGraph) or \
               isinstance(graph, networkx.classes.multigraph.MultiGraph) or \
               isinstance(graph, networkx.classes.multidigraph.MultiDiGraph):
                (e, n) = self.networkx2pandas(graph)
                return self._pandas2dataset(e, n)
        except ImportError:
            pass

        return None

    def _pandas2dataset(self, edges, nodes):
        def bind(df, pbname, attrib, default=None):
            bound = getattr(self, attrib)
            if bound:
                if bound in df.columns.tolist():
                    df[pbname] = df[bound]
                else:
                    util.warn('Attribute "%s" bound to %s does not exist.' % (bound, attrib))
            elif default:
                df[pbname] = df[default]

        self._check_bound_attribs(edges, ['source', 'destination'], 'Edge')
        nodeid = self._node or Plotter._defaultNodeId
        elist = edges.reset_index(drop=True)
        bind(elist, 'edgeColor', '_edge_color')
        bind(elist, 'edgeLabel', '_edge_label')
        bind(elist, 'edgeTitle', '_edge_title')
        bind(elist, 'edgeWeight', '_edge_weight')
        if nodes is None:
            nodes = pandas.DataFrame()
            nodes[nodeid] = pandas.concat([edges[self._source], edges[self._destination]],
                                           ignore_index=True).drop_duplicates()
        else:
            self._check_bound_attribs(nodes, ['node'], 'Vertex')

        nlist = nodes.reset_index(drop=True)
        bind(nlist, 'pointColor', '_point_color')
        bind(nlist, 'pointLabel', '_point_label')
        bind(nlist, 'pointTitle', '_point_title', nodeid)
        bind(nlist, 'pointSize', '_point_size')
        return self._make_dataset(elist, nlist)

    def _make_dataset(self, elist, nlist=None):
        edict = elist.where((pandas.notnull(elist)), None).to_dict(orient='records')
        name = ''.join(random.choice(string.ascii_uppercase +
                                     string.digits) for _ in range(10))
        bindings = {'idField': self._node or Plotter._defaultNodeId,
                    'destinationField': self._destination, 'sourceField': self._source}
        dataset = {'name': pygraphistry.PyGraphistry._dataset_prefix + name,
                   'bindings': bindings, 'type': 'edgelist', 'graph': edict}
        if nlist is not None:
            ndict = nlist.where((pandas.notnull(nlist)), None).to_dict(orient='records')
            dataset['labels'] = ndict
        return dataset

    def _iframe(self, url):
        tag = '<iframe src="%s" style="width:100%%; height:%dpx; border: 1px solid #DDD"></iframe>'
        return tag % (url, self._height)
