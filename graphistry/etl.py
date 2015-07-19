# This is a Python Class established as a Data Loader for Graphistry


# This Loader support four types of data : Json (eg. '{"xxx":"xxx"}'),
# Json Pointer (eg. './xxx/xxx.json'), Pandas (eg. csv table),
# Pandas Pointer (eg. './xxx/xxx.csv')

# Class Load has the following public methods:

# settings(url, frameheight)
# This function does the work of setting proper server for user.
# url = 'proxy'/'localhost', the default value is 'proxy'.
# 'localhost' would let user connect to the local server, 'proxy'
# would let user connect with server from Graphistry
# frameheight = integer, the default value is 500.
# frameheight is used to set the size of returned figures
#
#...............................................................................

# plot(edge, node, graphname, sourcefield, destfield, nodefield,
#         edgetitle, edgelabel, edgecolor, edgeweight, pointtitle , pointlabel,
#         pointcolor, pointsize)
# plot function is the primeary method. It is a wrapper for loadpandassync()
# and loadjsonsync().
#
#...............................................................................
# loadpandassync (edge, node, graphname, sourcefield, destfield, nodefield,
#         edgetitle, edgelabel, edgecolor, edgeweight, pointtitle , pointlabel,
#         pointcolor, pointsize)
# loadpandassync is the primary method. it parses the data with pandas format
# or with pandas pointer and load it to proper Graphistry server, and send back
# the concerned graph. It is a wrapper for loadpandas().

# edge(compulsory) = dataset for edge, the format includes Pandas pointer/Pandas
# node(compulsory) = dataset for node, the format includes Pandas pointer/Pandas
# sourcefield(optional) = name of the column in edge for sourcePointers
#                         (eg. 'source'), the default value is 'src'.
# destfield(optional) = name of the column in edge for destinationPointers
#                      (eg. 'des'), the default value is 'dst'.
# nodefield(optional) = name of the column in node for all pointers,
#                       the default value is 'node'
# graphname(optional) = a name string the user prefer (eg. 'myGraph'), the
#                       default value is a random generagted ten-characters
#                       string with numbers and capital letters.
# edgetitle(optional) = name of the column in edge for edges' titles, the
#                       default value is 'edgeTitle'
# edgelabel(optional) = name of the column in edge for edges' labels, the
#                       default value is 'edgeLabel'
# edgecolor(optional) = name of the column in edge for edges' labels, the
#                       default value is 'edgeColor'
# edgeweight(optional) = name of the column in edge for edges' Weights, the
#                        default value is 'edgeWeight'
# pointtitle(optional) = name of the column in node for nodes' titles, the
#                        default value is 'pointTitle'
# pointlabel(optional) = name of the column in node for nodes' labels, the
#                        default value is 'pointLabel'
# pointColor(optional) = name of the column in node for nodes' colors, the
#                        default value is 'pointColor'
# pointSize(optional) = name of the column in node for nodes' sizes, the
#                       default value is 'pointSize'
#
#...............................................................................

# loadjsonsync (document)
# loadjsonsync is the primary method. it parses the data with Json format
# or with Json pointer and load it to proper Graphistry server, and send back
# the concerned graph. It is a wrapper for isjsonpointer(), loadjsonpointer().

# document(compulsory) = complete dataset for edge and node,
#                        the format includes Json pointer / Json
#
#...............................................................................

# loadjsonpointer(filedir)
# This function load Json file from give directory and returns it with
# Json format (Decoding)
#...............................................................................

# loadpandas(edge, node, graphname, sourcefield, destfield, nodefield,
#            edgeTitle, edgeLabel, edgecolor, edgeWeight, pointTitle, pointLabel,
#            pointColor, pointSize)
# This function load data with pandas format as edge and node seperately
# and returns it with Json format (Decoding)
#...............................................................................

# isjsonpointer(document)
# This function checks whether the input document is Json pointer
#...............................................................................

import sys
import random
import string
import json
import gzip
import StringIO

import requests
import pandas


def settings(server='labs', height=500, url_params=None, key=None):

    """ Configure plot settings

    Keywords arguments:
    server -- the Graphistry server. Possible values: "labs", "staging", "localhost" (default "labs")
    height -- the height of the plot in pixels (default 500)
    """

    if server is 'localhost':
        hostname = 'localhost:3000'
    elif server is 'staging':
        hostname = 'proxy-staging.graphistry.com'
    elif server is 'labs':
        hostname = 'proxy-labs.graphistry.com'
    else:
        hostname = server

    return Graphistry(hostname, height, url_params, key)


def plot(edges, nodes=None, graph_name=None, source=None, destination=None, node=None,
         edge_title=None, edge_label=None, edge_color=None, edge_weight=None,
         point_title=None, point_label=None, point_color=None, point_size=None):
    """
        Plot a node-link diagram (graph).
        If running in IPython, returns an iframe. Otherwise, open webbrowser to show the plot.

        Mandatory Keywords arguments:
        edges -- the pandas dataframe containing edges
        source -- name of the source column for edges
        destination -- name of the destination column for edges

        Optional Keywords arguments:
        graph_name -- the name of the graph (default to random string)
        edge_title -- the column name (inside edges) containing edge titles
        edge_label -- the column name (inside edges) containing edge labels
        edge_color -- the column name (inside edges) containing edge colors
        edge_weight -- the column name (inside edges) containing edge weights
        nodes -- pandas dataframe containing node labels
        point_title -- the column name (inside nodes) containing node titles
        point_label -- the column name (inside nodes) containing node labels
        point_size -- the column name (inside nodes) containing node sizes
        point_color -- the column name (inside nodes) containing node colors

        TODO WRITE ME
    """

    g = settings()
    return g.plot(edges, nodes, graph_name, source, destination, node, edge_title,
                  edge_label, edge_color, edge_weight, point_title, point_label, point_color, point_size)


def fingerprint():
    import platform as p
    import uuid
    import hashlib

    md5 = hashlib.md5()
    # Hostname, OS, CPU, MAC,
    data = [p.node(), p.system(), p.machine(), str(uuid.getnode())]
    md5.update(''.join(data))
    return "%s-pygraphistry-%s" % (md5.hexdigest()[:8], sys.modules['graphistry'].__version__)


def in_ipython():
        try:
            __IPYTHON__
            return True
        except NameError:
            return False


class Graphistry (object):
    tag = fingerprint()
    dataset_prefix = "PyGraphistry/"


    def __init__(self, hostname, height, url_params, key):

        self.height = height
        self.hostname = hostname
        self.url_params = url_params
        self.key = key


    def settings(self, hostname=None, height=None, url_params=None, key=None):

        hn = self.hostname if hostname == None else hostname
        ht = self.height if height == None else height
        up = self.url_params if url_params == None else url_params
        ky = self.key if key == None else key

        return settings(hn, ht, up, ky)


    def etl_url(self):
        return "http://%s/etl" % self.hostname


    def viz_url(self, dataset_name):
        extra = ''
        if not self.url_params == None and not self.url_params == {}:
            extra = '&' + '&'.join([ k + '=' + self.url_params[k] for k in self.url_params])
        return "http://%s/graph/graph.html?dataset=%s&info=true&usertag=%s%s" % (self.hostname, dataset_name, self.tag, extra)


    def iframe(self, url):
        return '<iframe src="%s" style="width:100%%; height:%dpx; border: 1px solid #DDD"></iframe>' % (url, self.height)


    def plot(self, edges, nodes=None, graph_name=None, source=None, destination=None, node=None,
             edge_title=None, edge_label=None, edge_color=None, edge_weight=None,
             point_title=None, point_label=None, point_color=None, point_size=None):

        if isinstance(edges, pandas.core.frame.DataFrame):
            return self.load_pandas(edges, nodes, graph_name, source, destination, node,
                                    edge_title, edge_label, edge_color, edge_weight,
                                    point_title, point_label, point_color, point_size)
        else:
            return self.loadjsonsync(edges)


    def etl(self, json_dataset):
        headers = {'Content-Encoding': 'gzip', 'Content-Type': 'application/json'}
        params = {'usertag': self.tag, 'agent': 'pygraphistry', 'apiversion' : '1',
                  'agentversion': sys.modules['graphistry'].__version__}
        if self.key:
            params['key'] = self.key

        out_file = StringIO.StringIO()
        with gzip.GzipFile(fileobj=out_file, mode='w', compresslevel=9) as f:
            f.write(json_dataset)

        try:
            response = requests.post(self.etl_url(), out_file.getvalue(), headers=headers, params=params)
        except requests.exceptions.ConnectionError as e:
            raise ValueError("Connection Error:", e.message)
        except requests.exceptions.HTTPError as e:
            raise ValueError("HTTPError:", e.message)

        jres = response.json()
        if (jres['success'] is not True):
            raise ValueError("Server reported error when processsing data:", jres['msg'])
        else:
            url = self.viz_url(jres['dataset'])
            return {'url': url, 'iframe': self.iframe(url)}


    def load_pandas(self, edge, node=None, graphname=None,
                       sourcefield=None, destfield=None,
                       nodefield=None, edgetitle=None,
                       edgelabel=None, edgecolor=None, edgeweight=None,
                       pointtitle=None, pointlabel=None,
                       pointcolor=None, pointsize=None):
        dataset = self.package_pandas(edge, node, graphname, sourcefield,
                              destfield, nodefield, edgetitle,
                              edgelabel, edgecolor, edgeweight, pointtitle,
                              pointlabel, pointcolor, pointsize)

        result = self.etl(json.dumps(dataset))
        print "Url: ", result['url']

        if in_ipython() is True:
            from IPython.core.display import HTML
            return HTML(result['iframe'])
        else:
            import webbrowser
            webbrowser.open(result['url'])


    def load_json(self, dataset):
        if self.isjsonpointer(dataset):
            json_dataset = json.dumps(self.loadjsonpointer(dataset))
        else:
            json_dataset = dataset

        result = self.etl(json_dataset)

        if in_ipython is True:
            from IPython.core.display import HTML
            print "Url: ", result['url']
            return HTML(result['iframe'])
        else:
            import webbrowser
            webbrowser.open(result['url'])


    def isjsonpointer(self, document):
        if isinstance(document, str):
            if document[-4:] == 'json':
                return True
            else:
                raise ValueError(" This Json Pointer is Invalid!")
        else:
            return False


    def loadjsonpointer(self, filedir):
        print 'Loading Json...'
        with open(filedir) as data_file:
            files = json.load(data_file)
        return files


    def package_pandas(self, edge, node=None, graphname=None,
                   sourcefield=None, destfield=None, nodefield=None,
                   edgetitle=None, edgelabel=None, edgecolor=None, edgeweight=None,
                   pointtitle=None, pointlabel=None,
                   pointcolor=None, pointsize=None):
        if isinstance(edge, str):
            if (edge[-4:] == '.csv'):
                edge = pandas.read_csv(edge, na_values=['-'], low_memory=False)
                edge.dropna(how='all', axis=1, inplace=True)
            else:
                raise ValueError("This Pandas Pointer is Invalid")

        if isinstance(node, str):
            if (node[-4:] == '.csv'):
                node = pandas.read_csv(node, na_values=['-'], low_memory=False)
                node.dropna(how='all', axis=1, inplace=True)
            else:
                raise ValueError("This Pandas Pointer is Invalid")

        if node is None:
            nodefield = 'id'
            node = pandas.DataFrame()
            node[nodefield] = pandas.concat([edge[sourcefield], edge[destfield]], ignore_index=True).drop_duplicates()
            node['pointTitle'] = node[nodefield].map(lambda x: x)

        if graphname is None:
            graphname = ''.join(random.choice(string.ascii_uppercase +
                                string.digits)for _ in range(10))

        files = {'name': self.dataset_prefix + graphname}
        files['bindings'] = {'idField': 'node', 'destinationField': 'dst',
                             'sourceField': 'src'}
        files['type'] = 'edgelist'

        # for Edge
        edge = edge.reset_index()
        del edge['index']
        edgejson = edge.to_json(orient='records')
        edgej = json.loads(edgejson)
        u_edgej = json.dumps(edgej)
        if sourcefield is None:
            for i in range(0, len(edgej)):
                edgej[i]['src'] = edge['src'][i]
        else:
            for i in range(0, len(edgej)):
                edgej[i]['src'] = edge[sourcefield][i]

        if destfield is None:
            for i in range(0, len(edgej)):
                edgej[i]['dst'] = edge['dst'][i]
        else:
            for i in range(0, len(edgej)):
                edgej[i]['dst'] = edge[destfield][i]

        # set edgelabel
        if edgelabel is None:
            if 'edgeLabel' in u_edgej:
                for i in range(0, len(edgej)):
                    edgej[i]['edgeLabel'] = edge['edgeLabel'][i]
        else:
            for i in range(0, len(edgej)):
                    edgej[i]['edgeLabel'] = edge[edgelabel][i]

        # Set edgecolor
        if edgecolor is None:
            if 'edgeColor' in u_edgej:
                for i in range(0, len(edgej)):
                    edgej[i]['edgeColor2'] = edge['edgeColor'][i]
        else:
            for i in range(0, len(edgej)):
                    edgej[i]['edgeColor2'] = edge[edgecolor][i]

        # Set edgeWeight
        if edgeweight is None:
            if 'edgeWeight' in u_edgej:
                for i in range(0, len(edgej)):
                    edgej[i]['edgeWeight'] = edge['edgeWeight'][i]
        else:
            for i in range(0, len(edgej)):
                    edgej[i]['edgeWeight'] = edge[edgeweight][i]

        # Set edgeTitle
        if edgetitle is None:
            if 'edgeTitle' in u_edgej:
                for i in range(0, len(edgej)):
                    edgej[i]['edgeTitle'] = edge['edgeTitle'][i]
        else:
            for i in range(0, len(edgej)):
                    edgej[i]['edgeTitle'] = edge[edgetitle][i]

        files['graph'] = edgej

        # for Label
        node = node.reset_index()
        del node['index']
        nodejson = node.to_json(orient='records')
        nodej = json.loads(nodejson)
        u_nodej = json.dumps(nodej)
        if nodefield is None:
            nodej = {each['node']: each for each in nodej}.values()
            for i in range(0, len(nodej)):
                nodej[i]['node'] = node['node'][i]
        else:
            #nodej = {each[nodefield]: each for each in nodej}.values()
            for i in range(0, len(nodej)):
                nodej[i]['node'] = node[nodefield][i]

        # u_nodej = json.dumps(nodej)
        # Set Point Title
        if pointtitle is None:
            if 'pointTitle' in u_nodej:
                for i in range(0, len(nodej)):
                    nodej[i]['pointTitle'] = node['pointTitle'][i]
        else:
            for i in range(0, len(nodej)):
                    nodej[i]['pointTitle'] = node[pointtitle][i]
        # Set Point Label
        if pointlabel is None:
            if 'pointLabel' in u_nodej:
                for i in range(0, len(nodej)):
                    nodej[i]['pointLabel'] = node['pointLabel'][i]
        else:
            for i in range(0, len(nodej)):
                    nodej[i]['pointLabel'] = node[pointlabel][i]
        # Set the point size
        if pointsize is None:
            if 'pointSize' in u_nodej:
                for i in range(0, len(nodej)):
                    nodej[i]['pointSize'] = node['pointSize'][i]
        else:
            for i in range(0, len(nodej)):
                    nodej[i]['pointSize'] = node[pointsize][i]
        # Set the point Color
        if pointcolor is None:
            if 'pointColor' in u_nodej:
                for i in range(0, len(nodej)):
                    nodej[i]['pointColor'] = node['pointColor'][i]
        else:
            for i in range(0, len(nodej)):
                    nodej[i]['pointColor'] = node[pointcolor][i]

        files['labels'] = nodej

        return files
