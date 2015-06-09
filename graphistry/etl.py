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

import pandas
import random
import string
import requests
import json
import yaml

def settings(server='labs', frameheight=500):
    height = frameheight

    if server is 'localhost':
        hostname = 'localhost:3000'
        url = 'http://localhost:3000/etl'
    elif server is 'proxy':
        hostname = 'proxy-staging.graphistry.com'
        url = 'http://proxy-staging.graphistry.com/etl'
    elif server is 'labs':
        hostname = 'proxy-labs.graphistry.com'
        url = 'http://proxy-labs.graphistry.com/etl'
    else:
        raise ValueError("Unknown server: " + server)

    return Graphistry(height, url, hostname)


def plot(edge, node=None, graphname=None, sourcefield=None, destfield=None,
         nodefield=None, edgetitle=None, edgelabel=None, edgecolor=None, edgeweight=None,
         pointtitle=None, pointlabel=None, pointcolor=None, pointsize=None):

    g = settings()
    return g.plot(edge, node, graphname, sourcefield, destfield, nodefield, edgetitle,
                  edgelabel, edgecolor, edgeweight, pointtitle, pointlabel, pointcolor, pointsize)


class Graphistry (object):

    def __init__ (self, height, url, hostname):
        self.height = height
        self.url = url
        self.hostname = hostname


    def plot(self, edge, node=None, graphname=None,
             sourcefield=None, destfield=None, nodefield=None,
             edgetitle=None, edgelabel=None, edgecolor=None , edgeweight=None,
             pointtitle=None, pointlabel=None, pointcolor=None,
             pointsize=None):

        if isinstance(edge, pandas.core.frame.DataFrame):
            return self.loadpandassync(edge, node, graphname, sourcefield, destfield,
                                       nodefield, edgetitle, edgelabel, edgecolor, edgeweight,
                                       pointtitle, pointlabel, pointcolor, pointsize)
        else:
            return self.loadjsonsync(edge)


    def loadpandassync(self, edge, node=None, graphname=None,
                       sourcefield=None, destfield=None,
                       nodefield=None, edgetitle=None,
                       edgelabel=None, edgecolor=None, edgeweight=None,
                       pointtitle=None, pointlabel=None,
                       pointcolor=None, pointsize=None):
        from IPython.core.display import HTML

        doc = self.loadpandas(edge, node, graphname, sourcefield,
                              destfield, nodefield, edgetitle,
                              edgelabel, edgecolor, edgeweight, pointtitle,
                              pointlabel, pointcolor, pointsize)

        docj = json.dumps(doc)

        headers = {'content type': 'json', 'content-Encoding': 'gzip',
                   'Accept-Encoding': 'gzip', 'Accept': 'application/json',
                   'Content-Type': 'application/json'}
        try:
            r = requests.post(self.url, docj, headers=headers)
        except requests.exceptions.ConnectionError as e:
            raise ValueError("Connection Error")
        except requests.exceptions.HTTPError as e:
            raise ValueError("HTTPError:", e.message)

        else:
            name = yaml.safe_load(doc['name'])
            l = "http://" + self.hostname + "/graph/graph.html?dataset=%s"%name
            print "Url: ", l
            return HTML('<iframe src="' + l + '" style="width:100%; height:' +
                        str(self.height) + 'px; border: 1px solid #DDD">')


    def loadjsonsync(self, document):
        print 'Loading Json...'
        from IPython.core.display import HTML

        if self.isjsonpointer(document):
            doc = self.loadjsonpointer(document)
            name = yaml.safe_load(doc['name'])
            doc = json.dumps(doc)

        else:
            doc = document
            name = yaml.safe_load(doc['name'])

        headers = {'content type': 'json', 'content-Encoding': 'gzip',
                   'Accept-Encoding': 'gzip', 'Accept': 'application/json',
                   'Content-Type': 'application/json'}
        try:
            r = requests.post(self.url, doc, headers=headers)
        except requests.exceptions.ConnectionError as e:
            raise ValueError("Connection Error")
        except requests.exceptions.HTTPError as e:
            raise ValueError("HTTPError:", e.message)

        else:
            l = "http://" + self.hostname + "/graph/graph.html?dataset=%s"%name
            print "Url:", l
            return HTML('<iframe src="' + l + '" style="width:100%; height:' +
                        str(self.height) + 'px; border: 1px solid #DDD">')


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


    def loadpandas(self, edge, node=None, graphname=None,
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

        files = {'name': graphname}
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
                    edgej[i]['edgeColor2'] = edge[edgeColor][i]

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
            nodej = {each[nodefield]: each for each in nodej}.values()
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
