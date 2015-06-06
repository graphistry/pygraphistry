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

# loadpandassync (edge, node, graphname, sourcefield, desfield, nodefield,
#         edgetitle, edgelabel, edgeweight, pointtitle , pointlabel,
#         pointcolor, pointsize)
# loadpandassync is the primary method. it parses the data with pandas format
# or with pandas pointer and load it to proper Graphistry server, and send back
# the concerned graph. It is a wrapper for loadpandas().

# edge(compulsory) = dataset for edge, the format includes Pandas pointer/Pandas
# node(compulsory) = dataset for node, the format includes Pandas pointer/Pandas
# sourcefield(optional) = name of the column in edge for sourcePointers
#                         (eg. 'source'), the default value is 'src'.
# desfield(optional) = name of the column in edge for destinationPointers
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

# loadpandas(edge, node, graphname, sourcefield, desfield, nodefield,
#            edgeTitle, edgeLabel, edgeWeight, pointTitle, pointLabel,
#            pointColor, pointSize)
# This function load data with pandas format as edge and node seperately
# and returns it with Json format (Decoding)
#...............................................................................

# isjsonpointer(document)
# This function checks whether the input document is Json pointer
#...............................................................................

class Graphistry (object):

    def settings(self, url='proxy', frameheight=500):

        self.height = frameheight
        print 'Setting server...'
        if url is 'localhost':
            self.hostname = 'localhost:3000'
            self.url = 'http://localhost:3000/etl'
        elif url is 'proxy':
            self.hostname = 'proxy-staging.graphistry.com'
            self.url = 'http://proxy-staging.graphistry.com/etl'
        else:
            self.url = '-1'
            raise ValueError("Can not find this server")
        return self

    def loadpandassync(self, edge, node, graphname=None,
                       sourcefield=None, desfield=None,
                       nodefield=None, edgetitle=None,
                       edgelabel=None, edgeweight=None,
                       pointtitle=None, pointlabel=None,
                       pointcolor=None, pointsize=None):

        import random
        import string
        import requests
        import json
        import yaml

        doc = self.loadpandas(edge, node, graphname,
                              sourcefield, desfield, nodefield, edgetitle,
                              edgelabel, edgeweight, pointtitle,
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
            print 'Loading Finished!'
            print 'Getting iframe...'
            from IPython.core.display import HTML
            l = "http://" + self.hostname + "/graph/graph.html?dataset=%s"%name
            print "url: ", l
            return HTML('<iframe src="' + l + '" style="width:100%; height:' +
                        str(self.height) + 'px; border: 1px solid #DDD">')

    def loadjsonsync(self, document):

        import requests
        import json
        import yaml

        print 'Loading Json...'

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
            print 'Loading Finished!'
            print 'Getting iframe...'
            from IPython.core.display import HTML
            l = "http://" + self.hostname + "/graph/graph.html?dataset=%s"%name
            print "url:", l
            return HTML('<iframe src="' + l + '" style="width:100%; height:' + str(self.height) + 'px; border: 1px solid #DDD">')


    def isjsonpointer(self, document):
        if isinstance(document, str):
            if document[-4:] == 'json':
                return True
            else:
                raise ValueError(" This Json Pointer is Invalid!")
        else:
            return False

    def loadjsonpointer(self, filedir):
        import json
        print 'Loading Json...'
        with open(filedir) as data_file:
            files = json.load(data_file)
        return files

    def loadpandas(self, edge, node, graphname=None,
                   sourcefield=None, desfield=None, nodefield=None,
                   edgetitle=None, edgelabel=None, edgeweight=None,
                   pointtitle=None, pointlabel=None,
                   pointcolor=None, pointsize=None,):

        import pandas as pd
        import json
        import random
        import string

        print 'Loading Pandas...'

        if isinstance(edge, str):
            if (edge[-4:] == '.csv'):
                edge = pd.read_csv(edge, na_values=['-'], low_memory=False)
                edge.dropna(how='all', axis=1, inplace=True)
            else:
                raise ValueError("This Json Pointer is Invalid")

        if isinstance(node, str):
            if (node[-4:] == '.csv'):
                node = pd.read_csv(node, na_values=['-'], low_memory=False)
                node.dropna(how='all', axis=1, inplace=True)
            else:
                raise ValueError("This Json Pointer is Invalid")

        if graphname is None:
            graphname = ''.join(random.choice(string.ascii_uppercase +
                                string.digits)for _ in range(10))

        files = {'name': graphname}
        files['bindings'] = {'idField': 'node', 'destinationField': 'dst',
                             'sourceField': 'src'}
        files['type'] = 'edgelist'

        # for Edge
        edgejson = edge.to_json(orient='records')
        edgej = json.loads(edgejson)
        if sourcefield is None:
            for i in range(0, len(edgej)):
                edgej[i]['src'] = edge['src'][i]
        else:
            for i in range(0, len(edgej)):
                edgej[i]['src'] = edge[sourcefield][i]

        if desfield is None:
            for i in range(0, len(edgej)):
                edgej[i]['dst'] = edge['dst'][i]
        else:
            for i in range(0, len(edgej)):
                edgej[i]['dst'] = edge[desfield][i]

        u_edgej = json.dumps(edgej)
        # set edgelabel
        if edgelabel is None:
            u_edgej = u_edgej.replace("edgeLabel", "edgeLabel")
        else:
            u_edgej = u_edgej.replace(edgelabel, "edgeLabel")
        # Set edgeWeight
        if edgeweight is None:
            u_edgej = u_edgej.replace("edgeWeight", "edgeWeight")
        else:
            u_edgej = u_edgej.replace(edgeweight, "edgeWeight")
        # Set edgeTitle
        if edgetitle is None:
            u_edgej = u_edgej.replace("edgeTitle", "edgeTitle")
        else:
            u_edgej = u_edgej.replace(edgetitle, "edgeTitle")

        files['graph'] = json.loads(u_edgej)

        # for Label
        nodejson = node.to_json(orient='records')
        nodej = json.loads(nodejson)
        if nodefield is None:
            nodej = {each['node']: each for each in nodej}.values()
            for i in range(0, len(nodej)):
                nodej[i]['node'] = node['node'][i]
        else:
            nodej = {each[nodefield]: each for each in nodej}.values()
            for i in range(0, len(nodej)):
                nodej[i]['node'] = node[nodefield][i]

        u_nodej = json.dumps(nodej)
        # Set Point Title
        if pointtitle is None:
            u_nodej = u_nodej.replace('pointTitle', 'pointTitle')
        else:
            u_nodej = u_nodej.replace(pointtitle, 'pointTitle')
        # Set Point Label
        if pointlabel is None:
            u_nodej = u_nodej.replace('pointLabel', 'pointLabel')
        else:
            u_nodej = u_nodej.replace(pointlabel, 'pointLabel')
        # Set the point size
        if pointsize is None:
            u_nodej = u_nodej.replace('pointSize', 'pointSize')
        else:
            u_nodej = u_nodej.replace(pointsize, 'pointSize')
        # Set the point Color
        if pointcolor is None:
            u_nodej = u_nodej.replace('pointColor', 'pointColor')
        else:
            u_nodej = u_nodej.replace(pointcolor, 'pointColor')

        files['labels'] = json.loads(u_nodej)

        return files

