# This is a Python Class established as a Data Loader for Graphistry

# This Loader support four types of data : Json (eg. '{"xxx":"xxx"}'),
# Json Pointer (eg. './xxx/xxx.json'), Pandas (eg. csv table),
# Pandas Pointer (eg. './xxx/xxx.csv')

# Class Load has the following public methods:

# settings(serveraddress)
# This function does the work of setting proper server for user.
# serveradress = 'proxy'/'localhost', the default value is 'proxy'.
# 'localhost' would let user connect to the local server, 'proxy'
# would let user connect with server from Graphistry
# ................................................................................

# loadFileSync (document, node, graphname, sourcefield, desfield, nodefield,
#         edgeTitle, edgeLabel, edgeWeight, pointTitle , pointLabel,
#         pointColor, pointSize)
# loadFileSync is the primary method. it parses the input data/file and load it
# to proper Graphistry server, and send back the concerned graph. It is a wrapper for
# loadJsonPointer(), loadPandasPointer(),loadPandas().

# r = loadFileSync (document) is used when the input data format is Json or Json Pointer

# r = laodFileSync (document, sourcefield, desfield) is used when the input data format is
# Pandas pointer.
# document = Pandas pointer (eg. '.Desktop/xxx.csv')
# sourcefield = name of the column in document for sourcePointers (eg. 'source'),
# desfield = name of the column in document for destinationPointers (eg. 'des'),
# All other input arguments are optional:
# nodefield = name of the column in document for all pointers, the default value is the
# same as sourcefield
# graphname = a name string the user prefer (eg. 'myGraph'), the default value is a random
# generagted ten-characters string with numbers and capital letters.
# edgeTitle, edgeLabel, edgeWeight, pointTitle, pointLabel, pointColor, pointSize =
# name of the column in document for edges'/pointers' titles, labels, weights, color.
#
# r = laodFileSync (document, node, sourcefield, desfield, nodefield) is used when the input
# data format is Pandas. The edge data and point data must be inputed as document and node seperately.
# document = Pandas format for edges.
# node = Pandas format for nodes.
# sourcefield = name of the column in document for sourcePointers (eg. 'source'),
# desfield = name of the column in document for destinationPointers (eg. 'des'),
# nodefield = name of the column in node for all pointers (eg. 'name').
# All other input arguments are optional:
# graphname = a name string the user prefer (eg. 'myGraph'), the default value is a random
# generagted ten-characters string with numbers and capital letters.
# edgeTitle, edgeLabel, edgeWeight = name of the column in document for edges' titles, labels, weights.
# pointTitle, pointLabel, pointColor, pointSize = name of the column in node for points'
# titles, labels, colors, sizes.
#...............................................................................

# loadJsonPointer(filedir)
# This function load Json file from give directory and returns it with Json format (Decoding)
#...............................................................................

# loadPandasPointer(filedir, graphname, sourcefield, desfield, edgeTitle, edgeLabel, edgeWeight,
#                   pointTitle, pointLabel, pointColor, pointSize)
# This function load Pandas file from give directory and returns it with Json format (Decoding)
#...............................................................................

# loadPandas(edge, node, graphname, sourcefield, desfield, nodefield, edgeTitle, edgeLabel,
#            edgeWeight, pointTitle, pointLabel, pointColor, pointSize)
# This function load data with pandas format as edge and node seperately
# and returns it with Json format (Decoding)
#...............................................................................

# isJson(document)
# This function checks whether the input document is Json
#...............................................................................

# isJsonPointer(document)
# This function checks whether the input document is Json pointer
#...............................................................................

# isPandasPointer(document)
# This function checks whether the input document is Pandas pointer
#...............................................................................


class Load (object):


    # This function checks whether the input document is Json
    def isJson(self,document):
        import json
        global json_ob
        print 'Loading Json...'
        try:
            json_ob = json.loads(document)
        except ValueError, e:
            return False
        return True


    # This function checks whether the input document is Json pointer
    def isJsonPointer(self,document):
        if document[-4:] == 'json':
            return True
        else:
            return False


    # This function checks whether the input document is Pandas pointer
    def isPandasPointer(self,document):
        if document[-4:] == '.csv':
            return True
        else:
            return False


    # This function load Json file from give directory and returns it with Json format
    def loadJsonPointer (self,filedir):
        import json
        print 'Loading Json...'
        with open(filedir) as data_file:
            files = json.load(data_file)
        return files


    # This function load Pandas file from give directory and returns it with Json format (Decoding)
    def loadPandasPointer (self,filedir, graphname,
                       sourcefield, desfield, edgeTitle,
                       edgeLabel, edgeWeight,
                       pointTitle, pointLabel,
                       pointColor, pointSize, ):
        import pandas as pd
        import json
        import random
        import string

        print 'Loading Pandas...'

        if graphname is 'Idontlikeanamelol':
            graphname = ''.join(random.choice(string.ascii_uppercase + string.digits)
                            for _ in range(10))

        data = pd.read_csv(filedir, na_values=['-'], low_memory=False)
        data.dropna(how='all', axis=1, inplace=True)
        datajson = data.to_json(orient = 'records')
        df = json.loads(datajson)
        files = {'name':graphname}
        files['bindings'] = {'idField':'node','destinationField':'dst','sourceField':'src'}
        files['type'] = 'edgelist'
        g_df = df
        for i in range (0,len(g_df)):
            g_df[i]['src'] = data[sourcefield][i]
            g_df[i]['dst'] = data[desfield][i]

        # for graph
        g_df = json.dumps(g_df)
        # set edgeLabel
        if edgeLabel is not 'donothaveone':
            g_df = g_df.replace(edgeLabel,"edgeLabel")
        # Set edgeWeight
        if edgeWeight is not 'donothaveone':
            g_df = g_df.replace(edgeWeight,"edgeWeight")
        # Set edgeTitle
        if edgeTitle is not 'donothaveone':
            g_df = g_df.replace(edgeTitle,"edgeTitle")
        files['graph'] = json.loads(g_df)

        # for Label
        u_df = {each[sourcefield]:each for each in df}.values()
        u_df = json.dumps(u_df)
        u_df = u_df.replace(sourcefield,'node')

        # Set PointTitle
        if pointTitle is not 'donothaveone':
            u_df = u_df.replace(pointTitle,'pointTitle')
        # Set PointLabel
        if pointLabel is not 'donothaveone':
            u_df = u_df.replace(pointLabel,'pointLabel')
        # Set the point size
        if pointSize is not 'donothaveone':
            u_df = u_df.replace(pointSize,'pointSize')
        # Set the point Color
        if pointColor is not 'donothaveone':
            u_df = u_df.replace(pointColor,'pointColor')

        files['labels'] = json.loads(u_df)

        return files


    # This function load data with pandas format as edge and node seperately
    # and returns it with Json format (Decoding)
    def loadPandas (self, edge, node, graphname,
                       sourcefield, desfield, nodefield, edgeTitle,
                       edgeLabel, edgeWeight,
                       pointTitle, pointLabel,
                       pointColor, pointSize,):
        import pandas as pd
        import json
        import random
        import string

        print 'Loading Pandas...'

        if graphname is 'Idontlikeanamelol':
            graphname = ''.join(random.choice(string.ascii_uppercase + string.digits)
                            for _ in range(10))

        files = {'name':graphname}
        files['bindings'] = {'idField':'node','destinationField':'dst','sourceField':'src'}
        files['type'] = 'edgelist'
        # for Edge
        edgejson = edge.to_json(orient = 'records')
        edgej = json.loads(edgejson)
        for i in range (0,len(edgej)):
            edgej[i]['src'] = edge[sourcefield][i]
            edgej[i]['dst'] = edge[desfield][i]

        edgej = json.dumps(edgej)
        # set edgeLabel
        if edgeLabel is not 'donothaveone':
            edgej = edgej.replace(edgeLabel,"edgeLabel")
        # Set edgeWeight
        if edgeWeight is not 'donothaveone':
            edgej = edgej.replace(edgeWeight,"edgeWeight")
        # Set edgeTitle
        if edgeTitle is not 'donothaveone':
            edgej = edgej.replace(edgeTitle,"edgeTitle")
        files['graph'] = json.loads(edgej)

        # for Label
        nodejson = node.to_json(orient = 'records')
        nodej = json.loads(nodejson)
        nodej = {each[nodefield]:each for each in nodej}.values()
        for i in range (0,len(nodej)):
            nodej[i]['node'] = node[nodefield][i]

        u_nodej = json.dumps(nodej)
        # Set PointTitle
        if pointTitle is not 'donothaveone':
            u_nodej = u_nodej.replace(pointTitle,'pointTitle')
        # Set PointLabel
        if pointLabel is not 'donothaveone':
            u_nodej = u_nodej.replace(pointLabel,'pointLabel')
        # Set the point size
        if pointSize is not 'donothaveone':
            u_nodej = u_nodej.replace(pointSize,'pointSize')
        # Set the point Color
        if pointColor is not 'donothaveone':
            u_nodej = u_nodej.replace(pointColor,'pointColor')

        files['labels'] = json.loads(u_nodej)

        return files


    # This function is to set server preference
    def settings(self,serveraddress = 'proxy'):
        global url
        global mid
        print 'Setting server...'
        if serveraddress is 'localhost':
            mid = 'localhost:3000'
            url = 'http://localhost:3000/etl'
        elif serveraddress is 'proxy':
            mid = 'proxy-staging.graphistry.com'
            url = 'http://proxy-staging.graphistry.com/etl'
        else:
            url = '-1';
            print 'Can not find this server'


    # This function is the prime function
    def loadFileSync (self, document, node = 'donothaveone', graphname = 'Idontlikeanamelol',
                  sourcefield = 'donothaveone', desfield = 'donothaveone',
                  nodefield = 'donothaveone', edgeTitle = 'donothaveone',
                  edgeLabel = 'donothaveone', edgeWeight = 'donothaveone',
                  pointTitle = 'donothaveone', pointLabel = 'donothaveone',
                  pointColor = 'donothaveone', pointSize = 'donothaveone'):
        import random
        import string
        import requests
        import json
        import yaml
    # check input types
        if node is 'donothaveone':
            if self.isJson(document):
                doc = json_ob # doc is with a valid json style
            elif self.isJsonPointer(document):
                doc = self.loadJsonPointer (document)
            elif self.isPandasPointer(document):
                doc = self.loadPandasPointer (document, graphname,
                                 sourcefield, desfield, edgeTitle,
                                 edgeLabel, edgeWeight, pointTitle,
                                 pointLabel, pointColor, pointSize)
            else:
                raise ValueError(" The File Type is Invalid ")
        else:
            # check if pandas format is valid
            if (document.empty or node.empty):
                raise ValueError("DataFrame is Empty!")
            else:
                doc = self.loadPandas(document,node, graphname,
                                 sourcefield, desfield, nodefield, edgeTitle,
                                 edgeLabel, edgeWeight, pointTitle,
                                 pointLabel, pointColor, pointSize)


        docj = json.dumps(doc)

        headers = {'content type' : 'json', 'content-Encoding':'gzip', 'Accept-Encoding': 'gzip', 'Accept': 'application/json', 'Content-Type': 'application/json'}
        try:
            r = requests.post(url,docj,headers = headers)
        except requests.exceptions.ConnectionError as e:
            raise ValueError("Connection Error")
        except requests.exceptions.HTTPError as e:
            raise ValueError("HTTPError:", e.message)

        else:
            name = yaml.safe_load(doc['name'])
            print 'Loading Finished!'
            print 'Getting iframe...'
            from IPython.core.display import HTML
            l = "http://" + mid + "/graph/graph.html?dataset=%s" %name
            print "url:", l
            #print docj
            return HTML('<iframe src="' + l + '" style="width:100%; height: 500px; border: 1px solid #DDD">')

