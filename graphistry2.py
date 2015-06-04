class Load (object):




    def isJson(self,document):
        import json
        global json_ob
        print 'Loading Json...'
        try:
            json_ob = json.loads(document)
        except ValueError, e:
            return False
        return True


    def isJsonPointer(self,document):
        if document[-4:] == 'json':
            return True
        else:
            return False


    def isPandasPointer(self,document):
        if document[-4:] == '.csv':
            return True
        else:
            return False


    def loadJsonPointer (self,filedir):
        import json
        print 'Loading Json...'
        with open(filedir) as data_file:
            files = json.load(data_file)
        return files


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

